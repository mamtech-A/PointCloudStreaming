"""StreamingSession: the unit of simulation.

Binds one User + EdgeNode + Server + ABRStrategy + AccessLink + BackhaulLink, and
owns the per-user state that wrongly lived flat on the legacy EdgeNode
(observed throughput history, quality history, last rep, cumulative time). One
`step()` per frame: build the ABR observation -> select a rep -> fetch across the
(unconstrained) backhaul -> TCP transfer across the access link -> push to the
buffer -> return a record for logging.
"""

from .abr import ABRState
from .manifest import coded_size_bytes, density_quality, DEFAULT_BITS_PER_POINT


class StreamingSession:
    def __init__(self, user, edge, server, abr, access_link, backhaul, bits_per_point=None):
        self.user = user
        self.edge = edge
        self.server = server
        self.abr = abr
        self.access_link = access_link
        self.backhaul = backhaul
        # Codec rate for the streamed (compressed) bitstream; None => package default.
        self.bits_per_point = DEFAULT_BITS_PER_POINT if bits_per_point is None else bits_per_point

        # Per-user ABR state (isolated — this is the multi-user fix).
        self.observed_throughput_history = []  # bps, completed transfers only
        self.quality_history = []
        self.last_rep_id = None
        self.cumulative_time_s = 0.0
        # Quality-aware QoE bookkeeping: chosen density per frame + the running
        # ladder endpoints over every rep this session has SEEN (equals the
        # manifest-wide endpoints once the full clip has streamed).
        self.chosen_densities = []
        self._density_lo = None
        self._density_hi = None

    def start(self):
        self.abr.reset()
        self.observed_throughput_history = []
        self.quality_history = []
        self.last_rep_id = None
        self.cumulative_time_s = 0.0
        self.chosen_densities = []
        self._density_lo = None
        self._density_hi = None
        self.access_link.establish()
        return self

    def _build_state(self, reps, frame_id):
        return ABRState(
            reps=reps,
            observed_throughput_history=tuple(self.observed_throughput_history),
            buffer_level_s=self.user.buffer.buffer_level_s,
            last_rep_id=self.last_rep_id,
            predicted_bandwidth_bps=None,
            frame_id=frame_id,
        )

    def step_segment(self, segment, first_frame_idx):
        """Process a SEGMENT (list of >= 1 manifest frames) with ONE ABR decision
        and ONE TCP transfer — DASH semantics: the per-request RTT cost is
        amortized over the whole segment. A 1-frame segment IS the legacy
        per-frame behavior (see step()). Returns a segment record with
        per-frame sub-records under 'frames'."""
        first = segment[0]
        reps = first['representations']

        # 1) ONE ABR decision from the observable state at segment start
        # (no oracle capacity); the chosen rep id applies to every frame.
        state = self._build_state(reps, first['id'])
        rep_id = self.abr.select(state)

        # Per-frame rep + coded size (densities differ slightly per frame).
        seg_reps, seg_sizes = [], []
        for fr in segment:
            rep_f = next((r for r in fr['representations'] if r['id'] == rep_id), None)
            seg_reps.append(rep_f)
            seg_sizes.append(coded_size_bytes(rep_f, self.bits_per_point) if rep_f else 0)
        data_bytes = sum(seg_sizes)

        # 2) Fetch from origin across the (unconstrained today) backhaul.
        self.edge.fetch_from_server(first['id'], rep_id)
        backhaul_time = self.backhaul.transfer(data_bytes).get('time_s', 0.0) if self.backhaul else 0.0

        # 3) ONE access-link transfer for the whole segment. The download starts
        # at the current cumulative download clock; the link maps elapsed time
        # within the transfer onto the trace's wall-clock axis, so capacity
        # varies DURING the download.
        download_start_s = self.cumulative_time_s
        metrics = self.access_link.transfer(data_bytes, first_frame_idx,
                                            start_time_s=download_start_s)
        access_time = metrics.get('time_s', 0.0)

        # 4) ONE achieved-throughput sample per segment (what the LSTM/DQN see).
        sent_bytes = metrics.get('sent_bytes', 0)
        if sent_bytes > 0 and access_time > 0:
            self.observed_throughput_history.append((sent_bytes * 8) / access_time)

        segment_time = access_time + backhaul_time
        self.cumulative_time_s += segment_time
        arrival_s = self.cumulative_time_s

        # 5) Push every frame of the segment to the buffer at the same arrival
        # time; the buffer's elapsed-time consumption runs once on the first add
        # (the rest of the batch sees elapsed = 0), so stall accounting is exact.
        frame_records = []
        for fr, rep_f, size_f in zip(segment, seg_reps, seg_sizes):
            buffer_result = self.user.receive_frame(fr['id'], rep_id, size_f,
                                                    segment_time, arrival_s)
            self.quality_history.append(rep_id)
            # Quality-aware QoE bookkeeping (per frame).
            for r in fr['representations']:
                d = r.get('density')
                if d:
                    if self._density_lo is None or d < self._density_lo:
                        self._density_lo = d
                    if self._density_hi is None or d > self._density_hi:
                        self._density_hi = d
            if rep_f is not None:
                self.chosen_densities.append(rep_f.get('density'))
            frame_records.append({'frame_id': fr['id'], 'rep': rep_f,
                                  'buffer_result': buffer_result})

        self.last_rep_id = rep_id
        return {
            'user_id': self.user.user_id,
            'rep_id': rep_id,
            'first_frame_id': first['id'],
            'capacity_bps': self.access_link.capacity_at_time(download_start_s),
            'segment_time_s': segment_time,
            'cumulative_time_s': self.cumulative_time_s,
            'data_bytes': data_bytes,
            'metrics': metrics,
            'frames': frame_records,
        }

    def step(self, frame, frame_idx):
        """Legacy per-frame step == a 1-frame segment; returns the legacy record
        shape (single code path, so segment_frames=1 is exactly the old model)."""
        rec = self.step_segment([frame], frame_idx)
        f0 = rec['frames'][0]
        return {
            'user_id': rec['user_id'],
            'frame_id': f0['frame_id'],
            'rep_id': rec['rep_id'],
            'rep': f0['rep'],
            'capacity_bps': rec['capacity_bps'],
            'frame_time_s': rec['segment_time_s'],
            'cumulative_time_s': rec['cumulative_time_s'],
            'metrics': rec['metrics'],
            'buffer_result': f0['buffer_result'],
        }

    def qoe(self):
        """LEGACY stall-only QoE (kept for continuity). Counts only stalls/drops,
        so a policy hiding at the lowest tier trivially maximizes it — always
        read it alongside qoe_quality()."""
        s = self.user.get_buffer_stats()
        return max(0, 100 - s['rebuffer_count'] * 10
                   - s['total_stall_time_s'] * 5 - s['frames_dropped'] * 2)

    def mean_quality(self):
        """Mean [0,1] log-density quality utility of the CHOSEN reps."""
        if not self.chosen_densities:
            return 0.0
        lo, hi = self._density_lo, self._density_hi
        qs = [density_quality(d, lo, hi) for d in self.chosen_densities]
        return sum(qs) / len(qs)

    def qoe_quality(self, w_stall=4.3, w_switch=1.0, w_slow=10.0):
        """Quality-aware QoE' (DQN_REPORT §6.5 + AMP term): RAW (unclipped) score

            100 * mean_quality - w_stall * total_stall_s - w_switch * sum|dq|
                               - w_slow * slowdown_integral

        Unlike the legacy stall-only qoe() it rewards delivered quality, so it
        discriminates between bottom-tier hiding and actual adaptation. The
        slowdown term charges adaptive playback: slowdown_integral = ∫(1−rate)dt,
        so 10 s played at the 0.9x floor costs w_slow*1.0 = 10 points (zero when
        AMP is off) — mild, per the subjective evidence that <=0.9x is
        imperceptible, but not free. Callers wanting a 0-100 scale should clamp
        with max(0, ...) — report both.
        """
        if not self.chosen_densities:
            return 0.0
        lo, hi = self._density_lo, self._density_hi
        qs = [density_quality(d, lo, hi) for d in self.chosen_densities]
        mean_q = sum(qs) / len(qs)
        switch_sum = sum(abs(qs[i] - qs[i - 1]) for i in range(1, len(qs)))
        s = self.user.get_buffer_stats()
        return (100.0 * mean_q - w_stall * s['total_stall_time_s']
                - w_switch * switch_sum
                - w_slow * s.get('slowdown_integral', 0.0))
