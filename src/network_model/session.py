"""StreamingSession: the unit of simulation.

Binds one User + EdgeNode + Server + ABRStrategy + AccessLink + BackhaulLink, and
owns the per-user state that wrongly lived flat on the legacy EdgeNode
(observed throughput history, quality history, last rep, cumulative time). One
segment step paces for buffer headroom, builds the ABR observation, selects a
representation, fetches across the unconstrained backhaul, transfers across the
access link, and atomically admits the segment to the playback buffer.
"""

from .abr import ABRState
from .finite_trace import FiniteTraceWindow, TraceWindowExhausted, TIME_EPSILON_S
from .manifest import coded_size_bytes, tier_quality, DEFAULT_BITS_PER_POINT


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
        # QoE bookkeeping. Densities stay available for compatibility and state
        # analysis; reward/QoE use only the fixed utility of each tier.
        self.chosen_densities = []
        self.chosen_qualities = []
        self.chosen_quality_segments = []
        self.total_request_pacing_s = 0.0
        self._pending_request = None

    def start(self):
        self.abr.reset()
        self.observed_throughput_history = []
        self.quality_history = []
        self.last_rep_id = None
        self.cumulative_time_s = 0.0
        self.chosen_densities = []
        self.chosen_qualities = []
        self.chosen_quality_segments = []
        self.total_request_pacing_s = 0.0
        self._pending_request = None
        self.access_link.establish()
        return self

    def _pace_request(self, segment_frames):
        """Wait until the complete next segment can fit in the client buffer."""
        buffer = self.user.buffer
        segment_duration_s = float(segment_frames) / buffer.target_fps
        if segment_duration_s > buffer.buffer_capacity_s + 1e-9:
            raise ValueError(
                f"segment duration {segment_duration_s:.6f}s exceeds buffer "
                f"capacity {buffer.buffer_capacity_s:.6f}s"
            )
        target_before_request_s = max(
            0.0, buffer.buffer_capacity_s - segment_duration_s
        )
        pacing_s = max(0.0, buffer.buffer_level_s - target_before_request_s)
        if pacing_s <= 1e-12:
            return 0.0
        if not buffer.is_playing or buffer.is_rebuffering:
            state = "startup" if not buffer.playback_started else "rebuffering"
            raise ValueError(
                f"cannot admit a {segment_duration_s:.6f}s segment during "
                f"{state}: buffered={buffer.buffer_level_s:.6f}s, "
                f"capacity={buffer.buffer_capacity_s:.6f}s, "
                f"resume_threshold={buffer.min_buffer_s:.6f}s; choose a "
                "smaller segment or a compatible buffer threshold"
            )
        paced_request_time_s = self.cumulative_time_s + pacing_s
        trace = self.access_link.trace
        if (isinstance(trace, FiniteTraceWindow)
                and paced_request_time_s >= trace.duration_s - TIME_EPSILON_S):
            raise TraceWindowExhausted(
                trace.name, paced_request_time_s, trace.duration_s
            )
        self.cumulative_time_s = paced_request_time_s
        buffer.advance_playback(
            pacing_s, self.cumulative_time_s, event='request_pacing'
        )
        self.total_request_pacing_s += pacing_s
        return pacing_s

    def prepare_request(self, segment_frames, first_frame_idx):
        """Pace once before an external controller observes its next state."""
        key = (int(first_frame_idx), int(segment_frames))
        if self._pending_request is not None:
            if self._pending_request['key'] != key:
                raise RuntimeError(
                    f"prepared request {self._pending_request['key']} does not "
                    f"match requested segment {key}"
                )
            return self._pending_request['pacing_s']
        pacing_s = self._pace_request(segment_frames)
        self._pending_request = {'key': key, 'pacing_s': pacing_s}
        return pacing_s

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

        # 1) Pace the request so even an instantaneous transfer cannot overflow
        # the finite playback buffer. Playback and the trace clock both advance
        # during this intentional wait. The ABR observes the post-wait buffer.
        request_key = (int(first_frame_idx), len(segment))
        if self._pending_request is not None:
            if self._pending_request['key'] != request_key:
                raise RuntimeError(
                    f"prepared request {self._pending_request['key']} does not "
                    f"match delivered segment {request_key}"
                )
            pacing_s = self._pending_request['pacing_s']
            self._pending_request = None
        else:
            pacing_s = self._pace_request(len(segment))

        # 2) ONE ABR decision from the observable state at request time (no
        # oracle capacity); the chosen rep id applies to every frame.
        state = self._build_state(reps, first['id'])
        rep_id = self.abr.select(state)

        # Per-frame rep + coded size (densities differ slightly per frame).
        seg_reps, seg_sizes = [], []
        for fr in segment:
            rep_f = next((r for r in fr['representations'] if r['id'] == rep_id), None)
            seg_reps.append(rep_f)
            seg_sizes.append(coded_size_bytes(rep_f, self.bits_per_point) if rep_f else 0)
        data_bytes = sum(seg_sizes)

        # 3) Fetch from origin across the (unconstrained today) backhaul.
        self.edge.fetch_from_server(first['id'], rep_id)
        backhaul_time = self.backhaul.transfer(data_bytes).get('time_s', 0.0) if self.backhaul else 0.0

        # 4) ONE access-link transfer for the whole segment. The download starts
        # at the current cumulative download clock; the link maps elapsed time
        # within the transfer onto the trace's wall-clock axis, so capacity
        # varies DURING the download.
        download_start_s = self.cumulative_time_s
        metrics = self.access_link.transfer(data_bytes, first_frame_idx,
                                            start_time_s=download_start_s)
        access_time = metrics.get('time_s', 0.0)

        # 5) ONE achieved-throughput sample per segment (what the LSTM/DQN see).
        # Intentional request-pacing time is excluded from this network sample.
        sent_bytes = metrics.get('sent_bytes', 0)
        if sent_bytes > 0 and access_time > 0:
            self.observed_throughput_history.append((sent_bytes * 8) / access_time)

        segment_time = access_time + backhaul_time
        self.cumulative_time_s += segment_time
        arrival_s = self.cumulative_time_s

        # 6) Push every frame of the segment to the buffer at the same arrival
        # time; the buffer's elapsed-time consumption runs once on the first add
        # (the rest of the batch sees elapsed = 0), so stall accounting is exact.
        frame_records = []
        segment_qualities = []
        for fr, rep_f, size_f in zip(segment, seg_reps, seg_sizes):
            buffer_result = self.user.receive_frame(fr['id'], rep_id, size_f,
                                                    segment_time, arrival_s)
            self.quality_history.append(rep_id)
            if rep_f is not None:
                self.chosen_densities.append(rep_f.get('density'))
                q = tier_quality(rep_f)
                self.chosen_qualities.append(q)
                segment_qualities.append(q)
            frame_records.append({'frame_id': fr['id'], 'rep': rep_f,
                                  'buffer_result': buffer_result})

        self.chosen_quality_segments.append(segment_qualities)

        self.last_rep_id = rep_id
        return {
            'user_id': self.user.user_id,
            'rep_id': rep_id,
            'first_frame_id': first['id'],
            'capacity_bps': self.access_link.capacity_at_time(download_start_s),
            'request_pacing_s': pacing_s,
            'request_start_s': download_start_s,
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
            'request_pacing_s': rec['request_pacing_s'],
            'request_start_s': rec['request_start_s'],
            'frame_time_s': rec['segment_time_s'],
            'cumulative_time_s': rec['cumulative_time_s'],
            'metrics': rec['metrics'],
            'buffer_result': f0['buffer_result'],
        }

    def qoe(self):
        """The one canonical raw QoE (compatibility entry point)."""
        return self.qoe_quality()

    def mean_quality(self):
        """Mean fixed [0,1] log-bitrate utility of the chosen tiers."""
        if not self.chosen_qualities:
            return 0.0
        return sum(self.chosen_qualities) / len(self.chosen_qualities)

    def qoe_quality_terms(self, w_stall=4.3, w_rebuffer=2.0, w_switch=1.0,
                          w_startup=1.0):
        """Canonical five-term QoE breakdown for direct simulator sessions:

            QoE″ = 100·mean_q − w_stall·total_stall_s − w_switch·Σ|dq|
                   − w_rebuffer·rebuffer_count − w_startup·startup_delay_s

        Startup is the wait before playback first begins. Stall duration and
        rebuffer events accrue only after playback has begun, so the terms are
        disjoint. Quality change is measured between ABR segment means, matching
        the DQN reward. Request pacing prevents buffer-overflow frame loss.

        Returns a dict of the SIGNED terms plus 'total' (raw, unclipped).
        """
        if not self.chosen_qualities:
            return {'quality': 0.0, 'stall_duration': 0.0, 'rebuffering': 0.0,
                    'quality_change': 0.0, 'startup_delay': 0.0, 'total': 0.0}
        qs = self.chosen_qualities
        mean_q = sum(qs) / len(qs)
        segment_qs = []
        for values in self.chosen_quality_segments:
            if values:
                segment_qs.append(sum(values) / len(values))
        switch_sum = sum(abs(segment_qs[i] - segment_qs[i - 1])
                         for i in range(1, len(segment_qs)))
        s = self.user.get_buffer_stats()
        terms = {
            'quality': 100.0 * mean_q,
            'stall_duration': -w_stall * s['total_stall_time_s'],
            'rebuffering': -w_rebuffer * s.get('rebuffer_count', 0),
            'quality_change': -w_switch * switch_sum,
            'startup_delay': -w_startup * s.get('startup_delay_s', 0.0),
        }
        terms['total'] = sum(terms.values())
        return terms

    def qoe_quality(self, w_stall=4.3, w_rebuffer=2.0, w_switch=1.0,
                    w_startup=1.0):
        """Canonical raw QoE total; ``qoe_quality`` is a compatibility name."""
        return self.qoe_quality_terms(
            w_stall, w_rebuffer, w_switch, w_startup
        )['total']
