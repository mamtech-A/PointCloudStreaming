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

    def step(self, frame, frame_idx):
        """Process one frame for this session; return a record dict for logging."""
        reps = frame['representations']
        frame_id = frame['id']

        # 1) ABR decision from the observable state only (no oracle capacity).
        state = self._build_state(reps, frame_id)
        rep_id = self.abr.select(state)
        rep = next((r for r in reps if r['id'] == rep_id), None)
        # Streamed bytes = compressed (coded) size, not the raw .ply size.
        data_bytes = coded_size_bytes(rep, self.bits_per_point) if rep else 0

        # 2) Fetch from origin across the (unconstrained today) backhaul.
        self.edge.fetch_from_server(frame_id, rep_id)
        backhaul_time = self.backhaul.transfer(data_bytes).get('time_s', 0.0) if self.backhaul else 0.0

        # 3) Deliver to the user over the access link (TCP). The download starts
        # at the current cumulative download clock; the link maps elapsed time
        # within the transfer onto the trace's wall-clock axis, so capacity
        # varies DURING the download.
        download_start_s = self.cumulative_time_s
        metrics = self.access_link.transfer(data_bytes, frame_idx,
                                            start_time_s=download_start_s)
        access_time = metrics.get('time_s', 0.0)

        # 4) Record achieved throughput (access only; matches legacy _record_observed_throughput).
        sent_bytes = metrics.get('sent_bytes', 0)
        if sent_bytes > 0 and access_time > 0:
            self.observed_throughput_history.append((sent_bytes * 8) / access_time)

        frame_time = access_time + backhaul_time
        self.cumulative_time_s += frame_time

        # 5) Push to the playback buffer.
        buffer_result = self.user.receive_frame(
            frame_id, rep_id, data_bytes, frame_time, self.cumulative_time_s
        )

        self.last_rep_id = rep_id
        self.quality_history.append(rep_id)

        # Quality-aware QoE bookkeeping.
        for r in reps:
            d = r.get('density')
            if d:
                if self._density_lo is None or d < self._density_lo:
                    self._density_lo = d
                if self._density_hi is None or d > self._density_hi:
                    self._density_hi = d
        if rep is not None:
            self.chosen_densities.append(rep.get('density'))

        return {
            'user_id': self.user.user_id,
            'frame_id': frame_id,
            'rep_id': rep_id,
            'rep': rep,
            'capacity_bps': self.access_link.capacity_at_time(download_start_s),
            'frame_time_s': frame_time,
            'cumulative_time_s': self.cumulative_time_s,
            'metrics': metrics,
            'buffer_result': buffer_result,
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

    def qoe_quality(self, w_stall=4.3, w_switch=1.0):
        """Quality-aware QoE' (DQN_REPORT §6.5): RAW (unclipped) score

            100 * mean_quality - w_stall * total_stall_s - w_switch * sum|dq|

        Unlike the legacy stall-only qoe() it rewards delivered quality, so it
        discriminates between bottom-tier hiding and actual adaptation. Callers
        wanting a 0-100 scale should clamp with max(0, ...) — report both.
        """
        if not self.chosen_densities:
            return 0.0
        lo, hi = self._density_lo, self._density_hi
        qs = [density_quality(d, lo, hi) for d in self.chosen_densities]
        mean_q = sum(qs) / len(qs)
        switch_sum = sum(abs(qs[i] - qs[i - 1]) for i in range(1, len(qs)))
        s = self.user.get_buffer_stats()
        return (100.0 * mean_q - w_stall * s['total_stall_time_s']
                - w_switch * switch_sum)
