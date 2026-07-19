"""Finite measured-throughput views for reproducible streaming windows."""

from __future__ import annotations

import bisect

from .trace import BandwidthTrace


TIME_EPSILON_S = 1e-9


class TraceWindowExhausted(RuntimeError):
    """Raised when a transfer needs throughput beyond a measured boundary."""

    def __init__(self, trace_name, at_time_s, trace_end_s, remaining_bits=None):
        self.trace_name = trace_name
        self.at_time_s = float(at_time_s)
        self.trace_end_s = float(trace_end_s)
        self.remaining_bits = (None if remaining_bits is None
                               else max(0.0, float(remaining_bits)))
        detail = ""
        if self.remaining_bits is not None:
            detail = f"; {self.remaining_bits:.0f} bits remain in the TCP round"
        super().__init__(
            f"trace window {trace_name!r} ends at {trace_end_s:.6f}s "
            f"before the transfer can finish{detail}"
        )


def sample_start_times(trace):
    """Return the simulator's strictly increasing per-sample start times."""
    return [trace.start_time_of_sample(index) for index in range(len(trace))]


def measured_end_time_s(trace):
    """Return the last measured timestamp, excluding any inferred final tail."""
    explicit_end = getattr(trace, "_audit_measured_end_s", None)
    if explicit_end is not None:
        return float(explicit_end)
    raw_times = getattr(trace, "_raw_times", None)
    return float(raw_times[-1]) if raw_times else 0.0


class FiniteTraceWindow:
    """Read-only trace view that never repeats its terminal sample."""

    def __init__(self, trace):
        if not isinstance(trace, BandwidthTrace):
            raise TypeError("FiniteTraceWindow requires a BandwidthTrace")
        if not trace.samples:
            raise ValueError("cannot use an empty bandwidth trace")
        self._trace = trace
        self.samples = trace.samples
        self.name = trace.name
        self._starts = sample_start_times(trace)

    @property
    def duration_s(self):
        return measured_end_time_s(self._trace)

    def capacity_at_time(self, t_s):
        t_s = float(t_s)
        if t_s >= self.duration_s - TIME_EPSILON_S:
            raise TraceWindowExhausted(
                self.name, max(t_s, self.duration_s), self.duration_s
            )
        return self._trace.capacity_at_time(t_s)

    def time_to_transmit(self, t_start_s, bits):
        """Integrate only over measured intervals; never clamp after the end."""
        bits = float(bits)
        if bits <= 0:
            return 0.0
        t = max(0.0, float(t_start_s))
        if t >= self.duration_s - TIME_EPSILON_S:
            raise TraceWindowExhausted(
                self.name, max(t, self.duration_s), self.duration_s, bits
            )

        index = max(0, bisect.bisect_right(self._starts, t) - 1)
        remaining = bits
        elapsed = 0.0
        while index < len(self.samples):
            interval_end = (
                self._starts[index + 1]
                if index + 1 < len(self.samples) else self.duration_s
            )
            interval_s = max(0.0, interval_end - t)
            rate_bps = max(float(self.samples[index]), 1e3)
            deliverable = rate_bps * interval_s
            if deliverable + 1e-6 >= remaining:
                return elapsed + remaining / rate_bps
            remaining -= deliverable
            elapsed += interval_s
            t = interval_end
            index += 1

        raise TraceWindowExhausted(
            self.name, self.duration_s, self.duration_s, remaining
        )

    def __len__(self):
        return len(self.samples)
