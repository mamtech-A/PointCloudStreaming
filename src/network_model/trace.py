"""Bandwidth trace loading and a per-user capacity provider.

Traces are cleaned Irish 5G Download CSVs (bandwidth_5g/, G-NetTrack columns;
`DL_bitrate` in kbps + `Timestamp`) loaded via `BandwidthTrace.from_file`.
(The legacy 4G `.log` parser was removed with the dataset migration —
see PLAN.md section 12; recover both from git history if ever needed.)

`BandwidthTrace` wraps a trace as the time-varying capacity of one access link.
Two lookups are exposed:

- `capacity_at_time(t_s)` — capacity at `t_s` seconds of download wall-clock,
  driven by the CSV `Timestamp` deltas. This is what the simulator uses: as a
  download progresses, elapsed time advances through the trace, so a long
  download sees several bandwidth samples instead of one frozen value.
- `capacity_bps(idx)` — legacy positional lookup (sample per frame index),
  retained for backward compatibility.

Time model (piecewise-constant, hold-previous):
- sample `i` is in effect over `[start_i, start_{i+1})`;
- a run of k rows sharing one timestamp T splits `[T, T_next)` evenly;
- a gap (`T_next - T > 1 s`) simply means the sample HOLDS until the next one
  (at t=1s -> 2 Mbps and t=3s -> 3 Mbps, the missing t=2s second is 2 Mbps);
- the final sample (no successor) gets a default 1.0 s width;
- lookups past the end clamp to the last sample;
- `time_to_transmit(t, bits)` integrates the piecewise-constant capacity from
  `t` so delivery time is exact even when the rate changes mid-transfer (a
  transfer starting in a fade completes when the trace recovers — real-link
  semantics, used by the TCP model for serialization).
"""

import bisect
import csv
import os
from datetime import datetime

TIMESTAMP_FMT = '%Y.%m.%d_%H.%M.%S'
DEFAULT_SAMPLE_PERIOD_S = 1.0


def load_5g_trace_with_time(csv_path, state_filter='D'):
    """Read a cleaned 5G CSV -> (bandwidths_bps, epoch_times_s or None).

    Reads `DL_bitrate` (kbps) and `Timestamp`. The State/non-numeric guards are
    no-ops on cleaned bandwidth_5g/ files (prepare_5g_traces.py already
    filtered) - they only protect against pointing at a raw dataset file.
    `epoch_times_s` is None when any kept row has no parseable timestamp (the
    caller then falls back to a uniform 1 s cadence).
    """
    bandwidths, times = [], []
    have_all_times = True
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if state_filter and row.get('State', state_filter) != state_filter:
                continue
            try:
                kbps = float(row['DL_bitrate'])
            except (KeyError, TypeError, ValueError):
                continue
            if kbps <= 0:
                continue
            bandwidths.append(kbps * 1000.0)
            try:
                times.append(datetime.strptime(row['Timestamp'], TIMESTAMP_FMT).timestamp())
            except (KeyError, TypeError, ValueError):
                times.append(None)
                have_all_times = False
    return bandwidths, (times if (have_all_times and times) else None)


def load_5g_trace(csv_path, state_filter='D'):
    """Back-compat wrapper: bandwidth values (bps) only."""
    return load_5g_trace_with_time(csv_path, state_filter)[0]


class BandwidthTrace:
    """Time-varying access-link capacity with a real (timestamp-derived) time axis.

    Single-user note: a user's trace is its access-link capacity directly. Once a
    shared-bottleneck model is added (see PLAN.md §2.9), the per-frame allocation
    will be applied on top of this provider instead of using it raw.
    """

    def __init__(self, samples_bps, name=None, sample_times_s=None):
        self.samples = list(samples_bps)
        self.name = name
        n = len(self.samples)
        # Rebased (t0 -> 0), monotonic non-decreasing raw times; kept so slices
        # can carry per-sample timing. Falls back to a uniform 1 s cadence when
        # no timestamps are available (e.g. synthetic sample lists).
        if sample_times_s is not None and len(sample_times_s) == n and n > 0:
            t0 = sample_times_s[0]
            raw, prev = [], 0.0
            for t in sample_times_s:
                x = float(t) - t0
                if x < prev:  # clock went backwards: clamp to monotonic
                    x = prev
                raw.append(x)
                prev = x
            self._raw_times = raw
        else:
            self._raw_times = [i * DEFAULT_SAMPLE_PERIOD_S for i in range(n)]
        self._start_s, self._duration_s = self._build_starts(self._raw_times)

    @staticmethod
    def _build_starts(raw):
        """Strictly increasing per-sample start times from possibly-duplicate
        raw times. A run of k rows at time T spans [T, T_next) split evenly;
        the final run has no successor -> DEFAULT_SAMPLE_PERIOD_S per row."""
        n = len(raw)
        starts = [0.0] * n
        i = 0
        last_width = DEFAULT_SAMPLE_PERIOD_S
        while i < n:
            j = i
            while j + 1 < n and raw[j + 1] == raw[i]:
                j += 1
            k = j - i + 1
            if j + 1 < n:
                width = (raw[j + 1] - raw[i]) / k
            else:
                width = DEFAULT_SAMPLE_PERIOD_S
                last_width = width
            for m in range(k):
                starts[i + m] = raw[i] + m * width
            i = j + 1
        duration = (starts[-1] + last_width) if n else 0.0
        return starts, duration

    @classmethod
    def from_csv(cls, csv_path):
        samples, times = load_5g_trace_with_time(csv_path)
        return cls(samples, name=os.path.basename(csv_path), sample_times_s=times)

    @classmethod
    def from_file(cls, path):
        """Extension-dispatched loader (currently `.csv` only)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            return cls.from_csv(path)
        raise ValueError(f"unsupported trace extension '{ext}': {path}")

    # --- time-based lookup (the simulator's path) ---

    def capacity_at_time(self, t_s):
        """Capacity in effect at download wall-clock `t_s` (piecewise-constant,
        hold-previous across gaps; clamps to first/last sample outside range)."""
        if not self.samples:
            return None
        idx = bisect.bisect_right(self._start_s, t_s) - 1
        if idx < 0:
            idx = 0
        return self.samples[idx]

    def time_to_transmit(self, t_start_s, bits):
        """Seconds to deliver `bits` starting at wall-clock `t_start_s`, with the
        link running at the trace's INSTANTANEOUS rate (exact integral over the
        piecewise-constant capacity) — a transfer that starts inside a fade
        finishes as soon as the trace recovers, like a real link, instead of
        being charged the fade rate for its whole duration.

        The final sample extends indefinitely (same clamp as capacity_at_time).
        Rates are floored at 1e3 bps purely to keep synthetic all-zero traces
        finite (cleaned bandwidth_5g traces are all > 0).
        """
        if not self.samples or bits <= 0:
            return 0.0
        n = len(self.samples)
        idx = bisect.bisect_right(self._start_s, t_start_s) - 1
        if idx < 0:
            idx = 0
        t = max(t_start_s, 0.0)
        remaining = float(bits)
        elapsed = 0.0
        while True:
            rate = max(float(self.samples[idx]), 1e3)
            if idx + 1 < n:
                window_end = self._start_s[idx + 1]
                window_left = window_end - t
                deliverable = rate * window_left
                if deliverable >= remaining:
                    return elapsed + remaining / rate
                remaining -= deliverable
                elapsed += window_left
                t = window_end
                idx += 1
            else:
                # Last sample holds forever.
                return elapsed + remaining / rate

    def start_time_of_sample(self, idx):
        """Wall-clock start time (s) of sample `idx` (clamped to valid range)."""
        if not self._start_s:
            return 0.0
        idx = max(0, min(int(idx), len(self._start_s) - 1))
        return self._start_s[idx]

    @property
    def duration_s(self):
        """Total wall-clock span covered by the trace (end of last sample)."""
        return self._duration_s

    def slice_from(self, start_idx):
        """New BandwidthTrace starting at sample `start_idx`, timing preserved
        and rebased to 0 (used for random-offset / coverage-tiled episodes)."""
        n = len(self.samples)
        start_idx = max(0, min(int(start_idx), max(0, n - 1)))
        return BandwidthTrace(self.samples[start_idx:], name=self.name,
                              sample_times_s=self._raw_times[start_idx:] or None)

    # --- legacy positional lookup (kept for back-compat) ---

    def capacity_bps(self, frame_idx):
        """Capacity for `frame_idx`; clamps to the last sample past the trace end."""
        if not self.samples:
            return None
        if frame_idx < len(self.samples):
            return self.samples[frame_idx]
        return self.samples[-1]

    def __len__(self):
        return len(self.samples)
