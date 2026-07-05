"""Bandwidth trace loading and a per-user capacity provider.

Two on-disk formats are supported, dispatched by extension via
`BandwidthTrace.from_file`:
- `.csv`: cleaned Irish 5G Download traces (bandwidth_5g/, G-NetTrack columns;
  only `DL_bitrate` in kbps is read) -> `load_5g_trace`.
- `.log`: legacy 4G format -> `load_bandwidth_trace` (unchanged).

`BandwidthTrace` wraps a trace as the time-varying capacity of one access link:
`capacity_bps(idx)` returns the sample for frame `idx` (clamped to the last
sample past the end), matching the legacy behavior.
"""

import csv
import os


def load_5g_trace(csv_path, state_filter='D'):
    """Read a cleaned 5G CSV and return a list of bandwidth values (bps).

    Reads the `DL_bitrate` column (kbps). The State/non-numeric guards are
    no-ops on cleaned bandwidth_5g/ files (prepare_5g_traces.py already
    filtered) - they only protect against pointing at a raw dataset file.
    """
    bandwidths = []
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
    return bandwidths


def load_bandwidth_trace(log_path):
    """Read a .log file and return a list of bandwidth values (bps) per interval.

    Format: timestamp ms_since_start lat lon bytes_received ms_interval
    """
    bandwidths = []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            bytes_received = int(parts[4])
            ms_interval = int(parts[5])
            if ms_interval == 0:
                continue
            bps = (bytes_received * 8) / (ms_interval / 1000.0)
            bandwidths.append(bps)
    return bandwidths


class BandwidthTrace:
    """Time-varying access-link capacity, one sample consumed per frame.

    Single-user note: a user's trace is its access-link capacity directly. Once a
    shared-bottleneck model is added (see PLAN.md §2.9), the per-frame allocation
    will be applied on top of this provider instead of using it raw.
    """

    def __init__(self, samples_bps, name=None):
        self.samples = list(samples_bps)
        self.name = name

    @classmethod
    def from_log(cls, log_path):
        return cls(load_bandwidth_trace(log_path), name=os.path.basename(log_path))

    @classmethod
    def from_csv(cls, csv_path):
        return cls(load_5g_trace(csv_path), name=os.path.basename(csv_path))

    @classmethod
    def from_file(cls, path):
        """Extension-dispatched loader: `.csv` (5G) or `.log` (legacy 4G)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            return cls.from_csv(path)
        if ext == '.log':
            return cls.from_log(path)
        raise ValueError(f"unsupported trace extension '{ext}': {path}")

    def capacity_bps(self, frame_idx):
        """Capacity for `frame_idx`; clamps to the last sample past the trace end."""
        if not self.samples:
            return None
        if frame_idx < len(self.samples):
            return self.samples[frame_idx]
        return self.samples[-1]

    def __len__(self):
        return len(self.samples)
