"""Bandwidth trace loading and a per-user capacity provider.

Traces are cleaned Irish 5G Download CSVs (bandwidth_5g/, G-NetTrack columns;
only `DL_bitrate` in kbps is read) loaded via `BandwidthTrace.from_file`.
(The legacy 4G `.log` parser was removed with the dataset migration —
see PLAN.md section 12; recover both from git history if ever needed.)

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
    def from_csv(cls, csv_path):
        return cls(load_5g_trace(csv_path), name=os.path.basename(csv_path))

    @classmethod
    def from_file(cls, path):
        """Extension-dispatched loader (currently `.csv` only)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            return cls.from_csv(path)
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
