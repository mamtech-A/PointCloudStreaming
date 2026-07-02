"""Bandwidth trace loading and a per-user capacity provider.

`load_bandwidth_trace` is unchanged from the legacy module. `BandwidthTrace`
wraps a trace as the time-varying capacity of one access link: `capacity_bps(idx)`
returns the sample for frame `idx` (clamped to the last sample past the end),
matching the legacy `bandwidths[idx] if idx < len else bandwidths[-1]` behavior.
"""

import os


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

    def capacity_bps(self, frame_idx):
        """Capacity for `frame_idx`; clamps to the last sample past the trace end."""
        if not self.samples:
            return None
        if frame_idx < len(self.samples):
            return self.samples[frame_idx]
        return self.samples[-1]

    def __len__(self):
        return len(self.samples)
