"""Shared state-feature builder + reward quality function.

CRITICAL: this is the SINGLE place state vectors are constructed, used by BOTH the
training environment (`env.py`) and inference (`network_model.abr.DQNABR`). The
agent persists `feature_spec` + `norm` in its checkpoint so train- and inference-time
vectors are identical (no skew).

Each feature block is individually toggleable via `feature_spec` for ablation.
"""

import math
import numpy as np

from ..network_model.manifest import (
    coded_bitrate_bps, density_quality, manifest_quality_endpoints,
)

DEFAULT_FEATURE_SPEC = [
    'lstm_pred', 'buffer', 'last_rep_onehot',
    'tput_last', 'tput_mean', 'tput_std', 'rep_bitrates', 'rep_densities',
]

DEFAULT_NORM = {
    # Scaled to the Irish 5G Download traces (bandwidth_5g/): bw_mbps ~ p99.5 of
    # DL_bitrate (307 Mbps), tput_std_mbps ~ p99 of the rolling-5 std (123 Mbps).
    # Changing these invalidates any previously trained DQN checkpoint.
    'bw_mbps': 300.0,        # divide Mbps values by this
    'tput_std_mbps': 125.0,
    'buffer_cap_s': 5.0,
    'density_log': 7.0,      # log10(density) / this
    'bits_per_point': 3.0,   # codec rate for the coded-bitrate feature
    'fps': 30.0,
    'num_reps': 3,
    'hist_len': 5,
}

# Quality utility: the log-density normalization lives in
# network_model.manifest (density_quality / manifest_quality_endpoints) so the
# quality-aware QoE and the RL reward share ONE definition. These wrappers keep
# the historical rl.features API.


def quality_endpoints(manifest_frames):
    """(min_density, max_density) across all reps of all frames of a manifest."""
    return manifest_quality_endpoints(manifest_frames)


def quality(rep, low_density=None, high_density=None):
    """Normalized quality utility in [0, 1] from a representation's density."""
    return density_quality(rep.get('density'), low_density, high_density)


_DIMS = {
    'lstm_pred': lambda nr: 1, 'buffer': lambda nr: 1, 'last_rep_onehot': lambda nr: nr,
    'tput_last': lambda nr: 1, 'tput_mean': lambda nr: 1, 'tput_std': lambda nr: 1,
    'rep_bitrates': lambda nr: nr, 'rep_densities': lambda nr: nr,
}


def state_dim(feature_spec, num_reps=3, hist_len=5):
    return sum(_DIMS[f](num_reps) for f in feature_spec)


def build_state(feature_spec, state, predicted_bandwidth_bps=None, norm=None):
    """Build the normalized feature vector for one ABRState.

    `state` is a network_model.abr.ABRState. `predicted_bandwidth_bps` is the LSTM
    feature (falls back to the harmonic mean of observed throughput if None).
    """
    n = {**DEFAULT_NORM, **(norm or {})}
    num_reps = int(n['num_reps'])
    hist_len = int(n['hist_len'])

    reps_sorted = sorted(state.reps, key=lambda r: r['id'])
    hist_mbps = [bw / 1e6 for bw in state.observed_throughput_history]
    recent = hist_mbps[-hist_len:] if hist_mbps else []

    feats = []
    for f in feature_spec:
        if f == 'lstm_pred':
            if predicted_bandwidth_bps is not None:
                v = predicted_bandwidth_bps / 1e6
            else:
                v = state.harmonic_mean_throughput() / 1e6
            feats.append(min(1.5, max(0.0, v / n['bw_mbps'])))
        elif f == 'buffer':
            feats.append(state.buffer_level_s / n['buffer_cap_s'])
        elif f == 'last_rep_onehot':
            oh = [0.0] * num_reps
            if state.last_rep_id is not None and 0 <= state.last_rep_id < num_reps:
                oh[state.last_rep_id] = 1.0
            feats.extend(oh)
        elif f == 'tput_last':
            feats.append((recent[-1] if recent else 0.0) / n['bw_mbps'])
        elif f == 'tput_mean':
            feats.append((sum(recent) / len(recent) if recent else 0.0) / n['bw_mbps'])
        elif f == 'tput_std':
            if len(recent) >= 2:
                m = sum(recent) / len(recent)
                std = (sum((x - m) ** 2 for x in recent) / len(recent)) ** 0.5
            else:
                std = 0.0
            feats.append(min(1.0, std / n['tput_std_mbps']))
        elif f == 'rep_bitrates':
            bpp = n.get('bits_per_point', 3.0)
            fps = n.get('fps', 30.0)
            for i in range(num_reps):
                if i < len(reps_sorted):
                    br_mbps = coded_bitrate_bps(reps_sorted[i], bpp, fps) / 1e6
                    feats.append(min(3.0, br_mbps / n['bw_mbps']))
                else:
                    feats.append(0.0)
        elif f == 'rep_densities':
            for i in range(num_reps):
                if i < len(reps_sorted):
                    d = max(1.0, float(reps_sorted[i]['density']))
                    feats.append(math.log10(d) / n['density_log'])
                else:
                    feats.append(0.0)
        else:
            raise ValueError(f"unknown feature '{f}'")
    return np.asarray(feats, dtype=np.float32)
