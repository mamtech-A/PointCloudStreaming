"""Unit tests for the observable-only adaptive baselines."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.network_model.abr import ABRState, BandwidthABR, BufferBasedABR, MPCABR


def _reps():
    # IDs run high-quality -> low-quality, matching the G-PCC manifests.
    return [
        {"id": 0, "bandwidth": 90e6, "density": 1_000_000, "quality": "high"},
        {"id": 1, "bandwidth": 45e6, "density": 500_000, "quality": "medhigh"},
        {"id": 2, "bandwidth": 20e6, "density": 200_000, "quality": "med"},
        {"id": 3, "bandwidth": 8e6, "density": 80_000, "quality": "medlow"},
        {"id": 4, "bandwidth": 3e6, "density": 30_000, "quality": "low"},
        {"id": 5, "bandwidth": 1e6, "density": 10_000, "quality": "vlow"},
    ]


def test_recent_throughput_ignores_stale_attach_sample():
    state = ABRState(
        _reps(),
        observed_throughput_history=(0.1e6, 50e6, 50e6, 50e6, 50e6, 50e6),
    )
    recent = BandwidthABR(safety_factor=0.8, history_window=5).select(state)
    all_history = BandwidthABR(safety_factor=0.8, history_window=None).select(state)
    assert recent == 2
    assert all_history == 5


def test_buffer_policy_moves_monotonically_up_the_ladder():
    policy = BufferBasedABR(reservoir_s=1.0, cushion_s=3.0)
    ids = [policy.select(ABRState(_reps(), buffer_level_s=level))
           for level in (0.0, 1.5, 2.5, 4.5)]
    # Smaller IDs are higher quality in this manifest.
    assert ids == sorted(ids, reverse=True)
    assert ids[0] == 5 and ids[-1] == 0


def test_mpc_uses_low_tier_without_history_and_higher_tier_on_fast_link():
    policy = MPCABR(horizon=2, segment_frames=8)
    cold = ABRState(_reps(), buffer_level_s=0.0)
    warm = ABRState(
        _reps(), observed_throughput_history=(120e6,) * 5, buffer_level_s=4.0,
        last_rep_id=2,
    )
    assert policy.select(cold) == 5
    assert policy.select(warm) < 5


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
