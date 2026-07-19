"""Production guards for frozen gap-split training/evaluation windows."""

import json
import os
import random
import sys
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from train_dqn import build_epoch_episodes
from src.evaluation import aggregate_records, evaluate_strategy
from src.experiment_protocol import load_protocol
from src.network_model.abr import ABRStrategy
from src.network_model.finite_trace import (
    FiniteTraceWindow,
    TraceWindowExhausted,
)
from src.network_model.trace import BandwidthTrace
from src.rl.env import StreamingEnv
from src.trace_registry import load_trace_registry


TRACE_DIR = os.path.join(ROOT, "bandwidth_5g")
PROTOCOL_PATH = os.path.join(ROOT, "configs", "experiment_protocol.json")
REGISTRY_PATH = os.path.join(ROOT, "configs", "trace_window_registry.json")


def _registry():
    protocol = load_protocol(PROTOCOL_PATH, TRACE_DIR)
    return protocol, load_trace_registry(
        REGISTRY_PATH, protocol, TRACE_DIR, verify_hashes=True
    )


def _rep(rep_id, quality, coded_bytes):
    return {
        "id": rep_id,
        "quality": quality,
        "density": 1000 if quality == "high" else 100,
        "coded_bytes": float(coded_bytes),
        "bandwidth": 0,
        "size": "1K",
        "base_url": "dummy.bin",
    }


def _frames(count):
    return [{
        "id": index,
        "representations": [
            _rep(0, "high", 1_000_000),
            _rep(1, "vlow", 100),
        ],
    } for index in range(count)]


def test_checked_in_registry_has_expected_frozen_counts():
    protocol, registry = _registry()
    assert registry.registry_id == "223c00899f3e86f8"
    assert registry.data["registry_type"] == "gap_split_finite_high_supported_windows"
    assert registry.settings()["minimum_maximum_tier_headroom_s"] == 5.0
    assert len(registry.windows("train")) == 413
    assert len(registry.windows("validation")) == 12
    assert len(registry.windows("test")) == 12
    for split in ("validation", "test"):
        grouped = registry.validate_evaluation_count(
            split, 3, protocol["trace_split"][split]
        )
        assert set(grouped) == set(protocol["trace_split"][split])
        assert {len(rows) for rows in grouped.values()} == {3}


def test_training_epoch_samples_every_parent_equally():
    protocol, registry = _registry()
    paths = [os.path.join(TRACE_DIR, name)
             for name in protocol["trace_split"]["train"]]
    episodes, per_parent = build_epoch_episodes(
        paths, registry, random.Random(123), randomize=True
    )
    counts = Counter(os.path.basename(path) for path, _window in episodes)
    assert per_parent == 32
    assert len(episodes) == 416
    assert set(counts.values()) == {32}
    eligible_ids = {row["id"] for row in registry.windows("train")}
    assert all(window["id"] in eligible_ids for _path, window in episodes)


def test_all_candidate_registry_is_audit_only():
    protocol = load_protocol(PROTOCOL_PATH, TRACE_DIR)
    candidate_path = os.path.join(
        ROOT, "reports", "trace_window_registry_all_candidates.json"
    )
    try:
        load_trace_registry(
            candidate_path, protocol, TRACE_DIR, verify_hashes=True
        )
    except ValueError as exc:
        assert "candidate audit registry" in str(exc)
    else:
        raise AssertionError("candidate registry was accepted for experiments")
    candidate = load_trace_registry(
        candidate_path, protocol, TRACE_DIR, verify_hashes=True,
        allow_candidate_registry=True,
    )
    assert sum(len(candidate.windows(split)) for split in
               ("train", "validation", "test")) == 759


def test_registry_materializes_finite_block_local_windows():
    _protocol, registry = _registry()
    target = next(
        row for row in registry.windows("train")
        if row["trace"] == "driving_B_2020.02.13_15.02.01.csv"
        and row["block_id"] == "block-000"
    )
    finite = registry.materialize(target)
    assert isinstance(finite, FiniteTraceWindow)
    assert finite.duration_s == target["duration_s"]
    try:
        finite.capacity_at_time(finite.duration_s)
    except TraceWindowExhausted:
        pass
    else:
        raise AssertionError("registry window clamped beyond its block")


def test_trace_exhaustion_is_an_unscored_protocol_failure():
    env = StreamingEnv(
        {"longdress": _frames(10)},
        tcp_params={
            "rtt_ms": 1.0, "rtt_jitter_ms": 0.0, "loss_prob": 0.0,
            "cwnd_packets": 1000, "mss_bytes": 1460, "log_packets": False,
        },
        segment_frames=5,
    )
    finite = FiniteTraceWindow(BandwidthTrace(
        [1e3, 1e3], sample_times_s=[0.0, 1.0]
    ))
    env.reset(finite, sequence="longdress")
    try:
        env.step(0)
    except TraceWindowExhausted:
        pass
    else:
        raise AssertionError("high action unexpectedly completed")
    assert env.frame_idx == 0
    assert env.qoe() == 0.0
    assert not hasattr(env, "terminate_trace_exhausted")


def test_macro_aggregation_does_not_overweight_parent_case_count():
    def row(trace, qoe):
        return {
            "trace": trace, "sequence": "longdress", "eval_seed": 42,
            "reward": qoe, "qoe": qoe, "qoe_quality": qoe,
            "mean_quality": 0.0, "stall_s": 0.0, "rebuffer_events": 0,
            "quality_change": 0.0, "startup_s": 0.0,
            "request_pacing_s": 0.0,
        }
    result = aggregate_records([
        row("short.csv", 0.0),
        row("long.csv", 100.0), row("long.csv", 100.0), row("long.csv", 100.0),
    ])
    assert result["aggregation"] == "macro_parent_trace"
    assert result["qoe"] == 50.0
    assert result["case_level_summary"]["qoe"] == 75.0


def _run_all():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
