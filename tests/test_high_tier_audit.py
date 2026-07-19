"""Unit and evidence tests for maximum-demand frozen-window auditing."""

import hashlib
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.high_tier_audit import (
    SUPPORTED,
    UNSUPPORTED,
    audit_high_case,
    high_action,
    summarize_high_audit,
)
from src.network_model.finite_trace import FiniteTraceWindow
from src.network_model.trace import BandwidthTrace
from src.rl.env import StreamingEnv


def _rep(rep_id, quality, coded_bytes):
    return {
        "id": rep_id, "quality": quality, "density": 1,
        "coded_bytes": coded_bytes, "bandwidth": 0,
        "size": "1K", "base_url": "dummy.bin",
    }


def _frames(count):
    return [{
        "id": index,
        "representations": [
            _rep(0, "high", 1_000_000),
            _rep(1, "vlow", 100),
        ],
    } for index in range(count)]


def _env(count=10):
    return StreamingEnv(
        {"longdress": _frames(count)},
        tcp_params={
            "rtt_ms": 1.0, "rtt_jitter_ms": 0.0, "loss_prob": 0.0,
            "cwnd_packets": 1000, "mss_bytes": 1460, "log_packets": False,
        },
        segment_frames=5,
    )


def test_high_action_uses_maximum_tier():
    env = _env()
    env.reset(FiniteTraceWindow(BandwidthTrace(
        [100e6, 100e6], sample_times_s=[0.0, 10.0]
    )), sequence="longdress")
    assert high_action(env) == 0


def test_high_case_reports_successful_headroom():
    case = audit_high_case(
        _env(),
        FiniteTraceWindow(BandwidthTrace(
            [100e6, 100e6], sample_times_s=[0.0, 10.0]
        )),
        "longdress", 42,
    )
    assert case["status"] == SUPPORTED
    assert case["completed_frames"] == 10
    assert case["headroom_s"] > 0


def test_high_case_reports_finite_exhaustion_without_synthetic_qoe():
    case = audit_high_case(
        _env(),
        FiniteTraceWindow(BandwidthTrace(
            [1e3, 1e3], sample_times_s=[0.0, 1.0]
        )),
        "longdress", 42,
    )
    assert case["status"] == UNSUPPORTED
    assert case["completed_frames"] == 0
    assert "headroom_s" not in case


def test_summary_counts_window_and_case_failures():
    windows = [
        {
            "trace": "a.csv", "split": "train", "status": SUPPORTED,
            "case_count": 1, "failure_count": 0,
            "cases": [{
                "status": SUPPORTED, "headroom_s": 5.0,
                "segment_frames": 5, "sequence": "longdress", "jitter_seed": 42,
            }],
        },
        {
            "trace": "b.csv", "split": "validation", "status": UNSUPPORTED,
            "case_count": 1, "failure_count": 1,
            "cases": [{
                "status": UNSUPPORTED,
                "segment_frames": 5, "sequence": "longdress", "jitter_seed": 42,
            }],
        },
    ]
    summary = summarize_high_audit(windows)
    assert summary["window_count"] == 2
    assert summary["high_supported_window_count"] == 1
    assert summary["failed_case_count"] == 1
    assert summary["minimum_successful_headroom_s"] == 5.0


def test_checked_in_candidate_audit_covers_every_vlow_eligible_window():
    report_path = os.path.join(ROOT, "reports", "high_tier_candidate_audit.json")
    registry_path = os.path.join(
        ROOT, "reports", "trace_window_registry_all_candidates.json"
    )
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    with open(registry_path, encoding="utf-8") as handle:
        registry = json.load(handle)
    unsigned = dict(report)
    audit_id = unsigned.pop("audit_id")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16] == audit_id
    assert report["audit_type"] == (
        "read_only_static_high_candidate_window_headroom"
    )
    assert report["trace_registry_id"] == registry["registry_id"]
    assert report["settings"]["case_details_included"] is False
    assert report["summary"]["window_count"] == 759
    assert report["summary"]["high_supported_window_count"] == 675
    assert report["summary"]["failed_case_count"] == 3210
    assert len(report["windows"]) == 759
    assert all("cases" not in window for window in report["windows"])


def test_checked_in_full_audit_matches_the_frozen_registry():
    report_path = os.path.join(ROOT, "reports", "high_tier_headroom_audit.json")
    registry_path = os.path.join(ROOT, "configs", "trace_window_registry.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    with open(registry_path, encoding="utf-8") as handle:
        registry = json.load(handle)

    unsigned = dict(report)
    audit_id = unsigned.pop("audit_id")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16] == audit_id
    assert report["trace_registry_id"] == registry["registry_id"]
    registered_ids = {
        window["id"]
        for split in ("train", "validation", "test")
        for window in registry["splits"][split]["windows"]
    }
    audited_ids = {window["id"] for window in report["windows"]}
    assert audited_ids == registered_ids
    assert report["summary"]["window_count"] == 437
    assert report["summary"]["case_count"] == 20_976
    assert report["summary"]["high_supported_window_count"] == 437
    assert report["summary"]["failed_case_count"] == 0
    assert report["summary"]["minimum_successful_headroom_s"] >= 5.0
    for relative_path, expected_hash in report["input_hashes"]["implementation"].items():
        digest = hashlib.sha256()
        with open(os.path.join(ROOT, relative_path), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected_hash
    for window in report["windows"]:
        assert window["case_count"] == 48
        assert len(window["cases"]) == 48
        combinations = {
            (case["segment_frames"], case["sequence"], case["jitter_seed"])
            for case in window["cases"]
        }
        assert len(combinations) == 48
        failures = sum(case["status"] == UNSUPPORTED for case in window["cases"])
        assert failures == window["failure_count"]
        assert (window["status"] == SUPPORTED) == (failures == 0)
        for case in window["cases"]:
            assert "frames_dropped" not in case
            if case["status"] == SUPPORTED:
                assert "request_pacing_s" in case
