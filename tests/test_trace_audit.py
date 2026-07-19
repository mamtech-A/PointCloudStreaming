"""Focused tests for the strict, timestamp-aware trace-window audit."""

import csv
import hashlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.network_model.trace import BandwidthTrace
from src.rl.env import StreamingEnv
from src.trace_audit import (
    ELIGIBLE,
    OUTAGE,
    FiniteTraceWindow,
    TraceWindowExhausted,
    audit_case,
    audit_protocol,
    audit_window,
    measured_end_time_s,
    read_trace_metadata,
    select_evenly_spaced_windows,
    slice_trace_at_time,
    timestamp_grid_starts,
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


def _frames(count, vlow_bytes=1000):
    return [
        {
            "id": index,
            "representations": [
                _rep(0, "high", 100_000),
                _rep(1, "vlow", vlow_bytes),
            ],
        }
        for index in range(count)
    ]


def _env(pool, segment_frames=1):
    return StreamingEnv(
        pool,
        tcp_params={
            "rtt_ms": 1.0,
            "rtt_jitter_ms": 0.0,
            "loss_prob": 0.0,
            "cwnd_packets": 1000,
            "mss_bytes": 1460,
            "log_packets": False,
        },
        segment_frames=segment_frames,
    )


def test_measured_end_excludes_compatibility_terminal_extension():
    trace = BandwidthTrace(
        [1e6, 2e6, 3e6], sample_times_s=[100.0, 101.0, 105.0]
    )
    assert trace.duration_s == 6.0       # production compatibility behaviour
    assert measured_end_time_s(trace) == 5.0  # audit stops at final timestamp
    assert FiniteTraceWindow(trace).duration_s == 5.0


def test_finite_window_never_clamps_terminal_sample():
    trace = BandwidthTrace([1e6, 1e6], sample_times_s=[10.0, 12.0])
    finite = FiniteTraceWindow(trace)
    assert finite.time_to_transmit(0.0, 2e6) == 2.0
    try:
        finite.time_to_transmit(0.0, 2e6 + 1.0)
    except TraceWindowExhausted as exc:
        assert exc.trace_end_s == 2.0
        assert exc.remaining_bits > 0
    else:
        raise AssertionError("terminal sample was extended past its timestamp")


def test_timestamp_grid_uses_seconds_not_row_positions():
    trace = BandwidthTrace(
        [1e6] * 5, sample_times_s=[0.0, 1.0, 61.0, 62.0, 121.0]
    )
    assert timestamp_grid_starts(trace, 60.0) == [0.0, 60.0, 120.0]


def test_exact_time_slice_keeps_capacity_in_effect_before_next_row():
    trace = BandwidthTrace(
        [1e6, 2e6, 3e6], sample_times_s=[0.0, 1.0, 61.0]
    )
    window, source_index = slice_trace_at_time(trace, 60.0)
    assert source_index == 1
    assert window.capacity_at_time(0.0) == 2e6
    assert window.capacity_at_time(0.999) == 2e6
    assert window.capacity_at_time(1.0) == 3e6
    assert measured_end_time_s(window) == 1.0


def test_fast_case_is_eligible_and_slow_case_is_outage():
    env = _env({"longdress": _frames(4, vlow_bytes=1000)})
    fast = BandwidthTrace([10e6, 10e6], sample_times_s=[0.0, 2.0])
    slow = BandwidthTrace([1e3, 1e3], sample_times_s=[0.0, 1.0])
    accepted = audit_case(env, fast, "longdress", 42)
    rejected = audit_case(env, slow, "longdress", 42)
    assert accepted["status"] == ELIGIBLE
    assert accepted["completion_time_s"] <= accepted["measured_duration_s"]
    assert rejected["status"] == OUTAGE
    assert rejected["reason"] == "vlow_exceeds_trace_end"
    assert rejected["completed_frames"] < rejected["required_frames"]


def test_all_four_sequences_are_required_for_window_eligibility():
    pool = {
        "longdress": _frames(2, 100),
        "loot": _frames(2, 100),
        "redandblack": _frames(2, 100),
        "soldier": _frames(2, 2_000_000),
    }
    env = _env(pool)
    trace = BandwidthTrace([1e6, 1e6], sample_times_s=[0.0, 1.0])
    result = audit_window({1: env}, trace, list(pool), [42])
    assert result["status"] == OUTAGE
    assert set(result["tested_sequences"]) == set(pool)
    failures = result["failed_cases"]
    assert [case["sequence"] for case in failures] == ["soldier"]


def test_difficult_but_feasible_case_is_not_rejected_for_stall():
    env = _env({"longdress": _frames(60, vlow_bytes=2000)})
    # 60 frames carry 0.96 Mbit.  This finishes inside the 20-second measured
    # window but substantially slower than the two-second media duration.
    trace = BandwidthTrace([0.2e6, 0.2e6], sample_times_s=[0.0, 20.0])
    result = audit_case(env, trace, "longdress", 42)
    assert result["status"] == ELIGIBLE
    assert result["stall_duration_s"] > 0.0


def test_selection_keeps_trace_with_fewer_than_requested_eligible_windows():
    windows = [
        {"id": "a", "status": ELIGIBLE, "start_time_s": 0.0,
         "sample_offset": 0},
        {"id": "b", "status": OUTAGE, "start_time_s": 60.0,
         "sample_offset": 10},
    ]
    selected = select_evenly_spaced_windows(windows, 3)
    assert [row["id"] for row in selected] == ["a"]


def test_selection_is_even_in_time_not_candidate_row_count():
    windows = [
        {"id": str(start), "status": ELIGIBLE, "start_time_s": start,
         "sample_offset": index}
        for index, start in enumerate([0.0, 60.0, 120.0, 180.0, 600.0])
    ]
    selected = select_evenly_spaced_windows(windows, 3)
    assert [row["start_time_s"] for row in selected] == [0.0, 180.0, 600.0]


def _write_trace(path, kbps_values):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timestamp", "DL_bitrate", "State", "NetworkMode"])
        for second, kbps in enumerate(kbps_values):
            writer.writerow([
                f"2020.01.01_00.00.{second:02d}", kbps, "D", "5G"
            ])


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_trace_is_excluded_only_when_no_candidate_window_is_eligible():
    with tempfile.TemporaryDirectory() as directory:
        names = {
            "train": "driving_train.csv",
            "validation": "driving_validation.csv",
            "test": "static_test.csv",
        }
        mixed = os.path.join(directory, names["train"])
        _write_trace(mixed, [10_000, 10_000, 1, 1, 1])
        _write_trace(os.path.join(directory, names["validation"]), [1] * 5)
        _write_trace(os.path.join(directory, names["test"]), [1] * 5)
        before = {name: _digest(os.path.join(directory, name))
                  for name in names.values()}
        protocol = {
            "trace_split": {split: [name] for split, name in names.items()},
            "evaluation": {
                "validation": {"offsets_per_trace": 1},
                "test": {"offsets_per_trace": 1},
            },
        }
        pool = {sequence: _frames(1, 1000) for sequence in (
            "longdress", "loot", "redandblack", "soldier"
        )}
        report = audit_protocol(
            protocol, directory, pool, grid_stride_s=2.0,
            selected_windows_per_trace=1, jitter_seeds=[42],
            segment_frames=[1],
        )
        by_name = {row["trace"]: row for row in report["traces"]}
        assert by_name[names["train"]]["status"] == ELIGIBLE
        assert by_name[names["train"]]["eligible_window_count"] == 1
        assert by_name[names["train"]]["outage_window_count"] == 1
        assert by_name[names["validation"]]["status"] == OUTAGE
        assert by_name[names["test"]]["status"] == OUTAGE
        registered = (
            report["registries"]["eligible_windows"]
            + report["registries"]["outage_windows"]
        )
        assert registered
        for window in registered:
            assert window["status"] in (ELIGIBLE, OUTAGE)
            assert window["reason"]
            assert window["case_count"] == 4
            assert len(window["case_evidence"]) == 4
            assert "failure_count" in window
        after = {name: _digest(os.path.join(directory, name))
                 for name in names.values()}
        assert after == before


def test_protocol_windows_restart_inside_blocks_and_never_cross_gap():
    with tempfile.TemporaryDirectory() as directory:
        names = {
            "train": "driving_train.csv",
            "validation": "driving_validation.csv",
            "test": "static_test.csv",
        }
        header = ["Timestamp", "DL_bitrate", "State", "NetworkMode"]
        for name in names.values():
            path = os.path.join(directory, name)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                for second in (0, 1, 2):
                    writer.writerow([
                        f"2020.01.01_00.00.{second:02d}", 10000, "D", "5G"
                    ])
                for second in (10, 11, 12):
                    writer.writerow([
                        f"2020.01.01_00.00.{second:02d}", 10000, "D", "5G"
                    ])
        protocol = {
            "trace_split": {split: [name] for split, name in names.items()},
            "evaluation": {
                "validation": {"offsets_per_trace": 1},
                "test": {"offsets_per_trace": 1},
            },
        }
        pool = {sequence: _frames(1, 1000) for sequence in (
            "longdress", "loot", "redandblack", "soldier"
        )}
        report = audit_protocol(
            protocol, directory, pool, grid_stride_s=1.0,
            selected_windows_per_trace=1, jitter_seeds=[42],
            segment_frames=[1], max_gap_s=5.0,
        )
        assert report["summary"]["split_gap_count"] == 3
        assert report["summary"]["contiguous_block_count"] == 6
        for trace in report["traces"]:
            assert trace["block_count"] == 2
            assert trace["excluded_gap_duration_s"] == 8.0
            assert [block["start_time_s"] for block in trace["blocks"]] == [
                0.0, 10.0
            ]
        registered = (
            report["registries"]["eligible_windows"]
            + report["registries"]["outage_windows"]
        )
        assert {row["block_id"] for row in registered} == {
            "block-000", "block-001"
        }
        for row in registered:
            assert row["duration_s"] <= 2.0
            assert row["block_start_time_s"] <= row["start_time_s"]
            assert row["start_time_s"] <= row["block_end_time_s"]


def test_invalid_or_backward_timestamps_are_rejected_explicitly():
    header = ["Timestamp", "DL_bitrate", "State", "NetworkMode"]
    with tempfile.TemporaryDirectory() as directory:
        invalid_path = os.path.join(directory, "invalid.csv")
        with open(invalid_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(["not-a-time", "1000", "D", "5G"])
        try:
            read_trace_metadata(invalid_path)
        except ValueError as exc:
            assert "invalid Timestamp" in str(exc)
        else:
            raise AssertionError("invalid timestamp silently accepted")

        backward_path = os.path.join(directory, "backward.csv")
        with open(backward_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(["2020.01.01_00.00.02", "1000", "D", "5G"])
            writer.writerow(["2020.01.01_00.00.01", "1000", "D", "4G"])
        try:
            read_trace_metadata(backward_path)
        except ValueError as exc:
            assert "move backwards" in str(exc)
        else:
            raise AssertionError("backward timestamp silently accepted")


def test_duplicate_timestamps_are_disclosed_and_terminal_extrema_are_ignored():
    header = ["Timestamp", "DL_bitrate", "State", "NetworkMode"]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "duplicates.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(["2020.01.01_00.00.00", "1000", "D", "5G"])
            writer.writerow(["2020.01.01_00.00.00", "2000", "D", "LTE"])
            writer.writerow(["2020.01.01_00.00.02", "999999", "D", "5G"])
        metadata, _ = read_trace_metadata(path)
        assert metadata["timestamps_valid"] is True
        assert metadata["duplicate_timestamp_count"] == 1
        assert metadata["min_capacity_mbps"] == 1.0
        assert metadata["max_capacity_mbps"] == 2.0


def test_invalid_or_empty_audit_dimensions_are_rejected():
    protocol = {"trace_split": {name: [f"{name}.csv"] for name in (
        "train", "validation", "test"
    )}}
    pool = {"longdress": _frames(1)}
    for kwargs in (
        {"segment_frames": []},
        {"segment_frames": [0]},
        {"sequences": []},
        {"jitter_seeds": []},
    ):
        try:
            audit_protocol(protocol, ".", pool, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid dimensions accepted: {kwargs}")


def _run_all():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
