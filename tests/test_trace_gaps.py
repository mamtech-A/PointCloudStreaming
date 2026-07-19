"""Tests for read-only timestamp-gap splitting and inventory."""

import csv
import hashlib
import os
import sys
import tempfile
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.network_model.trace import BandwidthTrace
from src.trace_audit import FiniteTraceWindow, TraceWindowExhausted
from src.trace_gaps import (
    block_for_source_index,
    build_gap_inventory,
    split_trace_at_gaps,
    threshold_sensitivity,
    validate_gap_threshold,
)


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _write_trace(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Timestamp", "DL_bitrate", "State", "NetworkMode", "Speed",
            "CellID", "Longitude", "Latitude",
        ])
        writer.writerows(rows)


def test_split_is_strictly_greater_than_threshold_and_preserves_input():
    trace = BandwidthTrace(
        [1e6, 2e6, 3e6, 4e6], sample_times_s=[100.0, 105.0, 111.0, 112.0]
    )
    samples_before = list(trace.samples)
    times_before = list(trace._raw_times)
    blocks = split_trace_at_gaps(trace, max_gap_s=5.0)
    assert len(blocks) == 2
    assert blocks[0].source_start_index == 0
    assert blocks[0].source_end_index == 1
    assert blocks[0].duration_s == 5.0
    assert blocks[0].gap_after_s == 6.0
    assert blocks[1].source_start_index == 2
    assert blocks[1].source_end_index == 3
    assert blocks[1].global_start_time_s == 11.0
    assert blocks[1].duration_s == 1.0
    assert blocks[1].gap_before_s == 6.0
    assert trace.samples == samples_before
    assert trace._raw_times == times_before


def test_finite_capacity_cannot_cross_a_split_gap():
    trace = BandwidthTrace(
        [1e6, 1e6, 100e6], sample_times_s=[0.0, 1.0, 628.0]
    )
    first = split_trace_at_gaps(trace, max_gap_s=5.0)[0]
    finite = FiniteTraceWindow(first.trace)
    assert finite.duration_s == 1.0
    assert finite.time_to_transmit(0.0, 1e6) == 1.0
    try:
        finite.time_to_transmit(0.0, 1e6 + 1.0)
    except TraceWindowExhausted as exc:
        assert exc.trace_end_s == 1.0
    else:
        raise AssertionError("transfer crossed an unmeasured 627-second gap")


def test_source_index_maps_to_exactly_one_block():
    trace = BandwidthTrace(
        [1e6] * 5, sample_times_s=[0.0, 1.0, 10.0, 11.0, 20.0]
    )
    blocks = split_trace_at_gaps(trace, max_gap_s=5.0)
    assert [block_for_source_index(blocks, index).block_index
            for index in range(5)] == [0, 0, 1, 1, 2]
    try:
        block_for_source_index(blocks, 5)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range source index was accepted")


def test_threshold_sensitivity_exposes_over_fragmentation():
    origin = "2020.01.01_00.00."
    rows = []
    for index, second in enumerate((0, 1, 4, 5, 11, 12)):
        rows.append({"timestamp": datetime.strptime(
            f"{origin}{second:02d}", "%Y.%m.%d_%H.%M.%S"
        )})
    result = threshold_sensitivity(
        [("train", "trace.csv", rows)], thresholds_s=(2.0, 5.0),
        grid_stride_s=60.0,
    )
    assert result[0]["split_gap_count"] == 2
    assert result[0]["block_count"] == 3
    assert result[1]["split_gap_count"] == 1
    assert result[1]["block_count"] == 2


def test_inventory_reports_context_and_never_rewrites_csvs():
    with tempfile.TemporaryDirectory() as directory:
        names = {
            "train": "driving_train.csv",
            "validation": "static_validation.csv",
            "test": "driving_test.csv",
        }
        ordinary = [
            ["2020.01.01_00.00.00", 1000, "D", "LTE", 0, 1, 0, 0],
            ["2020.01.01_00.00.01", 2000, "D", "5G", 0, 1, 0, 0],
        ]
        long_gap = [
            ["2020.01.01_00.00.00", 4550, "D", "5G", 32, 12, 0, 0],
            ["2020.01.01_00.10.27", 1465, "D", "HSPA+", 34, 10505, 0, 0],
        ]
        for split, name in names.items():
            _write_trace(
                os.path.join(directory, name),
                long_gap if split == "train" else ordinary,
            )
        before = {
            name: _digest(os.path.join(directory, name)) for name in names.values()
        }
        protocol = {"trace_split": {
            split: [name] for split, name in names.items()
        }}
        report = build_gap_inventory(protocol, directory, long_gap_s=5.0)
        assert report["summary"]["long_gap_count"] == 1
        assert report["summary"]["maximum_gap_s"] == 627.0
        gap = report["long_gaps"][0]
        assert gap["before"]["network_mode"] == "5G"
        assert gap["after"]["network_mode"] == "HSPA+"
        assert gap["cell_id_changed"] is True
        after = {
            name: _digest(os.path.join(directory, name)) for name in names.values()
        }
        assert after == before


def test_invalid_gap_threshold_is_rejected():
    for value in (0, -1, float("nan"), float("inf")):
        try:
            validate_gap_threshold(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid threshold accepted: {value}")


def _run_all():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
