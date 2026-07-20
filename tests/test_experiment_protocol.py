"""Tests for the checked-in final-experiment protocol."""

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiment_protocol import (
    evenly_spaced_offsets,
    load_protocol,
    protocol_digest,
)


def test_registered_protocol_is_complete_and_disjoint():
    protocol = load_protocol(
        os.path.join(ROOT, "configs", "experiment_protocol.json"),
        os.path.join(ROOT, "bandwidth_5g"),
    )
    split = protocol["trace_split"]
    assert [len(split[k]) for k in ("train", "validation", "test")] == [13, 4, 4]
    assert sum(name.startswith("static_") for name in split["validation"]) == 1
    assert sum(name.startswith("static_") for name in split["test"]) == 1


def test_protocol_digest_is_stable_across_key_order():
    a = {"b": 2, "a": {"y": 1, "x": 0}}
    b = {"a": {"x": 0, "y": 1}, "b": 2}
    assert protocol_digest(a) == protocol_digest(b)


def test_evenly_spaced_offsets_reserve_tail_and_remove_duplicates():
    assert evenly_spaced_offsets(1000, 3, min_tail_samples=100) == [0, 450, 900]
    assert evenly_spaced_offsets(50, 5, min_tail_samples=120) == [0]
    assert evenly_spaced_offsets(122, 5, min_tail_samples=120) == [0, 1, 2]


def test_overlap_is_rejected():
    bad = {
        "trace_split": {
            "train": ["a.csv"],
            "validation": ["a.csv"],
            "test": ["b.csv"],
        }
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "protocol.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        try:
            load_protocol(path)
        except ValueError as exc:
            assert "overlap" in str(exc)
        else:
            raise AssertionError("overlapping protocol was accepted")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
