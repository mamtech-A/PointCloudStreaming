#!/usr/bin/env python3
"""Build the compact production registry from the reviewed trace audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiment_protocol import load_protocol, protocol_digest


WINDOW_FIELDS = (
    "id", "split", "trace", "block_id", "block_index",
    "block_start_time_s", "block_end_time_s", "block_local_start_time_s",
    "source_sample_index", "start_time_s", "duration_s",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_window(window):
    missing = [field for field in WINDOW_FIELDS if field not in window]
    if missing:
        raise ValueError(f"audit window {window.get('id')} lacks {missing}")
    return {field: window[field] for field in WINDOW_FIELDS}


def main():
    parser = argparse.ArgumentParser(
        description="Freeze audited block windows for training and evaluation"
    )
    parser.add_argument(
        "--audit", default=os.path.join("reports", "trace_window_audit.json")
    )
    parser.add_argument(
        "--protocol", default=os.path.join("configs", "experiment_protocol.json")
    )
    parser.add_argument("--trace-dir", default="bandwidth_5g")
    parser.add_argument(
        "--out", default=os.path.join("configs", "trace_window_registry.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    absolute = lambda path: (
        path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    )
    audit_path = absolute(args.audit)
    protocol_path = absolute(args.protocol)
    trace_dir = absolute(args.trace_dir)
    out_path = absolute(args.out)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {out_path} without --overwrite")

    with open(audit_path, encoding="utf-8") as handle:
        audit = json.load(handle)
    protocol = load_protocol(protocol_path, trace_dir)
    expected_protocol = protocol_digest(protocol)
    if audit.get("schema_version") != 3:
        raise ValueError("production registry requires trace audit schema 3")
    if audit.get("source_tracked_files_dirty") is not False:
        raise ValueError("refusing to freeze an audit generated from dirty source")
    if audit.get("protocol_digest") != expected_protocol:
        raise ValueError("audit/protocol digest mismatch")
    eligible = audit["registries"]["eligible_windows"]
    if len(eligible) != int(audit["summary"]["eligible_window_count"]):
        raise ValueError("eligible-window count mismatch")
    by_id = {window["id"]: window for window in eligible}
    if len(by_id) != len(eligible):
        raise ValueError("eligible window IDs are not unique")

    trace_hashes = {row["trace"]: row["sha256"] for row in audit["traces"]}
    configured_files = {
        filename for split in ("train", "validation", "test")
        for filename in protocol["trace_split"][split]
    }
    if set(trace_hashes) != configured_files:
        raise ValueError("audit trace hashes do not cover the protocol exactly")

    train_windows = [
        compact_window(window) for window in eligible
        if window["split"] == "train"
    ]
    selected_ids = [
        row["id"] for row in audit["registries"]["selected_evaluation_windows"]
    ]
    selected = []
    for window_id in selected_ids:
        if window_id not in by_id:
            raise ValueError(f"selected window is not eligible: {window_id}")
        selected.append(compact_window(by_id[window_id]))

    payload = {
        "schema_version": 1,
        "registry_type": "gap_split_finite_trace_windows",
        "source_audit": os.path.relpath(audit_path, PROJECT_ROOT).replace("\\", "/"),
        "source_audit_sha256": sha256_file(audit_path),
        "source_audit_commit": audit["source_commit"],
        "protocol": os.path.relpath(protocol_path, PROJECT_ROOT).replace("\\", "/"),
        "protocol_digest": expected_protocol,
        "trace_hashes": dict(sorted(trace_hashes.items())),
        "settings": {
            "maximum_contiguous_gap_s": audit["settings"]["maximum_contiguous_gap_s"],
            "candidate_grid_stride_s": audit["settings"]["candidate_grid_stride_s"],
            "terminal_capacity_policy": audit["settings"]["terminal_capacity_policy"],
            "required_sequences": audit["settings"]["required_sequences"],
            "required_segment_frames": audit["settings"]["segment_frames"],
            "required_jitter_seeds": audit["settings"]["jitter_seeds"],
            "required_frames_per_sequence": audit["settings"]["required_frames_per_sequence"],
            "training_sampling": "uniform parent trace, then eligible window",
            "evaluation_aggregation": "macro average of parent-trace means",
            "policy_exhaustion": (
                "terminal failure; remaining frames enter the canonical frame-drop term"
            ),
        },
        "splits": {
            split: {
                "parent_files": list(protocol["trace_split"][split]),
                "windows": (
                    train_windows if split == "train" else
                    [window for window in selected if window["split"] == split]
                ),
            }
            for split in ("train", "validation", "test")
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["registry_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    counts = {split: len(payload["splits"][split]["windows"])
              for split in ("train", "validation", "test")}
    print(f"registry={payload['registry_id']} windows={counts} -> {out_path}")


if __name__ == "__main__":
    main()
