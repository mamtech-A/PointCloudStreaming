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
from src.high_tier_audit import SUPPORTED as HIGH_SUPPORTED
from src.high_tier_audit import UNSUPPORTED as HIGH_UNSUPPORTED
from src.provenance import sha256_file
from src.trace_registry import load_trace_registry
from src.trace_audit import select_evenly_spaced_windows


WINDOW_FIELDS = (
    "id", "split", "trace", "block_id", "block_index",
    "block_start_time_s", "block_end_time_s", "block_local_start_time_s",
    "source_sample_index", "start_time_s", "duration_s",
)


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
    parser.add_argument(
        "--high-audit", default=None,
        help=("maximum-tier audit covering every Very-low-eligible candidate; "
              "when supplied, retain only windows supported at both extremes"),
    )
    parser.add_argument(
        "--all-eligible-evaluation", action="store_true",
        help=("put every Very-low-eligible validation/test candidate in the "
              "output; intended only for constructing a maximum-tier audit"),
    )
    parser.add_argument(
        "--minimum-high-headroom-s", type=float, default=5.0,
        help=("minimum measured time remaining after every successful Static-High "
              "case; defaults to one 5-second client-buffer horizon"),
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
    high_audit_path = absolute(args.high_audit) if args.high_audit else None
    if args.minimum_high_headroom_s < 0:
        raise ValueError("--minimum-high-headroom-s must be non-negative")
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

    high_report = None
    if high_audit_path:
        if args.all_eligible_evaluation:
            raise ValueError(
                "--high-audit and --all-eligible-evaluation are mutually exclusive"
            )
        with open(high_audit_path, encoding="utf-8") as handle:
            high_report = json.load(handle)
        if high_report.get("schema_version") != 1:
            raise ValueError("unsupported maximum-tier audit schema")
        unsigned_high = dict(high_report)
        reported_audit_id = unsigned_high.pop("audit_id", None)
        canonical_high = json.dumps(
            unsigned_high, sort_keys=True, separators=(",", ":")
        )
        calculated_audit_id = hashlib.sha256(
            canonical_high.encode("utf-8")
        ).hexdigest()[:16]
        if not reported_audit_id or reported_audit_id != calculated_audit_id:
            raise ValueError("maximum-tier audit content/ID mismatch")
        high_windows = high_report.get("windows", [])
        audited = {window["id"]: window for window in high_windows}
        if len(audited) != len(high_windows):
            raise ValueError("maximum-tier audit contains duplicate window IDs")
        if set(audited) != set(by_id):
            missing = sorted(set(by_id) - set(audited))
            extra = sorted(set(audited) - set(by_id))
            raise ValueError(
                "maximum-tier audit must cover every Very-low-eligible window; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        expected_sequences = list(audit["settings"]["required_sequences"])
        expected_segments = [int(v) for v in audit["settings"]["segment_frames"]]
        expected_seeds = [int(v) for v in audit["settings"]["jitter_seeds"]]
        expected_frames = {
            int(v) for v in audit["settings"]["required_frames_per_sequence"].values()
        }
        high_settings = high_report.get("settings", {})
        expected_cases = (
            len(expected_sequences) * len(expected_segments) * len(expected_seeds)
        )
        expected_settings = (
            high_report.get("audit_type")
            == "read_only_static_high_candidate_window_headroom"
            and high_report.get("protocol_digest") == expected_protocol
            and high_settings.get("policy") == "Static High G-PCC tier"
            and int(high_settings.get("frames_per_sequence", -1))
            in expected_frames
            and list(high_settings.get("sequences", [])) == expected_sequences
            and [int(v) for v in high_settings.get("segment_frames", [])]
            == expected_segments
            and [int(v) for v in high_settings.get("jitter_seeds", [])]
            == expected_seeds
            and high_settings.get("terminal_capacity_policy")
            == "finite; no final-sample clamping"
            and int(high_settings.get("case_count_per_window", -1))
            == expected_cases
        )
        if not expected_settings or len(expected_frames) != 1:
            raise ValueError("maximum-tier audit settings do not match trace audit")
        for window in high_windows:
            status = window.get("status")
            failures = int(window.get("failure_count", -1))
            if int(window.get("case_count", -1)) != expected_cases:
                raise ValueError(
                    f"maximum-tier audit case count mismatch: {window['id']}"
                )
            if ((status == HIGH_SUPPORTED and failures != 0)
                    or (status == HIGH_UNSUPPORTED and failures <= 0)
                    or status not in (HIGH_SUPPORTED, HIGH_UNSUPPORTED)):
                raise ValueError(
                    f"maximum-tier audit status/count mismatch: {window['id']}"
                )

        candidate_rel = high_report.get("trace_registry")
        if not candidate_rel:
            raise ValueError("maximum-tier audit has no source registry")
        candidate_path = absolute(candidate_rel)
        if sha256_file(candidate_path) != high_report["input_hashes"]["registry"]:
            raise ValueError("maximum-tier audit/source registry hash mismatch")
        candidate_registry = load_trace_registry(
            candidate_path, protocol, trace_dir, verify_hashes=True,
            allow_candidate_registry=True,
        )
        if (candidate_registry.data.get("registry_type")
                != "candidate_high_tier_audit_windows"):
            raise ValueError("maximum-tier audit source is not a candidate registry")
        if candidate_registry.registry_id != high_report.get("trace_registry_id"):
            raise ValueError("maximum-tier audit/source registry ID mismatch")
        candidate_ids = {
            window["id"] for split in ("train", "validation", "test")
            for window in candidate_registry.windows(split)
        }
        if candidate_ids != set(by_id):
            raise ValueError("candidate registry does not cover the trace audit")
        if (candidate_registry.data.get("source_audit_sha256")
                != sha256_file(audit_path)):
            raise ValueError("candidate registry was built from another trace audit")
        for relative_path, expected_hash in high_report.get(
                "input_hashes", {}).get("implementation", {}).items():
            if sha256_file(absolute(relative_path)) != expected_hash:
                raise ValueError(
                    f"maximum-tier audit implementation is stale: {relative_path}"
                )
        supported_ids = {
            window_id for window_id, window in audited.items()
            if (window.get("status") == HIGH_SUPPORTED
                and float(window.get("minimum_headroom_s", -1.0))
                >= args.minimum_high_headroom_s)
        }
        eligible = [window for window in eligible if window["id"] in supported_ids]
        by_id = {window["id"]: window for window in eligible}

    train_windows = [
        compact_window(window) for window in eligible
        if window["split"] == "train"
    ]
    if args.all_eligible_evaluation:
        selected = [
            compact_window(window) for window in eligible
            if window["split"] in ("validation", "test")
        ]
    elif high_report is not None:
        count = int(audit["settings"]["selected_windows_per_trace"])
        selected = []
        for split in ("validation", "test"):
            for filename in protocol["trace_split"][split]:
                candidates = [
                    window for window in eligible
                    if window["split"] == split and window["trace"] == filename
                ]
                chosen = select_evenly_spaced_windows(candidates, count)
                if len(chosen) != count:
                    raise ValueError(
                        f"{split} trace {filename} has only {len(chosen)} "
                        f"maximum-tier-supported windows; {count} are required"
                    )
                selected.extend(compact_window(window) for window in chosen)
    else:
        selected_ids = [
            row["id"]
            for row in audit["registries"]["selected_evaluation_windows"]
        ]
        selected = []
        for window_id in selected_ids:
            if window_id not in by_id:
                raise ValueError(f"selected window is not eligible: {window_id}")
            selected.append(compact_window(by_id[window_id]))

    for split in ("train", "validation", "test"):
        available = {
            window["trace"] for window in
            (train_windows if split == "train" else selected)
            if window["split"] == split
        }
        missing_parents = sorted(set(protocol["trace_split"][split]) - available)
        if missing_parents:
            raise ValueError(
                f"{split} parents have no retained feasible window: {missing_parents}"
            )

    payload = {
        "schema_version": 1,
        "registry_type": (
            "candidate_high_tier_audit_windows"
            if args.all_eligible_evaluation else
            "gap_split_finite_high_supported_windows"
            if high_report is not None else
            "gap_split_finite_trace_windows"
        ),
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
                "fatal protocol failure; no QoE is assigned to an incomplete case"
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
    if high_report is not None:
        payload["settings"]["minimum_maximum_tier_headroom_s"] = float(
            args.minimum_high_headroom_s
        )
        payload.update({
            "source_high_audit": os.path.relpath(
                high_audit_path, PROJECT_ROOT
            ).replace("\\", "/"),
            "source_high_audit_sha256": sha256_file(high_audit_path),
            "source_high_audit_id": high_report.get("audit_id"),
        })
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
