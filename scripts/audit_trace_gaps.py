#!/usr/bin/env python3
"""Inventory timestamp discontinuities without modifying source traces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GENERATED_AUDIT_OUTPUTS = (
    os.path.join("reports", "trace_window_audit.json"),
    os.path.join("reports", "TRACE_WINDOW_AUDIT.md"),
    os.path.join("reports", "trace_gap_audit.json"),
    os.path.join("reports", "TRACE_GAP_AUDIT.md"),
)

from src.experiment_protocol import load_protocol, protocol_digest
from src.trace_audit import sha256_file
from src.trace_gaps import (
    DEFAULT_MAX_GAP_S,
    DEFAULT_SENSITIVITY_THRESHOLDS_S,
    build_gap_inventory,
    render_gap_markdown,
)


def _absolute(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _csv_floats(value):
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be numbers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("thresholds must be positive")
    return result


def _git_state(ignored_output_paths=()):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        tracked_changes = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD", "--"],
            cwd=PROJECT_ROOT,
            text=True,
        ).splitlines()
        ignored = {
            os.path.relpath(_absolute(path), PROJECT_ROOT).replace("\\", "/")
            for path in (*GENERATED_AUDIT_OUTPUTS, *ignored_output_paths)
        }
        dirty_source_paths = [
            path.strip().replace("\\", "/") for path in tracked_changes
            if path.strip().replace("\\", "/") not in ignored
        ]
        return commit, bool(dirty_source_paths)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _preflight(paths, overwrite):
    paths = [_absolute(path) for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("output paths must be different")
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {existing}; pass --overwrite to rerun"
        )
    return paths


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only timestamp-gap inventory and threshold audit"
    )
    parser.add_argument(
        "--protocol", default=os.path.join("configs", "experiment_protocol.json")
    )
    parser.add_argument("--trace-dir", default="bandwidth_5g")
    parser.add_argument("--long-gap-s", type=float, default=DEFAULT_MAX_GAP_S)
    parser.add_argument(
        "--thresholds",
        type=_csv_floats,
        default=list(DEFAULT_SENSITIVITY_THRESHOLDS_S),
    )
    parser.add_argument("--grid-stride-s", type=float, default=60.0)
    parser.add_argument(
        "--json-out", default=os.path.join("reports", "trace_gap_audit.json")
    )
    parser.add_argument(
        "--markdown-out", default=os.path.join("reports", "TRACE_GAP_AUDIT.md")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.long_gap_s <= 0 or args.grid_stride_s <= 0:
        parser.error("gap and grid values must be positive")

    json_path, markdown_path = _preflight(
        [args.json_out, args.markdown_out], args.overwrite
    )
    protocol_path = _absolute(args.protocol)
    trace_dir = _absolute(args.trace_dir)
    protocol = load_protocol(protocol_path, trace_dir)
    report = build_gap_inventory(
        protocol,
        trace_dir,
        long_gap_s=args.long_gap_s,
        sensitivity_thresholds_s=args.thresholds,
        grid_stride_s=args.grid_stride_s,
    )
    source_commit, dirty = _git_state([json_path, markdown_path])
    report.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_tracked_files_dirty": dirty,
        "protocol_digest": protocol_digest(protocol),
        "input_hashes": {
            "protocol": sha256_file(protocol_path),
            "traces": {
                filename: sha256_file(os.path.join(trace_dir, filename))
                for split in ("train", "validation", "test")
                for filename in protocol["trace_split"][split]
            },
            "implementation": {
                "scripts/audit_trace_gaps.py": sha256_file(__file__),
                "src/provenance.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "provenance.py")
                ),
                "src/trace_gaps.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "trace_gaps.py")
                ),
            },
        },
    })
    _write(json_path, json.dumps(report, indent=2) + "\n")
    _write(markdown_path, render_gap_markdown(report))
    summary = report["summary"]
    print(
        f"Found {summary['long_gap_count']} gaps > {args.long_gap_s:g}s in "
        f"{summary['affected_trace_count']}/{summary['trace_count']} traces; "
        f"largest={summary['maximum_gap_s']:.0f}s."
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print("Raw CSV files were not modified.")


if __name__ == "__main__":
    main()
