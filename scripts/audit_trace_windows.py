#!/usr/bin/env python3
"""Audit timestamp-derived trace windows without changing the experiment.

The command writes a detailed JSON registry and a compact Markdown review.  It
never edits source traces or makes the registry active in training/evaluation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiment_protocol import load_protocol
from src.network_model.manifest import parse_mpd_xml
from src.trace_audit import audit_protocol, render_markdown_report, sha256_file
from src.trace_gaps import DEFAULT_MAX_GAP_S


def _absolute(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _sequence_name(mpd_path):
    stem = os.path.splitext(os.path.basename(mpd_path))[0]
    if stem == "mpd_gpcc":
        return "longdress"
    return stem[len("mpd_gpcc_"):] if stem.startswith("mpd_gpcc_") else stem


def _load_manifest_pool(pattern, required_frames):
    paths = (sorted(glob.glob(pattern))
             if any(char in pattern for char in "*?[") else [pattern])
    if not paths or any(not os.path.isfile(path) for path in paths):
        raise FileNotFoundError(f"no manifest matches {pattern}")
    pool = {}
    for path in paths:
        sequence = _sequence_name(path)
        if sequence in pool:
            raise ValueError(
                f"multiple manifests resolve to sequence {sequence!r}; "
                "use an unambiguous --mpd pattern"
            )
        frames = parse_mpd_xml(path)
        if required_frames and len(frames) < required_frames:
            raise ValueError(
                f"{path} has {len(frames)} frames; audit requires "
                f"{required_frames}"
            )
        if required_frames:
            frames = frames[:required_frames]
        pool[sequence] = frames
    return pool, paths


def _csv_ints(value):
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _csv_strings(value):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one value")
    return result


def _git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        tracked_changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        return commit, bool(tracked_changes)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _default_sequences(protocol):
    values = set()
    for split in ("validation", "test"):
        values.update(protocol.get("evaluation", {}).get(split, {}).get(
            "sequences", []
        ))
    return sorted(values) or ["longdress", "loot", "redandblack", "soldier"]


def _default_seeds(protocol):
    values = set()
    for split in ("validation", "test"):
        values.update(protocol.get("evaluation", {}).get(split, {}).get(
            "jitter_seeds", []
        ))
    return sorted(int(value) for value in values) or [42, 43, 44]


def _default_candidate_count(protocol):
    values = []
    for split in ("validation", "test"):
        value = protocol.get("evaluation", {}).get(split, {}).get(
            "offsets_per_trace"
        )
        if value:
            values.append(int(value))
    return max(values) if values else 3


def _write_new(path, content):
    path = _absolute(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def _preflight_outputs(paths, overwrite):
    absolute_paths = [_absolute(path) for path in paths]
    if len(set(absolute_paths)) != len(absolute_paths):
        raise ValueError("--json-out and --markdown-out must be different files")
    existing = [path for path in absolute_paths if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing audit output(s): "
            f"{existing}; pass --overwrite for a deliberate rerun"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Read-only, timestamp-aware Very-low trace-window audit"
    )
    parser.add_argument(
        "--protocol", default=os.path.join("configs", "experiment_protocol.json")
    )
    parser.add_argument("--trace-dir", default="bandwidth_5g")
    parser.add_argument("--mpd", default=os.path.join("manifests", "mpd_gpcc*.xml"))
    parser.add_argument(
        "--frames", type=int, default=300,
        help="required frames per sequence (default: complete 300-frame session)",
    )
    parser.add_argument(
        "--grid-stride-s", type=float, default=60.0,
        help="seconds between candidate starts on the real timestamp axis",
    )
    parser.add_argument(
        "--max-gap-s", type=float, default=DEFAULT_MAX_GAP_S,
        help=("split before timestamp deltas above this value; "
              f"default: {DEFAULT_MAX_GAP_S:g} s"),
    )
    parser.add_argument(
        "--selected-windows", type=int, default=0,
        help="eligible eval starts selected per trace; 0 uses the protocol count",
    )
    parser.add_argument(
        "--segment-frames", type=_csv_ints, default=[5, 8, 10, 15],
        help="comma-separated segment sizes that every window must support",
    )
    parser.add_argument(
        "--sequences", type=_csv_strings, default=None,
        help="comma-separated content names; default is the registered set",
    )
    parser.add_argument(
        "--jitter-seeds", type=_csv_ints, default=None,
        help="comma-separated transport seeds; default is the registered set",
    )
    parser.add_argument(
        "--json-out", default=os.path.join("reports", "trace_window_audit.json")
    )
    parser.add_argument(
        "--markdown-out", default=os.path.join("reports", "TRACE_WINDOW_AUDIT.md")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trace_dir = _absolute(args.trace_dir)
    protocol = load_protocol(_absolute(args.protocol), trace_dir)
    sequences = args.sequences or _default_sequences(protocol)
    jitter_seeds = args.jitter_seeds or _default_seeds(protocol)
    selected_windows = (
        args.selected_windows or _default_candidate_count(protocol)
    )
    if args.frames <= 0:
        parser.error("--frames must be positive for a complete-session audit")
    if not args.grid_stride_s > 0:
        parser.error("--grid-stride-s must be positive")
    if not args.max_gap_s > 0:
        parser.error("--max-gap-s must be positive")
    if selected_windows <= 0:
        parser.error("--selected-windows must be positive")
    _preflight_outputs([args.json_out, args.markdown_out], args.overwrite)

    manifest_pool, manifest_paths = _load_manifest_pool(
        _absolute(args.mpd), args.frames
    )
    missing = sorted(set(sequences) - set(manifest_pool))
    if missing:
        parser.error(
            f"registered sequences missing from --mpd input: {missing}; "
            f"available={sorted(manifest_pool)}"
        )

    def progress(split, filename, count):
        print(
            f"[{split:10}] {filename}: auditing {count} timestamp windows",
            flush=True,
        )

    print("Trace-window audit (inputs remain unchanged)", flush=True)
    print(
        f"policy=Static Very-low | frames={args.frames} | "
        f"segments={args.segment_frames} | sequences={sequences} | "
        f"jitter_seeds={jitter_seeds} | grid={args.grid_stride_s:g}s",
        flush=True,
    )
    print(
        f"gap policy=split when timestamp delta > {args.max_gap_s:g}s",
        flush=True,
    )
    protocol_path = _absolute(args.protocol)
    input_hashes = {
        "protocol": {os.path.relpath(protocol_path, PROJECT_ROOT):
                     sha256_file(protocol_path)},
        "manifests": {
            os.path.relpath(path, PROJECT_ROOT): sha256_file(path)
            for path in manifest_paths
        },
        "audit_implementation": {
            path: sha256_file(_absolute(path))
            for path in (
                os.path.join("scripts", "audit_trace_windows.py"),
                os.path.join("src", "trace_audit.py"),
                os.path.join("src", "trace_gaps.py"),
                os.path.join("src", "network_model", "trace.py"),
                os.path.join("src", "network_model", "links.py"),
                os.path.join("src", "network_model", "tcp_protocol.py"),
                os.path.join("src", "network_model", "session.py"),
                os.path.join("src", "network_model", "manifest.py"),
                os.path.join("src", "rl", "env.py"),
            )
        },
    }
    source_commit, tracked_files_dirty = _git_state()
    report = audit_protocol(
        protocol,
        trace_dir,
        manifest_pool,
        grid_stride_s=args.grid_stride_s,
        selected_windows_per_trace=selected_windows,
        sequences=sequences,
        jitter_seeds=jitter_seeds,
        segment_frames=args.segment_frames,
        source_commit=source_commit,
        source_tracked_files_dirty=tracked_files_dirty,
        input_hashes=input_hashes,
        progress=progress,
        max_gap_s=args.max_gap_s,
    )
    json_text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    markdown_text = render_markdown_report(report)
    json_path = _write_new(args.json_out, json_text)
    markdown_path = _write_new(args.markdown_out, markdown_text)

    summary = report["summary"]
    print(
        f"Done: {summary['eligible_window_count']}/"
        f"{summary['candidate_window_count']} windows eligible; "
        f"{summary['excluded_trace_count']}/"
        f"{summary['trace_count']} traces classified as outages."
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print("No trace, split, training, or evaluation setting was changed.")


if __name__ == "__main__":
    main()
