#!/usr/bin/env python3
"""Audit every frozen registry window with the maximum-demand High tier."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiment_protocol import load_protocol, protocol_digest
from src.high_tier_audit import (
    SUPPORTED,
    audit_high_window,
    summarize_high_audit,
    validate_high_tier,
)
from src.network_model import DEFAULT_TCP_PARAMS
from src.network_model.manifest import parse_mpd_xml
from src.rl.env import StreamingEnv
from src.trace_registry import load_trace_registry, sha256_file


_WORKER = {}


def _absolute(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _sequence_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem == "mpd_gpcc":
        return "longdress"
    return stem[len("mpd_gpcc_"):] if stem.startswith("mpd_gpcc_") else stem


def _load_manifests(pattern, frames):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no manifest matches {pattern}")
    pool = {}
    for path in paths:
        sequence = _sequence_name(path)
        manifest_frames = parse_mpd_xml(path)
        if len(manifest_frames) < frames:
            raise ValueError(
                f"{path} has {len(manifest_frames)} frames; expected {frames}"
            )
        pool[sequence] = manifest_frames[:frames]
    return pool, paths


def _initialize_worker(context):
    protocol = load_protocol(context["protocol"], context["trace_dir"])
    registry = load_trace_registry(
        context["registry"], protocol, context["trace_dir"], verify_hashes=True,
        allow_candidate_registry=True,
    )
    pool, _paths = _load_manifests(context["mpd"], context["frames"])
    validate_high_tier(pool, context["sequences"])
    tcp_params = {**DEFAULT_TCP_PARAMS, "log_packets": False}
    _WORKER.update({
        "registry": registry,
        "envs": {
            size: StreamingEnv(
                pool, lstm_predictor=None, tcp_params=tcp_params,
                segment_frames=size,
            )
            for size in context["segment_frames"]
        },
        "sequences": context["sequences"],
        "seeds": context["jitter_seeds"],
    })


def _audit_one(item):
    index, window = item
    finite = _WORKER["registry"].materialize(window)
    result = audit_high_window(
        window, finite, _WORKER["envs"],
        _WORKER["sequences"], _WORKER["seeds"],
    )
    return index, result


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _content_digest(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _write_new(path, content, overwrite):
    path = _absolute(path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} without --overwrite")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def _markdown(report):
    summary = report["summary"]
    scope = (
        "all Very-low-eligible candidate windows"
        if report["audit_type"]
        == "read_only_static_high_candidate_window_headroom"
        else "the frozen production registry"
    )
    lines = [
        "# Static-High registry headroom audit", "",
        f"This read-only audit covers {scope} and does not modify traces, "
        "splits, training, or evaluation.", "", "## Decision rule", "",
        "- Every included window is tested with the maximum-demand "
        "static High G-PCC tier.",
        f"- Required sequences: {', '.join(report['settings']['sequences'])}.",
        f"- Segment sizes: {report['settings']['segment_frames']}.",
        f"- Transport seeds: {report['settings']['jitter_seeds']}.",
        "- A window is High-supported only when all required cases complete "
        "before its last measured timestamp.",
        "- No terminal sample is clamped and no post-trace capacity is invented.",
        "", "## Summary", "",
        f"- Registry: `{report['trace_registry_id']}`.",
        f"- Windows: {summary['window_count']} total; "
        f"{summary['high_supported_window_count']} High-supported; "
        f"{summary['high_unsupported_window_count']} unsupported.",
        f"- Cases: {summary['case_count']} total; "
        f"{summary['failed_case_count']} require unobserved capacity.",
        f"- Minimum successful headroom: "
        f"{summary['minimum_successful_headroom_s']:.3f} s.",
        "", "## Split results", "",
        "| Split | Windows | High-supported | Unsupported | Failed cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in summary["by_split"].items():
        lines.append(
            f"| {split} | {row['window_count']} | "
            f"{row['high_supported_window_count']} | "
            f"{row['high_unsupported_window_count']} | "
            f"{row['failed_case_count']} |"
        )
    lines += ["", "## Per-trace results", "",
              "| Split | Trace | Windows | High-supported | Unsupported | Failed cases |",
              "|---|---|---:|---:|---:|---:|"]
    for trace, row in summary["by_trace"].items():
        lines.append(
            f"| {row['split']} | `{trace}` | {row['window_count']} | "
            f"{row['high_supported_window_count']} | "
            f"{row['high_unsupported_window_count']} | "
            f"{row['failed_case_count']} |"
        )
    unsupported = [
        window for window in report["windows"] if window["status"] != SUPPORTED
    ]
    lines += ["", "## Unsupported frozen windows", "",
              "| Window | Duration (s) | Failed cases |",
              "|---|---:|---:|"]
    if unsupported:
        for window in unsupported:
            lines.append(
                f"| `{window['id']}` | {window['measured_duration_s']:.3f} | "
                f"{window['failure_count']} |"
            )
    else:
        lines.append("| None | — | — |")
    detail_note = (
        "The JSON companion retains every case outcome and headroom value."
        if report["settings"].get("case_details_included", True)
        else "The JSON companion retains per-window status, counts, and headroom; "
             "case rows are intentionally omitted for the all-candidate audit."
    )
    lines += ["", detail_note, ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only static-High audit of the frozen trace registry"
    )
    parser.add_argument("--protocol", default="configs/experiment_protocol.json")
    parser.add_argument("--registry", default="configs/trace_window_registry.json")
    parser.add_argument("--trace-dir", default="bandwidth_5g")
    parser.add_argument("--mpd", default="manifests/mpd_gpcc*.xml")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--json-out", default="reports/high_tier_headroom_audit.json")
    parser.add_argument("--markdown-out", default="reports/HIGH_TIER_HEADROOM_AUDIT.md")
    parser.add_argument(
        "--omit-case-details", action="store_true",
        help=("retain per-window status/count/headroom but omit the 48 case "
              "records per window; useful for the all-candidate selection audit"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0 or args.jobs <= 0:
        parser.error("--frames and --jobs must be positive")

    protocol_path = _absolute(args.protocol)
    registry_path = _absolute(args.registry)
    trace_dir = _absolute(args.trace_dir)
    mpd_pattern = _absolute(args.mpd)
    protocol = load_protocol(protocol_path, trace_dir)
    registry = load_trace_registry(
        registry_path, protocol, trace_dir, verify_hashes=True,
        allow_candidate_registry=True,
    )
    settings = registry.settings()
    sequences = list(settings["required_sequences"])
    segment_frames = [int(value) for value in settings["required_segment_frames"]]
    jitter_seeds = [int(value) for value in settings["required_jitter_seeds"]]
    if any(int(value) != args.frames
           for value in settings["required_frames_per_sequence"].values()):
        raise ValueError("--frames does not match the frozen registry audit")
    pool, manifest_paths = _load_manifests(mpd_pattern, args.frames)
    validate_high_tier(pool, sequences)
    registry.validate_experiment(sequences, segment_frames[0], args.frames)
    windows = [
        window for split in ("train", "validation", "test")
        for window in registry.windows(split)
    ]

    context = {
        "protocol": protocol_path, "registry": registry_path,
        "trace_dir": trace_dir, "mpd": mpd_pattern, "frames": args.frames,
        "sequences": sequences, "segment_frames": segment_frames,
        "jitter_seeds": jitter_seeds,
    }
    print(
        f"Static-High audit: registry={registry.registry_id} windows={len(windows)} "
        f"cases/window={len(sequences) * len(segment_frames) * len(jitter_seeds)} "
        f"jobs={args.jobs}", flush=True,
    )
    indexed = list(enumerate(windows))
    results = [None] * len(windows)
    if args.jobs == 1:
        _initialize_worker(context)
        iterator = map(_audit_one, indexed)
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_initialize_worker,
            initargs=(context,),
        )
        iterator = executor.map(_audit_one, indexed, chunksize=1)
    try:
        for completed, (index, result) in enumerate(iterator, start=1):
            results[index] = result
            if completed == 1 or completed % 10 == 0 or completed == len(windows):
                print(f"audited {completed}/{len(windows)} windows", flush=True)
    finally:
        if args.jobs != 1:
            executor.shutdown()

    summary = summarize_high_audit(results)
    report_windows = results
    if args.omit_case_details:
        report_windows = [
            {key: value for key, value in window.items() if key != "cases"}
            for window in results
        ]
    candidate_mode = (
        registry.data.get("registry_type") == "candidate_high_tier_audit_windows"
    )
    payload = {
        "schema_version": 1,
        "audit_type": (
            "read_only_static_high_candidate_window_headroom"
            if candidate_mode else
            "read_only_static_high_frozen_registry_headroom"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_commit(),
        "protocol": os.path.relpath(protocol_path, PROJECT_ROOT).replace("\\", "/"),
        "protocol_digest": protocol_digest(protocol),
        "trace_registry": os.path.relpath(registry_path, PROJECT_ROOT).replace("\\", "/"),
        "trace_registry_id": registry.registry_id,
        "input_hashes": {
            "protocol": sha256_file(protocol_path),
            "registry": sha256_file(registry_path),
            "manifests": {
                os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"): sha256_file(path)
                for path in manifest_paths
            },
            "implementation": {
                "scripts/audit_high_tier_registry.py": sha256_file(__file__),
                "src/high_tier_audit.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "high_tier_audit.py")
                ),
                "src/network_model/buffer.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "network_model", "buffer.py")
                ),
                "src/network_model/finite_trace.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "network_model", "finite_trace.py")
                ),
                "src/network_model/links.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "network_model", "links.py")
                ),
                "src/network_model/manifest.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "network_model", "manifest.py")
                ),
                "src/network_model/session.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "network_model", "session.py")
                ),
                "src/network_model/tcp_protocol.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "network_model", "tcp_protocol.py")
                ),
                "src/rl/env.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "rl", "env.py")
                ),
                "src/rl/reward.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "rl", "reward.py")
                ),
                "src/trace_registry.py": sha256_file(
                    os.path.join(PROJECT_ROOT, "src", "trace_registry.py")
                ),
            },
        },
        "settings": {
            "policy": "Static High G-PCC tier",
            "frames_per_sequence": args.frames,
            "sequences": sequences,
            "segment_frames": segment_frames,
            "jitter_seeds": jitter_seeds,
            "terminal_capacity_policy": "finite; no final-sample clamping",
            "eligibility": "all 48 maximum-demand cases finish in measured data",
            "case_details_included": not args.omit_case_details,
            "case_count_per_window": (
                len(sequences) * len(segment_frames) * len(jitter_seeds)
            ),
        },
        "summary": summary,
        "windows": report_windows,
    }
    unsigned = dict(payload)
    payload["audit_id"] = _content_digest(unsigned)
    json_path = _write_new(
        args.json_out, json.dumps(payload, indent=2) + "\n", args.overwrite
    )
    markdown_path = _write_new(args.markdown_out, _markdown(payload), args.overwrite)
    print(
        f"Done: {summary['high_supported_window_count']}/{summary['window_count']} "
        f"windows High-supported; failed cases={summary['failed_case_count']}",
        flush=True,
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print("No trace, split, registry, training, or evaluation setting was changed.")


if __name__ == "__main__":
    main()
