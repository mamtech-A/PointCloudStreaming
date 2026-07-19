"""Timestamp-aware feasibility audit for point-cloud streaming windows.

This module deliberately does not change training or evaluation.  It previews
which timestamp-derived trace windows can carry a complete point-cloud session
under the least demanding available action (the static Very-low G-PCC tier).

The production :class:`BandwidthTrace` retains its historical clamp-after-end
behaviour for backward compatibility.  ``FiniteTraceWindow`` is stricter: it
raises as soon as a transfer cannot be completed with capacity measured before
the final timestamp.  Audit results can therefore identify windows that would
otherwise depend on an invented, indefinitely repeated terminal sample.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from .experiment_protocol import evenly_spaced_offsets, protocol_digest
from .network_model import DEFAULT_TCP_PARAMS
from .network_model.manifest import canonical_tier_name
from .network_model.manifest import coded_size_bytes
from .network_model.trace import (
    BandwidthTrace,
    TIMESTAMP_FMT,
)
from .network_model.finite_trace import (
    FiniteTraceWindow,
    TraceWindowExhausted,
    measured_end_time_s,
    sample_start_times,
)
from .rl.env import StreamingEnv
from .trace_gaps import (
    DEFAULT_MAX_GAP_S,
    block_for_source_index,
    split_trace_at_gaps,
    validate_gap_threshold,
)


AUDIT_SCHEMA_VERSION = 3
ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
# Backward-compatible import name for callers of the first audit draft. The
# value deliberately no longer claims that trace exhaustion is a physical
# network outage.
OUTAGE = INELIGIBLE
_TIME_EPSILON_S = 1e-9


def timestamp_grid_starts(trace, stride_s=60.0):
    """Choose exact starts ``0, stride, 2*stride, ...`` before measured end."""
    if len(trace) == 0:
        return []
    stride_s = float(stride_s)
    if not np.isfinite(stride_s) or stride_s <= 0:
        raise ValueError("timestamp grid stride must be a positive finite value")
    measured_end_s = measured_end_time_s(trace)
    starts = []
    index = 0
    while index * stride_s < measured_end_s - _TIME_EPSILON_S:
        starts.append(index * stride_s)
        index += 1
    return starts


def slice_trace_at_time(trace, start_time_s):
    """Slice at an exact wall-clock time, retaining the in-effect sample.

    Starting between rows must not move the candidate forward to the next row:
    under the simulator's hold-previous semantics, the previous observation is
    still the capacity in effect at the requested start.
    """
    start_time_s = float(start_time_s)
    end_time_s = measured_end_time_s(trace)
    if not np.isfinite(start_time_s) or start_time_s < 0:
        raise ValueError("window start must be a non-negative finite time")
    if start_time_s >= end_time_s - _TIME_EPSILON_S:
        raise ValueError(
            f"window start {start_time_s} is not before measured end {end_time_s}"
        )
    starts = sample_start_times(trace)
    source_index = max(0, bisect.bisect_right(starts, start_time_s) - 1)
    if abs(starts[source_index] - start_time_s) <= _TIME_EPSILON_S:
        samples = trace.samples[source_index:]
        sample_times = starts[source_index:]
    else:
        samples = [trace.samples[source_index], *trace.samples[source_index + 1:]]
        sample_times = [start_time_s, *starts[source_index + 1:]]
    window = BandwidthTrace(
        samples,
        name=trace.name,
        sample_times_s=sample_times,
    )
    window._audit_measured_end_s = end_time_s - start_time_s
    return window, source_index


def select_evenly_spaced_windows(windows, count):
    """Select up to ``count`` eligible windows, spread across start time."""
    eligible = sorted(
        (window for window in windows if window["status"] == ELIGIBLE),
        key=lambda window: (
            window["start_time_s"],
            window.get("source_sample_index", window.get("sample_offset", -1)),
        ),
    )
    count = max(1, int(count))
    if len(eligible) <= count:
        return eligible
    if count == 1:
        return [eligible[0]]
    first_s = eligible[0]["start_time_s"]
    last_s = eligible[-1]["start_time_s"]
    selected = []
    remaining = list(eligible)
    for index in range(count):
        target_s = first_s + index * (last_s - first_s) / float(count - 1)
        choice = min(
            remaining,
            key=lambda window: (
                abs(window["start_time_s"] - target_s),
                window["start_time_s"],
                window.get(
                    "source_sample_index", window.get("sample_offset", -1)
                ),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return sorted(
        selected,
        key=lambda window: (
            window["start_time_s"],
            window.get("source_sample_index", window.get("sample_offset", -1)),
        ),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kept_trace_rows(csv_path, state_filter="D"):
    """Read exactly the rows retained by ``load_5g_trace_with_time``."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if state_filter and row.get("State", state_filter) != state_filter:
                continue
            try:
                kbps = float(row["DL_bitrate"])
            except (KeyError, TypeError, ValueError):
                continue
            if kbps <= 0:
                continue
            raw_timestamp = row.get("Timestamp", "")
            try:
                timestamp = datetime.strptime(raw_timestamp, TIMESTAMP_FMT)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path}:{line_number}: invalid Timestamp "
                    f"{raw_timestamp!r}"
                ) from exc
            rows.append({
                "timestamp": timestamp,
                "network_mode": str(row.get("NetworkMode") or "Unknown").strip(),
                "bandwidth_bps": kbps * 1000.0,
            })
    if not rows:
        raise ValueError(f"{csv_path}: no positive download samples")
    for previous, current in zip(rows, rows[1:]):
        if current["timestamp"] < previous["timestamp"]:
            raise ValueError(
                f"{csv_path}: timestamps move backwards at "
                f"{current['timestamp'].strftime(TIMESTAMP_FMT)}"
            )
    return rows


def _weighted_summary(trace, network_modes):
    starts = sample_start_times(trace)
    measured_end_s = measured_end_time_s(trace)
    rat_duration_s = defaultdict(float)
    capacity_seconds = 0.0
    usable_capacities = []
    for index, (start_s, capacity_bps) in enumerate(zip(starts, trace.samples)):
        end_s = starts[index + 1] if index + 1 < len(starts) else measured_end_s
        end_s = min(end_s, measured_end_s)
        width_s = max(0.0, end_s - start_s)
        mode = network_modes[index] if index < len(network_modes) else "Unknown"
        rat_duration_s[mode] += width_s
        capacity_seconds += float(capacity_bps) * width_s
        if width_s > _TIME_EPSILON_S:
            usable_capacities.append(float(capacity_bps))
    duration_s = measured_end_s
    rat_share = {
        mode: (seconds / duration_s if duration_s else 0.0)
        for mode, seconds in sorted(rat_duration_s.items())
    }
    return {
        "duration_s": duration_s,
        "min_capacity_mbps": (
            min(usable_capacities) / 1e6 if usable_capacities else 0.0
        ),
        "mean_capacity_mbps": (
            capacity_seconds / duration_s / 1e6 if duration_s else 0.0
        ),
        "max_capacity_mbps": (
            max(usable_capacities) / 1e6 if usable_capacities else 0.0
        ),
        "rat_duration_s": dict(sorted(rat_duration_s.items())),
        "rat_share": rat_share,
    }


def read_trace_metadata(csv_path, trace=None):
    """Validate timestamps and report mobility/RAT composition."""
    rows = _kept_trace_rows(csv_path)
    trace = trace or BandwidthTrace.from_file(csv_path)
    if len(rows) != len(trace):
        raise AssertionError(
            f"loader/audit row mismatch for {csv_path}: {len(trace)} != {len(rows)}"
        )
    modes = [row["network_mode"] for row in rows]
    summary = _weighted_summary(trace, modes)
    timestamp_gaps = [
        (current["timestamp"] - previous["timestamp"]).total_seconds()
        for previous, current in zip(rows, rows[1:])
    ]
    duplicate_count = sum(gap == 0.0 for gap in timestamp_gaps)
    filename = os.path.basename(csv_path)
    mobility = ("Driving" if filename.lower().startswith("driving_")
                else "Static" if filename.lower().startswith("static_")
                else "Unknown")
    summary.update({
        "sample_count": len(trace),
        "mobility": mobility,
        "timestamps_valid": True,
        "timestamp_validation": "parseable and non-decreasing",
        "duplicate_timestamp_count": duplicate_count,
        "duplicate_timestamp_policy": (
            "rows sharing a timestamp split the interval to the next distinct "
            "timestamp evenly; terminal duplicates have no auditable interval"
        ),
        "timestamp_source": "Timestamp",
        "timestamp_gap_policy": "hold previous sample until next timestamp",
        "max_timestamp_gap_s": max(timestamp_gaps, default=0.0),
        "timestamp_gaps_over_5s": sum(gap > 5.0 for gap in timestamp_gaps),
        "timestamp_gaps_over_30s": sum(gap > 30.0 for gap in timestamp_gaps),
        "final_sample_width_s": 0.0,
        "final_sample_width_policy": "no capacity inferred after final timestamp",
        "first_timestamp": rows[0]["timestamp"].strftime(TIMESTAMP_FMT),
        "last_timestamp": rows[-1]["timestamp"].strftime(TIMESTAMP_FMT),
        "sha256": sha256_file(csv_path),
    })
    return summary, modes


def validate_manifest_pool(manifest_pool, sequences):
    """Reject missing/zero-byte Very-low representations before simulation."""
    for sequence in sequences:
        frames = manifest_pool[sequence]
        if not frames:
            raise ValueError(f"manifest sequence {sequence!r} is empty")
        expected_rep_id = None
        for frame_index, frame in enumerate(frames):
            matches = [
                rep for rep in frame.get("representations", [])
                if canonical_tier_name(rep.get("quality")) == "vlow"
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{sequence} frame {frame_index} must have exactly one "
                    f"Very-low representation; found {len(matches)}"
                )
            rep = matches[0]
            if coded_size_bytes(rep) <= 0:
                raise ValueError(
                    f"{sequence} frame {frame_index} has a zero-byte Very-low payload"
                )
            if expected_rep_id is None:
                expected_rep_id = rep["id"]
            elif rep["id"] != expected_rep_id:
                raise ValueError(
                    f"{sequence} changes its Very-low representation id from "
                    f"{expected_rep_id} to {rep['id']} at frame {frame_index}"
                )


def _very_low_action(env):
    state = env.current_abr_state()
    reps = sorted(state.reps, key=lambda rep: rep["id"])
    matches = [index for index, rep in enumerate(reps)
               if canonical_tier_name(rep.get("quality")) == "vlow"]
    if len(matches) != 1:
        raise ValueError(
            "each manifest frame must contain exactly one Very-low G-PCC tier; "
            f"found {len(matches)}"
        )
    return matches[0]


def audit_case(env, trace_window, sequence, jitter_seed):
    """Run one strict static-Very-low session and return a case record."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(jitter_seed))
    np.random.seed(int(jitter_seed))
    finite_trace = FiniteTraceWindow(trace_window)
    try:
        env.reset(finite_trace, sequence=sequence)
        while env.frame_idx < len(env.frames):
            env.step(_very_low_action(env))
    except TraceWindowExhausted as exc:
        return {
            "sequence": sequence,
            "jitter_seed": int(jitter_seed),
            "status": INELIGIBLE,
            "reason": "vlow_requires_unobserved_post_block_time",
            "completed_frames": int(env.frame_idx),
            "required_frames": len(env.frames),
            "measured_duration_s": float(finite_trace.duration_s),
            "measured_time_used_s": float(env.session.cumulative_time_s),
            "exhausted_at_s": float(exc.at_time_s),
            "remaining_bits_in_tcp_round": exc.remaining_bits,
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    completion_s = float(env.session.cumulative_time_s)
    if completion_s > finite_trace.duration_s + _TIME_EPSILON_S:
        raise AssertionError("finite trace allowed a session past its measured end")
    stats = env.user.get_buffer_stats()
    return {
        "sequence": sequence,
        "jitter_seed": int(jitter_seed),
        "status": ELIGIBLE,
        "reason": "eligible",
        "completed_frames": int(env.frame_idx),
        "required_frames": len(env.frames),
        "measured_duration_s": float(finite_trace.duration_s),
        "completion_time_s": completion_s,
        "headroom_s": float(finite_trace.duration_s - completion_s),
        # Difficult-but-feasible windows remain visible and eligible even when
        # playback stalls; stall is descriptive, never an exclusion criterion.
        "stall_duration_s": float(stats.get("total_stall_time_s", 0.0)),
        "rebuffer_events": int(stats.get("rebuffer_count", 0)),
        "startup_delay_s": float(stats.get("startup_delay_s", 0.0)),
        "request_pacing_s": float(env.session.total_request_pacing_s),
    }


def audit_window(env_by_segment, trace_window, sequences, jitter_seeds):
    """A window is eligible only when every required Very-low case finishes."""
    cases = []
    for segment_frames, env in sorted(env_by_segment.items()):
        for sequence in sequences:
            for seed in jitter_seeds:
                case = audit_case(env, trace_window, sequence, seed)
                case["segment_frames"] = int(segment_frames)
                cases.append(case)
    failures = [case for case in cases if case["status"] == INELIGIBLE]
    failed_combinations = []
    for segment_frames in sorted(env_by_segment):
        for sequence in sequences:
            group = [
                case for case in cases
                if case["segment_frames"] == segment_frames
                and case["sequence"] == sequence
            ]
            group_failures = [
                case for case in group if case["status"] == INELIGIBLE
            ]
            completions = [case["completion_time_s"] for case in group
                           if case["status"] == ELIGIBLE]
            headrooms = [case["headroom_s"] for case in group
                         if case["status"] == ELIGIBLE]
            stalls = [case["stall_duration_s"] for case in group
                      if case["status"] == ELIGIBLE]
            if group_failures:
                failed_combinations.append({
                    "segment_frames": int(segment_frames),
                    "sequence": sequence,
                    "failed_jitter_seeds": [
                        case["jitter_seed"] for case in group_failures
                    ],
                    "successful_jitter_seeds": [
                        case["jitter_seed"] for case in group
                        if case["status"] == ELIGIBLE
                    ],
                    "maximum_successful_completion_time_s": max(
                        completions, default=None
                    ),
                    "minimum_successful_headroom_s": min(
                        headrooms, default=None
                    ),
                    "maximum_successful_stall_duration_s": max(
                        stalls, default=None
                    ),
                })
    completions = [case["completion_time_s"] for case in cases
                   if case["status"] == ELIGIBLE]
    headrooms = [case["headroom_s"] for case in cases
                 if case["status"] == ELIGIBLE]
    stalls = [case["stall_duration_s"] for case in cases
              if case["status"] == ELIGIBLE]
    case_evidence = []
    for case in cases:
        is_eligible = case["status"] == ELIGIBLE
        case_evidence.append([
            int(case["segment_frames"]),
            case["sequence"],
            int(case["jitter_seed"]),
            case["status"],
            int(case["completed_frames"]),
            int(case["required_frames"]),
            float(case["completion_time_s"] if is_eligible
                  else case["exhausted_at_s"]),
            (float(case["headroom_s"]) if is_eligible else None),
            (float(case["stall_duration_s"]) if is_eligible else None),
        ])
    return {
        "status": INELIGIBLE if failures else ELIGIBLE,
        "case_count": len(cases),
        "failure_count": len(failures),
        "maximum_completion_time_s": max(completions, default=None),
        "minimum_headroom_s": min(headrooms, default=None),
        "maximum_stall_duration_s": max(stalls, default=None),
        "tested_segment_frames": sorted(env_by_segment),
        "tested_sequences": list(sequences),
        "tested_jitter_seeds": list(jitter_seeds),
        "case_evidence": case_evidence,
        "failed_combinations": failed_combinations,
        "failed_cases": failures,
    }


def audit_protocol(protocol, trace_dir, manifest_pool, *, grid_stride_s=60.0,
                   selected_windows_per_trace=3,
                   sequences=None, jitter_seeds=(42, 43, 44),
                   segment_frames=(5, 8, 10, 15), source_commit=None,
                   source_tracked_files_dirty=None, input_hashes=None,
                   progress=None, max_gap_s=DEFAULT_MAX_GAP_S):
    """Audit every trace in all registered splits without changing any input."""
    sequences = (sorted(manifest_pool) if sequences is None else list(sequences))
    jitter_seeds = [int(seed) for seed in jitter_seeds]
    segment_frames = [int(value) for value in segment_frames]
    if not sequences:
        raise ValueError("audit requires at least one content sequence")
    if not jitter_seeds:
        raise ValueError("audit requires at least one transport-jitter seed")
    if not segment_frames or any(value <= 0 for value in segment_frames):
        raise ValueError("segment_frames must contain positive integers")
    if int(selected_windows_per_trace) <= 0:
        raise ValueError("selected_windows_per_trace must be positive")
    if len(segment_frames) != len(set(segment_frames)):
        raise ValueError("segment_frames contains duplicate values")
    if len(jitter_seeds) != len(set(jitter_seeds)):
        raise ValueError("jitter_seeds contains duplicate values")
    if len(sequences) != len(set(sequences)):
        raise ValueError("sequences contains duplicate values")
    max_gap_s = validate_gap_threshold(max_gap_s)
    segment_frames = sorted(set(segment_frames))
    missing = sorted(set(sequences) - set(manifest_pool))
    if missing:
        raise ValueError(f"audit sequences absent from manifest pool: {missing}")
    validate_manifest_pool(manifest_pool, sequences)

    tcp_params = {**DEFAULT_TCP_PARAMS, "log_packets": False}
    env_by_segment = {
        size: StreamingEnv(
            manifest_pool,
            lstm_predictor=None,
            tcp_params=tcp_params,
            segment_frames=size,
        )
        for size in segment_frames
    }

    trace_records = []
    eligible_registry = []
    ineligible_registry = []
    legacy_evaluation_windows = []
    selected_evaluation_windows = []
    by_split = {
        split: {"traces": 0, "eligible_traces": 0, "excluded_traces": 0,
                "candidate_windows": 0, "eligible_windows": 0,
                "ineligible_windows": 0, "selected_windows": 0}
        for split in ("train", "validation", "test")
    }
    corpus_rat_duration = defaultdict(float)

    for split in ("train", "validation", "test"):
        for filename in protocol["trace_split"][split]:
            path = os.path.join(os.fspath(trace_dir), filename)
            full_trace = BandwidthTrace.from_file(path)
            metadata, modes = read_trace_metadata(path, full_trace)
            blocks = split_trace_at_gaps(full_trace, max_gap_s=max_gap_s)
            raw_timestamp_span_s = metadata["duration_s"]
            block_summaries = []
            trace_rat_duration = defaultdict(float)
            for block in blocks:
                block_modes = modes[
                    block.source_start_index:block.source_end_index + 1
                ]
                summary = _weighted_summary(block.trace, block_modes)
                block_summaries.append(summary)
                for mode, seconds in summary["rat_duration_s"].items():
                    trace_rat_duration[mode] += seconds
                    corpus_rat_duration[mode] += seconds
            usable_duration_s = sum(
                summary["duration_s"] for summary in block_summaries
            )
            metadata["raw_timestamp_span_s"] = raw_timestamp_span_s
            metadata["duration_s"] = usable_duration_s
            metadata["excluded_gap_duration_s"] = max(
                0.0, raw_timestamp_span_s - usable_duration_s
            )
            metadata["rat_duration_s"] = dict(sorted(trace_rat_duration.items()))
            metadata["rat_share"] = {
                mode: (seconds / usable_duration_s if usable_duration_s else 0.0)
                for mode, seconds in sorted(trace_rat_duration.items())
            }
            positive_summaries = [
                summary for summary in block_summaries
                if summary["duration_s"] > _TIME_EPSILON_S
            ]
            metadata["min_capacity_mbps"] = min(
                (summary["min_capacity_mbps"] for summary in positive_summaries),
                default=0.0,
            )
            metadata["mean_capacity_mbps"] = (
                sum(summary["mean_capacity_mbps"] * summary["duration_s"]
                    for summary in positive_summaries) / usable_duration_s
                if usable_duration_s else 0.0
            )
            metadata["max_capacity_mbps"] = max(
                (summary["max_capacity_mbps"] for summary in positive_summaries),
                default=0.0,
            )
            metadata["timestamp_gap_policy"] = (
                f"split before every consecutive timestamp delta > {max_gap_s:g} s"
            )
            candidate_count = sum(
                len(timestamp_grid_starts(block.trace, grid_stride_s))
                for block in blocks
            )
            if progress:
                progress(split, filename, candidate_count)
            windows = []
            block_records = []
            for block, block_summary in zip(blocks, block_summaries):
                block_windows = []
                candidate_starts = timestamp_grid_starts(
                    block.trace, grid_stride_s
                )
                for local_start_s in candidate_starts:
                    window_trace, block_source_index = slice_trace_at_time(
                        block.trace, local_start_s
                    )
                    source_index = (
                        block.source_start_index + block_source_index
                    )
                    global_start_s = block.global_start_time_s + local_start_s
                    window_summary = _weighted_summary(
                        window_trace,
                        modes[source_index:block.source_end_index + 1],
                    )
                    result = audit_window(
                        env_by_segment, window_trace, sequences, jitter_seeds
                    )
                    window_id = (
                        f"{split}/{filename}#{block.block_id}"
                        f"@time-{float(global_start_s):.3f}s"
                    )
                    window = {
                        "id": window_id,
                        "split": split,
                        "trace": filename,
                        "block_id": block.block_id,
                        "block_index": block.block_index,
                        "block_start_time_s": block.global_start_time_s,
                        "block_end_time_s": block.global_end_time_s,
                        "block_local_start_time_s": float(local_start_s),
                        "source_sample_index": int(source_index),
                        "start_time_s": float(global_start_s),
                        **window_summary,
                        **result,
                    }
                    window["reason"] = (
                        "eligible" if result["status"] == ELIGIBLE
                        else "no_measured_interval_after_start"
                        if window_summary["duration_s"] <= 0
                        else "vlow_requires_unobserved_post_block_time"
                    )
                    windows.append(window)
                    block_windows.append(window)
                    if result["status"] == ELIGIBLE:
                        eligible_registry.append(window)
                    else:
                        ineligible_registry.append(window)
                block_eligible_count = sum(
                    row["status"] == ELIGIBLE for row in block_windows
                )
                block_records.append({
                    "block_id": block.block_id,
                    "block_index": block.block_index,
                    "source_start_index": block.source_start_index,
                    "source_end_index": block.source_end_index,
                    "start_time_s": block.global_start_time_s,
                    "end_time_s": block.global_end_time_s,
                    "duration_s": block.duration_s,
                    "gap_before_s": block.gap_before_s,
                    "gap_after_s": block.gap_after_s,
                    "candidate_window_count": len(block_windows),
                    "eligible_window_count": block_eligible_count,
                    "ineligible_window_count": (
                        len(block_windows) - block_eligible_count
                    ),
                    "candidate_window_ids": [
                        row["id"] for row in block_windows
                    ],
                    **{
                        key: value for key, value in block_summary.items()
                        if key != "duration_s"
                    },
                })

            eligible_count = sum(w["status"] == ELIGIBLE for w in windows)
            trace_status = ELIGIBLE if eligible_count else INELIGIBLE
            selected = (select_evenly_spaced_windows(
                windows, selected_windows_per_trace
            ) if split in ("validation", "test") else [])
            selected_ids = [window["id"] for window in selected]
            for window in selected:
                selected_evaluation_windows.append({
                    "id": window["id"],
                    "split": split,
                    "trace": filename,
                    "block_id": window["block_id"],
                    "block_index": window["block_index"],
                    "source_sample_index": window["source_sample_index"],
                    "start_time_s": window["start_time_s"],
                    "measured_duration_s": window["duration_s"],
                    "status": window["status"],
                    "reason": window["reason"],
                    "case_count": window["case_count"],
                    "failure_count": window["failure_count"],
                })
            legacy_windows = []
            if split in ("validation", "test"):
                for offset in evenly_spaced_offsets(
                        len(full_trace), selected_windows_per_trace,
                        min_tail_samples=120):
                    block = block_for_source_index(blocks, offset)
                    local_offset = offset - block.source_start_index
                    local_start_s = block.trace.start_time_of_sample(
                        local_offset
                    )
                    if local_start_s < measured_end_time_s(block.trace):
                        legacy_trace, _ = slice_trace_at_time(
                            block.trace, local_start_s
                        )
                    else:
                        legacy_trace = BandwidthTrace(
                            [block.trace.samples[local_offset]],
                            name=block.trace.name,
                            sample_times_s=[0.0],
                        )
                        legacy_trace._audit_measured_end_s = 0.0
                    legacy = {
                        "sample_offset": int(offset),
                        "source_sample_index": int(offset),
                        "block_id": block.block_id,
                        "block_index": block.block_index,
                        "start_time_s": float(
                            block.global_start_time_s + local_start_s
                        ),
                        **_weighted_summary(
                            legacy_trace,
                            modes[offset:block.source_end_index + 1],
                        ),
                        **audit_window(
                            env_by_segment, legacy_trace, sequences,
                            jitter_seeds,
                        ),
                    }
                    legacy["id"] = (
                        f"legacy-{split}/{filename}@sample-{offset}"
                    )
                    legacy["split"] = split
                    legacy["trace"] = filename
                    legacy["reason"] = (
                        "eligible" if legacy["status"] == ELIGIBLE
                        else "no_measured_interval_after_start"
                        if legacy["duration_s"] <= 0
                        else "vlow_requires_unobserved_post_block_time"
                    )
                    legacy_windows.append(legacy)
                    legacy_evaluation_windows.append(legacy)
            trace_records.append({
                "split": split,
                "trace": filename,
                "status": trace_status,
                "candidate_window_count": len(windows),
                "eligible_window_count": eligible_count,
                "ineligible_window_count": len(windows) - eligible_count,
                "selected_window_count": len(selected),
                "selected_window_ids": selected_ids,
                "candidate_window_ids": [window["id"] for window in windows],
                "block_count": len(blocks),
                "split_gap_count": max(0, len(blocks) - 1),
                "blocks": block_records,
                "legacy_120_sample_window_ids": [
                    window["id"] for window in legacy_windows
                ],
                **metadata,
            })
            counts = by_split[split]
            counts["traces"] += 1
            counts["candidate_windows"] += len(windows)
            counts["eligible_windows"] += eligible_count
            counts["ineligible_windows"] += len(windows) - eligible_count
            counts["selected_windows"] += len(selected)
            if trace_status == ELIGIBLE:
                counts["eligible_traces"] += 1
            else:
                counts["excluded_traces"] += 1

    total_rat_s = sum(corpus_rat_duration.values())
    excluded_traces = [
        {"split": row["split"], "trace": row["trace"]}
        for row in trace_records if row["status"] == INELIGIBLE
    ]
    eligible_traces = [
        {"split": row["split"], "trace": row["trace"]}
        for row in trace_records if row["status"] == ELIGIBLE
    ]
    summary = {
        "trace_count": len(trace_records),
        "eligible_trace_count": len(eligible_traces),
        "excluded_trace_count": len(excluded_traces),
        "candidate_window_count": (
            len(eligible_registry) + len(ineligible_registry)
        ),
        "eligible_window_count": len(eligible_registry),
        "ineligible_window_count": len(ineligible_registry),
        "legacy_evaluation_window_count": len(legacy_evaluation_windows),
        "legacy_evaluation_ineligible_count": sum(
            row["status"] == INELIGIBLE for row in legacy_evaluation_windows
        ),
        "minimum_selected_evaluation_duration_s": min(
            (row["measured_duration_s"]
             for row in selected_evaluation_windows), default=None
        ),
        "by_split": by_split,
        "mobility_trace_count": {
            mobility: sum(row["mobility"] == mobility for row in trace_records)
            for mobility in ("Static", "Driving", "Unknown")
        },
        "traces_with_timestamp_gaps_over_5s": sum(
            row["timestamp_gaps_over_5s"] > 0 for row in trace_records
        ),
        "traces_with_timestamp_gaps_over_30s": sum(
            row["timestamp_gaps_over_30s"] > 0 for row in trace_records
        ),
        "maximum_timestamp_gap_s": max(
            (row["max_timestamp_gap_s"] for row in trace_records), default=0.0
        ),
        "duplicate_timestamp_count": sum(
            row["duplicate_timestamp_count"] for row in trace_records
        ),
        "contiguous_block_count": sum(
            row["block_count"] for row in trace_records
        ),
        "split_gap_count": sum(
            row["split_gap_count"] for row in trace_records
        ),
        "raw_timestamp_span_s": sum(
            row["raw_timestamp_span_s"] for row in trace_records
        ),
        "usable_block_duration_s": sum(
            row["duration_s"] for row in trace_records
        ),
        "excluded_gap_duration_s": sum(
            row["excluded_gap_duration_s"] for row in trace_records
        ),
        "corpus_rat_duration_s": dict(sorted(corpus_rat_duration.items())),
        "corpus_rat_share": {
            mode: (seconds / total_rat_s if total_rat_s else 0.0)
            for mode, seconds in sorted(corpus_rat_duration.items())
        },
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_type": "read_only_trace_window_feasibility",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_tracked_files_dirty": source_tracked_files_dirty,
        "input_hashes": input_hashes or {},
        "protocol_digest": protocol_digest(protocol),
        "settings": {
            "candidate_method": "dense_exact_timestamp_grid",
            "candidate_anchor": (
                "block-local zero; each contiguous block restarts the exact "
                "timestamp grid"
            ),
            "candidate_grid_stride_s": float(grid_stride_s),
            "maximum_contiguous_gap_s": float(max_gap_s),
            "gap_split_rule": (
                "split a parent trace before every consecutive timestamp "
                f"delta greater than {max_gap_s:g} seconds"
            ),
            "excluded_gap_duration_definition": (
                "exclude the full endpoint-to-endpoint delta at every split; "
                "infer no terminal width after the last pre-gap observation"
            ),
            "selected_windows_per_trace": int(selected_windows_per_trace),
            "legacy_comparison": (
                "current evenly_spaced_offsets with min_tail_samples=120; "
                "comparison only, never used for the new registry"
            ),
            "policy": "Static Very-low G-PCC tier",
            "required_sequences": sequences,
            "jitter_seeds": jitter_seeds,
            "segment_frames": segment_frames,
            "tcp_params": tcp_params,
            "required_frames_per_sequence": {
                name: len(manifest_pool[name]) for name in sequences
            },
            "eligibility": (
                "every required case completes before the final measured "
                "timestamp; stalls do not make a feasible window ineligible"
            ),
            "ineligibility_interpretation": (
                "Static Very-low requires unobserved post-block time. This is "
                "insufficient observed support and is excluded from ABR "
                "ranking; it is not asserted to be a physical network outage"
            ),
            "trace_retention": (
                "retain a trace when at least one candidate window is eligible"
            ),
            "terminal_capacity_policy": "finite; no final-sample clamping",
            "between_sample_policy": (
                "within a contiguous block, hold the previous capacity until "
                "the next timestamp"
            ),
            "internal_gap_status": (
                "resolved by read-only contiguous-block splitting; no window "
                "may cross a split gap"
            ),
            "final_sample_width_s": 0.0,
            "case_evidence_columns": [
                "segment_frames", "sequence", "jitter_seed", "status",
                "completed_frames", "required_frames", "end_or_exhaustion_time_s",
                "headroom_s", "stall_duration_s",
            ],
            "registry_layout": (
                "eligible_windows and ineligible_windows are self-contained; "
                "trace records reference them by candidate_window_ids"
            ),
            "required_future_consumer_policy": (
                "evaluation uses identical selected windows for every ABR and "
                "macro-averages parent-trace means; training samples parent "
                "traces uniformly before blocks/windows"
            ),
        },
        "summary": summary,
        "registries": {
            "eligible_windows": eligible_registry,
            "ineligible_windows": ineligible_registry,
            "eligible_traces": eligible_traces,
            "excluded_traces": excluded_traces,
            "selected_evaluation_windows": selected_evaluation_windows,
            "legacy_evaluation_windows": legacy_evaluation_windows,
        },
        "traces": trace_records,
    }


def _rat_text(shares):
    return ", ".join(
        f"{mode} {100.0 * value:.1f}%" for mode, value in shares.items()
    ) or "n/a"


def render_markdown_report(report):
    """Render a compact human-reviewable companion to the JSON registry."""
    settings = report["settings"]
    summary = report["summary"]
    lines = [
        "# Trace-window feasibility audit",
        "",
        "This is a read-only audit. It does not modify trace CSVs, protocol "
        "splits, training episodes, checkpoints, or evaluation code.",
        "",
        "## Decision rule",
        "",
        f"- Policy: {settings['policy']}.",
        f"- Required content: {', '.join(settings['required_sequences'])}.",
        f"- Candidate starts: a {settings['candidate_grid_stride_s']:g}-second "
        "grid restarted inside each contiguous block; no 120-row tail "
        "assumption is used.",
        f"- Before window generation, each parent trace is split when a "
        f"timestamp delta exceeds {settings['maximum_contiguous_gap_s']:g} s. "
        "No candidate can cross such a gap.",
        f"- Feasibility is the intersection of segment sizes "
        f"{settings['segment_frames']} and jitter seeds "
        f"{settings['jitter_seeds']}.",
        "- A window is eligible only when every required case finishes all "
        "frames before measured capacity ends.",
        "- Stalls do not cause exclusion when the session still finishes inside "
        "the measured window.",
        "- A trace is retained when at least one candidate window is eligible. "
        "No source trace is deleted by this audit.",
        "",
        "## Summary",
        "",
        f"- Traces: {summary['trace_count']} total; "
        f"{summary['eligible_trace_count']} retained; "
        f"{summary['excluded_trace_count']} lack an eligible window.",
        f"- Candidate windows: {summary['candidate_window_count']} total; "
        f"{summary['eligible_window_count']} eligible; "
        f"{summary['ineligible_window_count']} lack sufficient observed "
        "support.",
        f"- Gap split: {summary['split_gap_count']} discontinuities produce "
        f"{summary['contiguous_block_count']} contiguous blocks; "
        f"{summary['excluded_gap_duration_s']:.1f} s of unobserved intervals "
        "are excluded from capacity integration.",
        "- Excluded-gap duration uses the full endpoint-to-endpoint delta and "
        "assigns no inferred terminal second to the last pre-gap sample.",
        f"- Old 120-row rule: "
        f"{summary['legacy_evaluation_ineligible_count']}/"
        f"{summary['legacy_evaluation_window_count']} registered validation/test "
        "windows lack sufficient observed support under the strict test.",
        f"- Mobility: {summary['mobility_trace_count']}.",
        f"- Duration-weighted RAT composition: "
        f"{_rat_text(summary['corpus_rat_share'])}.",
        f"- Timestamp inventory: "
        f"{summary['traces_with_timestamp_gaps_over_5s']} traces contain a gap "
        f">5 s; the largest is {summary['maximum_timestamp_gap_s']:.1f} s.",
        "",
        "Raw CSVs and their parent train/validation/test assignments remain "
        "unchanged. Splitting is a read-only audit view; within a block, the "
        "previous observation is held only until the next observed timestamp.",
        "A rejected window is not labeled a physical network outage: the audit "
        "only establishes that Static Very-low cannot finish without "
        "unobserved post-block throughput. It is excluded from ABR ranking and "
        "reported separately as insufficient observed support.",
        "",
        "## Per-trace results",
        "",
        "| Split | Trace | Mobility | Blocks | Usable / raw span (s) | RAT mix | Eligible | Selected | Decision |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in report["traces"]:
        lines.append(
            f"| {row['split']} | `{row['trace']}` | {row['mobility']} | "
            f"{row['block_count']} | {row['duration_s']:.1f} / "
            f"{row['raw_timestamp_span_s']:.1f} | {_rat_text(row['rat_share'])} | "
            f"{row['eligible_window_count']}/{row['candidate_window_count']} | "
            f"{row['selected_window_count']} | "
            f"{row['status']} |"
        )
    lines.extend([
        "",
        "## Selected evaluation windows",
        "",
        "| Window | Block | Global start (s) | Measured tail (s) |",
        "|---|---|---:|---:|",
    ])
    for window in report["registries"]["selected_evaluation_windows"]:
        lines.append(
            f"| `{window['id']}` | {window['block_id']} | "
            f"{window['start_time_s']:.1f} | "
            f"{window['measured_duration_s']:.1f} |"
        )
    lines.extend([
        "",
        "## Windows with insufficient observed support",
        "",
        "| Window | Block | Global start (s) | Measured tail (s) | Reason | Failed cases |",
        "|---|---|---:|---:|---|---:|",
    ])
    for window in report["registries"]["ineligible_windows"]:
        lines.append(
            f"| `{window['id']}` | {window['block_id']} | "
            f"{window['start_time_s']:.1f} | "
            f"{window['duration_s']:.1f} | {window['reason']} | "
            f"{window['failure_count']} |"
        )
    lines.extend([
        "",
        "## Old 120-row rule: strict comparison",
        "",
        "| Window | Start (s) | Measured tail (s) | Status | Failed cases |",
        "|---|---:|---:|---|---:|",
    ])
    for window in report["registries"]["legacy_evaluation_windows"]:
        lines.append(
            f"| `{window['id']}` | {window['start_time_s']:.1f} | "
            f"{window['duration_s']:.1f} | {window['status']} | "
            f"{window['failure_count']} |"
        )
    lines.extend([
        "",
        "Ineligible windows are reported separately and must not be used to rank "
        "ABR algorithms. Reviewed eligible windows feed the maximum-tier support "
        "audit and the finite-window registry consumed by training/evaluation. "
        "Results must be macro-averaged per trace so traces with "
        "more eligible windows do not receive extra weight.",
        "Training must first sample parent traces uniformly, then sample an "
        "eligible block/window inside that parent; otherwise fragmented or "
        "long traces would be overrepresented.",
        "",
        "Static Very-low eligibility certifies only the minimum-demand action. "
        "The production registry must additionally retain only windows whose "
        "maximum-demand Static High cases all finish. Any unexpected runtime "
        "exhaustion is then a fatal protocol failure with no complete-session "
        "QoE; it must never clamp the final trace sample.",
        "",
    ])
    return "\n".join(lines)
