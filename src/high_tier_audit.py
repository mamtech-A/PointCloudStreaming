"""Read-only maximum-demand audit for frozen trace-registry windows."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from .network_model.finite_trace import TraceWindowExhausted
from .network_model.manifest import canonical_tier_name, coded_size_bytes


SUPPORTED = "high_supported"
UNSUPPORTED = "high_requires_unobserved_post_block_time"


def validate_high_tier(manifest_pool, sequences):
    """Require one non-empty High representation in every audited frame."""
    for sequence in sequences:
        frames = manifest_pool.get(sequence)
        if not frames:
            raise ValueError(f"manifest sequence {sequence!r} is absent or empty")
        expected_rep_id = None
        for frame_index, frame in enumerate(frames):
            matches = [
                rep for rep in frame.get("representations", [])
                if canonical_tier_name(rep.get("quality")) == "high"
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{sequence} frame {frame_index} must have exactly one High "
                    f"representation; found {len(matches)}"
                )
            rep = matches[0]
            if coded_size_bytes(rep) <= 0:
                raise ValueError(
                    f"{sequence} frame {frame_index} has a zero-byte High payload"
                )
            if expected_rep_id is None:
                expected_rep_id = rep["id"]
            elif rep["id"] != expected_rep_id:
                raise ValueError(
                    f"{sequence} changes its High representation id from "
                    f"{expected_rep_id} to {rep['id']} at frame {frame_index}"
                )


def high_action(env):
    """Return the environment action index for the fixed High tier."""
    state = env.current_abr_state()
    reps = sorted(state.reps, key=lambda rep: rep["id"])
    matches = [
        index for index, rep in enumerate(reps)
        if canonical_tier_name(rep.get("quality")) == "high"
    ]
    if len(matches) != 1:
        raise ValueError(
            "each manifest frame must contain exactly one High G-PCC tier; "
            f"found {len(matches)}"
        )
    return matches[0]


def audit_high_case(env, finite_trace, sequence, jitter_seed):
    """Run one static-High session without inventing post-window capacity."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(jitter_seed))
    np.random.seed(int(jitter_seed))
    try:
        env.reset(finite_trace, sequence=sequence)
        while env.frame_idx < len(env.frames):
            env.step(high_action(env))
    except TraceWindowExhausted as exc:
        return {
            "sequence": sequence,
            "segment_frames": int(env.segment_frames),
            "jitter_seed": int(jitter_seed),
            "status": UNSUPPORTED,
            "completed_frames": int(env.frame_idx),
            "required_frames": len(env.frames),
            "measured_duration_s": float(finite_trace.duration_s),
            "completed_download_time_s": float(env.session.cumulative_time_s),
            "exhausted_at_s": float(exc.at_time_s),
            "remaining_bits_in_tcp_round": exc.remaining_bits,
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    completion_s = float(env.session.cumulative_time_s)
    if completion_s > finite_trace.duration_s + 1e-9:
        raise AssertionError("finite trace allowed High past the measured end")
    stats = env.user.get_buffer_stats()
    return {
        "sequence": sequence,
        "segment_frames": int(env.segment_frames),
        "jitter_seed": int(jitter_seed),
        "status": SUPPORTED,
        "completed_frames": int(env.frame_idx),
        "required_frames": len(env.frames),
        "measured_duration_s": float(finite_trace.duration_s),
        "completion_time_s": completion_s,
        "headroom_s": float(finite_trace.duration_s - completion_s),
        "stall_duration_s": float(stats.get("total_stall_time_s", 0.0)),
        "rebuffer_events": int(stats.get("rebuffer_count", 0)),
        "startup_delay_s": float(stats.get("startup_delay_s", 0.0)),
        "frames_dropped": int(stats.get("frames_dropped", 0)),
    }


def audit_high_window(window, finite_trace, env_by_segment, sequences, seeds):
    """Audit all registered cases for one frozen window."""
    cases = []
    for segment_frames, env in sorted(env_by_segment.items()):
        for sequence in sequences:
            for seed in seeds:
                cases.append(audit_high_case(env, finite_trace, sequence, seed))
    failures = [case for case in cases if case["status"] == UNSUPPORTED]
    successes = [case for case in cases if case["status"] == SUPPORTED]
    headrooms = [case["headroom_s"] for case in successes]
    completions = [case["completion_time_s"] for case in successes]
    return {
        "id": window["id"],
        "split": window["split"],
        "trace": window["trace"],
        "block_id": window["block_id"],
        "start_time_s": float(window["start_time_s"]),
        "measured_duration_s": float(window["duration_s"]),
        "status": SUPPORTED if not failures else UNSUPPORTED,
        "case_count": len(cases),
        "failure_count": len(failures),
        "minimum_headroom_s": min(headrooms, default=None),
        "maximum_completion_time_s": max(completions, default=None),
        "cases": cases,
    }


def _dimension_summary(cases, key):
    groups = defaultdict(list)
    for case in cases:
        groups[str(case[key])].append(case)
    result = {}
    for value, rows in sorted(groups.items()):
        failures = sum(row["status"] == UNSUPPORTED for row in rows)
        headrooms = [
            row["headroom_s"] for row in rows if row["status"] == SUPPORTED
        ]
        result[value] = {
            "case_count": len(rows),
            "failure_count": failures,
            "failure_rate": failures / len(rows) if rows else 0.0,
            "minimum_successful_headroom_s": min(headrooms, default=None),
        }
    return result


def summarize_high_audit(windows):
    """Build split/trace/case summaries without discarding case evidence."""
    cases = [case for window in windows for case in window["cases"]]
    successful_headrooms = [
        case["headroom_s"] for case in cases if case["status"] == SUPPORTED
    ]
    split_summary = {}
    for split in ("train", "validation", "test"):
        rows = [window for window in windows if window["split"] == split]
        failures = sum(row["status"] == UNSUPPORTED for row in rows)
        split_summary[split] = {
            "window_count": len(rows),
            "high_supported_window_count": len(rows) - failures,
            "high_unsupported_window_count": failures,
            "case_count": sum(row["case_count"] for row in rows),
            "failed_case_count": sum(row["failure_count"] for row in rows),
        }
    trace_summary = {}
    for trace in sorted({window["trace"] for window in windows}):
        rows = [window for window in windows if window["trace"] == trace]
        supported = sum(row["status"] == SUPPORTED for row in rows)
        trace_summary[trace] = {
            "split": rows[0]["split"],
            "window_count": len(rows),
            "high_supported_window_count": supported,
            "high_unsupported_window_count": len(rows) - supported,
            "failed_case_count": sum(row["failure_count"] for row in rows),
        }
    unsupported_windows = sum(
        window["status"] == UNSUPPORTED for window in windows
    )
    return {
        "window_count": len(windows),
        "high_supported_window_count": len(windows) - unsupported_windows,
        "high_unsupported_window_count": unsupported_windows,
        "case_count": len(cases),
        "successful_case_count": sum(case["status"] == SUPPORTED for case in cases),
        "failed_case_count": sum(case["status"] == UNSUPPORTED for case in cases),
        "minimum_successful_headroom_s": min(successful_headrooms, default=None),
        "by_split": split_summary,
        "by_trace": trace_summary,
        "by_segment_frames": _dimension_summary(cases, "segment_frames"),
        "by_sequence": _dimension_summary(cases, "sequence"),
        "by_jitter_seed": _dimension_summary(cases, "jitter_seed"),
    }
