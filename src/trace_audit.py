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
from .rl.env import StreamingEnv


AUDIT_SCHEMA_VERSION = 1
ELIGIBLE = "eligible"
OUTAGE = "outage"
_TIME_EPSILON_S = 1e-9


class TraceWindowExhausted(RuntimeError):
    """Raised when measured samples cannot finish the current transfer."""

    def __init__(self, trace_name, at_time_s, trace_end_s, remaining_bits=None):
        self.trace_name = trace_name
        self.at_time_s = float(at_time_s)
        self.trace_end_s = float(trace_end_s)
        self.remaining_bits = (None if remaining_bits is None
                               else max(0.0, float(remaining_bits)))
        detail = ""
        if self.remaining_bits is not None:
            detail = f"; {self.remaining_bits:.0f} bits remain in the TCP round"
        super().__init__(
            f"trace window {trace_name!r} ends at {trace_end_s:.6f}s "
            f"before the transfer can finish{detail}"
        )


def sample_start_times(trace):
    """Return the trace's real, strictly increasing per-sample start times."""
    return [trace.start_time_of_sample(i) for i in range(len(trace))]


def measured_end_time_s(trace):
    """Last observed timestamp relative to the window's first timestamp.

    ``BandwidthTrace.duration_s`` includes a compatibility-only one-second
    extension for its last row.  The audit intentionally excludes that inferred
    interval and ends exactly at the last timestamp present in the CSV.
    """
    raw_times = getattr(trace, "_raw_times", None)
    return float(raw_times[-1]) if raw_times else 0.0


class FiniteTraceWindow:
    """Read-only trace view whose final sample ends at ``duration_s``.

    Unlike ``BandwidthTrace.time_to_transmit``, the last sample is not held
    indefinitely.  The wrapper exposes the same small interface consumed by
    ``AccessLink`` and raises ``TraceWindowExhausted`` at the measured boundary.
    """

    def __init__(self, trace):
        if not isinstance(trace, BandwidthTrace):
            raise TypeError("FiniteTraceWindow requires a BandwidthTrace")
        if not trace.samples:
            raise ValueError("cannot audit an empty bandwidth trace")
        self._trace = trace
        self.samples = trace.samples
        self.name = trace.name
        self._starts = sample_start_times(trace)

    @property
    def duration_s(self):
        return measured_end_time_s(self._trace)

    def capacity_at_time(self, t_s):
        t_s = float(t_s)
        if t_s >= self.duration_s - _TIME_EPSILON_S:
            raise TraceWindowExhausted(
                self.name, max(t_s, self.duration_s), self.duration_s
            )
        return self._trace.capacity_at_time(t_s)

    def time_to_transmit(self, t_start_s, bits):
        """Integrate only over measured intervals; never clamp after the end."""
        bits = float(bits)
        if bits <= 0:
            return 0.0

        t = max(0.0, float(t_start_s))
        if t >= self.duration_s - _TIME_EPSILON_S:
            raise TraceWindowExhausted(
                self.name, max(t, self.duration_s), self.duration_s, bits
            )

        idx = bisect.bisect_right(self._starts, t) - 1
        idx = max(0, idx)
        remaining = bits
        elapsed = 0.0
        n = len(self.samples)

        while idx < n:
            interval_end = (self._starts[idx + 1]
                            if idx + 1 < n else self.duration_s)
            interval_s = max(0.0, interval_end - t)
            rate_bps = max(float(self.samples[idx]), 1e3)
            deliverable = rate_bps * interval_s
            if deliverable + 1e-6 >= remaining:
                return elapsed + remaining / rate_bps
            remaining -= deliverable
            elapsed += interval_s
            t = interval_end
            idx += 1

        raise TraceWindowExhausted(
            self.name, self.duration_s, self.duration_s, remaining
        )

    def __len__(self):
        return len(self.samples)


def timestamp_grid_offsets(trace, stride_s=60.0):
    """Choose a dense, deterministic grid on the real timestamp axis.

    Targets are ``0, stride_s, 2*stride_s, ... < duration`` and each maps to
    the first measured sample at or after that time.  Duplicate mapped offsets
    (possible across timestamp gaps) are removed.  This is intentionally
    independent of row count and has no minimum-tail-sample assumption.
    """
    if len(trace) == 0:
        return []
    stride_s = float(stride_s)
    if not np.isfinite(stride_s) or stride_s <= 0:
        raise ValueError("timestamp grid stride must be a positive finite value")
    starts = sample_start_times(trace)
    offsets = []
    target_s = 0.0
    measured_end_s = measured_end_time_s(trace)
    while target_s < measured_end_s:
        offset = min(bisect.bisect_left(starts, target_s), len(starts) - 1)
        if offset not in offsets:
            offsets.append(offset)
        target_s += stride_s
    return offsets


def select_evenly_spaced_windows(windows, count):
    """Select up to ``count`` eligible windows, spread across start time."""
    eligible = sorted(
        (window for window in windows if window["status"] == ELIGIBLE),
        key=lambda window: (window["start_time_s"], window["sample_offset"]),
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
                window["sample_offset"],
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return sorted(
        selected,
        key=lambda window: (window["start_time_s"], window["sample_offset"]),
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
    for index, (start_s, capacity_bps) in enumerate(zip(starts, trace.samples)):
        end_s = starts[index + 1] if index + 1 < len(starts) else measured_end_s
        end_s = min(end_s, measured_end_s)
        width_s = max(0.0, end_s - start_s)
        mode = network_modes[index] if index < len(network_modes) else "Unknown"
        rat_duration_s[mode] += width_s
        capacity_seconds += float(capacity_bps) * width_s
    duration_s = measured_end_s
    rat_share = {
        mode: (seconds / duration_s if duration_s else 0.0)
        for mode, seconds in sorted(rat_duration_s.items())
    }
    return {
        "duration_s": duration_s,
        "min_capacity_mbps": min(trace.samples) / 1e6,
        "mean_capacity_mbps": (
            capacity_seconds / duration_s / 1e6 if duration_s else 0.0
        ),
        "max_capacity_mbps": max(trace.samples) / 1e6,
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
    filename = os.path.basename(csv_path)
    mobility = ("Driving" if filename.lower().startswith("driving_")
                else "Static" if filename.lower().startswith("static_")
                else "Unknown")
    summary.update({
        "sample_count": len(trace),
        "mobility": mobility,
        "timestamps_valid": True,
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
            "status": OUTAGE,
            "reason": "vlow_exceeds_trace_end",
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
        "frames_dropped": int(stats.get("frames_dropped", 0)),
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
    failures = [case for case in cases if case["status"] == OUTAGE]
    failed_combinations = []
    for segment_frames in sorted(env_by_segment):
        for sequence in sequences:
            group = [
                case for case in cases
                if case["segment_frames"] == segment_frames
                and case["sequence"] == sequence
            ]
            group_failures = [case for case in group if case["status"] == OUTAGE]
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
    return {
        "status": OUTAGE if failures else ELIGIBLE,
        "case_count": len(cases),
        "failure_count": len(failures),
        "maximum_completion_time_s": max(completions, default=None),
        "minimum_headroom_s": min(headrooms, default=None),
        "maximum_stall_duration_s": max(stalls, default=None),
        "tested_segment_frames": sorted(env_by_segment),
        "tested_sequences": list(sequences),
        "tested_jitter_seeds": list(jitter_seeds),
        "failed_combinations": failed_combinations,
        "failed_cases": failures,
    }


def audit_protocol(protocol, trace_dir, manifest_pool, *, grid_stride_s=60.0,
                   selected_windows_per_trace=3,
                   sequences=None, jitter_seeds=(42, 43, 44),
                   segment_frames=(5, 8, 10, 15), source_commit=None,
                   input_hashes=None, progress=None):
    """Audit every trace in all registered splits without changing any input."""
    sequences = list(sequences or sorted(manifest_pool))
    jitter_seeds = [int(seed) for seed in jitter_seeds]
    segment_frames = sorted({max(1, int(value)) for value in segment_frames})
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
    outage_registry = []
    legacy_evaluation_windows = []
    by_split = {
        split: {"traces": 0, "eligible_traces": 0, "excluded_traces": 0,
                "candidate_windows": 0, "eligible_windows": 0,
                "outage_windows": 0, "selected_windows": 0}
        for split in ("train", "validation", "test")
    }
    corpus_rat_duration = defaultdict(float)

    for split in ("train", "validation", "test"):
        for filename in protocol["trace_split"][split]:
            path = os.path.join(os.fspath(trace_dir), filename)
            full_trace = BandwidthTrace.from_file(path)
            metadata, modes = read_trace_metadata(path, full_trace)
            for mode, seconds in metadata["rat_duration_s"].items():
                corpus_rat_duration[mode] += seconds
            offsets = timestamp_grid_offsets(full_trace, grid_stride_s)
            if progress:
                progress(split, filename, len(offsets))
            windows = []
            for offset in offsets:
                window_trace = full_trace.slice_from(offset)
                window_summary = _weighted_summary(window_trace, modes[offset:])
                result = audit_window(
                    env_by_segment, window_trace, sequences, jitter_seeds
                )
                start_time_s = full_trace.start_time_of_sample(offset)
                window_id = f"{split}/{filename}@sample-{offset}"
                window = {
                    "id": window_id,
                    "sample_offset": int(offset),
                    "start_time_s": float(start_time_s),
                    **window_summary,
                    **result,
                }
                window["reason"] = (
                    "eligible" if result["status"] == ELIGIBLE
                    else "no_measured_interval_after_start"
                    if window_summary["duration_s"] <= 0
                    else "vlow_exceeds_trace_end"
                )
                windows.append(window)
                registry_row = {
                    "id": window_id,
                    "split": split,
                    "trace": filename,
                    "sample_offset": int(offset),
                    "start_time_s": float(start_time_s),
                    "measured_duration_s": float(measured_end_time_s(window_trace)),
                }
                if result["status"] == ELIGIBLE:
                    eligible_registry.append(registry_row)
                else:
                    outage_registry.append(registry_row)

            eligible_count = sum(w["status"] == ELIGIBLE for w in windows)
            trace_status = ELIGIBLE if eligible_count else OUTAGE
            selected = select_evenly_spaced_windows(
                windows, selected_windows_per_trace
            )
            selected_ids = [window["id"] for window in selected]
            legacy_windows = []
            if split in ("validation", "test"):
                windows_by_offset = {
                    window["sample_offset"]: window for window in windows
                }
                for offset in evenly_spaced_offsets(
                        len(full_trace), selected_windows_per_trace,
                        min_tail_samples=120):
                    if offset in windows_by_offset:
                        legacy = dict(windows_by_offset[offset])
                    else:
                        legacy_trace = full_trace.slice_from(offset)
                        legacy = {
                            "sample_offset": int(offset),
                            "start_time_s": float(
                                full_trace.start_time_of_sample(offset)
                            ),
                            **_weighted_summary(legacy_trace, modes[offset:]),
                            **audit_window(
                                env_by_segment, legacy_trace, sequences,
                                jitter_seeds,
                            ),
                        }
                    legacy["id"] = (
                        f"legacy-{split}/{filename}@sample-{offset}"
                    )
                    legacy["reason"] = (
                        "eligible" if legacy["status"] == ELIGIBLE
                        else "no_measured_interval_after_start"
                        if legacy["duration_s"] <= 0
                        else "vlow_exceeds_trace_end"
                    )
                    legacy_windows.append(legacy)
                    legacy_evaluation_windows.append({
                        "id": legacy["id"],
                        "split": split,
                        "trace": filename,
                        "sample_offset": int(offset),
                        "start_time_s": legacy["start_time_s"],
                        "measured_duration_s": legacy["duration_s"],
                        "status": legacy["status"],
                        "failure_count": legacy["failure_count"],
                    })
            trace_records.append({
                "split": split,
                "trace": filename,
                "status": trace_status,
                "candidate_window_count": len(windows),
                "eligible_window_count": eligible_count,
                "outage_window_count": len(windows) - eligible_count,
                "selected_window_count": len(selected),
                "selected_window_ids": selected_ids,
                "legacy_120_sample_windows": legacy_windows,
                **metadata,
                "windows": windows,
            })
            counts = by_split[split]
            counts["traces"] += 1
            counts["candidate_windows"] += len(windows)
            counts["eligible_windows"] += eligible_count
            counts["outage_windows"] += len(windows) - eligible_count
            counts["selected_windows"] += len(selected)
            if trace_status == ELIGIBLE:
                counts["eligible_traces"] += 1
            else:
                counts["excluded_traces"] += 1

    total_rat_s = sum(corpus_rat_duration.values())
    excluded_traces = [
        {"split": row["split"], "trace": row["trace"]}
        for row in trace_records if row["status"] == OUTAGE
    ]
    eligible_traces = [
        {"split": row["split"], "trace": row["trace"]}
        for row in trace_records if row["status"] == ELIGIBLE
    ]
    selected_evaluation_windows = []
    for row in trace_records:
        if row["split"] not in ("validation", "test"):
            continue
        by_id = {window["id"]: window for window in row["windows"]}
        for window_id in row["selected_window_ids"]:
            window = by_id[window_id]
            selected_evaluation_windows.append({
                "id": window_id,
                "split": row["split"],
                "trace": row["trace"],
                "sample_offset": window["sample_offset"],
                "start_time_s": window["start_time_s"],
                "measured_duration_s": window["duration_s"],
            })
    summary = {
        "trace_count": len(trace_records),
        "eligible_trace_count": len(eligible_traces),
        "excluded_trace_count": len(excluded_traces),
        "candidate_window_count": len(eligible_registry) + len(outage_registry),
        "eligible_window_count": len(eligible_registry),
        "outage_window_count": len(outage_registry),
        "legacy_evaluation_window_count": len(legacy_evaluation_windows),
        "legacy_evaluation_outage_count": sum(
            row["status"] == OUTAGE for row in legacy_evaluation_windows
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
        "input_hashes": input_hashes or {},
        "protocol_digest": protocol_digest(protocol),
        "settings": {
            "candidate_method": "dense_timestamp_grid",
            "candidate_grid_stride_s": float(grid_stride_s),
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
            "trace_retention": (
                "retain a trace when at least one candidate window is eligible"
            ),
            "terminal_capacity_policy": "finite; no final-sample clamping",
            "between_sample_policy": "hold previous capacity until next timestamp",
            "internal_gap_status": (
                "reported but unresolved; registry must not be activated until "
                "a maximum-gap or contiguous-block rule is approved"
            ),
            "final_sample_width_s": 0.0,
        },
        "summary": summary,
        "registries": {
            "eligible_windows": eligible_registry,
            "outage_windows": outage_registry,
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
        "This is a read-only preview. It does not modify trace CSVs, protocol "
        "splits, training episodes, checkpoints, or evaluation code.",
        "",
        "## Decision rule",
        "",
        f"- Policy: {settings['policy']}.",
        f"- Required content: {', '.join(settings['required_sequences'])}.",
        f"- Candidate starts: a {settings['candidate_grid_stride_s']:g}-second "
        "grid on each trace's real timestamp axis; no 120-row tail assumption "
        "is used.",
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
        f"{summary['excluded_trace_count']} classified as outages.",
        f"- Candidate windows: {summary['candidate_window_count']} total; "
        f"{summary['eligible_window_count']} eligible; "
        f"{summary['outage_window_count']} outages.",
        f"- Old 120-row rule: {summary['legacy_evaluation_outage_count']}/"
        f"{summary['legacy_evaluation_window_count']} registered validation/test "
        "windows are outages under the strict test.",
        f"- Mobility: {summary['mobility_trace_count']}.",
        f"- Duration-weighted RAT composition: "
        f"{_rat_text(summary['corpus_rat_share'])}.",
        f"- Timestamp-gap warning: "
        f"{summary['traces_with_timestamp_gaps_over_5s']} traces contain a gap "
        f">5 s; the largest is {summary['maximum_timestamp_gap_s']:.1f} s.",
        "",
        "> **Audit caveat:** the current simulator holds the previous measured "
        "capacity between timestamps. This report prevents all capacity use "
        "after the final timestamp, but it does not resolve long internal gaps. "
        "A maximum-gap/splitting rule must be approved before the registry is "
        "activated.",
        "",
        "## Per-trace results",
        "",
        "| Split | Trace | Mobility | Duration (s) | RAT mix | Eligible | Selected | Decision |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in report["traces"]:
        lines.append(
            f"| {row['split']} | `{row['trace']}` | {row['mobility']} | "
            f"{row['duration_s']:.1f} | {_rat_text(row['rat_share'])} | "
            f"{row['eligible_window_count']}/{row['candidate_window_count']} | "
            f"{row['selected_window_count']} | "
            f"{row['status']} |"
        )
    lines.extend([
        "",
        "## Selected evaluation windows",
        "",
        "| Window | Start (s) | Measured tail (s) |",
        "|---|---:|---:|",
    ])
    for window in report["registries"]["selected_evaluation_windows"]:
        lines.append(
            f"| `{window['id']}` | {window['start_time_s']:.1f} | "
            f"{window['measured_duration_s']:.1f} |"
        )
    lines.extend([
        "",
        "## Outage windows",
        "",
        "| Window | Start (s) | Measured tail (s) | Reason | Failed cases |",
        "|---|---:|---:|---|---:|",
    ])
    for trace in report["traces"]:
        for window in trace["windows"]:
            if window["status"] != OUTAGE:
                continue
            lines.append(
                f"| `{window['id']}` | {window['start_time_s']:.1f} | "
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
            f"{window['measured_duration_s']:.1f} | {window['status']} | "
            f"{window['failure_count']} |"
        )
    lines.extend([
        "",
        "Outage windows are reported separately and must not be used to rank "
        "ABR algorithms. The registry is not consumed by training/evaluation "
        "until the audit is reviewed and a follow-up pipeline change is approved. "
        "When activated, results must be macro-averaged per trace so traces with "
        "more eligible windows do not receive extra weight.",
        "",
        "Static Very-low eligibility certifies that a window has a feasible "
        "action, not that every ABR action will finish. The follow-up runtime "
        "guard must record a policy that exhausts an eligible trace as "
        "`policy_trace_exhausted`; it must never clamp or silently remove that "
        "policy's case.",
        "",
    ])
    return "\n".join(lines)
