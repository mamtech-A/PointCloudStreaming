"""Timestamp-gap inventory and contiguous-block construction for 5G traces.

Raw CSV files are never rewritten.  A gap policy is represented as a list of
read-only ``TraceBlock`` objects; consecutive observations belong to different
blocks when their timestamp delta is greater than ``max_gap_s``.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from .network_model.trace import BandwidthTrace, TIMESTAMP_FMT


DEFAULT_MAX_GAP_S = 5.0
DEFAULT_SENSITIVITY_THRESHOLDS_S = (2.0, 3.0, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class TraceBlock:
    """One contiguous timestamp block inherited from a parent trace file."""

    block_id: str
    block_index: int
    trace: BandwidthTrace
    source_start_index: int
    source_end_index: int
    global_start_time_s: float
    global_end_time_s: float
    duration_s: float
    gap_before_s: float | None
    gap_after_s: float | None


def raw_observation_times_s(trace):
    """Return the timestamp-derived observation axis retained by the loader."""
    raw = getattr(trace, "_raw_times", None)
    if raw is None or len(raw) != len(trace):
        raise ValueError("trace does not expose a complete timestamp axis")
    return [float(value) for value in raw]


def validate_gap_threshold(max_gap_s):
    max_gap_s = float(max_gap_s)
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be a positive finite value")
    return max_gap_s


def split_trace_at_gaps(trace, max_gap_s=DEFAULT_MAX_GAP_S):
    """Split ``trace`` whenever a consecutive timestamp delta exceeds policy.

    The returned traces preserve all samples and their within-block timestamp
    spacing.  Capacity is never invented across a split: each block's auditable
    end is exactly its last measured timestamp.
    """
    max_gap_s = validate_gap_threshold(max_gap_s)
    if len(trace) == 0:
        return []
    raw = raw_observation_times_s(trace)
    boundary_indexes = [
        index + 1 for index, (left, right) in enumerate(zip(raw, raw[1:]))
        if right - left > max_gap_s
    ]
    block_starts = [0, *boundary_indexes]
    block_ends = [index - 1 for index in boundary_indexes] + [len(trace) - 1]
    blocks = []
    for block_index, (start, end) in enumerate(zip(block_starts, block_ends)):
        block_trace = BandwidthTrace(
            trace.samples[start:end + 1],
            name=trace.name,
            sample_times_s=raw[start:end + 1],
        )
        duration_s = max(0.0, raw[end] - raw[start])
        # Audit code checks this explicit boundary instead of BandwidthTrace's
        # compatibility-only one-second extension of the final sample.
        block_trace._audit_measured_end_s = duration_s
        gap_before = (raw[start] - raw[start - 1]) if start else None
        gap_after = (raw[end + 1] - raw[end]) if end + 1 < len(raw) else None
        blocks.append(TraceBlock(
            block_id=f"block-{block_index:03d}",
            block_index=block_index,
            trace=block_trace,
            source_start_index=start,
            source_end_index=end,
            global_start_time_s=raw[start],
            global_end_time_s=raw[end],
            duration_s=duration_s,
            gap_before_s=gap_before,
            gap_after_s=gap_after,
        ))
    return blocks


def block_for_source_index(blocks, source_index):
    source_index = int(source_index)
    for block in blocks:
        if block.source_start_index <= source_index <= block.source_end_index:
            return block
    raise IndexError(f"sample index {source_index} is outside every trace block")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_kept_rows(csv_path, state_filter="D"):
    """Read rows retained by the simulator while preserving audit context."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if state_filter and row.get("State", state_filter) != state_filter:
                continue
            bitrate_kbps = _number(row.get("DL_bitrate"))
            if bitrate_kbps is None or bitrate_kbps <= 0:
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
                "sample_index": len(rows),
                "csv_line": line_number,
                "timestamp": timestamp,
                "timestamp_text": raw_timestamp,
                "state": str(row.get("State") or ""),
                "network_mode": str(row.get("NetworkMode") or "Unknown"),
                "dl_bitrate_kbps": bitrate_kbps,
                "speed_kmh": _number(row.get("Speed")),
                "cell_id": str(row.get("CellID") or ""),
                "longitude": _number(row.get("Longitude")),
                "latitude": _number(row.get("Latitude")),
            })
    if not rows:
        raise ValueError(f"{csv_path}: no positive download samples")
    for previous, current in zip(rows, rows[1:]):
        if current["timestamp"] < previous["timestamp"]:
            raise ValueError(
                f"{csv_path}: timestamps move backwards at "
                f"{current['timestamp_text']}"
            )
    return rows


def gap_records(rows, *, split, trace_name, long_gap_s=DEFAULT_MAX_GAP_S):
    long_gap_s = validate_gap_threshold(long_gap_s)
    records = []
    for before, after in zip(rows, rows[1:]):
        gap_s = (after["timestamp"] - before["timestamp"]).total_seconds()
        if gap_s <= long_gap_s:
            continue
        records.append({
            "split": split,
            "trace": trace_name,
            "gap_s": float(gap_s),
            "unobserved_beyond_1hz_s": max(0.0, float(gap_s) - 1.0),
            "before": {key: value for key, value in before.items()
                       if key != "timestamp"},
            "after": {key: value for key, value in after.items()
                      if key != "timestamp"},
            "network_mode_changed": (
                before["network_mode"] != after["network_mode"]
            ),
            "cell_id_changed": before["cell_id"] != after["cell_id"],
        })
    return records


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def threshold_sensitivity(trace_rows, thresholds_s=DEFAULT_SENSITIVITY_THRESHOLDS_S,
                          grid_stride_s=60.0):
    """Summarize fragmentation without running the expensive stream audit."""
    results = []
    for threshold in thresholds_s:
        threshold = validate_gap_threshold(threshold)
        durations = []
        split_gap_count = 0
        affected_traces = 0
        candidate_windows = 0
        raw_span_s = 0.0
        for _split, _trace_name, rows in trace_rows:
            times = [row["timestamp"] for row in rows]
            deltas = [
                (right - left).total_seconds()
                for left, right in zip(times, times[1:])
            ]
            boundaries = [
                index + 1 for index, delta in enumerate(deltas)
                if delta > threshold
            ]
            if boundaries:
                affected_traces += 1
            split_gap_count += len(boundaries)
            starts = [0, *boundaries]
            ends = [index - 1 for index in boundaries] + [len(rows) - 1]
            raw_span_s += (times[-1] - times[0]).total_seconds()
            for start, end in zip(starts, ends):
                duration_s = max(0.0, (times[end] - times[start]).total_seconds())
                durations.append(duration_s)
                if duration_s > 0:
                    candidate_windows += int(math.ceil(duration_s / grid_stride_s))
        usable_s = sum(durations)
        results.append({
            "max_gap_s": threshold,
            "split_gap_count": split_gap_count,
            "affected_trace_count": affected_traces,
            "block_count": len(durations),
            "zero_duration_block_count": sum(value <= 0 for value in durations),
            "blocks_shorter_than_10s": sum(value < 10.0 for value in durations),
            "minimum_block_duration_s": min(durations, default=0.0),
            "median_block_duration_s": statistics.median(durations) if durations else 0.0,
            "p10_block_duration_s": _percentile(durations, 10),
            "p90_block_duration_s": _percentile(durations, 90),
            "maximum_block_duration_s": max(durations, default=0.0),
            "candidate_window_count_at_grid_stride": candidate_windows,
            "usable_block_duration_s": usable_s,
            "raw_timestamp_span_s": raw_span_s,
            "excluded_gap_duration_s": max(0.0, raw_span_s - usable_s),
            "usable_duration_fraction": usable_s / raw_span_s if raw_span_s else 0.0,
        })
    return results


def build_gap_inventory(protocol, trace_dir, *, long_gap_s=DEFAULT_MAX_GAP_S,
                        sensitivity_thresholds_s=DEFAULT_SENSITIVITY_THRESHOLDS_S,
                        grid_stride_s=60.0):
    trace_rows = []
    all_deltas = []
    traces = []
    long_gaps = []
    for split in ("train", "validation", "test"):
        for trace_name in protocol["trace_split"][split]:
            path = os.path.join(os.fspath(trace_dir), trace_name)
            rows = read_kept_rows(path)
            deltas = [
                (right["timestamp"] - left["timestamp"]).total_seconds()
                for left, right in zip(rows, rows[1:])
            ]
            all_deltas.extend(deltas)
            trace_gaps = gap_records(
                rows, split=split, trace_name=trace_name,
                long_gap_s=long_gap_s,
            )
            long_gaps.extend(trace_gaps)
            trace_rows.append((split, trace_name, rows))
            traces.append({
                "split": split,
                "trace": trace_name,
                "sample_count": len(rows),
                "timestamp_span_s": (
                    rows[-1]["timestamp"] - rows[0]["timestamp"]
                ).total_seconds(),
                "maximum_gap_s": max(deltas, default=0.0),
                "long_gap_count": len(trace_gaps),
                "gap_count_over_10s": sum(delta > 10.0 for delta in deltas),
                "gap_count_over_30s": sum(delta > 30.0 for delta in deltas),
            })
    exact_counts = Counter(int(delta) if float(delta).is_integer() else str(delta)
                           for delta in all_deltas)
    sensitivity = threshold_sensitivity(
        trace_rows, sensitivity_thresholds_s, grid_stride_s
    )
    cadence_0_to_3 = sum(0.0 <= delta <= 3.0 for delta in all_deltas)
    borderline_4_to_5 = sum(3.0 < delta <= 5.0 for delta in all_deltas)
    return {
        "schema_version": 1,
        "audit_type": "timestamp_gap_inventory",
        "settings": {
            "long_gap_s": float(long_gap_s),
            "split_rule": "split when consecutive timestamp delta > max_gap_s",
            "candidate_grid_stride_s": float(grid_stride_s),
            "sensitivity_thresholds_s": [
                float(value) for value in sensitivity_thresholds_s
            ],
            "source_rows": "State D with numeric DL_bitrate > 0",
        },
        "summary": {
            "trace_count": len(traces),
            "timestamp_delta_count": len(all_deltas),
            "duplicate_timestamp_count": sum(delta == 0 for delta in all_deltas),
            "one_second_delta_count": sum(delta == 1 for delta in all_deltas),
            "cadence_delta_count_0_to_3s": cadence_0_to_3,
            "cadence_delta_fraction_0_to_3s": (
                cadence_0_to_3 / len(all_deltas) if all_deltas else 0.0
            ),
            "borderline_delta_count_over_3_to_5s": borderline_4_to_5,
            "long_gap_count": len(long_gaps),
            "affected_trace_count": sum(row["long_gap_count"] > 0 for row in traces),
            "maximum_gap_s": max(all_deltas, default=0.0),
            "gap_count_over_10s": sum(delta > 10 for delta in all_deltas),
            "gap_count_over_30s": sum(delta > 30 for delta in all_deltas),
            "gap_count_over_60s": sum(delta > 60 for delta in all_deltas),
            "long_gap_count_6_to_10s": sum(
                5.0 < gap["gap_s"] <= 10.0 for gap in long_gaps
            ),
            "long_gap_count_11_to_30s": sum(
                10.0 < gap["gap_s"] <= 30.0 for gap in long_gaps
            ),
            "long_gap_count_over_30s": sum(
                gap["gap_s"] > 30.0 for gap in long_gaps
            ),
            "long_gaps_with_network_mode_change": sum(
                gap["network_mode_changed"] for gap in long_gaps
            ),
            "long_gaps_with_cell_id_change": sum(
                gap["cell_id_changed"] for gap in long_gaps
            ),
            "exact_delta_counts_s": {
                str(key): exact_counts[key]
                for key in sorted(exact_counts, key=lambda value: float(value))
            },
        },
        "threshold_sensitivity": sensitivity,
        "traces": traces,
        "long_gaps": sorted(
            long_gaps,
            key=lambda row: (-row["gap_s"], row["split"], row["trace"],
                             row["before"]["sample_index"]),
        ),
    }


def render_gap_markdown(report):
    settings = report["settings"]
    summary = report["summary"]
    long_gap_s = float(settings["long_gap_s"])
    is_primary_five_second_report = math.isclose(long_gap_s, 5.0)
    lines = [
        "# Timestamp-gap audit",
        "",
        "Raw trace CSVs are unchanged. A long gap is a consecutive timestamp "
        f"delta greater than {settings['long_gap_s']:g} seconds.",
        "",
        "## Findings",
        "",
        f"- {summary['long_gap_count']} long gaps across "
        f"{summary['affected_trace_count']}/{summary['trace_count']} traces.",
        f"- Largest gap: {summary['maximum_gap_s']:.0f} seconds.",
        f"- Gaps >10 s: {summary['gap_count_over_10s']}; >30 s: "
        f"{summary['gap_count_over_30s']}; >60 s: "
        f"{summary['gap_count_over_60s']}.",
        f"- Dominant cadence: {summary['one_second_delta_count']}/"
        f"{summary['timestamp_delta_count']} deltas are exactly one second; "
        f"{summary['duplicate_timestamp_count']} are duplicate timestamps.",
        "",
        "The cleaned corpus has no measured positive application-layer download "
        "throughput sample inside these intervals. Holding the previous rate "
        "across them would invent observations, so the recommended policy is "
        f"to split at gaps >{long_gap_s:g} s while preserving the parent file's "
        "train/validation/test assignment.",
        "",
        f"- {summary['cadence_delta_count_0_to_3s']}/"
        f"{summary['timestamp_delta_count']} deltas "
        f"({100.0 * summary['cadence_delta_fraction_0_to_3s']:.2f}%) are "
        "between 0 and 3 seconds; only "
        f"{summary['borderline_delta_count_over_3_to_5s']} are 4-5 seconds.",
        f"- Endpoint context changes across "
        f"{summary['long_gaps_with_network_mode_change']} gaps for radio mode "
        f"and {summary['long_gaps_with_cell_id_change']} gaps for cell ID.",
        "",
        "## Threshold sensitivity",
        "",
        "| Split when gap > (s) | Gaps split | Traces | Blocks | Blocks <10 s | "
        "60-s candidates | Usable duration |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["threshold_sensitivity"]:
        lines.append(
            f"| {row['max_gap_s']:g} | {row['split_gap_count']} | "
            f"{row['affected_trace_count']} | {row['block_count']} | "
            f"{row['blocks_shorter_than_10s']} | "
            f"{row['candidate_window_count_at_grid_stride']} | "
            f"{100.0 * row['usable_duration_fraction']:.2f}% |"
        )
    if is_primary_five_second_report:
        lines.extend([
            "",
            f"Long-gap bands: {summary['long_gap_count_6_to_10s']} at 6-10 s; "
            f"{summary['long_gap_count_11_to_30s']} at 11-30 s; "
            f"{summary['long_gap_count_over_30s']} above 30 s.",
            "",
            "The source paper describes one-second logging granularity. A "
            "2-second cutoff would split the 607 observed 3-second deltas and "
            "over-fragment the corpus. The primary rule conservatively "
            "tolerates 4-5 seconds of sampling irregularity and splits every "
            "interval of at least 6 seconds. This threshold is fixed from "
            "acquisition cadence, before any ABR/QoE comparison; 3- and "
            "10-second alternatives remain sensitivity checks.",
        ])
    else:
        lines.extend([
            "",
            f"This is a non-primary {long_gap_s:g}-second sensitivity run. "
            "Threshold choice must remain independent of ABR/QoE outcomes.",
        ])
    lines.extend([
        "",
        f"## Every gap greater than {long_gap_s:g} seconds",
        "",
        "| Split | Trace | Gap (s) | Before timestamp | Before mode / kbps | "
        "After timestamp | After mode / kbps | Cell change |",
        "|---|---|---:|---|---|---|---|---|",
    ])
    for gap in report["long_gaps"]:
        before, after = gap["before"], gap["after"]
        lines.append(
            f"| {gap['split']} | `{gap['trace']}` | {gap['gap_s']:.0f} | "
            f"{before['timestamp_text']} | {before['network_mode']} / "
            f"{before['dl_bitrate_kbps']:g} | {after['timestamp_text']} | "
            f"{after['network_mode']} / {after['dl_bitrate_kbps']:g} | "
            f"{'yes' if gap['cell_id_changed'] else 'no'} |"
        )
    lines.extend([
        "",
        "The official dataset describes production-network measurements logged "
        "with G-NetTrack Pro and identifies DL_bitrate as application-layer "
        "download rate. Neither the paper, repository, nor tool manual specifies "
        "an interpolation or forward-fill rule for missing timestamps. The "
        "paper permits splitting long traces for experiment needs. This audit's "
        "gap threshold is therefore a conservative experiment policy, not a "
        "rule supplied by the dataset authors.",
        "",
        "Sources: [original MMSys paper](https://cora.ucc.ie/bitstreams/"
        "a0782899-2f0b-4741-9493-2b203db55bea/download), [official dataset "
        "repository](https://github.com/uccmisl/5Gdataset), and [G-NetTrack "
        "manual](https://gyokovsolutions.com/manual-g-nettrack/).",
        "",
    ])
    return "\n".join(lines)
