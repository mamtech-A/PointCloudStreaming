"""Validated access to the frozen gap-split trace-window registry."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict

from .experiment_protocol import protocol_digest
from .network_model.finite_trace import FiniteTraceWindow, measured_end_time_s
from .network_model.trace import BandwidthTrace
from .trace_audit import slice_trace_at_time
from .trace_gaps import split_trace_at_gaps


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TraceWindowRegistry:
    """Frozen training/evaluation windows reconstructed from unchanged CSVs."""

    def __init__(self, path, protocol, trace_dir, verify_hashes=True):
        self.path = os.path.abspath(os.fspath(path))
        self.trace_dir = os.path.abspath(os.fspath(trace_dir))
        with open(self.path, encoding="utf-8") as handle:
            self.data = json.load(handle)
        if self.data.get("schema_version") != 1:
            raise ValueError("unsupported trace-window registry schema")
        expected_protocol = protocol_digest(protocol)
        if self.data.get("protocol_digest") != expected_protocol:
            raise ValueError("trace-window registry/protocol digest mismatch")
        self.registry_id = str(self.data.get("registry_id") or "")
        if not self.registry_id:
            raise ValueError("trace-window registry has no registry_id")
        unsigned = dict(self.data)
        unsigned.pop("registry_id", None)
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
        calculated_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        if calculated_id != self.registry_id:
            raise ValueError("trace-window registry content/ID mismatch")
        project_root = os.path.dirname(os.path.dirname(self.path))
        audit_path = os.path.join(project_root, self.data["source_audit"])
        if not os.path.isfile(audit_path):
            raise FileNotFoundError(f"registry source audit is absent: {audit_path}")
        if sha256_file(audit_path) != self.data["source_audit_sha256"]:
            raise ValueError("trace-window registry/source audit hash mismatch")
        self.max_gap_s = float(
            self.data["settings"]["maximum_contiguous_gap_s"]
        )
        self.grid_stride_s = float(
            self.data["settings"]["candidate_grid_stride_s"]
        )
        self._cache = {}

        window_ids = set()
        for split in ("train", "validation", "test"):
            registered = self.data["splits"][split]
            if list(registered["parent_files"]) != list(protocol["trace_split"][split]):
                raise ValueError(f"registry/protocol parent mismatch for {split}")
            allowed = set(registered["parent_files"])
            for window in registered["windows"]:
                if window["split"] != split or window["trace"] not in allowed:
                    raise ValueError(f"window assigned outside its parent split: {window}")
                if window["id"] in window_ids:
                    raise ValueError(f"duplicate registry window ID: {window['id']}")
                window_ids.add(window["id"])
        if verify_hashes:
            for filename, expected in self.data["trace_hashes"].items():
                path = os.path.join(self.trace_dir, filename)
                if sha256_file(path) != expected:
                    raise ValueError(f"trace hash mismatch: {filename}")

    def settings(self):
        return dict(self.data["settings"])

    def validate_experiment(self, sequences, segment_frames, max_frames=0):
        settings = self.data["settings"]
        missing = sorted(set(sequences) - set(settings["required_sequences"]))
        if missing:
            raise ValueError(f"registry was not audited for sequences: {missing}")
        if int(segment_frames) not in {
            int(value) for value in settings["required_segment_frames"]
        }:
            raise ValueError(
                f"registry was not audited for segment_frames={segment_frames}"
            )
        if max_frames:
            available = min(
                int(settings["required_frames_per_sequence"][sequence])
                for sequence in sequences
            )
            if int(max_frames) > available:
                raise ValueError(
                    f"registry supports at most {available} audited frames; "
                    f"requested {max_frames}"
                )

    def windows(self, split, parent_files=None):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown registry split {split!r}")
        rows = list(self.data["splits"][split]["windows"])
        if parent_files is None:
            return rows
        allowed = {os.path.basename(path) for path in parent_files}
        unknown = allowed - set(self.data["splits"][split]["parent_files"])
        if unknown:
            raise ValueError(f"requested files absent from {split} registry: {sorted(unknown)}")
        return [row for row in rows if row["trace"] in allowed]

    def windows_by_parent(self, split, parent_files=None):
        grouped = defaultdict(list)
        for window in self.windows(split, parent_files):
            grouped[window["trace"]].append(window)
        expected = (
            list(self.data["splits"][split]["parent_files"])
            if parent_files is None else [os.path.basename(path) for path in parent_files]
        )
        missing = [filename for filename in expected if not grouped[filename]]
        if missing:
            raise ValueError(f"parents have no registered {split} windows: {missing}")
        return {filename: grouped[filename] for filename in expected}

    def validate_evaluation_count(self, split, count, parent_files=None):
        grouped = self.windows_by_parent(split, parent_files)
        insufficient = {
            filename: len(rows) for filename, rows in grouped.items()
            if len(rows) < int(count)
        }
        if insufficient:
            raise ValueError(
                f"frozen {split} registry has fewer than {count} windows/parent: "
                f"{insufficient}"
            )
        selected = {}
        for filename, rows in grouped.items():
            if len(rows) == int(count):
                selected[filename] = rows
                continue
            if int(count) == 1:
                indexes = [0]
            else:
                indexes = [
                    int(round(index * (len(rows) - 1) / float(int(count) - 1)))
                    for index in range(int(count))
                ]
            selected[filename] = [rows[index] for index in indexes]
        return selected

    def _parent_blocks(self, filename):
        cached = self._cache.get(filename)
        if cached is not None:
            return cached
        path = os.path.join(self.trace_dir, filename)
        trace = BandwidthTrace.from_file(path)
        blocks = split_trace_at_gaps(trace, self.max_gap_s)
        by_id = {block.block_id: block for block in blocks}
        self._cache[filename] = (trace, by_id)
        return trace, by_id

    def materialize(self, window):
        """Reconstruct one finite window and verify every frozen boundary."""
        filename = window["trace"]
        _trace, blocks = self._parent_blocks(filename)
        block = blocks.get(window["block_id"])
        if block is None:
            raise ValueError(f"missing block {window['block_id']} in {filename}")
        checks = (
            (block.block_index, int(window["block_index"]), "block index"),
            (block.global_start_time_s, float(window["block_start_time_s"]), "block start"),
            (block.global_end_time_s, float(window["block_end_time_s"]), "block end"),
        )
        for actual, expected, label in checks:
            if not math.isclose(float(actual), float(expected), abs_tol=1e-9):
                raise ValueError(f"{window['id']} {label} mismatch: {actual} != {expected}")
        local_start = float(window["block_local_start_time_s"])
        trace_window, local_source = slice_trace_at_time(block.trace, local_start)
        source_index = block.source_start_index + local_source
        if source_index != int(window["source_sample_index"]):
            raise ValueError(f"{window['id']} source sample mismatch")
        finite = FiniteTraceWindow(trace_window)
        if not math.isclose(
            finite.duration_s, float(window["duration_s"]), abs_tol=1e-9
        ):
            raise ValueError(f"{window['id']} measured duration mismatch")
        finite.window_id = window["id"]
        finite.block_id = window["block_id"]
        return finite


def load_trace_registry(path, protocol, trace_dir, verify_hashes=True):
    return TraceWindowRegistry(path, protocol, trace_dir, verify_hashes)
