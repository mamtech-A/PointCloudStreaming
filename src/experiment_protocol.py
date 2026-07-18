"""Reproducible trace splits and evaluation-case helpers.

The final experiment uses an explicit, checked-in train/validation/test split.
Keeping the filenames in configuration (rather than regenerating a random
split inside each script) prevents accidental test reuse and makes a run on a
second PC describe exactly the same experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")


def load_protocol(path, trace_dir=None, require_complete=True):
    """Load and validate an experiment-protocol JSON file.

    Validation guarantees that split entries are non-empty, contain no
    duplicates, are pairwise disjoint, and (when ``trace_dir`` is supplied)
    refer to files that exist. With ``require_complete=True`` every CSV trace
    in ``trace_dir`` must appear in exactly one split.
    """
    path = os.fspath(path)
    with open(path, encoding="utf-8") as f:
        protocol = json.load(f)

    split = protocol.get("trace_split", {})
    missing_keys = [name for name in SPLIT_NAMES if name not in split]
    if missing_keys:
        raise ValueError(f"protocol is missing trace split(s): {missing_keys}")

    normalized = {}
    for name in SPLIT_NAMES:
        files = list(split[name])
        if not files:
            raise ValueError(f"protocol split '{name}' is empty")
        if len(files) != len(set(files)):
            raise ValueError(f"protocol split '{name}' contains duplicate filenames")
        normalized[name] = sorted(files)

    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1:]:
            overlap = sorted(set(normalized[left]) & set(normalized[right]))
            if overlap:
                raise ValueError(
                    f"protocol splits '{left}' and '{right}' overlap: {overlap}"
                )

    if trace_dir is not None:
        trace_root = Path(trace_dir)
        configured = set().union(*(set(normalized[name]) for name in SPLIT_NAMES))
        absent = sorted(name for name in configured if not (trace_root / name).is_file())
        if absent:
            raise FileNotFoundError(
                f"protocol references traces absent from {trace_root}: {absent}"
            )
        if require_complete:
            available = {p.name for p in trace_root.glob("*.csv")}
            omitted = sorted(available - configured)
            extra = sorted(configured - available)
            if omitted or extra:
                raise ValueError(
                    "protocol must cover every CSV trace exactly once; "
                    f"omitted={omitted}, extra={extra}"
                )

    protocol["trace_split"] = normalized
    return protocol


def protocol_digest(protocol):
    """Stable short digest recorded in every training/evaluation artifact."""
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def split_paths(protocol, split_name, trace_dir):
    """Return absolute paths for one validated split."""
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown split '{split_name}'; expected one of {SPLIT_NAMES}")
    return [os.path.join(os.fspath(trace_dir), name)
            for name in protocol["trace_split"][split_name]]


def evenly_spaced_offsets(trace_length, count, min_tail_samples=120):
    """Choose deterministic offsets spread across a trace.

    The final ``min_tail_samples`` are reserved so an evaluation episode does
    not immediately clamp to the last trace sample. Short traces safely fall
    back to offset zero. Duplicate rounded offsets are removed.
    """
    count = max(1, int(count))
    max_offset = max(0, int(trace_length) - int(min_tail_samples))
    if count == 1 or max_offset == 0:
        return [0]
    offsets = [int(round(i * max_offset / float(count - 1))) for i in range(count)]
    return sorted(set(offsets))


def evaluation_settings(protocol, split_name):
    """Return the registered evaluation settings for validation or test."""
    if split_name not in ("validation", "test"):
        raise ValueError("evaluation settings exist only for validation and test")
    settings = dict(protocol.get("evaluation", {}).get(split_name, {}))
    settings.setdefault("offsets_per_trace", 1)
    settings.setdefault("jitter_seeds", [42])
    settings.setdefault("sequences", ["longdress"])
    return settings
