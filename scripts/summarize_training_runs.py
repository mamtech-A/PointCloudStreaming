"""Inventory raw training runs and build compact, reviewable experiment records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6})_(?P<mode>[A-Za-z0-9_-]+)$")
KEY_FILES = {
    "RUN.log", "training.config.json", "experiment_protocol.json",
    "trace_window_registry.json", "30_dqn_sweep.log",
    "35_tune_baselines.log", "40_registered_evaluation.log",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def scalar_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): scalar_tree(v) for k, v in value.items() if scalar_tree(v) is not None}
    if isinstance(value, list):
        items = [scalar_tree(v) for v in value]
        return [v for v in items if v is not None]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def sample_series(values: list[Any], limit: int = 300) -> list[dict[str, Any]]:
    """Retain endpoints and evenly spaced samples without embedding the raw log."""
    if not values:
        return []
    step = max(1, (len(values) - 1) // max(1, limit - 1))
    indices = list(range(0, len(values), step))
    if indices[-1] != len(values) - 1:
        indices.append(len(values) - 1)
    return [{"index": index, "value": scalar_tree(values[index])} for index in indices[:limit]]


def compact_validation(records: list[Any]) -> list[dict[str, Any]]:
    """Keep checkpoint-level scalars; omit bulky per-case and per-trace records."""
    compact = []
    for record in records:
        if not isinstance(record, dict):
            continue
        item = {str(key): value for key, value in record.items()
                if value is None or isinstance(value, (str, int, float, bool))}
        compact.append(item)
    return compact


def find_histories(run: Path) -> list[dict[str, Any]]:
    histories = []
    for path in run.rglob("*.json"):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        reward = data.get("episode_reward")
        validation = data.get("validation")
        if not isinstance(reward, list) and not isinstance(validation, list):
            continue
        histories.append({
            "source": path.relative_to(run).as_posix(),
            "episode_reward_count": len(reward) if isinstance(reward, list) else 0,
            "episode_reward_samples": sample_series(reward) if isinstance(reward, list) else [],
            "validation": compact_validation(validation) if isinstance(validation, list) else [],
        })
    return histories


def infer_status(files: list[Path], run: Path, histories: list[dict[str, Any]]) -> str:
    names = {p.name for p in files}
    if "40_registered_evaluation.log" in names:
        return "completed_with_evaluation"
    if histories:
        return "training_completed_or_partial"
    runlog = run / "RUN.log"
    if runlog.exists():
        text = runlog.read_text(encoding="utf-8", errors="replace").lower()
        if "pipeline complete" in text or "pipeline done" in text:
            return "completed"
        if "fail" in text or "traceback" in text:
            return "failed_or_interrupted"
    return "unknown_or_partial"


def scan_run(run: Path, machine: str) -> dict[str, Any]:
    files = [p for p in run.rglob("*") if p.is_file()]
    match = RUN_RE.match(run.name)
    config_path = run / "training.config.json"
    config = load_json(config_path) if config_path.exists() else None
    histories = find_histories(run)
    timestamps = [p.stat().st_mtime for p in files]
    warnings = []
    if not config_path.exists():
        warnings.append("training.config.json missing")
    if not (run / "RUN.log").exists():
        warnings.append("RUN.log missing")
    if not histories:
        warnings.append("no machine-readable training history found")
    manifest = []
    for path in files:
        relative = path.relative_to(run).as_posix()
        if path.name in KEY_FILES or path.suffix.lower() in {".pkl", ".json", ".md"}:
            manifest.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "schema_version": 1,
        "run_id": run.name,
        "machine": machine,
        "timestamp": match.group("stamp") if match else None,
        "mode": match.group("mode") if match else "unknown",
        "source_path": str(run.resolve()),
        "file_count": len(files),
        "size_bytes": sum(p.stat().st_size for p in files),
        "first_file_utc": datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat() if timestamps else None,
        "last_file_utc": datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat() if timestamps else None,
        "status": infer_status(files, run, histories),
        "configuration": scalar_tree(config),
        "compact_histories": histories,
        "artifact_manifest": manifest,
        "warnings": warnings,
    }


def scan(args: argparse.Namespace) -> None:
    root = Path(args.run_root)
    runs = [scan_run(path, args.machine) for path in sorted(root.iterdir()) if path.is_dir()]
    payload = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
               "machine": args.machine, "run_root": str(root.resolve()), "runs": runs}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Inventoried {len(runs)} runs from {root} -> {output}")


def render_summary(run: dict[str, Any]) -> str:
    mib = run["size_bytes"] / (1024 * 1024)
    histories = run.get("compact_histories", [])
    episodes = sum(h.get("episode_reward_count", 0) for h in histories)
    validations = sum(len(h.get("validation", [])) for h in histories)
    warning_lines = "\n".join(f"- {w}" for w in run.get("warnings", [])) or "- None detected by inventory."
    return f"""# Training Run: {run['run_id']}

## Identity

- Machine: `{run['machine']}`
- Mode: `{run['mode']}`
- Observed status: `{run['status']}`
- Raw source: `{run['source_path']}`
- Files / size: {run['file_count']} / {mib:.2f} MiB
- File time range (UTC): {run['first_file_utc']} to {run['last_file_utc']}

## Retained diagnostic evidence

- Machine-readable histories: {len(histories)}
- Episode-reward samples: {episodes}
- Validation checkpoints: {validations}
- See `training_curves.json` and `run_manifest.json` in this directory.

## Inventory warnings

{warning_lines}

## Manual interpretation

- Objective/change: unknown; reconcile with git history and committed model summary.
- Outcome: not yet reviewed.
- Decision: not yet reviewed.

> This is an evidence-first inventory record. Missing values are intentionally not guessed.
"""


def merge(args: argparse.Namespace) -> None:
    all_runs = []
    sources = []
    for item in args.inventory:
        path = Path(item)
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
            raise SystemExit(f"Invalid inventory: {path}")
        sources.append(str(path))
        all_runs.extend(data["runs"])
    all_runs.sort(key=lambda r: (r.get("timestamp") or "", r["machine"], r["run_id"]))
    reports = Path(args.reports_dir)
    experiments = Path(args.experiments_dir)
    reports.mkdir(parents=True, exist_ok=True)
    experiments.mkdir(parents=True, exist_ok=True)
    inventory_runs = [{key: value for key, value in run.items() if key != "compact_histories"}
                      for run in all_runs]
    combined = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "sources": sources, "runs": inventory_runs}
    (reports / "training_run_inventory.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    fields = ["timestamp", "machine", "run_id", "mode", "status", "file_count", "size_bytes",
              "first_file_utc", "last_file_utc", "source_path"]
    with (reports / "training_run_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in all_runs:
            writer.writerow({key: run.get(key) for key in fields})
    for run in all_runs:
        folder = experiments / "runs" / f"{run['machine']}_{run['run_id']}"
        folder.mkdir(parents=True, exist_ok=True)
        curves = {"schema_version": 1, "run_id": run["run_id"], "machine": run["machine"],
                  "histories": run.get("compact_histories", [])}
        (folder / "training_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
        manifest = {key: value for key, value in run.items() if key != "compact_histories"}
        (folder / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (folder / "SUMMARY.md").write_text(render_summary(run), encoding="utf-8")
    print(f"Merged {len(all_runs)} runs into {reports} and {experiments / 'runs'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--run-root", required=True)
    scan_parser.add_argument("--machine", required=True)
    scan_parser.add_argument("--output", required=True)
    scan_parser.set_defaults(func=scan)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--inventory", action="append", required=True)
    merge_parser.add_argument("--reports-dir", default="reports")
    merge_parser.add_argument("--experiments-dir", default="experiments")
    merge_parser.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
