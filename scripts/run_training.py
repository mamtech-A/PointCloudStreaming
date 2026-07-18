#!/usr/bin/env python3
"""One-command, resumable overnight training pipeline.

The pipeline keeps the registered trace split unchanged.  Training, LSTM
selection, DQN/segment selection, and baseline tuning use train/validation
only.  The fixed, previously inspected test split is evaluated only after every
choice is frozen and the user explicitly supplies ``--run-test``.

Stages:
  1. Generate one achieved-throughput dataset per segment cadence.
  2. Tune/train one matching LSTM per cadence.
  3. Screen DQN configurations and confirm finalists over the full seed set.
  4. Tune Fixed, BufferBased, and MPC parameters on validation at winner S.
  5. With explicit --run-test, evaluate the frozen methods on the fixed,
     previously inspected test split exactly once.
  6. Write a versioned training summary.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODELS = os.path.join(project_root, "models")


class Runner:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.run_log = open(
            os.path.join(run_dir, "RUN.log"), "a", encoding="utf-8"
        )

    def note(self, message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {message}"
        print(line)
        self.run_log.write(line + "\n")
        self.run_log.flush()

    def stage(self, name, command, log_name):
        command = [str(value) for value in command]
        log_path = os.path.join(self.run_dir, log_name)
        self.note(f"START {name}: {' '.join(command)} (log: {log_name})")
        started = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("# " + " ".join(command) + "\n")
            log.flush()
            return_code = subprocess.call(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=project_root,
                env=env,
            )
        minutes = (time.time() - started) / 60.0
        if return_code == 0:
            self.note(f"DONE  {name} in {minutes:.1f} min")
            return
        self.note(f"FAIL  {name} (rc={return_code}) after {minutes:.1f} min; see {log_path}")
        raise SystemExit(1)


def py(*arguments):
    return [sys.executable, "-u"] + [str(value) for value in arguments]


def absolute(path):
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def relative(path):
    return os.path.relpath(path, project_root).replace("\\", "/")


def artifact_set(cfg, run_dir, smoke):
    configured = cfg.get("artifacts", {})
    defaults = {
        "dqn_model": "models/abr_dqn.pkl",
        "sweep_results": "models/dqn_sweep_results.json",
        "baseline_config": "models/baseline_config.json",
        "final_results": "models/final_test_results.json",
        "training_summary": "models/TRAINING_SUMMARY.md",
        "lstm_model_template": "models/bandwidth_lstm_s{segment_frames}.pkl",
    }
    if smoke:
        return {
            "dqn_model": os.path.join(run_dir, "smoke_abr_dqn.pkl"),
            "sweep_results": os.path.join(run_dir, "smoke_dqn_sweep_results.json"),
            "baseline_config": os.path.join(run_dir, "smoke_baseline_config.json"),
            "final_results": os.path.join(run_dir, "smoke_validation_results.json"),
            "training_summary": os.path.join(run_dir, "SMOKE_SUMMARY.md"),
            "lstm_model_template": os.path.join(
                run_dir, "lstm_models", "bandwidth_lstm_s{segment_frames}.pkl"
            ),
        }
    artifacts = {}
    for key, default in defaults.items():
        value = configured.get(key, default)
        artifacts[key] = (
            absolute(value) if key != "lstm_model_template" else
            (value if os.path.isabs(value) else os.path.join(project_root, value))
        )
    return artifacts


def model_for(template, segment_frames):
    return template.format(segment_frames=int(segment_frames))


def install_artifact_set(candidate_model, installed_model):
    os.makedirs(os.path.dirname(installed_model) or ".", exist_ok=True)
    for suffix in (".pkl", "_best.pkl", "_split.json", "_tuning_results.json"):
        source = candidate_model.replace(".pkl", suffix)
        target = installed_model.replace(".pkl", suffix)
        if os.path.exists(source):
            shutil.copyfile(source, target)


def add_config_args(command, values, excluded=()):
    for key, value in values.items():
        if key in excluded or key.startswith("_") or value is None:
            continue
        command.append(f"--{key.replace('_', '-')}")
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        elif isinstance(value, bool):
            if not value:
                command[-1] = f"--no-{key.replace('_', '-')}"
        else:
            command.append(str(value))


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_search_config(cfg):
    lstm_segments = [int(value) for value in cfg["lstm"]["segment_frames"]]
    dqn_segments = [
        int(value) for value in cfg["dqn_sweep"]["axes"]["segment-frames"]
    ]
    if lstm_segments != dqn_segments:
        raise ValueError(
            f"LSTM/DQN segment search mismatch: {lstm_segments} != {dqn_segments}"
        )
    if lstm_segments != [5, 8, 10, 15]:
        raise ValueError("the requested segment search must be exactly [5, 8, 10, 15]")
    base = cfg["dqn_sweep"].get("base_args", {})
    if float(base.get("gamma", 1.0)) != 1.0:
        raise ValueError("gamma must be 1 so optimized return equals reported QoE")
    if cfg.get("final_eval", {}).get("split", "test") != "test":
        raise ValueError("the full final evaluation must use the registered test split")
    if cfg["dqn_sweep"].get("stratify_by") != "segment-frames":
        raise ValueError("the wide search must be stratified by segment-frames")
    return lstm_segments


def verify_lstm(model_path, segment_frames):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"missing segment-specific LSTM: {model_path}")
    metadata_path = model_path.replace(".pkl", "_split.json")
    metadata = load_json(metadata_path)
    if int(metadata.get("segment_frames", -1)) != int(segment_frames):
        raise ValueError(
            f"{metadata_path} has segment_frames={metadata.get('segment_frames')}, "
            f"expected {segment_frames}"
        )
    if not metadata.get("final_test_excluded"):
        raise ValueError(f"{metadata_path} does not certify final-test exclusion")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Overnight training pipeline")
    parser.add_argument("--config", default=os.path.join("configs", "training.json"))
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="parallel DQN trials (4 is appropriate for an i5-13400)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--run-test", action="store_true",
        help="explicitly authorize the frozen full run to evaluate the fixed test split",
    )
    parser.add_argument(
        "--run-dir", default="",
        help="reuse the same directory to resume interrupted sweep trials",
    )
    parser.add_argument(
        "--skip-stages", default="",
        help="comma list from {gen,lstm,sweep,baselines,test,eval}",
    )
    args = parser.parse_args()

    config_path = absolute(args.config)
    with open(config_path, "rb") as handle:
        config_bytes = handle.read()
    cfg = json.loads(config_bytes.decode("utf-8"))
    config_digest = hashlib.sha256(config_bytes).hexdigest()[:16]
    all_segments = validate_search_config(cfg)
    active_segments = all_segments[:1] if args.smoke else all_segments
    skip = {item.strip() for item in args.skip_stages.split(",") if item.strip()}
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = (
        absolute(args.run_dir) if args.run_dir else
        os.path.join(
            project_root, "logs", "train_runs",
            f"{timestamp}_{'smoke' if args.smoke else 'wide'}",
        )
    )
    runner = Runner(run_dir)
    artifacts = artifact_set(cfg, run_dir, args.smoke)
    lstm_cfg = cfg["lstm"]
    protocol_rel = cfg.get("protocol", os.path.join("configs", "experiment_protocol.json"))

    runner.note(
        f"pipeline start | config={relative(config_path)} smoke={args.smoke} "
        f"jobs={args.jobs} segments={active_segments} config_digest={config_digest} "
        f"skip={sorted(skip) or 'none'} run_test={args.run_test}"
    )
    shutil.copyfile(config_path, os.path.join(run_dir, "training.config.json"))
    shutil.copyfile(absolute(protocol_rel), os.path.join(run_dir, "experiment_protocol.json"))

    achieved = lstm_cfg.get("signal", "capacity") == "achieved"
    data_dirs = {
        segment: (
            os.path.join(run_dir, "lstm_data", f"s{segment}") if args.smoke else
            os.path.join(project_root, "data", f"lstm_achieved_s{segment}")
        )
        for segment in active_segments
    }

    # 1. Segment-cadence-matched achieved-throughput datasets.
    if achieved and "gen" not in skip:
        for segment in active_segments:
            command = py(os.path.join("scripts", "gen_lstm_dataset.py"))
            add_config_args(
                command, lstm_cfg.get("gen_args", {}),
                excluded={"segment_frames", "out_dir"},
            )
            command += [
                "--segment-frames", segment,
                "--out-dir", data_dirs[segment],
            ]
            if args.smoke:
                command += [
                    "--offsets", 1, "--policies", "fixed3", "--max-frames", 120,
                ]
            runner.stage(
                f"generate_lstm_data[S={segment}]", command,
                f"10_gen_lstm_s{segment}.log",
            )

    # 2. A separately trained predictor for every observation cadence.  Its
    # architecture is frozen from the prior validation-selected LSTM.
    if "lstm" not in skip:
        transforms = list(lstm_cfg.get("transforms", ["log1p"]))
        if args.smoke:
            transforms = transforms[:1]
        for segment in active_segments:
            segment_dir = os.path.join(run_dir, "lstm", f"s{segment}")
            os.makedirs(segment_dir, exist_ok=True)
            candidates = []
            for transform in transforms:
                output = os.path.join(
                    segment_dir, f"bandwidth_lstm_{transform}.pkl"
                )
                command = py(
                    os.path.join("scripts", "train_model.py"),
                    "--transform", transform,
                    "--out", output,
                    "--sequence-length", lstm_cfg.get("sequence_length", 8),
                    "--epochs", 5 if args.smoke else lstm_cfg.get("epochs", 100),
                    "--test-size", 0.2,
                )
                if achieved:
                    split_path = os.path.join(data_dirs[segment], "split.json")
                    command += [
                        "--bandwidth-dir", data_dirs[segment],
                        "--split-from", split_path,
                    ]
                if args.smoke or not lstm_cfg.get("tune", True):
                    command.append("--no-tune")
                    add_config_args(
                        command, lstm_cfg.get("fixed_hyperparameters", {})
                    )
                runner.stage(
                    f"train_lstm[S={segment},{transform}]", command,
                    f"20_lstm_s{segment}_{transform}.log",
                )
                metadata = load_json(output.replace(".pkl", "_split.json"))
                candidates.append((transform, output, metadata["metrics"]))

            metric = lstm_cfg.get("select_metric", "low_bw_mae")
            candidates.sort(
                key=lambda row: (
                    row[2].get(metric, float("inf")), row[2].get("mae", float("inf"))
                )
            )
            transform, winner, metrics = candidates[0]
            installed = model_for(artifacts["lstm_model_template"], segment)
            install_artifact_set(winner, installed)
            runner.note(
                f"LSTM S={segment}: transform={transform}, {metric}="
                f"{metrics.get(metric, float('nan')):.3f} -> {relative(installed)}"
            )

    for segment in active_segments:
        verify_lstm(model_for(artifacts["lstm_model_template"], segment), segment)

    # 3. Wide staged validation search; test traces are inaccessible here.
    if "sweep" not in skip:
        command = py(
            os.path.join("scripts", "sweep.py"),
            "--config", config_path,
            "--jobs", args.jobs,
            "--sweep-dir", os.path.join(run_dir, "sweep"),
            "--out", artifacts["dqn_model"],
            "--results", artifacts["sweep_results"],
            "--lstm-template", artifacts["lstm_model_template"],
        )
        if args.smoke:
            command.append("--smoke")
        runner.stage("dqn_staged_sweep", command, "30_dqn_sweep.log")

    sweep = load_json(artifacts["sweep_results"])
    if sweep.get("experiment_config_digest") != config_digest:
        raise ValueError(
            "sweep artifact does not match the current training config; use a fresh "
            "artifact tag/run directory"
        )
    winner_config = sweep["winner"]["config"]
    winner_segment = int(
        winner_config.get(
            "segment-frames", sweep.get("base_args", {}).get("segment-frames", -1)
        )
    )
    if winner_segment not in active_segments:
        raise ValueError(f"sweep selected unexpected segment_frames={winner_segment}")
    selected_lstm = model_for(artifacts["lstm_model_template"], winner_segment)
    verify_lstm(selected_lstm, winner_segment)
    runner.note(
        f"frozen DQN choice: trial={sweep['winner']['trial']} S={winner_segment}; "
        f"matching LSTM={relative(selected_lstm)}"
    )

    # 4. Fair validation-only tuning for the requested non-learning baselines.
    if "baselines" not in skip and os.path.exists(artifacts["baseline_config"]):
        frozen_baseline = load_json(artifacts["baseline_config"])
        expected_families = {
            name.strip() for name in cfg.get("baselines", {}).get(
                "families", "fixed,buffer,mpc").split(",") if name.strip()
        }
        if (
            frozen_baseline.get("experiment_config_digest") != config_digest
            or frozen_baseline.get("protocol_digest") != sweep.get("protocol_digest")
            or int(frozen_baseline.get("segment_frames", -1)) != winner_segment
            or set(frozen_baseline.get("families", {})) != expected_families
        ):
            raise ValueError(
                "existing baseline artifact is incompatible with this frozen sweep; "
                "use a new artifact tag rather than overwriting evidence"
            )
        runner.note(
            f"RESUME validation_baseline_tuning: {relative(artifacts['baseline_config'])}"
        )
    elif "baselines" not in skip:
        command = py(
            os.path.join("scripts", "tune_baselines.py"),
            "--protocol", protocol_rel,
            "--segment-frames", winner_segment,
            "--families", cfg.get("baselines", {}).get(
                "families", "fixed,buffer,mpc"
            ),
            "--lstm", selected_lstm,
            "--out", artifacts["baseline_config"],
            "--experiment-config-digest", config_digest,
        )
        if args.smoke:
            command += [
                "--quick", "--max-frames", 40, "--offsets", 1,
                "--eval-seeds", 42, "--overwrite",
            ]
        runner.stage("validation_baseline_tuning", command, "35_tune_baselines.log")

    # 5. Validation for smoke; explicit fixed-test stage for a full run.
    should_evaluate = args.smoke or args.run_test
    if (not should_evaluate and "test" not in skip and "eval" not in skip):
        runner.note(
            "LOCKED test not run. Reinvoke this frozen run with --run-test after "
            "reviewing validation artifacts; make no method changes afterward."
        )
    elif "test" not in skip and "eval" not in skip:
        final_cfg = cfg.get("final_eval", {})
        if os.path.exists(artifacts["final_results"]):
            frozen_result = load_json(artifacts["final_results"])
            if (
                frozen_result.get("experiment_config_digest") != config_digest
                or frozen_result.get("protocol_digest") != sweep.get("protocol_digest")
                or frozen_result.get("winner_trial") != sweep["winner"]["trial"]
                or frozen_result.get("winner_config") != winner_config
                or frozen_result.get("split") !=
                    ("validation" if args.smoke else final_cfg.get("split", "test"))
            ):
                raise ValueError(
                    "existing evaluation artifact is incompatible; never overwrite or "
                    "mix it with a changed method"
                )
            runner.note(
                f"RESUME frozen_evaluation: {relative(artifacts['final_results'])}"
            )
        else:
            command = py(
                os.path.join("scripts", "evaluate_experiment.py"),
                "--protocol", protocol_rel,
                "--sweep-results", artifacts["sweep_results"],
                "--baseline-config", artifacts["baseline_config"],
                "--lstm", selected_lstm,
                "--strategies", final_cfg.get("strategies", "dqn,fixed,buffer,mpc"),
                "--out", artifacts["final_results"],
                "--experiment-config-digest", config_digest,
            )
            if args.smoke:
                command += [
                    "--split", "validation", "--max-frames", 40, "--offsets", 1,
                    "--eval-seeds", 42,
                    "--max-dqn-seeds", 1, "--overwrite",
                ]
                stage_name = "smoke_validation"
            else:
                command += [
                    "--split", final_cfg.get("split", "test"),
                    "--max-frames", final_cfg.get("max_frames", 0),
                ]
                stage_name = "fixed_previously_inspected_test"
            runner.stage(stage_name, command, "40_registered_evaluation.log")

    # 6. Compact, versioned evidence summary.
    lines = [
        "# TRAINING SUMMARY", "",
        f"- run directory: `{relative(run_dir)}`",
        f"- mode: {'SMOKE (not a paper result)' if args.smoke else 'full wide search'}",
        f"- experiment config digest: `{config_digest}`",
        f"- segment candidates: {all_segments}",
        f"- frozen segment size: **{winner_segment} frames**",
        f"- test split changed by pipeline: **no**", "",
        "## Segment-specific LSTMs", "",
    ]
    for segment in active_segments:
        path = model_for(artifacts["lstm_model_template"], segment)
        metadata = verify_lstm(path, segment)
        metrics = metadata["metrics"]
        lines.append(
            f"- S={segment}: `{relative(path)}`; MAE={metrics['mae']:.3f}, "
            f"low-bandwidth MAE={metrics.get('low_bw_mae', float('nan')):.3f}, "
            f"transform={metadata.get('transform')}"
        )
    select_by = sweep["select_by"]
    winner_metrics = sweep["winner"]["metrics"]
    lines += [
        "", "## DQN selection", "",
        f"- screened configurations: {len(sweep.get('screening_ranking', []))}",
        f"- confirmed configurations: {len(sweep.get('confirmed_config_trials', []))}",
        f"- confirmation seeds: {sweep['winner']['seeds']}",
        f"- winner config: `{json.dumps(winner_config, sort_keys=True)}`",
        f"- validation {select_by}: {winner_metrics[select_by]:.3f} +/- "
        f"{winner_metrics.get(f'{select_by}_std', 0.0):.3f}",
        f"- mean training steps per confirmed seed: "
        f"{winner_metrics.get('steps_run_mean', 0):.0f}",
        "", "## Frozen evaluation", "",
    ]
    if os.path.exists(artifacts["final_results"]):
        final = load_json(artifacts["final_results"])
        lines.append(
            f"- split: **{final['split']}**; protocol digest: `{final['protocol_digest']}`"
        )
        lines += [
            "", "| policy | QoE | mean quality | stall (s) |", "|---|---:|---:|---:|"
        ]
        for key, label in (
            ("dqn", "DQN"), ("best_global_fixed", "Best global fixed"),
            ("buffer", "BufferBased"), ("mpc", "MPC"),
        ):
            row = final.get("results", {}).get(key)
            if row:
                lines.append(
                    f"| {label} | {row['qoe_quality']:.3f} | "
                    f"{row['mean_quality']:.3f} | {row['stall_s']:.3f} |"
                )
    else:
        lines.append("- fixed test not evaluated; method remains frozen on validation")
    lines += [
        "", "## Evidence", "",
        f"- DQN sweep: `{relative(artifacts['sweep_results'])}`",
        f"- baseline tuning: `{relative(artifacts['baseline_config'])}`",
        f"- stage/trial logs: `{relative(run_dir)}`", "",
    ]
    if os.path.exists(artifacts["final_results"]):
        lines.insert(-2, f"- case-level evaluation: `{relative(artifacts['final_results'])}`")
    os.makedirs(os.path.dirname(artifacts["training_summary"]) or ".", exist_ok=True)
    with open(artifacts["training_summary"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    runner.note(f"summary -> {relative(artifacts['training_summary'])}; pipeline OK")
    if not args.smoke and not os.path.exists(artifacts["final_results"]):
        print(
            "\nValidation selection is frozen. To perform the one fixed-test run, use:\n"
            f"  python scripts/run_training.py --run-dir {relative(run_dir)} "
            "--jobs 4 --skip-stages gen,lstm,sweep,baselines --run-test"
        )
    else:
        print(
            "\nTraining/evaluation complete. Commit the versioned artifacts needed "
            "for paper analysis, then push for the writing PC."
        )


if __name__ == "__main__":
    main()
