#!/usr/bin/env python3
"""Registered final evaluation for DQN and causal ABR baselines.

Training and checkpoint selection never call this script on the final-test
split. The full pipeline invokes it once after the winning configuration has
been frozen. Its JSON output contains every case row, so tables and figures can
be regenerated without copying values by hand.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation import aggregate_records, evaluate_agent, evaluate_strategy
from src.experiment_protocol import (
    evaluation_settings, load_protocol, protocol_digest, split_paths,
)
from src.lstm_model import LSTMPredictor
from src.network_model import (
    DEFAULT_TCP_PARAMS, BandwidthABR, BufferBasedABR, LSTMABR, MPCABR,
)
from src.network_model.abr import ABRStrategy
from src.network_model.manifest import parse_mpd_xml
from src.rl.dqn import DQNAgent
from src.rl.env import StreamingEnv
from src.rl.features import DEFAULT_FEATURE_SPEC
from src.rl.reward import DEFAULT_REWARD_SPEC, OBJECTIVE_VERSION


class FixedABR(ABRStrategy):
    name = "fixed"

    def __init__(self, arm):
        self.arm = int(arm)

    def select(self, state):
        reps = sorted(state.reps, key=lambda rep: rep["id"])
        return reps[max(0, min(self.arm, len(reps) - 1))]["id"]


def sequence_name(mpd_path):
    stem = os.path.splitext(os.path.basename(mpd_path))[0]
    return stem[len("mpd_gpcc_"):] if stem.startswith("mpd_gpcc_") else stem


def load_manifest_pool(pattern, max_frames=0):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        raise FileNotFoundError(f"no manifest matches {pattern}")
    pool = {}
    for path in paths:
        frames = parse_mpd_xml(path)
        pool[sequence_name(path)] = frames[:max_frames] if max_frames else frames
    return pool


def find_winner_entry(sweep):
    trial = sweep["winner"]["trial"]
    return next(row for row in sweep["ranking"] if row["trial"] == trial)


def resolve_checkpoint(path):
    if os.path.exists(path):
        return path
    candidate = os.path.join(project_root, path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"winner checkpoint not found: {path}; final evaluation must run on the "
        "training PC before per-trial checkpoints are cleaned"
    )


def validate_lstm_provenance(path, segment_frames, expected_protocol):
    stem, _ = os.path.splitext(path)
    if stem.endswith("_best"):
        stem = stem[:-5]
    metadata_path = stem + "_split.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"matching LSTM provenance metadata is required: {metadata_path}"
        )
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if int(metadata.get("segment_frames", -1)) != int(segment_frames):
        raise ValueError(
            f"LSTM segment mismatch: metadata={metadata.get('segment_frames')} "
            f"evaluation={segment_frames}"
        )
    if metadata.get("protocol_digest") != expected_protocol:
        raise ValueError("LSTM/protocol digest mismatch")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return None


def summarize_training_seeds(per_seed):
    rows = []
    for seed, result in sorted(per_seed.items(), key=lambda item: int(item[0])):
        rows.append({"training_seed": int(seed),
                     **{k: result[k] for k in (
                         "reward", "qoe", "qoe_quality", "mean_quality", "stall_s"
                     )}})
    return rows


def build_oracle(fixed_results):
    """Hindsight best constant arm per trace (non-causal reference only)."""
    trace_names = sorted({case["trace"]
                          for result in fixed_results.values()
                          for case in result["cases"]})
    selected = {}
    oracle_cases = []
    for trace in trace_names:
        candidates = {}
        for arm_name, result in fixed_results.items():
            cases = [case for case in result["cases"] if case["trace"] == trace]
            candidates[arm_name] = sum(c["qoe_quality"] for c in cases) / len(cases)
        arm_name = max(candidates, key=candidates.get)
        selected[trace] = {
            "arm": arm_name,
            "selection_qoe_quality": candidates[arm_name],
        }
        oracle_cases.extend(
            case for case in fixed_results[arm_name]["cases"]
            if case["trace"] == trace
        )
    result = aggregate_records(oracle_cases)
    result["selected_arm_per_trace"] = selected
    result["causal"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description="Registered DQN/baseline evaluation")
    parser.add_argument("--protocol", default=os.path.join(
        "configs", "experiment_protocol.json"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--sweep-results", default=os.path.join(
        "models", "dqn_sweep_results.json"))
    parser.add_argument("--lstm", default=os.path.join("models", "bandwidth_lstm.pkl"))
    parser.add_argument("--baseline-config", default=os.path.join(
        "models", "baseline_config.json"),
        help="validation-selected causal baseline parameters")
    parser.add_argument("--mpd", default=os.path.join("manifests", "mpd_gpcc*.xml"))
    parser.add_argument("--out", default=os.path.join("models", "final_test_results.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-config-digest", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--offsets", type=int, default=0,
                        help="0 = registered value")
    parser.add_argument("--eval-seeds", default="", help="comma list; empty = protocol")
    parser.add_argument("--sequences", default="", help="comma list; empty = protocol")
    parser.add_argument("--max-dqn-seeds", type=int, default=0,
                        help="smoke convenience; 0 = every winner-config seed")
    parser.add_argument("--checkpoint", default="",
                        help="optional single DQN checkpoint for diagnostics")
    parser.add_argument(
        "--strategies", default="dqn,fixed,throughput,lstm_rule,buffer,mpc"
    )
    args = parser.parse_args()

    def absolute(path):
        return path if os.path.isabs(path) else os.path.join(project_root, path)

    out_path = absolute(args.out)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite registered evaluation {out_path}; use "
            "--overwrite only for a documented rerun with an unchanged method"
        )

    trace_dir = os.path.join(project_root, "bandwidth_5g")
    protocol = load_protocol(absolute(args.protocol), trace_dir)
    settings = evaluation_settings(protocol, args.split)
    trace_paths = split_paths(protocol, args.split, trace_dir)
    sequences = ([s.strip() for s in args.sequences.split(",") if s.strip()]
                 if args.sequences else list(settings["sequences"]))
    eval_seeds = ([int(s) for s in args.eval_seeds.split(",") if s.strip()]
                  if args.eval_seeds else list(settings["jitter_seeds"]))
    offsets = args.offsets or int(settings["offsets_per_trace"])
    manifest_pool = load_manifest_pool(absolute(args.mpd), args.max_frames)
    missing = sorted(set(sequences) - set(manifest_pool))
    if missing:
        raise ValueError(f"evaluation sequences absent from manifests: {missing}")

    with open(absolute(args.sweep_results), encoding="utf-8") as f:
        sweep = json.load(f)
    if sweep.get("objective_version") != OBJECTIVE_VERSION:
        raise ValueError(
            f"sweep objective {sweep.get('objective_version')!r} does not match "
            f"current {OBJECTIVE_VERSION!r}"
        )
    if (args.experiment_config_digest and
            sweep.get("experiment_config_digest") != args.experiment_config_digest):
        raise ValueError("sweep/current experiment config digest mismatch")
    winner_entry = find_winner_entry(sweep)
    winner_config = dict(winner_entry["config"])
    base_args = dict(sweep.get("base_args", {}))
    segment_frames = int(winner_config.get(
        "segment-frames", base_args.get("segment-frames", 8)))
    reward_spec = dict(DEFAULT_REWARD_SPEC)
    reward_spec.update(base_args.get("reward-spec", {}))
    reward_spec.update(winner_config.get("reward-spec", {}))
    reward_spec.setdefault("mu", float(winner_config.get(
        "mu", base_args.get("mu", 4.3))))
    reward_spec.setdefault("lam", float(base_args.get("lam", 1.0)))

    baseline_params = {
        "fixed": {"arm": 3},
        "throughput": {"history_window": 5, "safety_factor": 0.8},
        "lstm_rule": {"safety_factor": 0.8},
        "buffer": {"reservoir_s": 1.0, "cushion_s": 3.0},
        "mpc": {"horizon": 3, "history_window": 5, "safety_factor": 0.9},
    }
    baseline_config_path = absolute(args.baseline_config)
    baseline_config_rel = None
    if os.path.exists(baseline_config_path):
        with open(baseline_config_path, encoding="utf-8") as f:
            baseline_config = json.load(f)
        if baseline_config.get("selection_split") != "validation":
            raise ValueError("baseline parameters were not selected on validation")
        if baseline_config.get("protocol_digest") != protocol_digest(protocol):
            raise ValueError("baseline config/protocol digest mismatch")
        if int(baseline_config.get("segment_frames", -1)) != segment_frames:
            raise ValueError("baseline config/DQN segment_frames mismatch")
        if baseline_config.get("objective_version") != OBJECTIVE_VERSION:
            raise ValueError("baseline config/current objective mismatch")
        if baseline_config.get("reward_spec") != reward_spec:
            raise ValueError("baseline config/DQN reward specification mismatch")
        if (args.experiment_config_digest and
                baseline_config.get("experiment_config_digest") !=
                args.experiment_config_digest):
            raise ValueError("baseline/current experiment config digest mismatch")
        for name, family in baseline_config.get("families", {}).items():
            baseline_params[name] = dict(family["winner"]["params"])
        baseline_config_rel = os.path.relpath(baseline_config_path, project_root)
    else:
        print(f"WARNING: {baseline_config_path} absent; using documented defaults")

    requested = {s.strip() for s in args.strategies.split(",") if s.strip()}
    allowed = {"dqn", "fixed", "throughput", "lstm_rule", "buffer", "mpc"}
    unknown = sorted(requested - allowed)
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}; expected {sorted(allowed)}")
    tuned_required = sorted(
        (requested & {"fixed", "throughput", "lstm_rule", "buffer", "mpc"})
        - set(baseline_config.get("families", {}) if baseline_config_rel else {})
    )
    if baseline_config_rel and tuned_required:
        raise ValueError(f"requested baselines were not validation-tuned: {tuned_required}")

    lstm_path = absolute(args.lstm)
    if "dqn" in requested or "lstm_rule" in requested:
        validate_lstm_provenance(
            lstm_path, segment_frames, protocol_digest(protocol))

    print("=" * 96)
    print(f"REGISTERED EVALUATION | split={args.split} | protocol={protocol_digest(protocol)}")
    print(f"traces={len(trace_paths)} sequences={sequences} offsets/trace={offsets} "
          f"jitter_seeds={eval_seeds} segment_frames={segment_frames}")
    print("=" * 96)

    payload = {
        "schema_version": 1,
        "experiment_config_digest": args.experiment_config_digest or None,
        "split": args.split,
        "protocol": os.path.relpath(absolute(args.protocol), project_root),
        "protocol_digest": protocol_digest(protocol),
        "git_commit": git_commit(),
        "trace_files": [os.path.basename(p) for p in trace_paths],
        "sequences": sequences,
        "offsets_per_trace": offsets,
        "jitter_seeds": eval_seeds,
        "max_frames": args.max_frames,
        "winner_trial": sweep["winner"]["trial"],
        "winner_config": winner_config,
        "segment_frames": segment_frames,
        "reward_spec": reward_spec,
        "baseline_config": baseline_config_rel,
        "baseline_selection_split": "validation" if baseline_config_rel else None,
        "baseline_parameters": baseline_params,
        "results": {},
    }

    if "dqn" in requested:
        checkpoint_items = (
            [(sweep["winner"].get("seed", 0), args.checkpoint)]
            if args.checkpoint else
            sorted(winner_entry["checkpoints"].items(), key=lambda item: int(item[0]))
        )
        if args.max_dqn_seeds:
            checkpoint_items = checkpoint_items[:args.max_dqn_seeds]
        per_training_seed = {}
        all_cases = []
        for seed, checkpoint in checkpoint_items:
            checkpoint = resolve_checkpoint(checkpoint)
            agent = DQNAgent.load(checkpoint, device="cpu")
            if agent.reward_spec != reward_spec:
                raise ValueError(
                    f"checkpoint {checkpoint} reward spec does not match the frozen sweep"
                )
            predictor = None
            if "lstm_pred" in agent.feature_spec:
                predictor = LSTMPredictor().load(lstm_path)
            env = StreamingEnv(
                manifest_pool, lstm_predictor=predictor,
                tcp_params={**DEFAULT_TCP_PARAMS, "log_packets": False},
                feature_spec=agent.feature_spec, norm=agent.norm_constants,
                reward_spec=agent.reward_spec or reward_spec,
                segment_frames=segment_frames,
            )
            result = evaluate_agent(
                agent, env, trace_paths, eval_seeds, sequences,
                offsets_per_trace=offsets,
            )
            for case in result["cases"]:
                case["training_seed"] = int(seed)
            all_cases.extend(result["cases"])
            per_training_seed[str(seed)] = result
            print(f"DQN seed {seed}: qoe_quality={result['qoe_quality']:.3f} "
                  f"stall={result['stall_s']:.3f}s")
        dqn_result = aggregate_records(all_cases)
        dqn_result["per_training_seed"] = summarize_training_seeds(per_training_seed)
        dqn_result["training_seed_count"] = len(per_training_seed)
        dqn_result["checkpoint_selection_split"] = "validation"
        payload["results"]["dqn"] = dqn_result

    baseline_env = StreamingEnv(
        manifest_pool, lstm_predictor=None,
        tcp_params={**DEFAULT_TCP_PARAMS, "log_packets": False},
        feature_spec=DEFAULT_FEATURE_SPEC,
        reward_spec=reward_spec,
        segment_frames=segment_frames,
    )

    fixed_results = {}
    if "fixed" in requested:
        for arm in range(baseline_env.num_actions):
            name = f"fixed_arm_{arm}"
            result = evaluate_strategy(
                lambda arm=arm: FixedABR(arm), baseline_env,
                trace_paths, eval_seeds, sequences, offsets,
            )
            fixed_results[name] = result
            print(f"{name}: qoe_quality={result['qoe_quality']:.3f}")
        payload["results"]["fixed"] = fixed_results
        selected_name = f"fixed_arm_{int(baseline_params['fixed']['arm'])}"
        payload["results"]["best_global_fixed"] = {
            "arm": selected_name, **fixed_results[selected_name], "causal": True,
            "selection_split": "validation",
        }
        test_best = max(fixed_results, key=lambda name: fixed_results[name]["qoe_quality"])
        payload["results"]["best_test_global_fixed_oracle"] = {
            "arm": test_best, **fixed_results[test_best], "causal": False,
            "selection_split": "test_hindsight",
        }
        payload["results"]["per_trace_fixed_oracle"] = build_oracle(fixed_results)

    strategy_factories = {
        "throughput": lambda: BandwidthABR(**baseline_params["throughput"]),
        "lstm_rule": lambda: LSTMABR(
            LSTMPredictor().load(absolute(args.lstm)), **baseline_params["lstm_rule"]
        ),
        "buffer": lambda: BufferBasedABR(**baseline_params["buffer"]),
        "mpc": lambda: MPCABR(
            **baseline_params["mpc"],
            segment_frames=segment_frames,
            episode_frames=int(
                sum(map(len, manifest_pool.values())) / len(manifest_pool)
            ),
            mu=reward_spec["mu"],
            rebuffer_weight=reward_spec["rebuffer_weight"],
            lam=reward_spec["lam"],
            startup_weight=reward_spec["startup_weight"],
        ),
    }
    for name, factory in strategy_factories.items():
        if name not in requested:
            continue
        result = evaluate_strategy(
            factory, baseline_env, trace_paths, eval_seeds, sequences, offsets,
        )
        result["causal"] = True
        payload["results"][name] = result
        print(f"{name}: qoe_quality={result['qoe_quality']:.3f} "
              f"stall={result['stall_s']:.3f}s")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved case-level results: {out_path}")


if __name__ == "__main__":
    main()
