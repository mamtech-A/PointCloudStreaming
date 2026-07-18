#!/usr/bin/env python3
"""Select causal baseline parameters on the registered validation split."""

import argparse
import glob
import itertools
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation import evaluate_strategy
from src.experiment_protocol import (
    evaluation_settings, load_protocol, protocol_digest, split_paths,
)
from src.lstm_model import LSTMPredictor
from src.network_model import (
    DEFAULT_TCP_PARAMS, BandwidthABR, BufferBasedABR, LSTMABR, MPCABR,
)
from src.network_model.abr import ABRStrategy
from src.network_model.manifest import parse_mpd_xml
from src.rl.env import StreamingEnv
from src.rl.features import DEFAULT_FEATURE_SPEC
from src.rl.reward import DEFAULT_REWARD_SPEC, OBJECTIVE_VERSION


class FixedABR(ABRStrategy):
    name = "fixed"

    def __init__(self, arm):
        self.arm = int(arm)

    def select(self, state):
        reps = sorted(state.reps, key=lambda rep: rep["id"])
        return reps[self.arm]["id"]


def sequence_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("mpd_gpcc_"):] if stem.startswith("mpd_gpcc_") else stem


def load_pool(pattern, max_frames=0):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no manifest matches {pattern}")
    pool = {}
    for path in paths:
        frames = parse_mpd_xml(path)
        pool[sequence_name(path)] = frames[:max_frames] if max_frames else frames
    return pool


def rank(candidates):
    return sorted(
        candidates,
        key=lambda row: (-row["metrics"]["qoe_quality"], row["metrics"]["stall_s"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Validation-only baseline tuning")
    parser.add_argument("--protocol", default=os.path.join(
        "configs", "experiment_protocol.json"))
    parser.add_argument("--mpd", default=os.path.join("manifests", "mpd_gpcc*.xml"))
    parser.add_argument("--out", default=os.path.join("models", "baseline_config.json"))
    parser.add_argument("--lstm", default=os.path.join("models", "bandwidth_lstm.pkl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-config-digest", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--offsets", type=int, default=0)
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--segment-frames", type=int, default=8)
    parser.add_argument(
        "--families", default="fixed,buffer,mpc",
        help="comma-separated validation-tuned families",
    )
    parser.add_argument("--quick", action="store_true",
                        help="one candidate per family for pipeline smoke tests")
    args = parser.parse_args()

    def absolute(path):
        return path if os.path.isabs(path) else os.path.join(project_root, path)

    out_path = absolute(args.out)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {out_path} without --overwrite")

    trace_dir = os.path.join(project_root, "bandwidth_5g")
    protocol = load_protocol(absolute(args.protocol), trace_dir)
    settings = evaluation_settings(protocol, "validation")
    trace_paths = split_paths(protocol, "validation", trace_dir)
    sequences = ([s.strip() for s in args.sequences.split(",") if s.strip()]
                 if args.sequences else list(settings["sequences"]))
    seeds = ([int(s) for s in args.eval_seeds.split(",") if s.strip()]
             if args.eval_seeds else list(settings["jitter_seeds"]))
    offsets = args.offsets or int(settings["offsets_per_trace"])
    pool = load_pool(absolute(args.mpd), args.max_frames)

    reward_spec = dict(DEFAULT_REWARD_SPEC)
    env = StreamingEnv(
        pool, lstm_predictor=None,
        tcp_params={**DEFAULT_TCP_PARAMS, "log_packets": False},
        feature_spec=DEFAULT_FEATURE_SPEC, reward_spec=reward_spec,
        segment_frames=args.segment_frames,
    )

    grids = {
        "fixed": [{"arm": arm} for arm in range(env.num_actions)],
        "throughput": [
            {"history_window": w, "safety_factor": sf}
            for w, sf in itertools.product([3, 5, 8], [0.75, 0.85, 0.95])
        ],
        "lstm_rule": [
            {"safety_factor": sf} for sf in [0.75, 0.85, 0.95]
        ],
        "buffer": [
            {"reservoir_s": r, "cushion_s": c}
            for r, c in itertools.product(
                [0.25, 0.5, 1.0, 1.5, 2.0], [1.5, 2.0, 3.0, 4.0]
            )
        ],
        "mpc": [
            {"horizon": h, "history_window": w, "safety_factor": sf}
            for h, w, sf in itertools.product(
                [2, 3, 4], [3, 5, 8], [0.75, 0.85, 0.95]
            )
        ],
    }
    requested = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(grids))
    if unknown:
        raise ValueError(f"unknown baseline families {unknown}; expected {sorted(grids)}")
    grids = {name: grids[name] for name in requested}
    if args.quick:
        grids = {name: values[:1] for name, values in grids.items()}

    def factory(name, params):
        if name == "fixed":
            return lambda: FixedABR(**params)
        if name == "throughput":
            return lambda: BandwidthABR(**params)
        if name == "lstm_rule":
            return lambda: LSTMABR(
                LSTMPredictor().load(absolute(args.lstm)), **params
            )
        if name == "buffer":
            return lambda: BufferBasedABR(**params)
        if name == "mpc":
            return lambda: MPCABR(
                **params, segment_frames=args.segment_frames,
                episode_frames=int(sum(map(len, pool.values())) / len(pool)),
                mu=reward_spec["mu"],
                rebuffer_weight=reward_spec["rebuffer_weight"],
                lam=reward_spec["lam"],
                startup_weight=reward_spec["startup_weight"],
            )
        raise KeyError(name)

    families = {}
    for name, candidates in grids.items():
        rows = []
        for params in candidates:
            metrics = evaluate_strategy(
                factory(name, params), env, trace_paths, seeds, sequences, offsets,
            )
            rows.append({"params": params, "metrics": metrics})
            print(f"{name} {params}: qoe_quality={metrics['qoe_quality']:.3f} "
                  f"stall={metrics['stall_s']:.3f}s")
        ranking = rank(rows)
        families[name] = {"winner": ranking[0], "ranking": ranking}
        print(f"WINNER {name}: {ranking[0]['params']}")

    payload = {
        "schema_version": 1,
        "experiment_config_digest": args.experiment_config_digest or None,
        "objective_version": OBJECTIVE_VERSION,
        "reward_spec": reward_spec,
        "selection_split": "validation",
        "selection_metric": "qoe_quality",
        "protocol": os.path.relpath(absolute(args.protocol), project_root),
        "protocol_digest": protocol_digest(protocol),
        "trace_files": [os.path.basename(p) for p in trace_paths],
        "sequences": sequences,
        "offsets_per_trace": offsets,
        "jitter_seeds": seeds,
        "segment_frames": args.segment_frames,
        "families": families,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved frozen baseline parameters: {out_path}")


if __name__ == "__main__":
    main()
