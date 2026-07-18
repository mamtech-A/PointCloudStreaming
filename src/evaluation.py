"""Shared, case-level evaluation for DQN and rule-based ABR policies."""

from __future__ import annotations

import os
import random
from collections import defaultdict

import numpy as np

from .experiment_protocol import evenly_spaced_offsets
from .network_model.trace import BandwidthTrace


METRICS = ("reward", "qoe", "qoe_quality", "mean_quality", "stall_s")


def _metric_summary(records):
    result = {"n_cases": len(records)}
    for metric in METRICS:
        values = [float(r[metric]) for r in records]
        result[metric] = float(np.mean(values)) if values else 0.0
        result[f"{metric}_std"] = float(np.std(values)) if values else 0.0
    return result


def _grouped(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: _metric_summary(rows) for name, rows in sorted(groups.items())}


def aggregate_records(records):
    """Aggregate case rows while preserving the dimensions needed by figures."""
    result = _metric_summary(records)
    result["per_trace"] = _grouped(records, "trace")
    result["per_sequence"] = _grouped(records, "sequence")
    result["per_seed"] = _grouped(records, "eval_seed")
    result["cases"] = records
    return result


def _evaluate(env, trace_paths, eval_seeds, sequences, offsets_per_trace,
              choose_action, reset_policy=None, min_tail_samples=120):
    """Evaluate one policy over the full Cartesian registered case set.

    ``choose_action`` receives ``(observation, env)`` and returns an action
    index. All global RNG state is restored after evaluation so checkpoint
    selection does not perturb subsequent training episodes.
    """
    rng_state = random.getstate()
    np_state = np.random.get_state()
    records = []
    traces = {path: BandwidthTrace.from_file(path) for path in trace_paths}
    try:
        for eval_seed in eval_seeds:
            for sequence in sequences:
                for path in trace_paths:
                    full_trace = traces[path]
                    offsets = evenly_spaced_offsets(
                        len(full_trace), offsets_per_trace, min_tail_samples
                    )
                    for offset in offsets:
                        # Reuse the same seed for the same registered case across
                        # all policies, giving paired transport-jitter outcomes.
                        random.seed(int(eval_seed))
                        np.random.seed(int(eval_seed))
                        trace = full_trace.slice_from(offset) if offset else full_trace
                        observation = env.reset(trace, sequence=sequence)
                        if reset_policy is not None:
                            reset_policy()
                        done = False
                        total_reward = 0.0
                        while not done:
                            action = int(choose_action(observation, env))
                            observation, reward, done, _ = env.step(action)
                            total_reward += reward
                        records.append({
                            "trace": os.path.basename(path),
                            "sequence": sequence,
                            "offset": int(offset),
                            "eval_seed": int(eval_seed),
                            "reward": float(total_reward),
                            "qoe": float(env.qoe()),
                            "qoe_quality": float(env.qoe_quality()),
                            "mean_quality": float(env.mean_quality()),
                            "stall_s": float(env.total_stall_s()),
                        })
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    return aggregate_records(records)


def evaluate_agent(agent, env, trace_paths, eval_seeds, sequences,
                   offsets_per_trace=1, min_tail_samples=120):
    """Greedy DQN evaluation over registered traces, contents, and offsets."""
    return _evaluate(
        env, trace_paths, eval_seeds, sequences, offsets_per_trace,
        choose_action=lambda observation, _env: agent.act(observation, epsilon=0.0),
        min_tail_samples=min_tail_samples,
    )


def evaluate_strategy(strategy_factory, env, trace_paths, eval_seeds, sequences,
                      offsets_per_trace=1, min_tail_samples=120):
    """Evaluate an ``ABRStrategy`` through the same StreamingEnv as the DQN."""
    strategy = strategy_factory()

    def choose_action(_observation, active_env):
        state = active_env.current_abr_state()
        rep_id = strategy.select(state)
        reps = sorted(state.reps, key=lambda rep: rep["id"])
        for index, rep in enumerate(reps):
            if rep["id"] == rep_id:
                return index
        raise ValueError(f"strategy selected unknown representation id {rep_id}")

    return _evaluate(
        env, trace_paths, eval_seeds, sequences, offsets_per_trace,
        choose_action=choose_action,
        reset_policy=strategy.reset,
        min_tail_samples=min_tail_samples,
    )
