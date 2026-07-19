"""Shared, case-level evaluation for DQN and rule-based ABR policies."""

from __future__ import annotations

import os
import random
from collections import defaultdict

import numpy as np

from .network_model.finite_trace import TraceWindowExhausted


METRICS = (
    "reward", "qoe", "qoe_quality", "mean_quality", "stall_s",
    "rebuffer_events", "quality_change", "frames_dropped", "startup_s",
)


def _metric_summary(records):
    result = {"n_cases": len(records)}
    for metric in METRICS:
        values = [float(r[metric]) for r in records]
        result[metric] = float(np.mean(values)) if values else 0.0
        result[f"{metric}_std"] = float(np.std(values)) if values else 0.0
    failures = sum(bool(row.get("policy_trace_exhausted")) for row in records)
    result["policy_trace_exhausted_cases"] = failures
    result["completed_cases"] = len(records) - failures
    result["policy_trace_exhausted_rate"] = (
        failures / len(records) if records else 0.0
    )
    return result


def _grouped(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: _metric_summary(rows) for name, rows in sorted(groups.items())}


def aggregate_records(records):
    """Macro-average parent-trace means and retain every case-level row."""
    case_level = _metric_summary(records)
    per_trace = _grouped(records, "trace")
    result = {
        "n_cases": len(records),
        "n_parent_traces": len(per_trace),
        "aggregation": "macro_parent_trace",
        "case_level_summary": case_level,
        "per_trace": per_trace,
        "policy_trace_exhausted_cases": case_level[
            "policy_trace_exhausted_cases"
        ],
        "completed_cases": case_level["completed_cases"],
    }
    for metric in (*METRICS, "policy_trace_exhausted_rate"):
        values = [float(summary[metric]) for summary in per_trace.values()]
        result[metric] = float(np.mean(values)) if values else 0.0
        result[f"{metric}_std"] = float(np.std(values)) if values else 0.0
    result["per_sequence"] = _grouped(records, "sequence")
    result["per_seed"] = _grouped(records, "eval_seed")
    result["cases"] = records
    return result


def _evaluate(env, trace_paths, eval_seeds, sequences, offsets_per_trace,
              choose_action, reset_policy=None, window_registry=None,
              split_name=None):
    """Evaluate one policy over the full Cartesian registered case set.

    ``choose_action`` receives ``(observation, env)`` and returns an action
    index. All global RNG state is restored after evaluation so checkpoint
    selection does not perturb subsequent training episodes.
    """
    rng_state = random.getstate()
    np_state = np.random.get_state()
    records = []
    if window_registry is None:
        raise ValueError("evaluation requires the frozen trace-window registry")
    if split_name not in ("validation", "test"):
        raise ValueError("evaluation split_name must be validation or test")
    grouped_windows = window_registry.validate_evaluation_count(
        split_name, offsets_per_trace, trace_paths
    )
    try:
        for eval_seed in eval_seeds:
            for sequence in sequences:
                for path in trace_paths:
                    filename = os.path.basename(path)
                    for window in grouped_windows[filename]:
                        # Reuse the same seed for the same registered case across
                        # all policies, giving paired transport-jitter outcomes.
                        random.seed(int(eval_seed))
                        np.random.seed(int(eval_seed))
                        trace = window_registry.materialize(window)
                        observation = env.reset(trace, sequence=sequence)
                        if reset_policy is not None:
                            reset_policy()
                        done = False
                        total_reward = 0.0
                        policy_failure = False
                        failure_detail = None
                        while not done:
                            action = int(choose_action(observation, env))
                            try:
                                observation, reward, done, _ = env.step(action)
                            except TraceWindowExhausted as exc:
                                observation, reward, done, failure_detail = (
                                    env.terminate_trace_exhausted()
                                )
                                failure_detail.update({
                                    "reason": "policy_trace_exhausted",
                                    "trace_end_s": exc.trace_end_s,
                                    "remaining_bits_in_tcp_round": exc.remaining_bits,
                                })
                                policy_failure = True
                            total_reward += reward
                        qoe = float(env.qoe())
                        if abs(total_reward - qoe) > 1e-8:
                            raise AssertionError(
                                f"reward/QoE mismatch: return={total_reward} qoe={qoe}"
                            )
                        stats = env.user.get_buffer_stats()
                        records.append({
                            "trace": os.path.basename(path),
                            "sequence": sequence,
                            "window_id": window["id"],
                            "block_id": window["block_id"],
                            "start_time_s": float(window["start_time_s"]),
                            "measured_duration_s": float(window["duration_s"]),
                            "offset": int(window["source_sample_index"]),
                            "eval_seed": int(eval_seed),
                            "reward": float(total_reward),
                            "qoe": qoe,
                            # Compatibility alias for existing result readers.
                            "qoe_quality": qoe,
                            "mean_quality": float(env.mean_quality()),
                            "stall_s": float(env.total_stall_s()),
                            "rebuffer_events": int(stats.get('rebuffer_count', 0)),
                            "quality_change": float(env.quality_change_sum()),
                            "frames_dropped": env.frames_dropped(),
                            "startup_s": float(stats.get('startup_delay_s', 0.0)),
                            "qoe_terms": env.qoe_terms(),
                            "policy_trace_exhausted": policy_failure,
                            "policy_failure": failure_detail,
                        })
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    return aggregate_records(records)


def evaluate_agent(agent, env, trace_paths, eval_seeds, sequences,
                   offsets_per_trace=1, window_registry=None,
                   split_name="validation"):
    """Greedy DQN evaluation over registered traces, contents, and offsets."""
    return _evaluate(
        env, trace_paths, eval_seeds, sequences, offsets_per_trace,
        choose_action=lambda observation, _env: agent.act(observation, epsilon=0.0),
        window_registry=window_registry,
        split_name=split_name,
    )


def evaluate_strategy(strategy_factory, env, trace_paths, eval_seeds, sequences,
                      offsets_per_trace=1, window_registry=None,
                      split_name="validation"):
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
        window_registry=window_registry,
        split_name=split_name,
    )
