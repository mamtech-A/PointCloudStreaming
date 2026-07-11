#!/usr/bin/env python3
"""Compare ABR strategies: Baseline (rule) vs LSTM-rule vs DQN.

Runs three independent simulations on the SAME trace + manifest, each writing to
its own log dir (logs/<label>/), and prints a comparison table. Each strategy is
one policy applied to all of an edge's users (here a single user).

Usage:
    python compare.py                      # full 300-frame manifest
    python compare.py --max-frames 30      # quick run
    python compare.py --trace bandwidth_5g/static_B_2020.01.16_10.43.34.csv
"""

import os
import sys
import csv
import math
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model import (
    Server, EdgeNode, User, Topology, Simulator, BandwidthABR, LSTMABR, DQNABR,
    DEFAULT_TCP_PARAMS,
)
from src.network_model.trace import BandwidthTrace
from src.lstm_model import LSTMPredictor
from src.rl.dqn import DQNAgent

TCP_PARAMS = dict(DEFAULT_TCP_PARAMS)


def quality_metrics(results_csv, q_lo, q_hi, mu=4.3, lam=1.0):
    """Quality-aware metrics from a run's per-frame results.csv.

    mean_quality: mean log-density utility of the CHOSEN reps (0..1 over the
    manifest's density range). pensieve_reward: sum(q) - mu*stall - lam*|dq| —
    the quality-aware objective (the plain QoE column only counts stalls, so it
    rewards hiding at the lowest quality).
    """
    lo, hi = math.log10(q_lo), math.log10(q_hi)
    qs, reward, prev_q = [], 0.0, None
    with open(results_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            q = (math.log10(max(1.0, float(row['density']))) - lo) / (hi - lo)
            q = max(0.0, min(1.0, q))
            reward += q - mu * float(row['stall_duration_s'])
            if prev_q is not None:
                reward -= lam * abs(q - prev_q)
            qs.append(q)
            prev_q = q
    return (sum(qs) / len(qs) if qs else 0.0), reward


def build_sim(label, abr_factory, trace_path, playback_rate_min=1.0):
    server = Server(base_url="http://localhost/")
    topo = Topology(server)
    edge = EdgeNode("edge-1", server=server, tcp_params=TCP_PARAMS, abr_factory=abr_factory)
    topo.add_edge(edge)
    user = User("User", target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0,
                playback_rate_min=playback_rate_min)
    topo.add_user(user, edge, trace=BandwidthTrace.from_file(trace_path))
    return Simulator(topo)


def main():
    p = argparse.ArgumentParser(description="Compare baseline vs LSTM vs DQN ABR")
    p.add_argument('--trace', default=os.path.join('bandwidth_5g', 'static_B_2020.01.16_10.43.34.csv'),
                   help='bandwidth trace (default: an unseen test-split 5G trace)')
    p.add_argument('--max-frames', type=int, default=0, help='0 = full manifest')
    p.add_argument('--segment-frames', type=int, default=10,
                   help='frames per DASH-style segment (1 = legacy per-frame)')
    p.add_argument('--playback-rate-min', type=float, default=1.0,
                   help='adaptive-playback floor (1.0 = off; 0.9 recommended)')
    args = p.parse_args()

    mpd_path = os.path.join(project_root, 'config', 'mpd_gpcc_longdress.xml')
    trace_path = os.path.join(project_root, args.trace)
    lstm_path = os.path.join(project_root, 'models', 'bandwidth_lstm.pkl')
    dqn_path = os.path.join(project_root, 'models', 'abr_dqn.pkl')
    max_frames = args.max_frames or None

    print("=" * 100)
    print("BASELINE vs LSTM vs DQN comparison")
    print(f"trace: {trace_path} | frames: {'all' if not max_frames else max_frames} | "
          f"segment: {args.segment_frames} | amp_floor: {args.playback_rate_min}")
    print("=" * 100)

    runs = []

    # 1) Baseline rule.
    sim = build_sim('baseline', lambda: BandwidthABR(), trace_path, args.playback_rate_min)
    runs.append(sim.run(mpd_path, run_label='baseline', return_summary=True, max_frames=max_frames, segment_frames=args.segment_frames)[0])

    # 2) LSTM-rule (one shared predictor; predictions rebuild their window each call).
    predictor_lstm = LSTMPredictor().load(lstm_path)
    sim = build_sim('lstm', lambda: LSTMABR(predictor_lstm, use_prediction=True), trace_path, args.playback_rate_min)
    runs.append(sim.run(mpd_path, run_label='lstm', return_summary=True, max_frames=max_frames, segment_frames=args.segment_frames)[0])

    # 3) DQN (if a trained policy exists).
    if os.path.exists(dqn_path):
        agent = DQNAgent.load(dqn_path)
        predictor_dqn = LSTMPredictor().load(lstm_path)
        sim = build_sim('dqn', lambda: DQNABR(policy=agent, lstm_provider=LSTMABR(predictor_dqn)), trace_path, args.playback_rate_min)
        runs.append(sim.run(mpd_path, run_label='dqn', return_summary=True, max_frames=max_frames, segment_frames=args.segment_frames)[0])
    else:
        print(f"\n(skipping DQN — no trained model at {dqn_path}; run train_dqn.py first)")

    # --- Comparison table (stall-only QoE AND quality-aware metrics) ---
    from src.network_model.manifest import parse_mpd_xml
    from src.rl.features import quality_endpoints
    q_lo, q_hi = quality_endpoints(parse_mpd_xml(mpd_path))

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    hdr = (f"{'strategy':<10} {'QoE':>6} {'QoE_q':>8} {'rebuf':>6} {'stall_s':>9} {'slow_s':>7} {'dropped':>8} "
           f"{'mean_rep':>9} {'switches':>9} {'mean_qual':>10} {'reward':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        mq, rw = quality_metrics(os.path.join(project_root, 'logs', r['abr'] if r['abr'] != 'bandwidth' else 'baseline', 'results.csv'), q_lo, q_hi)
        print(f"{r['abr']:<10} {r['qoe']:>6.1f} {r.get('qoe_quality', 0.0):>8.1f} "
              f"{r['rebuffer_count']:>6d} {r['total_stall_time_s']:>9.1f} "
              f"{r.get('slowdown_time_s', 0.0):>7.1f} "
              f"{r['frames_dropped']:>8d} {r['mean_rep_id']:>9.2f} {r['quality_switches']:>9d} "
              f"{mq:>10.3f} {rw:>9.1f}")
    print("\nQoE counts only stalls/drops (favors the lowest quality); QoE_q is the")
    print("quality-aware QoE' = 100*mean_quality - 4.3*stall_s - 1.0*sum|dq| (raw, unclipped);")
    print("'reward' is the Pensieve objective: sum(quality) - 4.3*stall_s - 1.0*|quality change|.")
    print("Logs written to logs/baseline/, logs/lstm/, logs/dqn/")


if __name__ == "__main__":
    main()
