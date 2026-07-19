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
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def build_sim(label, abr_factory, trace_path):
    server = Server(base_url="http://localhost/")
    topo = Topology(server)
    edge = EdgeNode("edge-1", server=server, tcp_params=TCP_PARAMS, abr_factory=abr_factory)
    topo.add_edge(edge)
    user = User("User", target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0)
    topo.add_user(user, edge, trace=BandwidthTrace.from_file(trace_path))
    return Simulator(topo)


def main():
    p = argparse.ArgumentParser(description="Compare baseline vs LSTM vs DQN ABR")
    p.add_argument('--trace', default=os.path.join('bandwidth_5g', 'static_B_2020.01.16_10.43.34.csv'),
                   help='bandwidth trace (default: an unseen test-split 5G trace)')
    p.add_argument('--max-frames', type=int, default=0, help='0 = full manifest')
    p.add_argument('--segment-frames', type=int, default=5,
                   help='frames per DASH-style segment (1 = legacy per-frame)')
    args = p.parse_args()

    mpd_path = os.path.join(project_root, 'manifests', 'mpd_gpcc_longdress.xml')
    trace_path = os.path.join(project_root, args.trace)
    lstm_path = os.path.join(project_root, 'models', 'bandwidth_lstm.pkl')
    dqn_path = os.path.join(project_root, 'models', 'abr_dqn.pkl')
    max_frames = args.max_frames or None

    print("=" * 100)
    print("BASELINE vs LSTM vs DQN comparison")
    print(f"trace: {trace_path} | frames: {'all' if not max_frames else max_frames} | "
          f"segment: {args.segment_frames}")
    print("=" * 100)

    runs = []

    # 1) Baseline rule.
    sim = build_sim('baseline', lambda: BandwidthABR(), trace_path)
    runs.append(sim.run(mpd_path, run_label='baseline', return_summary=True, max_frames=max_frames, segment_frames=args.segment_frames, verbose=False)[0])

    # 2) LSTM-rule (one shared predictor; predictions rebuild their window each call).
    predictor_lstm = LSTMPredictor().load(lstm_path)
    sim = build_sim('lstm', lambda: LSTMABR(predictor_lstm, use_prediction=True), trace_path)
    runs.append(sim.run(mpd_path, run_label='lstm', return_summary=True, max_frames=max_frames, segment_frames=args.segment_frames, verbose=False)[0])

    # 3) DQN (if a trained policy exists).
    if os.path.exists(dqn_path):
        agent = DQNAgent.load(dqn_path)
        predictor_dqn = LSTMPredictor().load(lstm_path)
        sim = build_sim('dqn', lambda: DQNABR(policy=agent, lstm_provider=LSTMABR(predictor_dqn)), trace_path)
        runs.append(sim.run(mpd_path, run_label='dqn', return_summary=True, max_frames=max_frames, segment_frames=args.segment_frames, verbose=False)[0])
    else:
        print(f"\n(skipping DQN — no trained model at {dqn_path}; run train_dqn.py first)")

    # --- Comparison table under the one canonical reward/QoE ---
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    hdr = (f"{'strategy':<10} {'QoE':>8} {'rebuf':>6} {'stall_s':>9} {'pacing_s':>9} "
           f"{'mean_rep':>9} {'switches':>9} {'mean_qual':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        print(f"{r['abr']:<10} {r['qoe']:>8.1f} "
              f"{r['rebuffer_count']:>6d} {r['total_stall_time_s']:>9.1f} "
              f"{r['request_pacing_s']:>9.1f} {r['mean_rep_id']:>9.2f} {r['quality_switches']:>9d} "
              f"{r['mean_quality']:>10.3f}")
    print("\nreward = QoE = 100*mean_q - 4.3*stall_s - 2.0*rebuffer_events")
    print("             - 1.0*sum|dq_segment| - 1.0*startup_s")
    print("             (raw, unclipped; startup and post-start stalls are disjoint).")
    print("Logs written to logs/baseline/, logs/lstm/, logs/dqn/")


if __name__ == "__main__":
    main()
