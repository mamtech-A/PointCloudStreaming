#!/usr/bin/env python3
"""Fixed-arm baselines: run each constant-representation policy over the held-out
test traces and print/save the per-arm reward table.

This is the reusable replacement for the throwaway code behind the original
fixed-policy bars (DQN_REPORT section 4.1). Methodology mirrors
`train_dqn.evaluate`: fixed jitter seed per trace, greedy constant action,
sum of Pensieve-style env rewards. The best arm's mean reward is the bar a
trained DQN must beat (`models/fixed_arm_baseline.json`).

Usage:
    python eval_fixed.py                 # full 300-frame episodes, 4 test traces
    python eval_fixed.py --max-frames 60 # quick run
"""

import os
import sys
import json
import argparse
import random

import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model import DEFAULT_TCP_PARAMS
from src.network_model.manifest import parse_mpd_xml
from src.network_model.trace import BandwidthTrace
from src.lstm_model import split_bandwidth_files
from src.rl.env import StreamingEnv

TCP_PARAMS = dict(DEFAULT_TCP_PARAMS)


def run_arm(env, arm, trace_path, seed):
    """One constant-arm episode with the same RNG discipline as train_dqn.evaluate."""
    rng_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        env.reset(BandwidthTrace.from_file(trace_path))
        done = False
        total_reward = 0.0
        total_stall = 0.0
        qualities = []
        while not done:
            _, r, done, info = env.step(arm)
            total_reward += r
            total_stall += info['stall_s']
            qualities.append(info['quality'])
        return {
            'reward': total_reward,
            'qoe': env.qoe(),
            'qoe_quality': env.qoe_quality(),
            'stall_s': total_stall,
            'mean_quality': float(np.mean(qualities)) if qualities else 0.0,
        }
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)


def main():
    p = argparse.ArgumentParser(description="Evaluate fixed-representation ABR arms")
    p.add_argument('--trace-dir', default=os.path.join(project_root, 'bandwidth_5g'))
    p.add_argument('--mpd', default=os.path.join(project_root, 'config', 'mpd_gpcc_longdress.xml'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--mu', type=float, default=4.3)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--segment-frames', type=int, default=10,
                   help='frames per DASH-style segment (1 = legacy per-frame)')
    p.add_argument('--playback-rate-min', type=float, default=1.0,
                   help='adaptive-playback floor (1.0 = off)')
    p.add_argument('--out', default=os.path.join(project_root, 'models', 'fixed_arm_baseline.json'))
    args = p.parse_args()

    _, test_files = split_bandwidth_files(args.trace_dir, test_size=args.test_size,
                                          random_state=args.seed)
    test_paths = [os.path.join(args.trace_dir, f) for f in test_files]

    # Split-consistency guard: the LSTM/DQN test set must be the same files.
    split_json = os.path.join(project_root, 'models', 'bandwidth_lstm_split.json')
    if os.path.exists(split_json):
        with open(split_json, encoding='utf-8') as f:
            recorded = json.load(f).get('test_files', [])
        if all(os.path.exists(os.path.join(args.trace_dir, t)) for t in recorded):
            assert sorted(recorded) == sorted(test_files), (
                f"split mismatch: {split_json} test_files {recorded} != {test_files}")
        else:
            print(f"WARNING: {split_json} references traces not in {args.trace_dir} "
                  f"(stale pre-migration split?) - consistency check skipped")

    frames = parse_mpd_xml(args.mpd)
    if args.max_frames:
        frames = frames[:args.max_frames]

    env = StreamingEnv(frames, lstm_predictor=None, tcp_params=TCP_PARAMS,
                       mu=args.mu, lam=args.lam,
                       segment_frames=args.segment_frames,
                       playback_rate_min=args.playback_rate_min)

    # Arm a selects reps_sorted_by_id[a]; report each arm's manifest identity.
    reps0 = sorted(frames[0]['representations'], key=lambda r: r['id'])

    print("=" * 100)
    print(f"Fixed-arm baselines | {len(test_paths)} test traces | {len(frames)} frames/episode "
          f"| segment={args.segment_frames} amp_floor={args.playback_rate_min} "
          f"| mu={args.mu} lam={args.lam} seed={args.seed}")
    for t in test_files:
        print(f"   test trace: {t}")
    print("=" * 100)

    results = {}
    for arm in range(env.num_actions):
        per_trace = [run_arm(env, arm, tp, args.seed) for tp in test_paths]
        agg = {k: float(np.mean([r[k] for r in per_trace]))
               for k in ('reward', 'qoe', 'qoe_quality', 'stall_s', 'mean_quality')}
        agg['per_trace'] = {os.path.basename(tp): r for tp, r in zip(test_paths, per_trace)}
        rep = reps0[arm]
        agg['rep_id'] = rep['id']
        agg['density'] = rep.get('density')
        agg['bitrate_mbps'] = (rep.get('bandwidth') or 0) / 1e6
        results[f'arm_{arm}'] = agg
        print(f"arm {arm} (rep {rep['id']}, {agg['bitrate_mbps']:6.1f} Mbps): "
              f"mean_reward={agg['reward']:8.2f}  mean_qoe={agg['qoe']:6.1f}  "
              f"mean_qoe_q={agg['qoe_quality']:8.1f}  "
              f"mean_stall={agg['stall_s']:7.2f}s  mean_quality={agg['mean_quality']:.3f}")

    best = max(results, key=lambda k: results[k]['reward'])
    print("-" * 100)
    print(f"BEST FIXED ARM: {best} (rep {results[best]['rep_id']}, "
          f"{results[best]['bitrate_mbps']:.1f} Mbps) mean_reward={results[best]['reward']:.2f}")

    payload = {
        'seed': args.seed, 'mu': args.mu, 'lam': args.lam,
        'segment_frames': args.segment_frames,
        'playback_rate_min': args.playback_rate_min,
        'frames_per_episode': len(frames), 'test_files': test_files,
        'tcp_params': TCP_PARAMS, 'results': results, 'best_arm': best,
        'best_reward': results[best]['reward'],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
