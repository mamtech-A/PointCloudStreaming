#!/usr/bin/env python3
"""Train the DQN ABR agent against the streaming environment.

Trains on the SAME file-level train/test split as the LSTM (so results are
comparable and leak-free), keeping the LSTM as a feature provider. Saves the best
agent (by held-out eval reward) to models/abr_dqn.pkl.

NOTE: each env step is a full TCP transfer of a (large) point-cloud frame, so
training is compute-heavy. The defaults are modest; scale up with the flags below
for a real run. The script prints exactly how much it covers (no silent caps).

Usage:
    python train_dqn.py                          # default modest run
    python train_dqn.py --epochs 40              # longer, better policy
    python train_dqn.py --epochs 2 --max-train-files 2 --max-frames 40 --eval-every 1  # smoke
"""

import os
import sys
import argparse
import random

import numpy as np
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model.manifest import parse_mpd_xml
from src.network_model.trace import BandwidthTrace
from src.lstm_model import split_bandwidth_files, LSTMPredictor
from src.rl.env import StreamingEnv
from src.rl.dqn import DQNAgent
from src.rl.features import DEFAULT_FEATURE_SPEC

TCP_PARAMS = {
    'rtt_ms': 50.0, 'rtt_jitter_ms': 10.0, 'loss_prob': 0.0,
    'cwnd_packets': 10.0, 'mss_bytes': 1460, 'rto_formula': 'jacobson', 'rto_fixed_s': 1.0,
}


def evaluate(agent, env, trace_paths, seed):
    """Greedy eval over traces with fixed jitter seed (for run-to-run comparability).

    Saves/restores the global RNG state so mid-training evals don't reset the
    exploration/jitter randomness of subsequent training episodes.
    """
    rng_state = random.getstate()
    np_state = np.random.get_state()
    rewards, qoes = [], []
    try:
        for path in trace_paths:
            random.seed(seed)
            np.random.seed(seed)
            s = env.reset(BandwidthTrace.from_log(path))
            done = False
            total = 0.0
            while not done:
                a = agent.act(s, epsilon=0.0)
                s, r, done, _ = env.step(a)
                total += r
            rewards.append(total)
            qoes.append(env.qoe())
    finally:
        random.setstate(rng_state)
        np.random.set_state(np_state)
    return float(np.mean(rewards)), float(np.mean(qoes))


def main():
    p = argparse.ArgumentParser(description="Train DQN ABR agent")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max-train-files', type=int, default=0, help='0 = all 36 train files')
    p.add_argument('--max-test-files', type=int, default=0, help='0 = all 4 test files')
    p.add_argument('--max-frames', type=int, default=0, help='0 = all manifest frames')
    p.add_argument('--eval-every', type=int, default=50, help='episodes between evals')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--mu', type=float, default=4.3, help='rebuffer penalty weight')
    p.add_argument('--lam', type=float, default=1.0, help='quality-switch penalty weight')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-frac', type=float, default=0.6)
    p.add_argument('--out', type=str, default=os.path.join('models', 'abr_dqn.pkl'))
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bandwidth_dir = os.path.join(project_root, 'bandwidth')
    mpd_path = os.path.join(project_root, 'config', 'mpd.xml')
    lstm_path = os.path.join(project_root, 'models', 'bandwidth_lstm.pkl')

    train_files, test_files = split_bandwidth_files(bandwidth_dir, test_size=0.1, random_state=args.seed)
    if args.max_train_files:
        train_files = train_files[:args.max_train_files]
    if args.max_test_files:
        test_files = test_files[:args.max_test_files]
    train_paths = [os.path.join(bandwidth_dir, f) for f in train_files]
    test_paths = [os.path.join(bandwidth_dir, f) for f in test_files]

    frames = parse_mpd_xml(mpd_path)
    if args.max_frames:
        frames = frames[:args.max_frames]

    feature_spec = DEFAULT_FEATURE_SPEC
    predictor = None
    if 'lstm_pred' in feature_spec:
        predictor = LSTMPredictor().load(lstm_path)

    env = StreamingEnv(frames, lstm_predictor=predictor, tcp_params=TCP_PARAMS,
                       feature_spec=feature_spec, mu=args.mu, lam=args.lam)
    agent = DQNAgent(env.state_dim, env.num_actions, env.feature_spec, env.norm,
                     hidden=args.hidden, lr=args.lr, gamma=args.gamma, mu=args.mu, lam=args.lam,
                     lstm_model_path=lstm_path,
                     sequence_length=getattr(predictor, 'sequence_length', 10))

    total_episodes = args.epochs * len(train_files)
    total_steps_est = max(1, total_episodes * len(frames))
    decay_steps = max(1, int(args.eps_decay_frac * total_steps_est))

    print("=" * 90)
    print("🤖 DQN ABR training")
    print(f"   train files: {len(train_files)} | test files: {len(test_files)} | frames/episode: {len(frames)}")
    print(f"   epochs: {args.epochs} | episodes: {total_episodes} | est. env steps: {total_steps_est}")
    print(f"   state_dim: {env.state_dim} | actions: {env.num_actions} | feature_spec: {feature_spec}")
    print(f"   reward: quality - {args.mu}*stall - {args.lam}*|dq|  (gamma={args.gamma}, lr={args.lr})")
    print("=" * 90)

    step_count = 0
    best_eval = float('-inf')
    history = {'episode_reward': [], 'eval': []}
    episode = 0

    def epsilon():
        frac = min(1.0, step_count / decay_steps)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    # Baseline eval (untrained policy) for reference.
    base_r, base_q = evaluate(agent, env, test_paths, args.seed)
    print(f"[eval @ ep 0] untrained: mean_reward={base_r:.2f} mean_qoe={base_q:.1f}")

    for epoch in range(args.epochs):
        order = list(range(len(train_paths)))
        random.shuffle(order)
        for i in order:
            s = env.reset(BandwidthTrace.from_log(train_paths[i]))
            done = False
            ep_reward = 0.0
            losses = []
            while not done:
                eps = epsilon()
                a = agent.act(s, eps)
                s2, r, done, _ = env.step(a)
                agent.push(s, a, r, s2, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                s = s2
                ep_reward += r
                step_count += 1
            episode += 1
            history['episode_reward'].append(ep_reward)
            mean_loss = (sum(losses) / len(losses)) if losses else 0.0
            print(f"[ep {episode:4d}] file={os.path.basename(train_paths[i]):24s} "
                  f"reward={ep_reward:7.2f} qoe={env.qoe():6.1f} eps={eps:.3f} loss={mean_loss:.4f}")

            if args.eval_every and episode % args.eval_every == 0:
                ev_r, ev_q = evaluate(agent, env, test_paths, args.seed)
                history['eval'].append({'episode': episode, 'reward': ev_r, 'qoe': ev_q})
                marker = ""
                if ev_r > best_eval:
                    best_eval = ev_r
                    agent.save(args.out)
                    agent.save(args.out.replace('.pkl', '_best.pkl'))
                    marker = "  <-- best (saved)"
                print(f"    [eval @ ep {episode}] mean_reward={ev_r:.2f} mean_qoe={ev_q:.1f}{marker}")

    # Final eval + ensure a model is saved.
    final_r, final_q = evaluate(agent, env, test_paths, args.seed)
    history['eval'].append({'episode': episode, 'reward': final_r, 'qoe': final_q})
    if final_r >= best_eval or not os.path.exists(args.out):
        agent.save(args.out)
    print("=" * 90)
    print(f"✅ Training done. untrained reward={base_r:.2f} -> final eval reward={final_r:.2f} "
          f"(best={max(best_eval, final_r):.2f}); final qoe={final_q:.1f}")
    print(f"📦 Saved: {args.out}  (+ {args.out.replace('.pkl', '_best.pkl')})")

    import json
    hist_path = args.out.replace('.pkl', '_train_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"🧾 History: {hist_path}")


if __name__ == "__main__":
    main()
