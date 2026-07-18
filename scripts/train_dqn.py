#!/usr/bin/env python3
"""Train the DQN ABR agent against the streaming environment.

Trains on the registered training traces and uses only registered validation
traces for checkpoint selection. The final test split is never evaluated here.
The LSTM remains an optional feature provider (drop it with --no-lstm-pred for
the preregistered ablation).

Content: every manifest matching --mpd (default manifests/mpd_gpcc*.xml) forms the
training pool; each episode draws a (sequence, trace, offset) triple. Validation
uses deterministic offsets registered in configs/experiment_protocol.json.

Coverage: each epoch TILES every training trace end-to-end with windows every
--coverage-stride samples (episode count per file proportional to its length),
instead of one arbitrary window per file per epoch.

NOTE: each env step is a full TCP transfer of a (large) point-cloud frame, so
training is compute-heavy. The defaults are modest; scale up with the flags
below for a real run. The script prints exactly how much it covers.

Usage:
    python train_dqn.py                          # default modest run
    python train_dqn.py --epochs 8               # registered training cap
    python train_dqn.py --epochs 1 --max-train-files 1 --max-frames 40 --eval-every 2 --coverage-stride 1500   # smoke
"""

import os
import sys
import glob
import json
import argparse
import random

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model import DEFAULT_TCP_PARAMS
from src.network_model.manifest import parse_mpd_xml
from src.network_model.trace import BandwidthTrace
from src.lstm_model import LSTMPredictor
from src.experiment_protocol import (
    evaluation_settings, load_protocol, protocol_digest, split_paths,
)
from src.evaluation import evaluate_agent
from src.rl.env import StreamingEnv
from src.rl.dqn import DQNAgent
from src.rl.features import DEFAULT_FEATURE_SPEC

TCP_PARAMS = dict(DEFAULT_TCP_PARAMS)

# Keep at least this many trace samples ahead of an episode's start offset
# (~>=2 min of wall-clock: plenty for a 300-frame episode even with stalls).
MIN_TAIL_SAMPLES = 120


def sequence_name(mpd_path):
    """manifests/mpd_gpcc.xml -> 'longdress'; manifests/mpd_gpcc_<seq>.xml -> '<seq>'."""
    stem = os.path.splitext(os.path.basename(mpd_path))[0]
    if stem == 'mpd_gpcc':
        return 'longdress'
    return stem[len('mpd_gpcc_'):] if stem.startswith('mpd_gpcc_') else stem


def load_manifest_pool(mpd_arg, max_frames=0):
    """Load every manifest matching the --mpd glob into {sequence: frames}."""
    paths = sorted(glob.glob(mpd_arg)) if any(c in mpd_arg for c in '*?[') else [mpd_arg]
    if not paths:
        raise FileNotFoundError(f"no manifest matches --mpd {mpd_arg}")
    pool = {}
    for p in paths:
        frames = parse_mpd_xml(p)
        if max_frames:
            frames = frames[:max_frames]
        pool[sequence_name(p)] = frames
    return pool


def build_epoch_episodes(train_paths, traces, stride, rng, random_phase):
    """Coverage-tiled (trace_path, offset) windows: every train trace is tiled
    end-to-end each epoch, so episodes-per-file is proportional to file length
    (the long static traces stop being under-used). A per-epoch random phase
    decorrelates window boundaries across epochs."""
    episodes = []
    for path in train_paths:
        n = len(traces[path])
        k = max(1, int(round(n / float(stride))))
        phase = rng.randrange(stride) if (random_phase and stride > 1) else 0
        max_off = max(0, n - MIN_TAIL_SAMPLES)
        for j in range(k):
            episodes.append((path, min(j * stride + phase, max_off)))
    rng.shuffle(episodes)
    return episodes


def evaluate(agent, env, trace_paths, eval_seeds, sequences, offsets_per_trace=1):
    """Greedy case-level evaluation on validation traces only."""
    if isinstance(sequences, str):
        sequences = [sequences]
    return evaluate_agent(
        agent, env, trace_paths, eval_seeds, sequences,
        offsets_per_trace=offsets_per_trace,
        min_tail_samples=MIN_TAIL_SAMPLES,
    )


def main():
    p = argparse.ArgumentParser(description="Train DQN ABR agent")
    p.add_argument('--protocol', default=os.path.join('configs', 'experiment_protocol.json'),
                   help='registered explicit train/validation/test protocol')
    p.add_argument('--epochs', type=int, default=8,
                   help='maximum passes over the coverage-tiled training set')
    p.add_argument('--mpd', type=str, default=os.path.join('manifests', 'mpd_gpcc*.xml'),
                   help='manifest path or glob; every match is one content sequence')
    p.add_argument('--eval-sequences', type=str, default='',
                   help='comma-separated validation contents; default comes from protocol')
    p.add_argument('--eval-offsets', type=int, default=0,
                   help='deterministic offsets per validation trace; 0 = protocol value')
    p.add_argument('--max-train-files', type=int, default=0, help='0 = all train files')
    p.add_argument('--max-validation-files', type=int, default=0,
                   help='0 = all validation files (smoke-test convenience only)')
    p.add_argument('--coverage-stride', type=int, default=60,
                   help='samples between episode start offsets when tiling each '
                        'training trace (episodes per file ~= file length / stride)')
    p.add_argument('--random-offset', action=argparse.BooleanOptionalAction, default=True,
                   help='randomize the per-epoch training tiling phase; validation '
                        'offsets remain deterministic')
    p.add_argument('--reward-scale', type=float, default=1.0,
                   help='LEGACY fixed learner-side reward scale (argmax-invariant). '
                        'Superseded by --reward-norm; kept for reproducing old runs')
    p.add_argument('--reward-norm', action=argparse.BooleanOptionalAction, default=True,
                   help='normalize learner-side rewards by a running std '
                        '(argmax-invariant; replaces the hand-tuned --reward-scale 0.1)')
    p.add_argument('--reward-spec', type=str, default=None,
                   help='JSON dict overriding the reward spec (see src/rl/reward.py), '
                        'e.g. \'{"stall_mode":"bounded","stall_cap_s":2.0,"mu":6}\'')
    p.add_argument('--no-lstm-pred', action='store_true',
                   help='ABLATION: drop the lstm_pred feature from the state')
    p.add_argument('--segment-frames', type=int, default=10,
                   help='frames per DASH-style segment (one decision + one transfer); '
                        '1 = legacy per-frame fetching')
    p.add_argument('--playback-rate-min', type=float, default=1.0,
                   help='adaptive-playback floor (1.0 = off; 0.9 = research-backed '
                        'imperceptible slowdown instead of stalling)')
    p.add_argument('--eval-seeds', type=str, default='',
                   help='comma list of jitter seeds averaged per eval (default: just '
                        '--seed); e.g. 42,43,44 de-noises checkpoint selection')
    p.add_argument('--max-frames', type=int, default=0, help='0 = all manifest frames')
    p.add_argument('--eval-every', type=int, default=100, help='episodes between validations')
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='validation checks without improvement before stopping; 0 = disabled')
    p.add_argument('--select-by', choices=['qoe_quality', 'reward'], default='qoe_quality',
                   help='validation metric that picks the best checkpoint (qoe_quality is '
                        'comparable across reward specs; reward is not)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--mu', type=float, default=4.3, help='rebuffer penalty weight')
    p.add_argument('--lam', type=float, default=1.0, help='quality-switch penalty weight')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--target-update', type=int, default=500,
                   help='learn-steps between target-network syncs (larger = more stable)')
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-frac', type=float, default=0.6)
    p.add_argument('--out', type=str, default=os.path.join('models', 'abr_dqn.pkl'))
    p.add_argument('--run-dir', type=str, default=None,
                   help='directory for train_summary.json / history / episodes.csv '
                        '(default: the --out directory)')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.run_dir or (os.path.dirname(args.out) or '.')
    os.makedirs(run_dir, exist_ok=True)

    bandwidth_dir = os.path.join(project_root, 'bandwidth_5g')
    lstm_path = os.path.join(project_root, 'models', 'bandwidth_lstm.pkl')

    protocol_path = (args.protocol if os.path.isabs(args.protocol)
                     else os.path.join(project_root, args.protocol))
    protocol = load_protocol(protocol_path, bandwidth_dir)
    protocol_id = protocol_digest(protocol)
    train_files = list(protocol['trace_split']['train'])
    validation_files = list(protocol['trace_split']['validation'])
    registered_test_files = list(protocol['trace_split']['test'])
    if args.max_train_files:
        train_files = train_files[:args.max_train_files]
    if args.max_validation_files:
        validation_files = validation_files[:args.max_validation_files]
    train_paths = [os.path.join(bandwidth_dir, f) for f in train_files]
    validation_paths = [os.path.join(bandwidth_dir, f) for f in validation_files]

    validation_cfg = evaluation_settings(protocol, 'validation')
    eval_sequences = ([s.strip() for s in args.eval_sequences.split(',') if s.strip()]
                      if args.eval_sequences else list(validation_cfg['sequences']))
    eval_offsets = args.eval_offsets or int(validation_cfg['offsets_per_trace'])

    manifest_pool = load_manifest_pool(
        args.mpd if os.path.isabs(args.mpd) else os.path.join(project_root, args.mpd),
        args.max_frames)
    seq_names = sorted(manifest_pool)
    missing_eval_sequences = sorted(set(eval_sequences) - set(seq_names))
    if missing_eval_sequences:
        raise ValueError(
            f"validation sequences absent from manifest pool: {missing_eval_sequences}; "
            f"available={seq_names}"
        )
    mean_frames = int(np.mean([len(f) for f in manifest_pool.values()]))

    feature_spec = [f for f in DEFAULT_FEATURE_SPEC
                    if not (args.no_lstm_pred and f == 'lstm_pred')]
    predictor = None
    if 'lstm_pred' in feature_spec:
        predictor = LSTMPredictor().load(lstm_path)

    reward_spec = {'mu': args.mu, 'lam': args.lam}
    if args.reward_spec:
        reward_spec.update(json.loads(args.reward_spec))

    eval_seeds = ([int(s) for s in args.eval_seeds.split(',') if s.strip()]
                  if args.eval_seeds else list(validation_cfg['jitter_seeds']))

    env = StreamingEnv(manifest_pool, lstm_predictor=predictor, tcp_params=TCP_PARAMS,
                       feature_spec=feature_spec, reward_spec=reward_spec,
                       segment_frames=args.segment_frames,
                       playback_rate_min=args.playback_rate_min)
    agent = DQNAgent(env.state_dim, env.num_actions, env.feature_spec, env.norm,
                     hidden=args.hidden, lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size,
                     mu=env.mu, lam=env.lam, reward_norm=args.reward_norm,
                     target_update_freq=args.target_update,
                     lstm_model_path=(lstm_path if predictor is not None else None),
                     sequence_length=getattr(predictor, 'sequence_length', 10),
                     reward_spec=env.reward_fn.spec)

    # Cache traces once (timestamp axis preserved by slice_from per episode).
    traces = {path: BandwidthTrace.from_file(path) for path in train_paths}
    episodes_per_epoch = sum(max(1, int(round(len(traces[p]) / float(args.coverage_stride))))
                             for p in train_paths)
    total_episodes = args.epochs * episodes_per_epoch
    steps_per_episode = max(1, -(-mean_frames // max(1, args.segment_frames)))  # ceil
    total_steps_est = max(1, total_episodes * steps_per_episode)
    decay_steps = max(1, int(args.eps_decay_frac * total_steps_est))

    print("=" * 90)
    print("🤖 DQN ABR training")
    print(f"   sequences: {seq_names} | validation sequences: {eval_sequences}")
    print(f"   protocol: {protocol_id} | train files: {len(train_files)} "
          f"| validation files: {len(validation_files)} | registered test files: "
          f"{len(registered_test_files)} (NOT evaluated) "
          f"| frames/episode: {mean_frames}")
    print(f"   coverage stride: {args.coverage_stride} samples -> "
          f"{episodes_per_epoch} episodes/epoch x {args.epochs} epochs = {total_episodes} episodes "
          f"(~{total_steps_est} env steps)")
    print(f"   segment: {args.segment_frames} frames/request "
          f"({steps_per_episode} decisions/episode) | playback-rate-min: "
          f"{args.playback_rate_min} | validation offsets/trace: {eval_offsets} "
          f"| validation seeds: {eval_seeds}")
    print(f"   state_dim: {env.state_dim} | actions: {env.num_actions} | feature_spec: {feature_spec}")
    print(f"   reward: {env.reward_fn.describe()}  (gamma={args.gamma}, lr={args.lr}, "
          f"reward_norm={args.reward_norm}, reward_scale={args.reward_scale})")
    print(f"   select-by: {args.select_by} | out: {args.out} | run-dir: {run_dir}")
    print("=" * 90)

    step_count = 0
    best_metric = float('-inf')
    best_entry = None
    history = {'episode_reward': [], 'validation': []}
    episode = 0
    ep_rng = random.Random(args.seed + 1)

    ep_csv_path = os.path.join(run_dir, 'episodes.csv')
    ep_csv = open(ep_csv_path, 'w', encoding='utf-8')
    ep_csv.write('episode,sequence,file,offset,reward,qoe,qoe_quality,epsilon,mean_loss\n')

    def epsilon():
        frac = min(1.0, step_count / decay_steps)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    validations_without_improvement = 0

    def run_eval(tag):
        nonlocal best_metric, best_entry, validations_without_improvement
        ev = evaluate(
            agent, env, validation_paths, eval_seeds, eval_sequences,
            offsets_per_trace=eval_offsets,
        )
        entry = {'episode': episode, **ev}
        history['validation'].append(entry)
        marker = ""
        if ev[args.select_by] > best_metric:
            best_metric = ev[args.select_by]
            best_entry = entry
            validations_without_improvement = 0
            agent.save(args.out)
            agent.save(args.out.replace('.pkl', '_best.pkl'))
            marker = "  <-- best (saved)"
        else:
            validations_without_improvement += 1
        print(f"    [eval @ {tag}] reward={ev['reward']:.2f} qoe={ev['qoe']:.1f} "
              f"qoe_q={ev['qoe_quality']:.1f} mean_q={ev['mean_quality']:.3f} "
              f"stall={ev['stall_s']:.1f}s{marker}")
        return ev

    # Untrained validation is a diagnostic, never a test-set observation.
    base = evaluate(
        agent, env, validation_paths, eval_seeds, eval_sequences,
        offsets_per_trace=eval_offsets,
    )
    print(f"[validation @ ep 0] untrained: reward={base['reward']:.2f} qoe={base['qoe']:.1f} "
          f"qoe_q={base['qoe_quality']:.1f}")

    stopped_early = False
    for epoch in range(args.epochs):
        epoch_eps = build_epoch_episodes(train_paths, traces, args.coverage_stride,
                                         ep_rng, args.random_offset)
        for k, (path, off) in enumerate(epoch_eps):
            # Round-robin content over the epoch's shuffled windows (offset by
            # epoch so a (window % sequence) correlation can't persist).
            seq = seq_names[(epoch + k) % len(seq_names)]
            trace = traces[path].slice_from(off) if off else traces[path]
            s = env.reset(trace, sequence=seq)
            done = False
            ep_reward = 0.0
            losses = []
            while not done:
                eps = epsilon()
                a = agent.act(s, eps)
                s2, r, done, _ = env.step(a)
                agent.push(s, a, r * args.reward_scale, s2, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                s = s2
                ep_reward += r
                step_count += 1
            episode += 1
            history['episode_reward'].append(ep_reward)
            mean_loss = (sum(losses) / len(losses)) if losses else 0.0
            print(f"[ep {episode:4d}] {seq:<12s} {os.path.basename(path):>32s}@{off:<5d} "
                  f"reward={ep_reward:8.2f} qoe={env.qoe():6.1f} qoe_q={env.qoe_quality():7.1f} "
                  f"eps={eps:.3f} loss={mean_loss:.4f}")
            ep_csv.write(f"{episode},{seq},{os.path.basename(path)},{off},"
                         f"{ep_reward:.4f},{env.qoe():.2f},{env.qoe_quality():.2f},"
                         f"{eps:.4f},{mean_loss:.6f}\n")
            ep_csv.flush()

            if args.eval_every and episode % args.eval_every == 0:
                run_eval(f"ep {episode}")
                if (args.early_stop_patience and
                        validations_without_improvement >= args.early_stop_patience):
                    print(
                        f"    [early stop] no validation improvement for "
                        f"{validations_without_improvement} checks"
                    )
                    stopped_early = True
                    break
        if stopped_early:
            break

    # Save the unstable last iterate separately; args.out remains validation-best.
    last_out = args.out.replace('.pkl', '_last.pkl')
    agent.save(last_out)
    if history['validation'] and history['validation'][-1]['episode'] == episode:
        final = history['validation'][-1]
    else:
        final = run_eval("final")
    if not os.path.exists(args.out):
        agent.save(args.out)
    ep_csv.close()
    print("=" * 90)
    print(f"✅ Training done. untrained {args.select_by}={base[args.select_by]:.2f} -> "
          f"final {final[args.select_by]:.2f} (best={best_metric:.2f})")
    print(f"📦 Saved: {args.out}  (+ {args.out.replace('.pkl', '_best.pkl')})")

    hist_path = args.out.replace('.pkl', '_train_history.json')
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    summary = {
        'args': {k: v for k, v in vars(args).items()},
        'reward': env.reward_fn.describe(),
        'reward_spec': env.reward_fn.spec,
        'feature_spec': feature_spec,
        'sequences': seq_names,
        'validation_sequences': eval_sequences,
        'validation_offsets_per_trace': eval_offsets,
        'validation_seeds': eval_seeds,
        'protocol': os.path.relpath(protocol_path, project_root),
        'protocol_digest': protocol_id,
        'train_files': train_files,
        'validation_files': validation_files,
        'registered_test_files_not_evaluated': registered_test_files,
        'episodes_per_epoch': episodes_per_epoch,
        'episodes_run': episode,
        'stopped_early': stopped_early,
        'untrained': base,
        'final': final,
        'best': best_entry,
        'best_metric': best_metric,
        'select_by': args.select_by,
        'out': os.path.abspath(args.out),
        'last_out': os.path.abspath(last_out),
        'history': os.path.abspath(hist_path),
        'episodes_csv': os.path.abspath(ep_csv_path),
    }
    summary_path = os.path.join(run_dir, 'train_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"🧾 History: {hist_path}")
    print(f"🧾 Summary: {summary_path}")


if __name__ == "__main__":
    main()
