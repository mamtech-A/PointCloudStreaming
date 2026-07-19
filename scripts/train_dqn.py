#!/usr/bin/env python3
"""Train the DQN ABR agent against the streaming environment.

Trains on the registered training traces and uses only registered validation
traces for checkpoint selection. The final test split is never evaluated here.
The LSTM remains an optional feature provider (drop it with --no-lstm-pred for
the preregistered ablation).

Content: every manifest matching --mpd (default manifests/mpd_gpcc*.xml) forms
the training pool. Each episode draws a content sequence and one feasible,
gap-split window from the frozen trace registry.

Coverage: every parent trace contributes the same number of episodes per epoch.
Within a parent, only audited windows are shuffled and sampled. Validation uses
the registry's three frozen windows per parent and macro-averages parent means.

NOTE: each env step is a full TCP transfer of a (large) point-cloud frame, so
training is compute-heavy. The defaults are modest; scale up with the flags
below for a real run. The script prints exactly how much it covers.

Usage:
    python train_dqn.py                          # default modest run
    python train_dqn.py --epochs 8               # registered training cap
    python train_dqn.py --epochs 1 --max-train-files 1 --max-validation-files 1 --max-frames 40 --eval-offsets 1 --no-lstm-pred   # smoke
"""

import os
import sys
import glob
import json
import argparse
import random
import math

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
from src.lstm_model import LSTMPredictor
from src.experiment_protocol import (
    evaluation_settings, load_protocol, protocol_digest, split_paths,
)
from src.evaluation import evaluate_agent
from src.trace_registry import load_trace_registry
from src.rl.env import StreamingEnv
from src.rl.dqn import DQNAgent
from src.rl.features import DEFAULT_FEATURE_SPEC
from src.rl.reward import OBJECTIVE_VERSION

TCP_PARAMS = dict(DEFAULT_TCP_PARAMS)

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


def build_epoch_episodes(train_paths, registry, rng, randomize=True):
    """Sample every parent equally, then sample its eligible windows.

    The per-parent episode count is the ceiling of the mean eligible-window
    count. Windows are shuffled and cycled within a parent when necessary, so
    long or fragmented parents cannot dominate the training objective.
    """
    grouped = registry.windows_by_parent("train", train_paths)
    episodes_per_parent = int(math.ceil(
        sum(len(rows) for rows in grouped.values()) / float(len(grouped))
    ))
    path_by_name = {os.path.basename(path): path for path in train_paths}
    episodes = []
    for filename, rows in grouped.items():
        selected = []
        while len(selected) < episodes_per_parent:
            cycle = list(rows)
            if randomize:
                rng.shuffle(cycle)
            selected.extend(cycle)
        episodes.extend(
            (path_by_name[filename], window)
            for window in selected[:episodes_per_parent]
        )
    if randomize:
        rng.shuffle(episodes)
    return episodes, episodes_per_parent


def evaluate(agent, env, trace_paths, eval_seeds, sequences, offsets_per_trace,
             registry):
    """Greedy case-level evaluation on validation traces only."""
    if isinstance(sequences, str):
        sequences = [sequences]
    return evaluate_agent(
        agent, env, trace_paths, eval_seeds, sequences,
        offsets_per_trace=offsets_per_trace,
        window_registry=registry,
        split_name="validation",
    )


def resolve_lstm_path(requested, segment_frames):
    """Resolve the predictor for this decision cadence.

    New experiments install one predictor per segment size.  There is no
    canonical fallback: silently pairing another cadence is invalid.
    """
    if requested:
        return requested if os.path.isabs(requested) else os.path.join(project_root, requested)
    specific = os.path.join(
        project_root, 'models',
        f'bandwidth_lstm_request_pacing_s{int(segment_frames)}.pkl')
    return specific


def validate_lstm_provenance(lstm_path, segment_frames, protocol_id):
    """Reject a predictor known to have incompatible sampling/protocol data."""
    stem, _ = os.path.splitext(lstm_path)
    if stem.endswith('_best'):
        stem = stem[:-5]
    metadata_path = stem + '_split.json'
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"LSTM provenance metadata is required: {metadata_path}"
        )
    with open(metadata_path, encoding='utf-8') as f:
        metadata = json.load(f)
    trained_segment = metadata.get('segment_frames')
    if trained_segment is not None and int(trained_segment) != int(segment_frames):
        raise ValueError(
            f"LSTM segment mismatch: {lstm_path} was generated for "
            f"{trained_segment} frames, DQN requested {segment_frames}"
        )
    trained_protocol = metadata.get('protocol_digest')
    if trained_protocol and trained_protocol != protocol_id:
        raise ValueError(
            f"LSTM/protocol digest mismatch: {trained_protocol} != {protocol_id}"
        )


def main():
    p = argparse.ArgumentParser(description="Train DQN ABR agent")
    p.add_argument('--protocol', default=os.path.join('configs', 'experiment_protocol.json'),
                   help='registered explicit train/validation/test protocol')
    p.add_argument('--trace-registry', default=os.path.join(
        'configs', 'trace_window_registry.json'),
        help='frozen gap-split finite-window registry')
    p.add_argument('--epochs', type=int, default=8,
                   help='maximum passes over the parent-balanced window registry')
    p.add_argument('--mpd', type=str, default=os.path.join('manifests', 'mpd_gpcc*.xml'),
                   help='manifest path or glob; every match is one content sequence')
    p.add_argument('--eval-sequences', type=str, default='',
                   help='comma-separated validation contents; default comes from protocol')
    p.add_argument('--eval-offsets', type=int, default=0,
                   help='frozen windows per validation parent; 0 = protocol value')
    p.add_argument('--max-train-files', type=int, default=0, help='0 = all train files')
    p.add_argument('--max-validation-files', type=int, default=0,
                   help='0 = all validation files (smoke-test convenience only)')
    p.add_argument('--coverage-stride', type=int, default=60,
                   help='compatibility assertion: must equal the registry timestamp '
                        'grid stride (training windows are already frozen)')
    p.add_argument('--random-offset', action=argparse.BooleanOptionalAction, default=True,
                   help='shuffle eligible windows within each uniformly sampled parent')
    p.add_argument('--reward-spec', type=str, default=None,
                   help='JSON dict overriding the reward spec (see src/rl/reward.py), '
                        'e.g. \'{"mu":4.3,"rebuffer_weight":2.0}\'')
    p.add_argument('--no-lstm-pred', action='store_true',
                   help='ABLATION: drop the lstm_pred feature from the state')
    p.add_argument('--lstm', type=str, default='',
                   help='LSTM checkpoint. Empty selects '
                        'models/bandwidth_lstm_request_pacing_s<segment-frames>.pkl')
    p.add_argument('--segment-frames', type=int, default=10,
                   help='frames per DASH-style segment (one decision + one transfer); '
                        '1 = legacy per-frame fetching')
    p.add_argument('--eval-seeds', type=str, default='',
                   help='comma list of jitter seeds averaged per eval (default: just '
                        '--seed); e.g. 42,43,44 de-noises checkpoint selection')
    p.add_argument('--max-frames', type=int, default=0, help='0 = all manifest frames')
    p.add_argument('--eval-every', type=int, default=100, help='episodes between validations')
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='validation checks without improvement before stopping; 0 = disabled')
    p.add_argument('--select-by', choices=['qoe_quality', 'reward'], default='qoe_quality',
                   help='validation metric that picks the best checkpoint; both names '
                        'are identical under the canonical objective')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--torch-threads', type=int, default=1,
                   help='intra-op CPU threads for this trial; use 2 with four '
                        'parallel trials on an i5-13400')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=1.0,
                   help='discount factor; 1.0 makes the optimized return equal QoE')
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

    if abs(args.gamma - 1.0) > 1e-12:
        p.error("--gamma must be 1.0 so the DQN return equals the reported QoE")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    run_dir = args.run_dir or (os.path.dirname(args.out) or '.')
    os.makedirs(run_dir, exist_ok=True)

    bandwidth_dir = os.path.join(project_root, 'bandwidth_5g')

    protocol_path = (args.protocol if os.path.isabs(args.protocol)
                     else os.path.join(project_root, args.protocol))
    protocol = load_protocol(protocol_path, bandwidth_dir)
    protocol_id = protocol_digest(protocol)
    registry_path = (args.trace_registry if os.path.isabs(args.trace_registry)
                     else os.path.join(project_root, args.trace_registry))
    trace_registry = load_trace_registry(
        registry_path, protocol, bandwidth_dir, verify_hashes=True
    )
    if not math.isclose(
        float(args.coverage_stride), trace_registry.grid_stride_s, abs_tol=1e-9
    ):
        raise ValueError(
            f"--coverage-stride={args.coverage_stride} does not match frozen "
            f"registry grid {trace_registry.grid_stride_s:g}s"
        )
    lstm_path = resolve_lstm_path(args.lstm, args.segment_frames)
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
    trace_registry.validate_experiment(
        seq_names, args.segment_frames, args.max_frames
    )

    feature_spec = [f for f in DEFAULT_FEATURE_SPEC
                    if not (args.no_lstm_pred and f == 'lstm_pred')]
    predictor = None
    if 'lstm_pred' in feature_spec:
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(
                f"LSTM predictor not found for segment_frames={args.segment_frames}: "
                f"{lstm_path}. Run scripts/run_training.py through its LSTM stage first."
            )
        validate_lstm_provenance(lstm_path, args.segment_frames, protocol_id)
        predictor = LSTMPredictor().load(lstm_path)

    reward_spec = {'mu': args.mu, 'lam': args.lam}
    if args.reward_spec:
        reward_spec.update(json.loads(args.reward_spec))

    eval_seeds = ([int(s) for s in args.eval_seeds.split(',') if s.strip()]
                  if args.eval_seeds else list(validation_cfg['jitter_seeds']))

    env = StreamingEnv(manifest_pool, lstm_predictor=predictor, tcp_params=TCP_PARAMS,
                       feature_spec=feature_spec, reward_spec=reward_spec,
                       segment_frames=args.segment_frames)
    agent = DQNAgent(env.state_dim, env.num_actions, env.feature_spec, env.norm,
                     hidden=args.hidden, lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size,
                     mu=env.mu, lam=env.lam,
                     target_update_freq=args.target_update,
                     lstm_model_path=(lstm_path if predictor is not None else None),
                     sequence_length=getattr(predictor, 'sequence_length', 10),
                     reward_spec=env.reward_fn.spec)

    preview_episodes, episodes_per_parent = build_epoch_episodes(
        train_paths, trace_registry, random.Random(args.seed + 1),
        args.random_offset,
    )
    episodes_per_epoch = len(preview_episodes)
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
    print(f"   registry: {trace_registry.registry_id} | grid: "
          f"{trace_registry.grid_stride_s:g}s | {episodes_per_parent} episodes/parent -> "
          f"{episodes_per_epoch} episodes/epoch x {args.epochs} epochs = {total_episodes} episodes "
          f"(~{total_steps_est} env steps)")
    print(f"   segment: {args.segment_frames} frames/request "
          f"({steps_per_episode} decisions/episode) | validation offsets/trace: {eval_offsets} "
          f"| validation seeds: {eval_seeds}")
    print(f"   state_dim: {env.state_dim} | actions: {env.num_actions} | feature_spec: {feature_spec}")
    print(f"   PyTorch CPU threads/trial: {torch.get_num_threads()}")
    if predictor is not None:
        print(f"   LSTM: {os.path.relpath(lstm_path, project_root)}")
    print(f"   reward = QoE: {env.reward_fn.describe()}  "
          f"(gamma={args.gamma}, lr={args.lr}, no learner-side scaling)")
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
    ep_csv.write(
        'episode,sequence,file,window_id,block_id,start_time_s,reward,qoe,'
        'qoe_quality,request_pacing_s,epsilon,mean_loss\n'
    )

    def epsilon():
        frac = min(1.0, step_count / decay_steps)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    validations_without_improvement = 0

    def run_eval(tag):
        nonlocal best_metric, best_entry, validations_without_improvement
        ev = evaluate(
            agent, env, validation_paths, eval_seeds, eval_sequences,
            offsets_per_trace=eval_offsets,
            registry=trace_registry,
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
        print(f"    [eval @ {tag}] reward=QoE={ev['qoe']:.2f} "
              f"mean_q={ev['mean_quality']:.3f} "
              f"stall={ev['stall_s']:.1f}s{marker}")
        return ev

    # Untrained validation is a diagnostic, never a test-set observation.
    base = evaluate(
        agent, env, validation_paths, eval_seeds, eval_sequences,
        offsets_per_trace=eval_offsets,
        registry=trace_registry,
    )
    print(f"[validation @ ep 0] untrained: reward=QoE={base['qoe']:.2f}")

    stopped_early = False
    for epoch in range(args.epochs):
        epoch_eps, _ = build_epoch_episodes(
            train_paths, trace_registry, ep_rng, args.random_offset
        )
        for k, (path, window) in enumerate(epoch_eps):
            # Round-robin content over the epoch's shuffled windows (offset by
            # epoch so a (window % sequence) correlation can't persist).
            seq = seq_names[(epoch + k) % len(seq_names)]
            trace = trace_registry.materialize(window)
            s = env.reset(trace, sequence=seq)
            done = False
            ep_reward = 0.0
            losses = []
            request_pacing_s = 0.0
            while not done:
                eps = epsilon()
                a = agent.act(s, eps)
                s2, r, done, info = env.step(a)
                request_pacing_s += float(info.get('request_pacing_s', 0.0))
                agent.push(s, a, r, s2, done)
                loss = agent.learn()
                if loss is not None:
                    if not math.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite DQN loss at episode={episode + 1}, "
                            f"step={step_count}: {loss}"
                        )
                    losses.append(loss)
                s = s2
                ep_reward += r
                step_count += 1
            episode += 1
            history['episode_reward'].append(ep_reward)
            mean_loss = (sum(losses) / len(losses)) if losses else 0.0
            episode_qoe = env.qoe()
            if abs(ep_reward - episode_qoe) > 1e-8:
                raise AssertionError(
                    f"reward/QoE mismatch: return={ep_reward} qoe={episode_qoe}"
                )
            print(f"[ep {episode:4d}] {seq:<12s} {os.path.basename(path):>32s} "
                  f"{window['block_id']}@{window['start_time_s']:.0f}s "
                  f"reward=QoE={ep_reward:8.2f} "
                  f"pacing={request_pacing_s:.2f}s "
                  f"eps={eps:.3f} loss={mean_loss:.4f}")
            ep_csv.write(
                         f"{episode},{seq},{os.path.basename(path)},"
                         f"{window['id']},{window['block_id']},"
                         f"{window['start_time_s']:.3f},"
                         f"{ep_reward:.4f},{episode_qoe:.2f},{episode_qoe:.2f},"
                         f"{request_pacing_s:.4f},{eps:.4f},"
                         f"{mean_loss:.6f}\n")
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
        'objective_version': OBJECTIVE_VERSION,
        'args': {k: v for k, v in vars(args).items()},
        'reward': env.reward_fn.describe(),
        'reward_spec': env.reward_fn.spec,
        'feature_spec': feature_spec,
        'lstm_model': (os.path.abspath(lstm_path) if predictor is not None else None),
        'sequences': seq_names,
        'validation_sequences': eval_sequences,
        'validation_offsets_per_trace': eval_offsets,
        'validation_seeds': eval_seeds,
        'protocol': os.path.relpath(protocol_path, project_root),
        'protocol_digest': protocol_id,
        'trace_registry': os.path.relpath(registry_path, project_root),
        'trace_registry_id': trace_registry.registry_id,
        'train_files': train_files,
        'validation_files': validation_files,
        'registered_test_files_not_evaluated': registered_test_files,
        'episodes_per_epoch': episodes_per_epoch,
        'episodes_per_parent_per_epoch': episodes_per_parent,
        'episodes_run': episode,
        'steps_run': step_count,
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
