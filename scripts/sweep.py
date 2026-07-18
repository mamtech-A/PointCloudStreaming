#!/usr/bin/env python3
"""DQN reward/hyperparameter sweep harness.

Each trial is one `train_dqn.py` subprocess with its own --out/--run-dir; its
stdout goes to <trial>/train.log and its result is <trial>/train_summary.json.
Trials are ranked by the validation selection metric (the compatibility name
``qoe_quality`` equals raw episodic reward/QoE), averaged over --seeds. The
winner's checkpoint is copied to models/abr_dqn.pkl and the full
ranking + the quality-vs-stall Pareto front go to models/dqn_sweep_results.json.

Resume: a trial whose train_summary.json already exists is skipped, so an
interrupted overnight sweep continues where it stopped.

Config: JSON (see configs/training.json, key "dqn_sweep"):
  {
    "mode": "grid" | "random" | "stratified_random",
    "budget": 24,                       // random mode: number of sampled configs
    "seeds": [42],                      // each config runs once per seed
    "base_args": {"epochs": 8, "coverage-stride": 60},
    "axes": {
      "mu": [2.0, 4.3, 8.0],
      "lam": [0.5, 1.0],
      "lr": [1e-4, 5e-4],
      "no-lstm-pred": [false, true],    // the LSTM-feature ablation axis
      "reward-spec": [{"rebuffer_weight": 2.0, "startup_weight": 1.0}]
    }
  }

Usage:
    python sweep.py --config configs/training.json --jobs 2
    python sweep.py --config configs/training.json --smoke   # tiny sanity run
"""

import os
import sys
import json
import time
import shutil
import argparse
import itertools
import subprocess
import random as _random
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Flags that are plain store_true on train_dqn.py (False => omit).
STORE_TRUE_FLAGS = {'no-lstm-pred'}

SMOKE_BASE = {
    'epochs': 1, 'max-frames': 30, 'coverage-stride': 2000, 'eval-every': 3,
    'max-validation-files': 1, 'eval-offsets': 1, 'eval-seeds': '42',
    'eval-sequences': 'longdress,loot,redandblack,soldier',
}

_FILE_DIGEST_CACHE = {}


def file_digest(path):
    path = os.path.abspath(path)
    cached = _FILE_DIGEST_CACHE.get(path)
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    value = digest.hexdigest()[:16]
    _FILE_DIGEST_CACHE[path] = value
    return value


def training_code_digest():
    paths = [os.path.join(project_root, 'scripts', 'train_dqn.py')]
    for root, _dirs, files in os.walk(os.path.join(project_root, 'src')):
        paths.extend(os.path.join(root, name) for name in files if name.endswith('.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.relpath(path, project_root).encode('utf-8'))
        with open(path, 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()[:16]


def argv_value(argv, flag):
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def trial_provenance(argv, config_digest, code_digest):
    files = {}
    for flag in ('--protocol', '--lstm'):
        value = argv_value(argv, flag)
        if not value:
            continue
        path = value if os.path.isabs(value) else os.path.join(project_root, value)
        files[flag] = {'path': os.path.relpath(path, project_root),
                       'sha256': file_digest(path)}
        if flag == '--lstm':
            metadata = os.path.splitext(path)[0] + '_split.json'
            files['--lstm-metadata'] = {
                'path': os.path.relpath(metadata, project_root),
                'sha256': file_digest(metadata),
            }
    return {
        'experiment_config_digest': config_digest,
        'training_code_digest': code_digest,
        'input_files': files,
    }


def expand_trials(cfg, smoke=False):
    axes = cfg.get('axes', {})
    names = sorted(axes)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*(axes[n] for n in names))]
    mode = cfg.get('mode', 'grid')
    if mode == 'random':
        rng = _random.Random(cfg.get('sample_seed', 0))
        budget = int(cfg.get('budget', len(combos)))
        combos = rng.sample(combos, min(budget, len(combos)))
    elif mode == 'stratified_random':
        rng = _random.Random(cfg.get('sample_seed', 0))
        key = cfg.get('stratify_by', 'segment-frames')
        if key not in axes:
            raise ValueError(f"stratify_by {key!r} is not a search axis")
        budget = int(cfg.get('budget_per_value', 1))
        # Draw the non-stratification hyperparameters once and pair that same
        # subset with every segment size.  Segment comparisons therefore do
        # not depend on one size receiving a luckier random sub-grid.
        other_names = [name for name in names if name != key]
        base_combos = [
            dict(zip(other_names, values))
            for values in itertools.product(*(axes[name] for name in other_names))
        ]
        required = [dict(combo) for combo in cfg.get('required_configs', [])]
        for combo in required:
            if set(combo) != set(other_names):
                raise ValueError(
                    f"required config keys {sorted(combo)} do not match "
                    f"non-stratified axes {other_names}"
                )
            if any(combo[name] not in axes[name] for name in other_names):
                raise ValueError(f"required config is outside the declared axes: {combo}")
        if len(required) > budget:
            raise ValueError("required_configs exceeds budget_per_value")
        remaining = [combo for combo in base_combos if combo not in required]
        draw_count = min(budget - len(required), len(remaining))
        sampled_base = None
        for _ in range(10000):
            chosen = sorted(rng.sample(range(len(remaining)), draw_count))
            candidate = required + [remaining[index] for index in chosen]
            if (not cfg.get('require_axis_coverage', False) or all(
                    set(combo[name] for combo in candidate) == set(axes[name])
                    for name in other_names)):
                sampled_base = candidate
                break
        if sampled_base is None:
            raise ValueError(
                f"cannot cover every declared axis value with budget_per_value={budget}"
            )
        sampled = []
        for value in axes[key]:
            sampled.extend({**combo, key: value} for combo in sampled_base)
        combos = sampled
    elif mode != 'grid':
        raise ValueError(f"unknown sweep mode {mode!r}")
    if smoke:
        combos = combos[:2]
    return combos


def aggregate_trials(combos, results, indices, select_by, metric_seeds=None):
    """Aggregate per-seed validation summaries for selected config indices."""
    table = []
    for idx in indices:
        combo = combos[idx]
        all_per_seed = results.get(idx, {})
        per_seed = (
            {seed: all_per_seed[seed] for seed in metric_seeds if seed in all_per_seed}
            if metric_seeds is not None else all_per_seed
        )
        if not per_seed:
            continue
        bests = [s['best'] or s['final'] for s in per_seed.values()]
        for best in bests:
            for metric in ('reward', 'qoe', 'qoe_quality', 'mean_quality', 'stall_s'):
                if not math.isfinite(float(best[metric])):
                    raise ValueError(
                        f"non-finite {metric} in trial {idx}: {best[metric]}"
                    )
        agg = {k: sum(b[k] for b in bests) / len(bests)
               for k in ('reward', 'qoe', 'qoe_quality', 'mean_quality', 'stall_s')}
        steps = [int(summary.get('steps_run', 0)) for summary in per_seed.values()]
        episodes = [int(summary.get('episodes_run', 0)) for summary in per_seed.values()]
        agg['steps_run_mean'] = sum(steps) / len(steps)
        agg['episodes_run_mean'] = sum(episodes) / len(episodes)
        agg['steps_run_per_seed'] = {
            seed: int(per_seed[seed].get('steps_run', 0)) for seed in sorted(per_seed)
        }
        sel_vals = [b[select_by] for b in bests]
        n = len(sel_vals)
        if n >= 2:
            mean = sum(sel_vals) / n
            agg[f'{select_by}_std'] = (
                sum((v - mean) ** 2 for v in sel_vals) / n
            ) ** 0.5
        else:
            agg[f'{select_by}_std'] = 0.0
        agg[f'{select_by}_per_seed'] = {
            seed: (per_seed[seed]['best'] or per_seed[seed]['final'])[select_by]
            for seed in sorted(per_seed)
        }
        trace_keys = set().union(*(b.get('per_trace', {}).keys() for b in bests))
        if trace_keys:
            agg[f'{select_by}_per_trace'] = {
                trace: (lambda values: sum(values) / len(values))(
                    [b['per_trace'][trace][select_by] for b in bests
                     if trace in b.get('per_trace', {})]
                )
                for trace in sorted(trace_keys)
            }
        table.append({
            'trial': idx,
            'config': combo,
            'seeds': sorted(per_seed),
            'metrics': agg,
            'checkpoints': {
                seed: os.path.relpath(all_per_seed[seed]['out'], project_root)
                for seed in all_per_seed
            },
        })
    table.sort(key=lambda row: row['metrics'][select_by], reverse=True)
    return table


def confirmation_trials(screening_table, cfg):
    """Choose the validation-best candidates to receive the full seed set."""
    per_value = int(cfg.get('confirm_top_per_value', 0))
    if per_value:
        key = cfg.get('stratify_by', 'segment-frames')
        selected = []
        values = cfg.get('axes', {}).get(key, [])
        for value in values:
            rows = [row for row in screening_table if row['config'].get(key) == value]
            selected.extend(row['trial'] for row in rows[:per_value])
        return selected
    top_k = int(cfg.get('confirm_top_k', 0))
    return [row['trial'] for row in screening_table[:top_k]]


def trial_args(base_args, combo, seed):
    """Merge base + axis values into a train_dqn.py CLI argument list."""
    merged = dict(base_args or {})
    merged.update(combo)
    merged['seed'] = seed
    argv = []
    for k, v in sorted(merged.items()):
        if k.startswith('_'):
            continue  # config-file comment key (e.g. "_epochs"), not a CLI flag
        flag = k.replace('_', '-')
        if isinstance(v, bool):
            if flag in STORE_TRUE_FLAGS:
                if v:
                    argv.append(f"--{flag}")
            else:  # BooleanOptionalAction pair
                argv.append(f"--{flag}" if v else f"--no-{flag}")
        elif isinstance(v, dict):
            argv += [f"--{flag}", json.dumps(v)]
        else:
            argv += [f"--{flag}", str(v)]
    return argv


def run_trial(idx, seed, argv, sweep_dir, provenance):
    tdir = os.path.join(sweep_dir, f"trial_{idx:03d}_s{seed}")
    os.makedirs(tdir, exist_ok=True)
    summary_path = os.path.join(tdir, 'train_summary.json')
    request_path = os.path.join(tdir, 'trial_request.json')
    request = {
        'trial': idx, 'seed': seed, 'argv': argv, 'provenance': provenance,
    }
    if os.path.exists(summary_path):
        if not os.path.exists(request_path):
            return idx, seed, None, 'FAILED (resume metadata missing)'
        with open(request_path, encoding='utf-8') as f:
            saved_request = json.load(f)
        if saved_request != request:
            return idx, seed, None, 'FAILED (resume command/config mismatch)'
        with open(summary_path, encoding='utf-8') as f:
            summary = json.load(f)
        if os.path.exists(summary.get('out', '')):
            return idx, seed, summary, 'resumed'
    with open(request_path, 'w', encoding='utf-8') as f:
        json.dump(request, f, indent=2)
    cmd = [sys.executable, '-u', os.path.join(project_root, 'scripts', 'train_dqn.py'),
           '--out', os.path.join(tdir, 'abr_dqn.pkl'), '--run-dir', tdir] + argv
    with open(os.path.join(tdir, 'train.log'), 'w', encoding='utf-8') as log:
        log.write('# ' + ' '.join(cmd) + '\n')
        log.flush()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=project_root)
    if rc != 0 or not os.path.exists(summary_path):
        return idx, seed, None, f'FAILED (rc={rc}, see {tdir}/train.log)'
    with open(summary_path, encoding='utf-8') as f:
        return idx, seed, json.load(f), 'ok'


def pareto_front(points):
    """Indices of trials not dominated on (maximize mean_quality, minimize stall_s)."""
    front = []
    for i, (q_i, s_i) in enumerate(points):
        dominated = any((q_j >= q_i and s_j <= s_i and (q_j > q_i or s_j < s_i))
                        for j, (q_j, s_j) in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front


def main():
    p = argparse.ArgumentParser(description="DQN reward/hyperparameter sweep")
    p.add_argument('--config', default=os.path.join('configs', 'training.json'))
    p.add_argument('--jobs', type=int, default=1, help='parallel trials (CPU-bound; '
                   'keep <= physical cores / 2)')
    p.add_argument('--sweep-dir', default=None,
                   help='trial output root (default: logs/train_runs/<ts>/sweep)')
    p.add_argument('--out', default=os.path.join('models', 'abr_dqn.pkl'),
                   help='where the WINNING checkpoint is copied')
    p.add_argument('--results', default=os.path.join('models', 'dqn_sweep_results.json'))
    p.add_argument('--lstm-template', default='',
                   help='optional checkpoint template containing {segment_frames}; '
                        'passed explicitly to every DQN trial')
    p.add_argument('--smoke', action='store_true', help='2 trials, tiny episodes')
    args = p.parse_args()

    config_path = (args.config if os.path.isabs(args.config)
                   else os.path.join(project_root, args.config))
    with open(config_path, 'rb') as f:
        config_bytes = f.read()
    cfg = json.loads(config_bytes.decode('utf-8')).get('dqn_sweep', {})
    config_digest = hashlib.sha256(config_bytes).hexdigest()[:16]
    code_digest = training_code_digest()
    base_args = dict(cfg.get('base_args', {}))
    screening_base_args = dict(base_args)
    screening_base_args.update(cfg.get('screen_overrides', {}))
    if args.smoke:
        base_args.update(SMOKE_BASE)
        screening_base_args = dict(base_args)
    screen_seeds = list(cfg.get('screen_seeds', cfg.get('seeds', [42])))
    confirm_seeds = list(cfg.get('confirm_seeds', screen_seeds))
    confirmation_selection_seeds = list(cfg.get(
        'confirmation_selection_seeds',
        [seed for seed in confirm_seeds if seed not in screen_seeds],
    ))
    if args.smoke:
        screen_seeds = screen_seeds[:1]
        confirm_seeds = screen_seeds
        confirmation_selection_seeds = screen_seeds
    if not set(confirmation_selection_seeds).issubset(confirm_seeds):
        raise ValueError("confirmation_selection_seeds must be a subset of confirm_seeds")
    if not args.smoke and not confirmation_selection_seeds:
        raise ValueError("confirmation_selection_seeds cannot be empty")
    select_by = base_args.get('select-by', 'qoe_quality')

    sweep_dir = args.sweep_dir or os.path.join(
        project_root, 'logs', 'train_runs',
        time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '_sweep', 'sweep')
    os.makedirs(sweep_dir, exist_ok=True)

    combos = expand_trials(cfg, smoke=args.smoke)
    screening_results = {}   # idx -> {seed: summary}
    confirmation_results = {}
    def execute(indices, seeds, label, stage_base_args, stage_dir, result_store):
        jobs = []
        for idx in indices:
            combo = combos[idx]
            for seed in seeds:
                argv = trial_args(stage_base_args, combo, seed)
                if args.lstm_template:
                    segment_frames = combo.get(
                        'segment-frames', stage_base_args.get('segment-frames'))
                    if segment_frames is None:
                        raise ValueError(
                            '--lstm-template requires segment-frames in base_args or axes')
                    argv += [
                        '--lstm',
                        args.lstm_template.format(segment_frames=segment_frames),
                    ]
                provenance = trial_provenance(
                    argv, config_digest=config_digest, code_digest=code_digest)
                jobs.append((idx, seed, argv, combo, provenance))
        print(f"{label}: {len(indices)} configs x {len(seeds)} seeds = "
              f"{len(jobs)} trial requests | jobs={args.jobs}")
        failures = []
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = {
                ex.submit(run_trial, idx, seed, argv, stage_dir, provenance):
                    (idx, seed, combo)
                for idx, seed, argv, combo, provenance in jobs
            }
            for fut in as_completed(futs):
                idx, seed, combo = futs[fut]
                _, _, summary, status = fut.result()
                print(f"  [trial {idx:03d} seed {seed}] {status}  {combo}")
                if summary is None:
                    failures.append((idx, seed, status))
                else:
                    result_store.setdefault(idx, {})[seed] = summary
        if failures:
            raise RuntimeError(f"{label} has failed trials: {failures}")

    print(f"sweep: {len(combos)} screened configs | jobs={args.jobs} | "
          f"select-by={select_by}\n  dir: {sweep_dir}")
    all_indices = list(range(len(combos)))
    execute(
        all_indices, screen_seeds, 'SCREEN', screening_base_args,
        os.path.join(sweep_dir, 'screen'), screening_results,
    )

    screening_table = aggregate_trials(
        combos, screening_results, all_indices, select_by)
    if not screening_table:
        sys.exit("no successful screening trials")

    # Smoke also confirms its one available finalist, exercising the separate
    # high-fidelity directory/provenance path with tiny one-epoch arguments.
    confirmed = confirmation_trials(screening_table, cfg)
    if confirmed:
        print(f"CONFIRM selected config trials: {confirmed}")
        execute(
            confirmed, confirm_seeds, 'CONFIRM', base_args,
            os.path.join(sweep_dir, 'confirm'), confirmation_results,
        )
        table = aggregate_trials(
            combos, confirmation_results, confirmed, select_by,
            metric_seeds=confirmation_selection_seeds,
        )
        final_seeds = confirmation_selection_seeds
        final_results = confirmation_results
    else:
        table = screening_table
        final_seeds = screen_seeds
        final_results = screening_results
    if not table:
        sys.exit("no successful trials - check the trial train.log files")

    table.sort(key=lambda t: t['metrics'][select_by], reverse=True)
    pts = [(t['metrics']['mean_quality'], t['metrics']['stall_s']) for t in table]
    front = [table[i]['trial'] for i in pareto_front(pts)]

    winner = table[0]
    # Install the representative (closest-to-mean) seed, not the lucky best
    # seed. All winner-config seeds are evaluated once on final test later.
    winner_mean = winner['metrics'][select_by]
    representative_seed = min(
        winner['seeds'],
        key=lambda s: (
            abs((final_results[winner['trial']][s]['best'] or
                 final_results[winner['trial']][s]['final'])[select_by] - winner_mean),
            int(s),
        ),
    )
    src = final_results[winner['trial']][representative_seed]['out']
    os.makedirs(os.path.dirname(os.path.join(project_root, args.out)), exist_ok=True)
    shutil.copyfile(src, os.path.join(project_root, args.out))
    best_src = src.replace('.pkl', '_best.pkl')
    if os.path.exists(best_src):
        shutil.copyfile(best_src,
                        os.path.join(project_root, args.out.replace('.pkl', '_best.pkl')))

    payload = {
        'objective_version': final_results[winner['trial']][representative_seed].get(
            'objective_version'),
        'protocol_digest': final_results[winner['trial']][representative_seed].get(
            'protocol_digest'),
        'experiment_config_digest': config_digest,
        'training_code_digest': code_digest,
        'select_by': select_by, 'selection_split': 'validation',
        'search_mode': cfg.get('mode', 'grid'),
        'seeds': final_seeds,
        'screen_seeds': screen_seeds,
        'confirm_seeds': confirm_seeds if confirmed else [],
        'confirmation_selection_seeds': (
            confirmation_selection_seeds if confirmed else []),
        'base_args': base_args,
        'screening_base_args': screening_base_args,
        'lstm_template': args.lstm_template or None,
        'sweep_dir': os.path.relpath(sweep_dir, project_root),
        'n_configs': len(combos),
        'screening_ranking': screening_table,
        'confirmed_config_trials': confirmed,
        'ranking': table,
        'pareto_front_trials': front,
        'winner': {'trial': winner['trial'], 'config': winner['config'],
                   'seed': representative_seed, 'seed_role': 'closest_to_validation_mean',
                   'seeds': winner['seeds'],
                   'checkpoint_seeds': sorted(final_results[winner['trial']]),
                   'metrics': winner['metrics'],
                   'checkpoint': os.path.relpath(
                       os.path.abspath(args.out), project_root)},
    }
    with open(os.path.join(project_root, args.results), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    std_key = f'{select_by}_std'
    print("\n" + "=" * 96)
    print(f"{'rank':<5}{'trial':<7}{select_by:>12}{'+/-std':>8}{'mean_q':>9}{'stall_s':>9}  config")
    for rank, t in enumerate(table[:15], 1):
        m = t['metrics']
        star = ' *pareto' if t['trial'] in front else ''
        print(f"{rank:<5}{t['trial']:<7}{m[select_by]:>12.2f}{m.get(std_key, 0.0):>8.2f}"
              f"{m['mean_quality']:>9.3f}{m['stall_s']:>9.1f}  {t['config']}{star}")
    wm = winner['metrics']
    print(f"\nWINNER trial {winner['trial']} -> {args.out} "
          f"(representative seed {representative_seed})")
    print(f"  {select_by} = {wm[select_by]:.2f} +/- {wm.get(std_key, 0.0):.2f} "
          f"over {len(winner['seeds'])} seeds {winner['seeds']}")
    print(f"  per-seed: {wm.get(f'{select_by}_per_seed', {})}")
    print(f"results: {args.results}")


if __name__ == '__main__':
    main()
