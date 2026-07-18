#!/usr/bin/env python3
"""DQN reward/hyperparameter sweep harness (grid or random search).

Each trial is one `train_dqn.py` subprocess with its own --out/--run-dir; its
stdout goes to <trial>/train.log and its result is <trial>/train_summary.json.
Trials are ranked by the validation selection metric (default qoe_quality — the
only metric comparable ACROSS reward specs; raw reward is not), averaged over
--seeds. The winner's checkpoint is copied to models/abr_dqn.pkl and the full
ranking + the quality-vs-stall Pareto front go to models/dqn_sweep_results.json.

Resume: a trial whose train_summary.json already exists is skipped, so an
interrupted overnight sweep continues where it stopped.

Config: JSON (see configs/training.json, key "dqn_sweep"):
  {
    "mode": "grid" | "random",
    "budget": 24,                       // random mode: number of sampled configs
    "seeds": [42],                      // each config runs once per seed
    "base_args": {"epochs": 8, "coverage-stride": 60},
    "axes": {
      "mu": [2.0, 4.3, 8.0],
      "lam": [0.5, 1.0],
      "lr": [1e-4, 5e-4],
      "no-lstm-pred": [false, true],    // the LSTM-feature ablation axis
      "reward-spec": [{}, {"stall_mode": "bounded", "stall_cap_s": 2.0}]
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
    'eval-sequences': 'longdress',
}


def expand_trials(cfg, smoke=False):
    axes = cfg.get('axes', {})
    names = sorted(axes)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*(axes[n] for n in names))]
    if cfg.get('mode', 'grid') == 'random':
        rng = _random.Random(cfg.get('sample_seed', 0))
        budget = int(cfg.get('budget', len(combos)))
        combos = rng.sample(combos, min(budget, len(combos)))
    if smoke:
        combos = combos[:2]
    return combos


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


def run_trial(idx, seed, argv, sweep_dir):
    tdir = os.path.join(sweep_dir, f"trial_{idx:03d}_s{seed}")
    os.makedirs(tdir, exist_ok=True)
    summary_path = os.path.join(tdir, 'train_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            return idx, seed, json.load(f), 'resumed'
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
    p.add_argument('--smoke', action='store_true', help='2 trials, tiny episodes')
    args = p.parse_args()

    with open(os.path.join(project_root, args.config), encoding='utf-8') as f:
        cfg = json.load(f).get('dqn_sweep', {})
    base_args = dict(cfg.get('base_args', {}))
    if args.smoke:
        base_args.update(SMOKE_BASE)
    seeds = cfg.get('seeds', [42])
    if args.smoke:
        seeds = seeds[:1]
    select_by = base_args.get('select-by', 'qoe_quality')

    sweep_dir = args.sweep_dir or os.path.join(
        project_root, 'logs', 'train_runs',
        time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '_sweep', 'sweep')
    os.makedirs(sweep_dir, exist_ok=True)

    combos = expand_trials(cfg, smoke=args.smoke)
    jobs = [(i, s, trial_args(base_args, combo, s), combo)
            for i, combo in enumerate(combos) for s in seeds]
    print(f"sweep: {len(combos)} configs x {len(seeds)} seed(s) = {len(jobs)} trials "
          f"| jobs={args.jobs} | select-by={select_by}\n  dir: {sweep_dir}")

    results = {}   # idx -> {seed: summary}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(run_trial, i, s, argv, sweep_dir): (i, s, combo)
                for i, s, argv, combo in jobs}
        for fut in as_completed(futs):
            i, s, combo = futs[fut]
            idx, seed, summary, status = fut.result()
            print(f"  [trial {idx:03d} seed {seed}] {status}  {combo}")
            if summary is not None:
                results.setdefault(idx, {})[seed] = summary

    # Aggregate per config (mean of the best-eval metric over seeds).
    table = []
    for idx, combo in enumerate(combos):
        per_seed = results.get(idx, {})
        if not per_seed:
            continue
        bests = [s['best'] or s['final'] for s in per_seed.values()]
        agg = {k: sum(b[k] for b in bests) / len(bests)
               for k in ('reward', 'qoe', 'qoe_quality', 'mean_quality', 'stall_s')}
        # Across-seed std of the SELECTION metric: turns the "seed 42 vs 43 spread"
        # caveat into a reported number (population std; n>=2 or 0.0).
        sel_vals = [b[select_by] for b in bests]
        n = len(sel_vals)
        if n >= 2:
            mean = sum(sel_vals) / n
            agg[f'{select_by}_std'] = (sum((v - mean) ** 2 for v in sel_vals) / n) ** 0.5
        else:
            agg[f'{select_by}_std'] = 0.0
        agg[f'{select_by}_per_seed'] = {s: (results[idx][s]['best'] or
                                            results[idx][s]['final'])[select_by]
                                        for s in sorted(per_seed)}
        # Per-TRACE mean of the selection metric across seeds (train_dqn's
        # evaluate() stores per_trace in each best entry). The validation set is
        # heterogeneous (3 driving incl. a 137s blackout + 1 static), so charts
        # need WHERE a config wins/loses, not just the aggregate. Missing key =
        # pre-round-4 summary (skipped quietly).
        trace_keys = set().union(*(b.get('per_trace', {}).keys() for b in bests))
        if trace_keys:
            agg[f'{select_by}_per_trace'] = {
                t: (lambda vs: sum(vs) / len(vs))(
                    [b['per_trace'][t][select_by] for b in bests
                     if t in b.get('per_trace', {})])
                for t in sorted(trace_keys)}
        table.append({
            'trial': idx, 'config': combo, 'seeds': sorted(per_seed),
            'metrics': agg,
            'checkpoints': {s: per_seed[s]['out'] for s in per_seed},
        })
    if not table:
        sys.exit("no successful trials — check the trial train.log files")

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
            abs((results[winner['trial']][s]['best'] or
                 results[winner['trial']][s]['final'])[select_by] - winner_mean),
            int(s),
        ),
    )
    src = results[winner['trial']][representative_seed]['out']
    os.makedirs(os.path.dirname(os.path.join(project_root, args.out)), exist_ok=True)
    shutil.copyfile(src, os.path.join(project_root, args.out))
    best_src = src.replace('.pkl', '_best.pkl')
    if os.path.exists(best_src):
        shutil.copyfile(best_src,
                        os.path.join(project_root, args.out.replace('.pkl', '_best.pkl')))

    payload = {
        'select_by': select_by, 'selection_split': 'validation',
        'seeds': seeds, 'base_args': base_args,
        'sweep_dir': sweep_dir, 'n_configs': len(combos),
        'ranking': table,
        'pareto_front_trials': front,
        'winner': {'trial': winner['trial'], 'config': winner['config'],
                   'seed': representative_seed, 'seed_role': 'closest_to_validation_mean',
                   'seeds': winner['seeds'],
                   'metrics': winner['metrics'], 'checkpoint': args.out},
    }
    with open(os.path.join(project_root, args.results), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    std_key = f'{select_by}_std'
    print("\n" + "=" * 96)
    print(f"{'rank':<5}{'trial':<7}{select_by:>12}{'±std':>8}{'mean_q':>9}{'stall_s':>9}  config")
    for rank, t in enumerate(table[:15], 1):
        m = t['metrics']
        star = ' *pareto' if t['trial'] in front else ''
        print(f"{rank:<5}{t['trial']:<7}{m[select_by]:>12.2f}{m.get(std_key, 0.0):>8.2f}"
              f"{m['mean_quality']:>9.3f}{m['stall_s']:>9.1f}  {t['config']}{star}")
    wm = winner['metrics']
    print(f"\nWINNER trial {winner['trial']} -> {args.out} "
          f"(representative seed {representative_seed})")
    print(f"  {select_by} = {wm[select_by]:.2f} ± {wm.get(std_key, 0.0):.2f} "
          f"over {len(winner['seeds'])} seeds {winner['seeds']}")
    print(f"  per-seed: {wm.get(f'{select_by}_per_seed', {})}")
    print(f"results: {args.results}")


if __name__ == '__main__':
    main()
