#!/usr/bin/env python3
"""One-command overnight training pipeline (second-PC entrypoint).

Stages (each a subprocess; stdout/stderr -> its own log under
logs/train_runs/<UTC-timestamp>/, narrated in RUN.log):

  1. gen    — achieved-throughput LSTM dataset (gen_lstm_dataset.py), if
              configs/training.json lstm.signal == "achieved"
  2. lstm   — train one LSTM per transform candidate (train_model.py), pick the
              winner by the configured low-bandwidth metric, install it as
              models/bandwidth_lstm.pkl (+ _best/_split/_tuning companions)
  3. sweep  — DQN reward/hyperparameter sweep (sweep.py); winner ->
              models/abr_dqn.pkl, ranking -> models/dqn_sweep_results.json
  4. eval   — eval_fixed.py + compare.py on the held-out static trace
  5. report — models/TRAINING_SUMMARY.md tying it all together

Everything is resumable: rerun with --skip-stages gen,lstm to continue after a
crash (the sweep also resumes internally per trial).

Usage (see TRAINING.md for the full second-PC runbook):
    python run_training.py --jobs 2            # the real overnight run
    python run_training.py --smoke             # ~minutes end-to-end sanity run
    python run_training.py --skip-stages gen,lstm --jobs 4
"""

import os
import sys
import json
import glob
import time
import shutil
import argparse
import subprocess

project_root = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODELS = os.path.join(project_root, 'models')


class Runner:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.run_log = open(os.path.join(run_dir, 'RUN.log'), 'a', encoding='utf-8')

    def note(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}"
        print(line)
        self.run_log.write(line + '\n')
        self.run_log.flush()

    def stage(self, name, cmd, log_name):
        """Run one stage subprocess; returns True on success."""
        log_path = os.path.join(self.run_dir, log_name)
        self.note(f"START {name}: {' '.join(cmd)}  (log: {log_name})")
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write('# ' + ' '.join(cmd) + '\n')
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=project_root, env=env)
        dt = time.time() - t0
        if rc == 0:
            self.note(f"DONE  {name} in {dt/60:.1f} min")
            return True
        self.note(f"FAIL  {name} (rc={rc}) after {dt/60:.1f} min — see {log_path}")
        return False


def py(*args):
    return [sys.executable, '-u'] + [str(a) for a in args]


def install_lstm_winner(candidate_out, final_out):
    """Copy a candidate LSTM's artifact set to the canonical model paths."""
    for suffix in ('.pkl', '_best.pkl', '_split.json', '_tuning_results.json'):
        src = candidate_out.replace('.pkl', suffix)
        dst = final_out.replace('.pkl', suffix)
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def main():
    p = argparse.ArgumentParser(description="Overnight training pipeline")
    p.add_argument('--config', default=os.path.join('configs', 'training.json'))
    p.add_argument('--jobs', type=int, default=2, help='parallel sweep trials')
    p.add_argument('--smoke', action='store_true',
                   help='tiny end-to-end run (~minutes) to validate the pipeline')
    p.add_argument('--skip-stages', default='',
                   help='comma list from {gen,lstm,sweep,eval} to skip (resume)')
    args = p.parse_args()
    skip = {s.strip() for s in args.skip_stages.split(',') if s.strip()}

    with open(os.path.join(project_root, args.config), encoding='utf-8') as f:
        cfg = json.load(f)
    lstm_cfg = cfg.get('lstm', {})
    ts = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    run_dir = os.path.join(project_root, 'logs', 'train_runs',
                           f"{ts}_{'smoke' if args.smoke else 'full'}")
    R = Runner(run_dir)
    R.note(f"pipeline start | config={args.config} smoke={args.smoke} "
           f"jobs={args.jobs} skip={sorted(skip) or 'none'}")
    shutil.copyfile(os.path.join(project_root, args.config),
                    os.path.join(run_dir, 'training.config.json'))

    ok = True

    # --- Stage 1: achieved-throughput dataset -------------------------------
    achieved = lstm_cfg.get('signal', 'capacity') == 'achieved'
    achieved_dir = os.path.join(project_root, 'data', 'lstm_achieved')
    if achieved and 'gen' not in skip:
        cmd = py('gen_lstm_dataset.py')
        for k, v in lstm_cfg.get('gen_args', {}).items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        if args.smoke:
            # 120 frames -> 12 segment samples at S=10: still enough sequences
            # for the seq_len-8 smoke LSTM fit.
            cmd += ['--offsets', '1', '--policies', 'fixed3', '--max-frames', '120']
        ok = R.stage('gen', cmd, '10_gen_lstm_dataset.log')
        if not ok:
            sys.exit(1)

    # --- Stage 2: LSTM candidates + winner selection ------------------------
    if 'lstm' not in skip:
        transforms = lstm_cfg.get('transforms', ['none'])
        if args.smoke:
            transforms = transforms[:1]
        lstm_dir = os.path.join(run_dir, 'lstm')
        os.makedirs(lstm_dir, exist_ok=True)
        candidates = []
        for tf in transforms:
            out = os.path.join(lstm_dir, f'bandwidth_lstm_{tf}.pkl')
            cmd = py('train_model.py', '--transform', tf, '--out', out,
                     '--sequence-length', lstm_cfg.get('sequence_length', 20),
                     '--epochs', 5 if args.smoke else lstm_cfg.get('epochs', 100),
                     '--test-size', 0.2)
            if achieved:
                cmd += ['--bandwidth-dir', achieved_dir,
                        '--split-from', os.path.join(achieved_dir, 'split.json')]
            if args.smoke or not lstm_cfg.get('tune', True):
                cmd += ['--no-tune']
            if not R.stage(f'lstm[{tf}]', cmd, f'20_lstm_{tf}.log'):
                sys.exit(1)
            with open(out.replace('.pkl', '_split.json'), encoding='utf-8') as f:
                metrics = json.load(f)['metrics']
            candidates.append((tf, out, metrics))
            R.note(f"lstm[{tf}] metrics: mae={metrics['mae']:.2f} "
                   f"low_bw_mae={metrics.get('low_bw_mae', 0):.3f} "
                   f"dir_acc={metrics.get('directional_accuracy', 0):.3f}")

        sel = lstm_cfg.get('select_metric', 'low_bw_mae')
        candidates.sort(key=lambda c: (c[2].get(sel, float('inf')), c[2]['mae']))
        winner_tf, winner_out, winner_metrics = candidates[0]
        install_lstm_winner(winner_out, os.path.join(MODELS, 'bandwidth_lstm.pkl'))
        R.note(f"LSTM winner: transform={winner_tf} ({sel}="
               f"{winner_metrics.get(sel, float('nan')):.3f}) -> models/bandwidth_lstm.pkl")

    # --- Stage 3: DQN sweep ---------------------------------------------------
    if 'sweep' not in skip:
        cmd = py('sweep.py', '--config', args.config, '--jobs', args.jobs,
                 '--sweep-dir', os.path.join(run_dir, 'sweep'))
        if args.smoke:
            cmd += ['--smoke']
        if not R.stage('sweep', cmd, '30_sweep.log'):
            sys.exit(1)

    # --- Stage 4: final eval ---------------------------------------------------
    if 'eval' not in skip:
        fe = cfg.get('final_eval', {})
        fe_flags = ['--segment-frames', str(fe.get('segment_frames', 10)),
                    '--playback-rate-min', str(fe.get('playback_rate_min', 1.0))]
        mf = ['--max-frames', '60'] if args.smoke else []
        if not R.stage('eval_fixed', py('eval_fixed.py', *(fe_flags + mf)),
                       '40_eval_fixed.log'):
            ok = False
        mf = ['--max-frames', '40'] if args.smoke else []
        if not R.stage('compare', py('compare.py', *(fe_flags + mf)), '41_compare.log'):
            ok = False

    # --- Stage 5: TRAINING_SUMMARY.md -----------------------------------------
    summary_lines = [
        "# TRAINING SUMMARY", "",
        f"- run dir: `{os.path.relpath(run_dir, project_root)}`",
        f"- finished: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- mode: {'SMOKE (not a real result)' if args.smoke else 'full'}", "",
    ]
    split_json = os.path.join(MODELS, 'bandwidth_lstm_split.json')
    if os.path.exists(split_json):
        with open(split_json, encoding='utf-8') as f:
            sj = json.load(f)
        m = sj['metrics']
        summary_lines += [
            "## LSTM (models/bandwidth_lstm.pkl)", "",
            f"- transform: **{m.get('transform')}** | seq_len {sj.get('sequence_length')}",
            f"- MAE {m['mae']:.2f} (persistence {m['persistence_mae']:.2f}) | "
            f"RMSE {m['rmse']:.2f} (pers. {m['persistence_rmse']:.2f})",
            f"- low-bw (<{m.get('low_bw_threshold_mbps', 5)} Mbps) MAE "
            f"{m.get('low_bw_mae', float('nan')):.3f} "
            f"(pers. {m.get('persistence_low_bw_mae', float('nan')):.3f}) | "
            f"directional accuracy {m.get('directional_accuracy', float('nan')):.3f}",
            f"- split: {sj['train_file_count']} train / {sj['test_file_count']} test files", "",
        ]
    sweep_json = os.path.join(MODELS, 'dqn_sweep_results.json')
    if os.path.exists(sweep_json):
        with open(sweep_json, encoding='utf-8') as f:
            sw = json.load(f)
        w = sw['winner']
        summary_lines += [
            "## DQN sweep (models/abr_dqn.pkl)", "",
            f"- {sw['n_configs']} configs x seeds {sw['seeds']} | select-by **{sw['select_by']}**",
            f"- winner: trial {w['trial']} seed {w['seed']} — config `{json.dumps(w['config'])}`",
            f"- winner metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in w['metrics'].items()),
            f"- Pareto front (quality vs stall): trials {sw['pareto_front_trials']}",
            "", "| rank | trial | " + sw['select_by'] + " | mean_q | stall_s | config |",
            "|---|---|---|---|---|---|",
        ]
        for rank, t in enumerate(sw['ranking'][:10], 1):
            m = t['metrics']
            summary_lines.append(
                f"| {rank} | {t['trial']} | {m[sw['select_by']]:.2f} | "
                f"{m['mean_quality']:.3f} | {m['stall_s']:.1f} | `{json.dumps(t['config'])}` |")
        summary_lines.append("")
    fixed_json = os.path.join(MODELS, 'fixed_arm_baseline.json')
    if os.path.exists(fixed_json):
        with open(fixed_json, encoding='utf-8') as f:
            fx = json.load(f)
        summary_lines += [
            "## Fixed-arm baseline",
            "",
            f"- best arm: {fx.get('best_arm')} (mean reward {fx.get('best_reward', 0):.2f}) — "
            f"see models/fixed_arm_baseline.json",
            "",
        ]
    summary_lines += [
        "## Logs", "",
        f"- stage logs + sweep trial dirs: `{os.path.relpath(run_dir, project_root)}/`",
        "- per-trial: `sweep/trial_*/train.log`, `train_summary.json`, `episodes.csv`", "",
    ]
    with open(os.path.join(MODELS, 'TRAINING_SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    R.note(f"summary -> models/TRAINING_SUMMARY.md | pipeline {'OK' if ok else 'FINISHED WITH FAILURES'}")
    print("\nNext: git add -A && git commit -m 'training run' && git push  "
          "(see TRAINING.md)")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
