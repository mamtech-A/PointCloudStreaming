# TRAINING.md — second-PC overnight training runbook

The full retraining pipeline (LSTM → DQN sweep → eval → report) is one command
and fully described by repo state (`configs/training.json`). Prep happens on
the dev PC, training on the fast PC, review back on the dev PC — all via git.

## 0. One-time on the fast PC

```
git clone <repo> && cd PointCloudStreaming
python -m venv .venv && .venv\Scripts\activate     # (or your env)
python -m pip install -r requirements.txt          # torch/numpy/pandas/scikit-learn (CPU is fine)
python -c "import torch, numpy, sklearn, pandas; print('deps ok', torch.__version__)"
python tests/test_time_varying.py                  # must print "9 tests passed"
python run_training.py --smoke                     # ~minutes; validates the whole pipeline
```

Note: use `python -m pip` (not bare `pip`) so the packages land in the SAME
interpreter that runs `run_training.py`. If `run_training.py` fails instantly
with `ModuleNotFoundError: No module named 'torch'`, the install went to a
different Python — rerun the `python -m pip install` line above with the exact
`python` you launch training with.

## 1. Start the overnight run

```
git pull
python run_training.py --jobs 2        # --jobs ~= physical cores / 4 (each trial uses torch threads)
```

Then sleep. Everything is logged under `logs/train_runs/<UTC-timestamp>_full/`:
`RUN.log` (stage narration), `10_gen…/20_lstm…/30_sweep/40_eval…` stage logs,
and per-trial dirs `sweep/trial_*/` with `train.log`, `train_summary.json`,
`episodes.csv`.

Interrupted? Rerun and skip finished stages — the sweep also resumes per trial:

```
python run_training.py --skip-stages gen,lstm --jobs 2
```

What the stages do:

1. **gen** — `gen_lstm_dataset.py`: achieved-throughput series (the signal the
   LSTM is FED at inference) under the current transport model → `data/lstm_achieved/`.
2. **lstm** — `train_model.py` per transform candidate (`none`, `log1p`),
   winner picked by low-bandwidth-regime MAE → `models/bandwidth_lstm.pkl`.
3. **sweep** — `sweep.py`: grid over reward spec (μ, λ, bounded-stall shape) ×
   DQN hyperparameters × the `lstm_pred` ablation, ranked by held-out
   quality-aware QoE on longdress; winner → `models/abr_dqn.pkl`, full ranking +
   quality-vs-stall Pareto front → `models/dqn_sweep_results.json`.
4. **eval** — `eval_fixed.py` (fixed-arm bars) + `compare.py` (baseline/LSTM/DQN)
   on the held-out static trace.
5. **report** — `models/TRAINING_SUMMARY.md`.

## 2. Push the results back

```
git add models logs/train_runs data/lstm_achieved
git commit -m "overnight training run: <one-line result>"
git push
```

Committed artifacts: winning checkpoints (`models/*.pkl`), all JSON results,
`TRAINING_SUMMARY.md`, stage logs and per-trial `train_summary.json`/`train.log`
(the bulky per-trial checkpoints under `logs/train_runs/*/sweep/trial_*/` are
gitignored — only the installed winners in `models/` travel).

## 3. Review on the dev PC

```
git pull
```

Read `models/TRAINING_SUMMARY.md` first, then `logs/train_runs/<ts>/RUN.log`
for the narrative, `models/dqn_sweep_results.json` for the full ranking /
Pareto front, and any trial's `train.log` for its episode-by-episode history.

## Content sequences (multi-sequence training)

Encode a new 8i sequence on the fast PC with the SAME 6-tier ladder, then push
the manifest + coded json (bitstreams stay out of git):

```
python gpcc/encode_frames.py --dir "F:/path/to/loot/Ply" --name loot --jobs 8
git add config/mpd_gpcc_loot.xml gpcc/coded_frames_loot.json && git commit && git push
```

`train_dqn.py` picks up every `config/mpd_gpcc*.xml` automatically; held-out
eval stays longdress-only for comparability.

## Scaling the search

Edit `configs/training.json` (commit it — the run is then reproducible):

- more/other reward shapes → `dqn_sweep.axes.reward-spec` (see `src/rl/reward.py`)
- longer training → `dqn_sweep.base_args.epochs` (no cap; more is fine while
  held-out metrics improve)
- multiple seeds → `dqn_sweep.seeds: [42, 43, 44]` (results average over seeds)
- random subsample of a huge grid → `"mode": "random", "budget": N`
