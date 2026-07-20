# Clean IST-2026 experiment: second-PC runbook

This branch implements the final, leakage-free experiment. The complete run is
defined by two tracked files:

- `configs/training.json`: frozen training/model/baseline settings.
- `configs/experiment_protocol.json`: explicit train/validation/test traces and
  registered evaluation cases.

The four traces used during earlier Round-4 development are now **validation**.
Four different traces are reserved for the final test. Training and checkpoint
selection never execute the final-test split.

## 1. Pull the experiment branch

On the fast PC:

```powershell
git fetch origin
git switch exp/clean-eval-retrain
git pull --ff-only origin exp/clean-eval-retrain
git status --short --branch
```

The status should show `exp/clean-eval-retrain` with no local code changes.

## 2. Verify the environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tests/test_experiment_protocol.py
python tests/test_abr_baselines.py
python tests/test_qoe_reward.py
python tests/test_time_varying.py
```

The protocol test checks that all 21 traces occur in exactly one partition and
that validation/test are disjoint and stratified.

## 3. Optional smoke test

```powershell
python scripts/run_training.py --smoke --jobs 1
```

The smoke run uses validation, never final test. It is only a pipeline check and
must not be reported as a paper result.

## 4. Run the frozen experiment

Use a clean clone/worktree without an old `models/final_test_results.json`, then:

```powershell
python scripts/run_training.py --jobs 2
```

Use approximately one job per four physical CPU cores. The pipeline performs:

1. Generate achieved-throughput LSTM series from training and validation trace
   sources. Final-test sources are excluded.
2. Select/train the LSTM using validation only.
3. Train two preregistered DQN configurations (with and without the LSTM
   prediction feature), each with 12 seeds.
4. Select the configuration by mean validation QoE and install the seed closest
   to that validation mean—not the luckiest seed.
5. Select the global fixed tier and recent-throughput, LSTM-rule, buffer-based,
   and MPC parameters using validation.
6. Run the locked clean-run test once across four traces, all four 8i sequences, three
   offsets per trace, and three paired jitter seeds.
7. Evaluate DQN, six fixed tiers, recent-throughput, LSTM-rule, buffer-based,
   MPC, and the non-causal per-trace fixed oracle.

The final-test script refuses to overwrite an existing result. Do not add
`--overwrite` after examining the result unless it is a documented exact rerun
of the unchanged commit and configuration.

## 5. Resume an interrupted run

The first line of `RUN.log` contains the run directory. Reuse that exact path so
completed seed/configuration trials are discovered:

```powershell
python scripts/run_training.py `
  --run-dir logs/train_runs/<timestamp>_full `
  --skip-stages gen,lstm `
  --jobs 2
```

If baseline tuning already completed, add `baselines` to `--skip-stages`. If the
registered final test also completed and only report generation was interrupted,
add `test` as well.

## 6. Inspect before committing results

Read these in order:

1. `models/TRAINING_SUMMARY.md`
2. `models/final_test_results.json`
3. `models/dqn_sweep_results.json`
4. `logs/train_runs/<timestamp>_full/RUN.log`

Confirm that `models/final_test_results.json` says:

- `split: "test"`
- four trace files
- four sequences
- three offsets per trace
- three jitter seeds
- the same protocol digest printed in the training logs

## 7. Commit and push the training artifacts

After checking that the full run succeeded:

```powershell
git status --short
git add models/abr_dqn.pkl models/abr_dqn_best.pkl `
  models/bandwidth_lstm.pkl models/bandwidth_lstm_best.pkl `
  models/bandwidth_lstm_split.json models/bandwidth_lstm_tuning_results.json `
  models/dqn_sweep_results.json models/baseline_config.json `
  models/final_test_results.json `
  models/TRAINING_SUMMARY.md data/lstm_achieved logs/train_runs
git commit -m "results: clean registered IST-2026 experiment"
git push origin exp/clean-eval-retrain
```

Per-trial checkpoints remain ignored because they are large; their case-level
metrics and logs are retained. The installed representative DQN checkpoint is
tracked.

## Scientific interpretation rules

- Validation numbers explain model/checkpoint selection; they are not final
  performance claims.
- Report the locked-test result and uncertainty across cases/training seeds.
- Call quality a normalized log-density utility or proxy, not a validated
  perceptual metric.
- “Best global fixed” and “per-trace fixed oracle” are different. The oracle is
  hindsight-only and non-causal.
- Do not claim stochastic dominance from four traces.
- Do not use a running maximum as evidence of convergence.
