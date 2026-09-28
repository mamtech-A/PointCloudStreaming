# TRAINING SUMMARY

- run directory: `logs/train_runs/20260721_222523_wide`
- mode: full wide search
- experiment config digest: `6775697dada8a106`
- trace registry: `configs/trace_window_registry.json`
- trace registry ID: `8297ac83fa031471`
- segment candidates: [10]
- frozen segment size: **10 frames**
- test split changed by pipeline: **no**

## Segment-specific LSTMs

- S=10: `models/bandwidth_lstm_request_pacing_s10.pkl`; MAE=4.951, low-bandwidth MAE=0.951, transform=log1p

## DQN selection

- screened configurations: 1
- confirmed configurations: 1
- confirmation seeds: [45, 46, 47, 48, 49, 50, 51, 52, 53]
- winner config: `{"batch-size": 64, "eps-decay-frac": 0.5, "hidden": 256, "lr": 0.0003, "segment-frames": 10, "target-update": 6000}`
- validation qoe_quality: 45.534 +/- 0.287
- validation behavior: quality=0.594, stall=1.077 s, startup=4.559 s, rebuffer events=0.505
- mean training steps per confirmed seed: 68000

## Frozen evaluation

- fixed test not evaluated; method remains frozen on validation

## Evidence

- DQN sweep: `models/dqn_sweep_results_qoe_v4_startup_w15_s10.json`
- winner convergence: embedded in the DQN sweep artifact under `winner.validation_history_per_seed`
- baseline tuning: `models/baseline_config_qoe_v4_startup_w15_s10.json`
- stage/trial logs: `logs/train_runs/20260721_222523_wide`
