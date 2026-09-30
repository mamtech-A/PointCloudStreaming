# TRAINING SUMMARY

- run directory: `logs/train_runs/20260928_211458_wide`
- mode: full wide search
- experiment config digest: `eb2438c9799f2a76`
- trace registry: `configs/trace_window_registry.json`
- trace registry ID: `8297ac83fa031471`
- segment candidates: [5, 8, 10, 15]
- frozen segment size: **8 frames**
- test split changed by pipeline: **no**

## Segment-specific LSTMs

- S=5: `models/bandwidth_lstm_request_pacing_s5.pkl`; MAE=2.802, low-bandwidth MAE=0.430, transform=log1p
- S=8: `models/bandwidth_lstm_request_pacing_s8.pkl`; MAE=4.451, low-bandwidth MAE=0.610, transform=log1p
- S=10: `models/bandwidth_lstm_request_pacing_s10.pkl`; MAE=4.951, low-bandwidth MAE=0.951, transform=log1p
- S=15: `models/bandwidth_lstm_request_pacing_s15.pkl`; MAE=7.320, low-bandwidth MAE=1.616, transform=log1p

## DQN selection

- screened configurations: 4
- confirmed configurations: 1
- confirmation seeds: [45, 46, 47, 48, 49, 50, 51, 52, 53]
- winner config: `{"batch-size": 64, "eps-decay-frac": 0.5, "hidden": 256, "lr": 0.0003, "segment-frames": 8, "target-update": 6000}`
- validation qoe_quality: 43.749 +/- 0.673
- validation behavior: quality=0.589, stall=0.920 s, startup=4.506 s, rebuffer events=0.414
- mean training steps per confirmed seed: 84444

## Frozen evaluation

- fixed test not evaluated; method remains frozen on validation

## Evidence

- DQN sweep: `models/dqn_sweep_results_qoe_v4_startup_w2_segments.json`
- winner convergence: embedded in the DQN sweep artifact under `winner.validation_history_per_seed`
- baseline tuning: `models/baseline_config_qoe_v4_startup_w2_segments.json`
- stage/trial logs: `logs/train_runs/20260928_211458_wide`
