# TRAINING SUMMARY

- run directory: `logs/train_runs/20260721_203138_wide`
- mode: full wide search
- experiment config digest: `bbe86f854667c2cd`
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
- validation qoe_quality: 43.644 +/- 0.602
- mean training steps per confirmed seed: 76000

## Frozen evaluation

- split: **test**; protocol digest: `4071599372abeb61`

| policy | QoE | mean quality | stall (s) |
|---|---:|---:|---:|
| DQN | 46.940 | 0.594 | 0.875 |
| Best global fixed | 26.707 | 0.613 | 5.071 |
| BufferBased | 43.261 | 0.531 | 0.241 |
| MPC | 43.366 | 0.517 | 0.282 |

## Evidence

- DQN sweep: `models/dqn_sweep_results_qoe_v4_startup_w2_s10.json`
- baseline tuning: `models/baseline_config_qoe_v4_startup_w2_s10.json`
- case-level evaluation: `models/final_test_results_qoe_v4_startup_w2_s10.json`
- stage/trial logs: `logs/train_runs/20260721_203138_wide`
