# TRAINING SUMMARY

- run directory: `logs/train_runs/20260720_005851_wide`
- mode: full wide search
- experiment config digest: `f7af2fd8e9aaaf81`
- trace registry: `configs/trace_window_registry.json`
- trace registry ID: `8297ac83fa031471`
- segment candidates: [5, 8, 10, 15]
- frozen segment size: **10 frames**
- test split changed by pipeline: **no**

## Segment-specific LSTMs

- S=5: `models/bandwidth_lstm_request_pacing_s5.pkl`; MAE=2.802, low-bandwidth MAE=0.430, transform=log1p
- S=8: `models/bandwidth_lstm_request_pacing_s8.pkl`; MAE=4.451, low-bandwidth MAE=0.610, transform=log1p
- S=10: `models/bandwidth_lstm_request_pacing_s10.pkl`; MAE=4.951, low-bandwidth MAE=0.951, transform=log1p
- S=15: `models/bandwidth_lstm_request_pacing_s15.pkl`; MAE=7.320, low-bandwidth MAE=1.616, transform=log1p

## DQN selection

- screened configurations: 20
- confirmed configurations: 4
- confirmation seeds: [45, 46, 47, 48, 49, 50, 51, 52, 53]
- winner config: `{"batch-size": 64, "eps-decay-frac": 0.5, "hidden": 256, "lr": 0.0003, "segment-frames": 10, "target-update": 6000}`
- validation qoe_quality: 48.933 +/- 0.519
- mean training steps per confirmed seed: 76000

## Frozen evaluation

- split: **test**; protocol digest: `4071599372abeb61`

| policy | QoE | mean quality | stall (s) |
|---|---:|---:|---:|
| DQN | 49.946 | 0.615 | 1.107 |
| Best global fixed | 31.412 | 0.613 | 5.071 |
| BufferBased | 45.692 | 0.531 | 0.241 |
| MPC | 45.797 | 0.517 | 0.282 |

## Evidence

- DQN sweep: `models/dqn_sweep_results_qoe_v4_request_pacing.json`
- baseline tuning: `models/baseline_config_qoe_v4_request_pacing.json`
- case-level evaluation: `models/final_test_results_qoe_v4_request_pacing.json`
- stage/trial logs: `logs/train_runs/20260720_005851_wide`
