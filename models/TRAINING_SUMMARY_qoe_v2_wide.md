# TRAINING SUMMARY

- run directory: `logs/train_runs/qoe_v2_wide`
- mode: full wide search
- experiment config digest: `725d4b220174d141`
- segment candidates: [5, 8, 10, 15]
- frozen segment size: **8 frames**
- test split changed by pipeline: **no**

## Segment-specific LSTMs

- S=5: `models/bandwidth_lstm_s5.pkl`; MAE=2.001, low-bandwidth MAE=0.414, transform=log1p
- S=8: `models/bandwidth_lstm_s8.pkl`; MAE=3.141, low-bandwidth MAE=0.574, transform=log1p
- S=10: `models/bandwidth_lstm_s10.pkl`; MAE=3.442, low-bandwidth MAE=0.739, transform=log1p
- S=15: `models/bandwidth_lstm_s15.pkl`; MAE=4.743, low-bandwidth MAE=1.095, transform=log1p

## DQN selection

- screened configurations: 20
- confirmed configurations: 4
- confirmation seeds: [45, 46, 47, 48, 49, 50, 51, 52, 53]
- winner config: `{"batch-size": 64, "eps-decay-frac": 0.6, "hidden": 128, "lr": 0.0001, "segment-frames": 8, "target-update": 3000}`
- validation qoe_quality: -315.286 +/- 80.134
- mean training steps per confirmed seed: 94578

## Frozen evaluation

- split: **test**; protocol digest: `4071599372abeb61`

| policy | QoE | mean quality | stall (s) |
|---|---:|---:|---:|
| DQN | 32.570 | 0.460 | 1.166 |
| Best global fixed | -20.033 | 0.000 | 0.000 |
| BufferBased | 33.360 | 0.412 | 0.001 |
| MPC | 37.660 | 0.444 | 0.327 |

## Evidence

- DQN sweep: `models/dqn_sweep_results_qoe_v2_wide.json`
- baseline tuning: `models/baseline_config_qoe_v2_wide.json`
- case-level evaluation: `models/final_test_results_qoe_v2_wide.json`
- stage/trial logs: `logs/train_runs/qoe_v2_wide`
