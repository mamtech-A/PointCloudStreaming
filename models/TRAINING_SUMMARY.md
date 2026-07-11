# TRAINING SUMMARY

- run dir: `logs\train_runs\20260711_073843_full`
- finished: 2026-07-11 11:00:23 UTC
- mode: full

## LSTM (models/bandwidth_lstm.pkl)

- transform: **log1p** | seq_len 8
- MAE 5.80 (persistence 5.36) | RMSE 12.88 (pers. 14.02)
- low-bw (<5.0 Mbps) MAE 1.368 (pers. 1.539) | directional accuracy 0.611
- split: 128 train / 32 test files

## DQN sweep (models/abr_dqn.pkl)

- 24 configs x seeds [42, 43] | select-by **qoe_quality**
- winner: trial 13 seed 42 — config `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 5}`
- winner metrics: reward=238.689, qoe=92.202, qoe_quality=77.273, mean_quality=0.809, stall_s=0.560
- Pareto front (quality vs stall): trials [13, 14, 19, 15, 21, 20, 16, 7, 1, 8, 4, 23, 10, 17, 5, 11]

| rank | trial | qoe_quality | mean_q | stall_s | config |
|---|---|---|---|---|---|
| 1 | 13 | 77.27 | 0.809 | 0.6 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 5}` |
| 2 | 14 | 75.98 | 0.811 | 0.7 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 8}` |
| 3 | 19 | 75.94 | 0.813 | 0.9 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 5}` |
| 4 | 15 | 68.94 | 0.845 | 2.8 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 10}` |
| 5 | 21 | 64.74 | 0.853 | 4.1 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 10}` |
| 6 | 20 | 64.61 | 0.884 | 4.6 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 8}` |
| 7 | 16 | 64.31 | 0.899 | 5.5 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 15}` |
| 8 | 22 | 63.48 | 0.894 | 5.7 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 15}` |
| 9 | 7 | 63.15 | 0.887 | 5.0 | `{"mu": 2.0, "no-lstm-pred": true, "segment-frames": 5}` |
| 10 | 1 | 61.41 | 0.890 | 5.2 | `{"mu": 2.0, "no-lstm-pred": false, "segment-frames": 5}` |

## Fixed-arm baseline

- best arm: arm_1 (mean reward 275.97) — see models/fixed_arm_baseline.json

## Logs

- stage logs + sweep trial dirs: `logs\train_runs\20260711_073843_full/`
- per-trial: `sweep/trial_*/train.log`, `train_summary.json`, `episodes.csv`
