# TRAINING SUMMARY

- run dir: `logs\train_runs\20260711_130607_full`
- finished: 2026-07-11 14:28:01 UTC
- mode: full

## LSTM (models/bandwidth_lstm.pkl)

- transform: **log1p** | seq_len 8
- MAE 6.21 (persistence 5.76) | RMSE 13.63 (pers. 14.65)
- low-bw (<5.0 Mbps) MAE 1.310 (pers. 1.447) | directional accuracy 0.612
- split: 128 train / 32 test files

## DQN sweep (models/abr_dqn.pkl)

- 4 configs x seeds [42, 43, 44, 45, 46, 47, 48, 49] | select-by **qoe_quality**
- winner: trial 3 — config `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 8}`
- **qoe_quality = 74.60 ± 19.79** over 8 seeds (per-seed: {'42': 90.94552657478353, '43': 38.27038062095867, '44': 95.11091865217524, '45': 87.93558763650542, '46': 56.8986781135384, '47': 67.90453818338757, '48': 63.53712643427891, '49': 96.17907099633108})
- winner metrics: reward=253.633, qoe=73.526, qoe_quality=74.598, mean_quality=0.909, stall_s=3.195, qoe_quality_std=19.786
- Pareto front (quality vs stall): trials [3]

| rank | trial | qoe_quality | ±std | mean_q | stall_s | config |
|---|---|---|---|---|---|---|
| 1 | 3 | 74.60 | 19.79 | 0.909 | 3.2 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 8}` |
| 2 | 0 | 74.23 | 20.26 | 0.909 | 3.3 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 5}` |
| 3 | 2 | 73.77 | 20.63 | 0.905 | 3.2 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 5}` |
| 4 | 1 | 72.60 | 19.03 | 0.893 | 3.3 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 8}` |

## Fixed-arm baseline

- best arm: arm_1 (mean reward 275.94) — see models/fixed_arm_baseline.json

## Logs

- stage logs + sweep trial dirs: `logs\train_runs\20260711_130607_full/`
- per-trial: `sweep/trial_*/train.log`, `train_summary.json`, `episodes.csv`
