# TRAINING SUMMARY

- run dir: `logs\train_runs\20260714_131016_full`
- finished: 2026-07-14 21:46:30 UTC
- mode: full

## LSTM (models/bandwidth_lstm.pkl)

- transform: **log1p** | seq_len 8
- MAE 2.67 (persistence 2.75) | RMSE 9.12 (pers. 10.27)
- low-bw (<5.0 Mbps) MAE 0.497 (pers. 0.506) | directional accuracy 0.616
- split: 536 train / 128 test files

## DQN sweep (models/abr_dqn.pkl)

- 4 configs x seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] | select-by **qoe_quality**
- winner: trial 1 — config `{"mu": 4.3, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0, "startup_weight": 1.0, "drop_weight": 1.0}, "segment-frames": 8}`
- **qoe_quality = 54.50 ± 0.95** over 12 seeds (per-seed: {'42': 55.292351959975264, '43': 51.9440962600954, '44': 55.27339464800906, '45': 55.062627724828864, '46': 53.44293693159978, '47': 54.83911983928189, '48': 54.931816796657166, '49': 54.8837701042047, '50': 55.11641256925078, '51': 54.93891548616009, '52': 53.69469675464233, '53': 54.551778621535625})
- winner metrics: reward=205.332, qoe=79.287, qoe_quality=54.498, mean_quality=0.775, stall_s=1.831, qoe_quality_std=0.952
- Pareto front (quality vs stall): trials [1, 2, 3]

| rank | trial | qoe_quality | ±std | mean_q | stall_s | config |
|---|---|---|---|---|---|---|
| 1 | 1 | 54.50 | 0.95 | 0.775 | 1.8 | `{"mu": 4.3, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0, "startup_weight": 1.0, "drop_weight": 1.0}, "segment-frames": 8}` |
| 2 | 2 | 54.33 | 0.86 | 0.782 | 1.8 | `{"mu": 4.3, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}, "segment-frames": 8}` |
| 3 | 3 | 53.56 | 1.61 | 0.759 | 1.7 | `{"mu": 4.3, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0, "startup_weight": 1.0, "drop_weight": 1.0}, "segment-frames": 8}` |
| 4 | 0 | 53.43 | 1.43 | 0.777 | 1.9 | `{"mu": 4.3, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}, "segment-frames": 8}` |

## Fixed-arm baseline

- best arm: arm_3 (mean reward 187.63) — see models/fixed_arm_baseline.json

## Logs

- stage logs + sweep trial dirs: `logs\train_runs\20260714_131016_full/`
- per-trial: `sweep/trial_*/train.log`, `train_summary.json`, `episodes.csv`
