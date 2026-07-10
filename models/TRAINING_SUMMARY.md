# TRAINING SUMMARY

- run dir: `logs\train_runs\20260710_083943_full`
- finished: 2026-07-10 15:32:13 UTC
- mode: full

## LSTM (models/bandwidth_lstm.pkl)

- transform: **log1p** | seq_len 20
- MAE 0.68 (persistence 0.80) | RMSE 1.95 (pers. 2.39)
- low-bw (<5.0 Mbps) MAE 0.190 (pers. 0.195) | directional accuracy 0.703
- split: 64 train / 16 test files

## DQN sweep (models/abr_dqn.pkl)

- 24 configs x seeds [42] | select-by **qoe_quality**
- winner: trial 13 seed 42 — config `{"lam": 1.0, "mu": 2.0, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}`
- winner metrics: reward=219.984, qoe=0.000, qoe_quality=11.962, mean_quality=0.951, stall_s=18.400
- Pareto front (quality vs stall): trials [13, 7, 19, 1, 17, 18]

| rank | trial | qoe_quality | mean_q | stall_s | config |
|---|---|---|---|---|---|
| 1 | 13 | 11.96 | 0.951 | 18.4 | `{"lam": 1.0, "mu": 2.0, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 2 | 22 | -28.64 | 0.950 | 28.1 | `{"lam": 1.0, "mu": 8.0, "no-lstm-pred": true, "reward-spec": {}}` |
| 3 | 7 | -29.46 | 0.954 | 28.4 | `{"lam": 0.5, "mu": 4.3, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 4 | 19 | -31.08 | 0.956 | 28.7 | `{"lam": 1.0, "mu": 4.3, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 5 | 1 | -31.11 | 0.953 | 28.2 | `{"lam": 0.5, "mu": 2.0, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 6 | 11 | -31.18 | 0.674 | 22.8 | `{"lam": 0.5, "mu": 8.0, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 7 | 23 | -31.18 | 0.674 | 22.8 | `{"lam": 1.0, "mu": 8.0, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 8 | 17 | -31.19 | 0.962 | 29.2 | `{"lam": 1.0, "mu": 4.3, "no-lstm-pred": false, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 9 | 15 | -34.59 | 0.904 | 27.7 | `{"lam": 1.0, "mu": 2.0, "no-lstm-pred": true, "reward-spec": {"stall_mode": "bounded", "stall_cap_s": 2.0, "event_penalty": 2.0}}` |
| 10 | 16 | -35.08 | 0.947 | 29.2 | `{"lam": 1.0, "mu": 4.3, "no-lstm-pred": false, "reward-spec": {}}` |

## Fixed-arm baseline

- best arm: arm_3 (mean reward 63.64) — see models/fixed_arm_baseline.json

## Logs

- stage logs + sweep trial dirs: `logs\train_runs\20260710_083943_full/`
- per-trial: `sweep/trial_*/train.log`, `train_summary.json`, `episodes.csv`
