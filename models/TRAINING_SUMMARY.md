# TRAINING SUMMARY

- run dir: `logs\train_runs\20260718_122736_full`
- finished: 2026-07-18 14:19:51 UTC
- mode: full

## LSTM (models/bandwidth_lstm.pkl)

- transform: **log1p** | seq_len 8
- MAE 2.75 (persistence 2.67) | RMSE 8.83 (pers. 9.76)
- low-bw (<5.0 Mbps) MAE 0.482 (pers. 0.502) | directional accuracy 0.610
- split: 412 train / 128 validation files (final test excluded)

## DQN sweep (models/abr_dqn.pkl)

- 2 configs x seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] | select-by validation **qoe_quality**
- winner: trial 0 — config `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 8}`
- **qoe_quality = -845.37 ± 354.79** over 12 seeds (per-seed: {'42': -355.42926954475354, '43': -1269.4195296062646, '44': -318.3477803280542, '45': -1088.7388057540293, '46': -1270.9641997976532, '47': -573.2530815198318, '48': -445.1114858280679, '49': -511.0224898682254, '50': -1010.0173925565248, '51': -1065.1096034318916, '52': -1087.4231027415053, '53': -1149.5859203239238})
- winner metrics: reward=-5.402, qoe=65.585, qoe_quality=-845.369, mean_quality=0.570, stall_s=179.448, qoe_quality_std=354.794
- Pareto front (quality vs stall): trials [0, 1]

| rank | trial | qoe_quality | ±std | mean_q | stall_s | config |
|---|---|---|---|---|---|---|
| 1 | 0 | -845.37 | 354.79 | 0.570 | 179.4 | `{"mu": 4.3, "no-lstm-pred": false, "segment-frames": 8}` |
| 2 | 1 | -1290.52 | 1043.17 | 0.637 | 279.5 | `{"mu": 4.3, "no-lstm-pred": true, "segment-frames": 8}` |

## Untouched final test

- protocol digest: `73916fc9adae6232` | 4 traces x 4 contents x 3 offsets

| policy | QoE quality | case std | mean quality | stall (s) |
|---|---:|---:|---:|---:|
| DQN | 26.020 | 44.221 | 0.651 | 5.905 |
| Best global fixed | -18.543 | 7.895 | 0.019 | 0.000 |
| Recent-throughput | -12.377 | 11.403 | 0.067 | 0.000 |
| LSTM rule | -16.187 | 9.470 | 0.037 | 0.000 |
| Buffer-based | 36.165 | 20.749 | 0.442 | 0.007 |
| MPC | 38.819 | 20.286 | 0.496 | 0.480 |
| Per-trace fixed oracle* | 37.497 | 27.261 | 0.558 | 1.847 |

* Non-causal hindsight reference; not a deployable baseline.

## Logs

- stage logs + sweep trial dirs: `logs\train_runs\20260718_122736_full/`
- per-trial: `sweep/trial_*/train.log`, `train_summary.json`, `episodes.csv`
