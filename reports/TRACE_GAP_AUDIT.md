# Timestamp-gap audit

Raw trace CSVs are unchanged. A long gap is a consecutive timestamp delta greater than 5 seconds.

## Findings

- 34 long gaps across 13/21 traces.
- Largest gap: 627 seconds.
- Gaps >10 s: 8; >30 s: 2; >60 s: 1.
- Dominant cadence: 30493/40942 deltas are exactly one second; 3664 are duplicate timestamps.

The cleaned corpus has no measured positive application-layer download throughput sample inside these intervals. Holding the previous rate across them would invent observations, so the recommended policy is to split at gaps >5 s while preserving the parent file's train/validation/test assignment.

- 40837/40942 deltas (99.74%) are between 0 and 3 seconds; only 71 are 4-5 seconds.
- Endpoint context changes across 9 gaps for radio mode and 12 gaps for cell ID.

## Threshold sensitivity

| Split when gap > (s) | Gaps split | Traces | Blocks | Blocks <10 s | 60-s candidates | Usable duration |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 712 | 21 | 733 | 70 | 1125 | 93.28% |
| 3 | 105 | 20 | 126 | 22 | 803 | 97.26% |
| 5 | 34 | 13 | 55 | 9 | 774 | 97.91% |
| 10 | 8 | 7 | 29 | 3 | 766 | 98.32% |
| 30 | 2 | 2 | 23 | 1 | 764 | 98.53% |

Long-gap bands: 26 at 6-10 s; 6 at 11-30 s; 2 above 30 s.

The source paper describes one-second logging granularity. A 2-second cutoff would split the 607 observed 3-second deltas and over-fragment the corpus. The primary rule conservatively tolerates 4-5 seconds of sampling irregularity and splits every interval of at least 6 seconds. This threshold is fixed from acquisition cadence, before any ABR/QoE comparison; 3- and 10-second alternatives remain sensitivity checks.

## Every gap greater than 5 seconds

| Split | Trace | Gap (s) | Before timestamp | Before mode / kbps | After timestamp | After mode / kbps | Cell change |
|---|---|---:|---|---|---|---|---|
| train | `driving_B_2020.02.13_15.02.01.csv` | 627 | 2020.02.13_15.54.19 | 5G / 4550 | 2020.02.13_16.04.46 | HSPA+ / 1465 | yes |
| train | `driving_B_2020.02.13_13.03.24.csv` | 44 | 2020.02.13_13.17.54 | 5G / 6984 | 2020.02.13_13.18.38 | HSPA+ / 2469 | yes |
| validation | `driving_B_2019.12.16_07.22.43.csv` | 24 | 2019.12.16_07.40.11 | 5G / 1441 | 2019.12.16_07.40.35 | LTE / 1626 | yes |
| train | `driving_B_2020.02.14_12.58.17.csv` | 18 | 2020.02.14_13.04.17 | HSPA+ / 44 | 2020.02.14_13.04.35 | HSPA+ / 15 | yes |
| test | `driving_B_2019.12.16_14.23.32.csv` | 17 | 2019.12.16_15.01.37 | LTE / 11 | 2019.12.16_15.01.54 | LTE / 11 | no |
| train | `driving_B_2019.12.17_07.32.39.csv` | 15 | 2019.12.17_07.57.50 | LTE / 11 | 2019.12.17_07.58.05 | LTE / 11 | no |
| test | `driving_B_2019.12.16_14.23.32.csv` | 11 | 2019.12.16_15.01.54 | LTE / 11 | 2019.12.16_15.02.05 | 5G / 3 | yes |
| train | `static_B_2020.02.27_18.39.27.csv` | 11 | 2020.02.27_18.39.28 | 5G / 12 | 2020.02.27_18.39.39 | 5G / 2044 | no |
| train | `driving_B_2020.02.13_13.03.24.csv` | 10 | 2020.02.13_13.37.53 | HSPA+ / 457 | 2020.02.13_13.38.03 | HSPA+ / 135 | yes |
| train | `static_B_2019.12.16_13.40.04.csv` | 9 | 2019.12.16_14.01.33 | 5G / 55 | 2019.12.16_14.01.42 | 5G / 34 | no |
| validation | `driving_B_2020.02.27_20.35.57.csv` | 9 | 2020.02.27_20.41.21 | 5G / 11 | 2020.02.27_20.41.30 | HSPA+ / 154 | yes |
| test | `driving_B_2019.12.16_14.23.32.csv` | 8 | 2019.12.16_14.54.40 | LTE / 194 | 2019.12.16_14.54.48 | LTE / 34 | no |
| train | `driving_B_2019.12.17_07.32.39.csv` | 8 | 2019.12.17_07.57.42 | LTE / 11 | 2019.12.17_07.57.50 | LTE / 11 | no |
| train | `driving_B_2019.12.17_07.32.39.csv` | 8 | 2019.12.17_07.58.05 | LTE / 11 | 2019.12.17_07.58.13 | 5G / 132 | yes |
| train | `driving_B_2019.12.17_07.32.39.csv` | 8 | 2019.12.17_08.09.48 | LTE / 11 | 2019.12.17_08.09.56 | LTE / 11 | no |
| train | `driving_B_2020.01.16_09.56.56.csv` | 8 | 2020.01.16_10.05.27 | LTE / 1805 | 2020.01.16_10.05.35 | LTE / 1 | no |
| train | `driving_B_2020.02.13_13.03.24.csv` | 8 | 2020.02.13_13.37.39 | LTE / 19083 | 2020.02.13_13.37.47 | HSPA+ / 819 | yes |
| train | `driving_B_2020.02.27_17.30.15.csv` | 8 | 2020.02.27_17.35.00 | HSPA+ / 55 | 2020.02.27_17.35.08 | HSPA+ / 13 | yes |
| train | `static_B_2019.12.16_13.40.04.csv` | 8 | 2019.12.16_14.01.07 | 5G / 11 | 2019.12.16_14.01.15 | 5G / 279 | no |
| validation | `static_B_2020.01.16_10.43.34.csv` | 8 | 2020.01.16_11.10.44 | 5G / 1494 | 2020.01.16_11.10.52 | 5G / 359 | no |
| test | `driving_B_2019.12.16_14.23.32.csv` | 7 | 2019.12.16_15.01.30 | LTE / 11 | 2019.12.16_15.01.37 | LTE / 11 | no |
| train | `driving_B_2020.02.13_13.03.24.csv` | 7 | 2020.02.13_13.39.22 | HSPA+ / 11 | 2020.02.13_13.39.29 | HSPA+ / 100 | no |
| train | `driving_B_2020.02.14_12.58.17.csv` | 7 | 2020.02.14_13.07.06 | HSPA+ / 11 | 2020.02.14_13.07.13 | HSUPA / 11 | yes |
| train | `static_B_2019.12.16_13.40.04.csv` | 7 | 2019.12.16_13.49.52 | 5G / 312 | 2019.12.16_13.49.59 | 5G / 3 | no |
| validation | `driving_B_2019.12.14_10.16.30.csv` | 7 | 2019.12.14_10.28.37 | 5G / 12669 | 2019.12.14_10.28.44 | HSPA+ / 302 | yes |
| train | `driving_B_2020.02.13_13.03.24.csv` | 6 | 2020.02.13_13.38.34 | HSPA+ / 33 | 2020.02.13_13.38.40 | HSPA+ / 44 | no |
| train | `driving_B_2020.02.13_13.03.24.csv` | 6 | 2020.02.13_13.40.56 | HSPA+ / 11 | 2020.02.13_13.41.02 | HSPA+ / 12 | no |
| train | `driving_B_2020.02.13_13.03.24.csv` | 6 | 2020.02.13_13.41.28 | HSPA+ / 55 | 2020.02.13_13.41.34 | HSPA+ / 378 | no |
| train | `driving_B_2020.02.27_17.30.15.csv` | 6 | 2020.02.27_17.30.17 | 5G / 30 | 2020.02.27_17.30.23 | 5G / 773 | no |
| train | `static_B_2019.12.16_13.40.04.csv` | 6 | 2019.12.16_14.00.45 | 5G / 189 | 2019.12.16_14.00.51 | 5G / 114 | no |
| train | `static_B_2019.12.16_13.40.04.csv` | 6 | 2019.12.16_14.02.24 | 5G / 11 | 2019.12.16_14.02.30 | 5G / 336 | no |
| train | `static_B_2019.12.16_13.40.04.csv` | 6 | 2019.12.16_14.16.10 | 5G / 11 | 2019.12.16_14.16.16 | 5G / 246 | no |
| validation | `driving_B_2019.12.14_10.16.30.csv` | 6 | 2019.12.14_10.16.33 | 5G / 9 | 2019.12.14_10.16.39 | 5G / 149 | no |
| validation | `driving_B_2019.12.16_07.22.43.csv` | 6 | 2019.12.16_07.47.36 | 5G / 11 | 2019.12.16_07.47.42 | 5G / 1 | no |

The official dataset describes production-network measurements logged with G-NetTrack Pro and identifies DL_bitrate as application-layer download rate. Neither the paper, repository, nor tool manual specifies an interpolation or forward-fill rule for missing timestamps. The paper permits splitting long traces for experiment needs. This audit's gap threshold is therefore a conservative experiment policy, not a rule supplied by the dataset authors.

Sources: [original MMSys paper](https://cora.ucc.ie/bitstreams/a0782899-2f0b-4741-9493-2b203db55bea/download), [official dataset repository](https://github.com/uccmisl/5Gdataset), and [G-NetTrack manual](https://gyokovsolutions.com/manual-g-nettrack/).
