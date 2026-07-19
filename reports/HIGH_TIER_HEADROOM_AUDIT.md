# Static-High registry headroom audit

This read-only audit covers the frozen production registry and does not modify traces, splits, training, or evaluation.

## Decision rule

- Every included window is tested with the maximum-demand static High G-PCC tier.
- Required sequences: longdress, loot, redandblack, soldier.
- Segment sizes: [5, 8, 10, 15].
- Transport seeds: [42, 43, 44].
- A window is High-supported only when all required cases complete before its last measured timestamp.
- No terminal sample is clamped and no post-trace capacity is invented.

## Summary

- Registry: `223c00899f3e86f8`.
- Windows: 437 total; 437 High-supported; 0 unsupported.
- Cases: 20976 total; 0 require unobserved capacity.
- Minimum successful headroom: 5.118 s.

## Split results

| Split | Windows | High-supported | Unsupported | Failed cases |
|---|---:|---:|---:|---:|
| train | 413 | 413 | 0 | 0 |
| validation | 12 | 12 | 0 | 0 |
| test | 12 | 12 | 0 | 0 |

## Per-trace results

| Split | Trace | Windows | High-supported | Unsupported | Failed cases |
|---|---|---:|---:|---:|---:|
| validation | `driving_B_2019.12.14_10.16.30.csv` | 3 | 3 | 0 | 0 |
| validation | `driving_B_2019.12.16_07.22.43.csv` | 3 | 3 | 0 | 0 |
| test | `driving_B_2019.12.16_11.49.59.csv` | 3 | 3 | 0 | 0 |
| train | `driving_B_2019.12.16_12.27.05.csv` | 22 | 22 | 0 | 0 |
| test | `driving_B_2019.12.16_14.23.32.csv` | 3 | 3 | 0 | 0 |
| train | `driving_B_2019.12.17_07.32.39.csv` | 42 | 42 | 0 | 0 |
| train | `driving_B_2020.01.16_07.26.43.csv` | 32 | 32 | 0 | 0 |
| train | `driving_B_2020.01.16_09.56.56.csv` | 28 | 28 | 0 | 0 |
| train | `driving_B_2020.01.16_12.10.03.csv` | 7 | 7 | 0 | 0 |
| train | `driving_B_2020.02.13_13.03.24.csv` | 32 | 32 | 0 | 0 |
| train | `driving_B_2020.02.13_15.02.01.csv` | 51 | 51 | 0 | 0 |
| test | `driving_B_2020.02.14_07.29.00.csv` | 3 | 3 | 0 | 0 |
| train | `driving_B_2020.02.14_09.38.22.csv` | 27 | 27 | 0 | 0 |
| train | `driving_B_2020.02.14_12.58.17.csv` | 5 | 5 | 0 | 0 |
| train | `driving_B_2020.02.27_17.30.15.csv` | 5 | 5 | 0 | 0 |
| validation | `driving_B_2020.02.27_20.35.57.csv` | 3 | 3 | 0 | 0 |
| train | `static_B_2019.12.16_13.40.04.csv` | 32 | 32 | 0 | 0 |
| validation | `static_B_2020.01.16_10.43.34.csv` | 3 | 3 | 0 | 0 |
| test | `static_B_2020.02.13_13.57.29.csv` | 3 | 3 | 0 | 0 |
| train | `static_B_2020.02.14_13.21.26.csv` | 19 | 19 | 0 | 0 |
| train | `static_B_2020.02.27_18.39.27.csv` | 111 | 111 | 0 | 0 |

## Unsupported frozen windows

| Window | Duration (s) | Failed cases |
|---|---:|---:|
| None | — | — |

The JSON companion retains every case outcome and headroom value.
