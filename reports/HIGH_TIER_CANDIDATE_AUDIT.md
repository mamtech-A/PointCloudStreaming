# Static-High registry headroom audit

This read-only audit covers all Very-low-eligible candidate windows and does not modify traces, splits, training, or evaluation.

## Decision rule

- Every included window is tested with the maximum-demand static High G-PCC tier.
- Required sequences: longdress, loot, redandblack, soldier.
- Segment sizes: [5, 8, 10, 15].
- Transport seeds: [42, 43, 44].
- A window is High-supported only when all required cases complete before its last measured timestamp.
- No terminal sample is clamped and no post-trace capacity is invented.

## Summary

- Registry: `e1d52528a7129527`.
- Windows: 759 total; 675 High-supported; 84 unsupported.
- Cases: 36432 total; 3210 require unobserved capacity.
- Minimum successful headroom: 0.111 s.

## Split results

| Split | Windows | High-supported | Unsupported | Failed cases |
|---|---:|---:|---:|---:|
| train | 468 | 415 | 53 | 2076 |
| validation | 137 | 118 | 19 | 699 |
| test | 154 | 142 | 12 | 435 |

## Per-trace results

| Split | Trace | Windows | High-supported | Unsupported | Failed cases |
|---|---|---:|---:|---:|---:|
| validation | `driving_B_2019.12.14_10.16.30.csv` | 17 | 11 | 6 | 240 |
| validation | `driving_B_2019.12.16_07.22.43.csv` | 44 | 41 | 3 | 120 |
| test | `driving_B_2019.12.16_11.49.59.csv` | 7 | 4 | 3 | 84 |
| train | `driving_B_2019.12.16_12.27.05.csv` | 23 | 22 | 1 | 37 |
| test | `driving_B_2019.12.16_14.23.32.csv` | 50 | 46 | 4 | 156 |
| train | `driving_B_2019.12.17_07.32.39.csv` | 44 | 42 | 2 | 56 |
| train | `driving_B_2020.01.16_07.26.43.csv` | 35 | 32 | 3 | 108 |
| train | `driving_B_2020.01.16_09.56.56.csv` | 31 | 29 | 2 | 72 |
| train | `driving_B_2020.01.16_12.10.03.csv` | 7 | 7 | 0 | 0 |
| train | `driving_B_2020.02.13_13.03.24.csv` | 45 | 33 | 12 | 486 |
| train | `driving_B_2020.02.13_15.02.01.csv` | 54 | 51 | 3 | 120 |
| test | `driving_B_2020.02.14_07.29.00.csv` | 38 | 34 | 4 | 147 |
| train | `driving_B_2020.02.14_09.38.22.csv` | 29 | 27 | 2 | 72 |
| train | `driving_B_2020.02.14_12.58.17.csv` | 17 | 5 | 12 | 501 |
| train | `driving_B_2020.02.27_17.30.15.csv` | 15 | 5 | 10 | 396 |
| validation | `driving_B_2020.02.27_20.35.57.csv` | 15 | 6 | 9 | 303 |
| train | `static_B_2019.12.16_13.40.04.csv` | 38 | 32 | 6 | 228 |
| validation | `static_B_2020.01.16_10.43.34.csv` | 61 | 60 | 1 | 36 |
| test | `static_B_2020.02.13_13.57.29.csv` | 59 | 58 | 1 | 48 |
| train | `static_B_2020.02.14_13.21.26.csv` | 19 | 19 | 0 | 0 |
| train | `static_B_2020.02.27_18.39.27.csv` | 111 | 111 | 0 | 0 |

## Unsupported frozen windows

| Window | Duration (s) | Failed cases |
|---|---:|---:|
| `train/driving_B_2019.12.16_12.27.05.csv#block-000@time-1320.000s` | 28.000 | 37 |
| `train/driving_B_2019.12.17_07.32.39.csv#block-003@time-2194.000s` | 35.000 | 8 |
| `train/driving_B_2019.12.17_07.32.39.csv#block-004@time-2597.000s` | 26.000 | 48 |
| `train/driving_B_2020.01.16_07.26.43.csv#block-000@time-1920.000s` | 168.000 | 12 |
| `train/driving_B_2020.01.16_07.26.43.csv#block-000@time-1980.000s` | 108.000 | 48 |
| `train/driving_B_2020.01.16_07.26.43.csv#block-000@time-2040.000s` | 48.000 | 48 |
| `train/driving_B_2020.01.16_09.56.56.csv#block-000@time-420.000s` | 91.000 | 24 |
| `train/driving_B_2020.01.16_09.56.56.csv#block-000@time-480.000s` | 31.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-000@time-720.000s` | 150.000 | 3 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-000@time-780.000s` | 90.000 | 36 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-000@time-840.000s` | 30.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-003@time-2079.000s` | 31.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-004@time-2116.000s` | 42.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-005@time-2165.000s` | 87.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-005@time-2225.000s` | 27.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-007@time-2410.000s` | 256.000 | 24 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-007@time-2470.000s` | 196.000 | 39 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-007@time-2530.000s` | 136.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-007@time-2590.000s` | 76.000 | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-007@time-2650.000s` | 16.000 | 48 |
| `train/driving_B_2020.02.13_15.02.01.csv#block-000@time-3060.000s` | 74.000 | 24 |
| `train/driving_B_2020.02.13_15.02.01.csv#block-000@time-3120.000s` | 14.000 | 48 |
| `train/driving_B_2020.02.13_15.02.01.csv#block-001@time-3761.000s` | 9.000 | 48 |
| `train/driving_B_2020.02.14_09.38.22.csv#block-000@time-1620.000s` | 92.000 | 24 |
| `train/driving_B_2020.02.14_09.38.22.csv#block-000@time-1680.000s` | 32.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-000@time-60.000s` | 300.000 | 21 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-000@time-120.000s` | 240.000 | 24 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-000@time-180.000s` | 180.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-000@time-240.000s` | 120.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-000@time-300.000s` | 60.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-001@time-378.000s` | 151.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-001@time-438.000s` | 91.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-001@time-498.000s` | 31.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-002@time-776.000s` | 218.000 | 24 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-002@time-836.000s` | 158.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-002@time-896.000s` | 98.000 | 48 |
| `train/driving_B_2020.02.14_12.58.17.csv#block-002@time-956.000s` | 38.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-001@time-8.000s` | 277.000 | 24 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-001@time-68.000s` | 217.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-001@time-128.000s` | 157.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-001@time-188.000s` | 97.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-001@time-248.000s` | 37.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-002@time-593.000s` | 292.000 | 12 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-002@time-653.000s` | 232.000 | 24 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-002@time-713.000s` | 172.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-002@time-773.000s` | 112.000 | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-002@time-833.000s` | 52.000 | 48 |
| `train/static_B_2019.12.16_13.40.04.csv#block-000@time-480.000s` | 108.000 | 24 |
| `train/static_B_2019.12.16_13.40.04.csv#block-001@time-1195.000s` | 46.000 | 36 |
| `train/static_B_2019.12.16_13.40.04.csv#block-002@time-1247.000s` | 16.000 | 48 |
| `train/static_B_2019.12.16_13.40.04.csv#block-004@time-1298.000s` | 42.000 | 48 |
| `train/static_B_2019.12.16_13.40.04.csv#block-005@time-2066.000s` | 100.000 | 24 |
| `train/static_B_2019.12.16_13.40.04.csv#block-005@time-2126.000s` | 40.000 | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-001@time-549.000s` | 178.000 | 24 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-001@time-609.000s` | 118.000 | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-001@time-669.000s` | 58.000 | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-854.000s` | 308.000 | 24 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-914.000s` | 248.000 | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-974.000s` | 188.000 | 48 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-000@time-1020.000s` | 28.000 | 48 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-002@time-2519.000s` | 77.000 | 24 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-002@time-2579.000s` | 17.000 | 48 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-000@time-300.000s` | 24.000 | 39 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-393.000s` | 455.000 | 12 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-453.000s` | 395.000 | 24 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-513.000s` | 335.000 | 24 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-573.000s` | 275.000 | 24 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-633.000s` | 215.000 | 36 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-693.000s` | 155.000 | 48 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-753.000s` | 95.000 | 48 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-813.000s` | 35.000 | 48 |
| `validation/static_B_2020.01.16_10.43.34.csv#block-000@time-1560.000s` | 65.000 | 36 |
| `test/driving_B_2019.12.16_11.49.59.csv#block-000@time-240.000s` | 152.000 | 12 |
| `test/driving_B_2019.12.16_11.49.59.csv#block-000@time-300.000s` | 92.000 | 24 |
| `test/driving_B_2019.12.16_11.49.59.csv#block-000@time-360.000s` | 32.000 | 48 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-000@time-1800.000s` | 68.000 | 48 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-000@time-1860.000s` | 8.000 | 48 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-004@time-2853.000s` | 94.000 | 12 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-004@time-2913.000s` | 34.000 | 48 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-2040.000s` | 230.000 | 24 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-2100.000s` | 170.000 | 27 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-2160.000s` | 110.000 | 48 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-2220.000s` | 50.000 | 48 |
| `test/static_B_2020.02.13_13.57.29.csv#block-000@time-3480.000s` | 17.000 | 48 |

The JSON companion retains per-window status, counts, and headroom; case rows are intentionally omitted for the all-candidate audit.
