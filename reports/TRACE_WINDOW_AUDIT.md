# Trace-window feasibility audit

This is a read-only preview. It does not modify trace CSVs, protocol splits, training episodes, checkpoints, or evaluation code.

## Decision rule

- Policy: Static Very-low G-PCC tier.
- Required content: longdress, loot, redandblack, soldier.
- Candidate starts: a 60-second grid restarted inside each contiguous block; no 120-row tail assumption is used.
- Before window generation, each parent trace is split when a timestamp delta exceeds 5 s. No candidate can cross such a gap.
- Feasibility is the intersection of segment sizes [5, 8, 10, 15] and jitter seeds [42, 43, 44].
- A window is eligible only when every required case finishes all frames before measured capacity ends.
- Stalls do not cause exclusion when the session still finishes inside the measured window.
- A trace is retained when at least one candidate window is eligible. No source trace is deleted by this audit.

## Summary

- Traces: 21 total; 21 retained; 0 lack an eligible window.
- Candidate windows: 774 total; 759 eligible; 15 lack sufficient observed support.
- Gap split: 34 discontinuities produce 55 contiguous blocks; 956.0 s of unobserved intervals are excluded from capacity integration.
- Excluded-gap duration uses the full endpoint-to-endpoint delta and assigns no inferred terminal second to the last pre-gap sample.
- Old 120-row rule: 2/24 registered validation/test windows lack sufficient observed support under the strict test.
- Mobility: {'Static': 5, 'Driving': 16, 'Unknown': 0}.
- Duration-weighted RAT composition: 5G 70.5%, HSDPA 0.1%, HSPA+ 8.2%, HSUPA 0.0%, LTE 21.1%.
- Timestamp inventory: 13 traces contain a gap >5 s; the largest is 627.0 s.

Raw CSVs and their parent train/validation/test assignments remain unchanged. Splitting is a read-only audit view; within a block, the previous observation is held only until the next observed timestamp.
A rejected window is not labeled a physical network outage: the audit only establishes that Static Very-low cannot finish without unobserved post-block throughput. It is excluded from ABR ranking and reported separately as insufficient observed support.

## Per-trace results

| Split | Trace | Mobility | Blocks | Usable / raw span (s) | RAT mix | Eligible | Selected | Decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| train | `driving_B_2019.12.16_12.27.05.csv` | Driving | 1 | 1348.0 / 1348.0 | 5G 73.5%, HSPA+ 25.1%, LTE 1.3% | 23/23 | 0 | eligible |
| train | `driving_B_2019.12.17_07.32.39.csv` | Driving | 5 | 2584.0 / 2623.0 | 5G 63.0%, LTE 37.0% | 44/45 | 0 | eligible |
| train | `driving_B_2020.01.16_07.26.43.csv` | Driving | 1 | 2088.0 / 2088.0 | 5G 45.6%, LTE 54.4% | 35/35 | 0 | eligible |
| train | `driving_B_2020.01.16_09.56.56.csv` | Driving | 2 | 1816.0 / 1824.0 | 5G 72.2%, LTE 27.8% | 31/31 | 0 | eligible |
| train | `driving_B_2020.01.16_12.10.03.csv` | Driving | 1 | 381.0 / 381.0 | 5G 100.0% | 7/7 | 0 | eligible |
| train | `driving_B_2020.02.13_13.03.24.csv` | Driving | 8 | 2579.0 / 2666.0 | 5G 34.4%, HSPA+ 22.1%, LTE 43.5% | 45/48 | 0 | eligible |
| train | `driving_B_2020.02.13_15.02.01.csv` | Driving | 2 | 3143.0 / 3770.0 | 5G 71.1%, HSPA+ 0.1%, LTE 28.8% | 54/54 | 0 | eligible |
| train | `driving_B_2020.02.14_09.38.22.csv` | Driving | 1 | 1712.0 / 1712.0 | 5G 31.0%, LTE 69.0% | 29/29 | 0 | eligible |
| train | `driving_B_2020.02.14_12.58.17.csv` | Driving | 3 | 969.0 / 994.0 | 5G 15.9%, HSPA+ 67.3%, HSUPA 1.1%, LTE 15.7% | 17/17 | 0 | eligible |
| train | `driving_B_2020.02.27_17.30.15.csv` | Driving | 3 | 871.0 / 885.0 | 5G 15.3%, HSPA+ 72.7%, LTE 12.1% | 15/16 | 0 | eligible |
| train | `static_B_2019.12.16_13.40.04.csv` | Static | 7 | 2239.0 / 2281.0 | 5G 100.0% | 38/40 | 0 | eligible |
| train | `static_B_2020.02.14_13.21.26.csv` | Static | 1 | 1101.0 / 1101.0 | 5G 100.0% | 19/19 | 0 | eligible |
| train | `static_B_2020.02.27_18.39.27.csv` | Static | 2 | 6642.0 / 6653.0 | 5G 100.0% | 111/112 | 0 | eligible |
| validation | `driving_B_2019.12.14_10.16.30.csv` | Driving | 3 | 1149.0 / 1162.0 | 5G 45.1%, HSPA+ 37.2%, LTE 17.6% | 17/21 | 3 | eligible |
| validation | `driving_B_2019.12.16_07.22.43.csv` | Driving | 3 | 2566.0 / 2596.0 | 5G 58.7%, LTE 41.3% | 44/45 | 3 | eligible |
| validation | `driving_B_2020.02.27_20.35.57.csv` | Driving | 2 | 839.0 / 848.0 | 5G 38.6%, HSPA+ 61.4% | 15/15 | 3 | eligible |
| validation | `static_B_2020.01.16_10.43.34.csv` | Static | 2 | 3666.0 / 3674.0 | 5G 100.0% | 61/63 | 3 | eligible |
| test | `driving_B_2019.12.16_11.49.59.csv` | Driving | 1 | 392.0 / 392.0 | HSDPA 11.0%, HSPA+ 89.0% | 7/7 | 3 | eligible |
| test | `driving_B_2019.12.16_14.23.32.csv` | Driving | 5 | 2904.0 / 2947.0 | 5G 70.9%, LTE 29.1% | 50/50 | 3 | eligible |
| test | `driving_B_2020.02.14_07.29.00.csv` | Driving | 1 | 2270.0 / 2270.0 | 5G 35.9%, HSPA+ 8.9%, LTE 55.2% | 38/38 | 3 | eligible |
| test | `static_B_2020.02.13_13.57.29.csv` | Static | 1 | 3497.0 / 3497.0 | 5G 100.0% | 59/59 | 3 | eligible |

## Selected evaluation windows

| Window | Block | Global start (s) | Measured tail (s) |
|---|---|---:|---:|
| `validation/driving_B_2019.12.14_10.16.30.csv#block-001@time-9.000s` | block-001 | 9.0 | 718.0 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-001@time-489.000s` | block-001 | 489.0 | 238.0 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-974.000s` | block-002 | 974.0 | 188.0 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-000@time-0.000s` | block-000 | 0.0 | 1048.0 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-001@time-1312.000s` | block-001 | 1312.0 | 181.0 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-002@time-2579.000s` | block-002 | 2579.0 | 17.0 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-000@time-0.000s` | block-000 | 0.0 | 324.0 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-393.000s` | block-001 | 393.0 | 455.0 |
| `validation/driving_B_2020.02.27_20.35.57.csv#block-001@time-813.000s` | block-001 | 813.0 | 35.0 |
| `validation/static_B_2020.01.16_10.43.34.csv#block-000@time-0.000s` | block-000 | 0.0 | 1625.0 |
| `validation/static_B_2020.01.16_10.43.34.csv#block-001@time-1813.000s` | block-001 | 1813.0 | 1861.0 |
| `validation/static_B_2020.01.16_10.43.34.csv#block-001@time-3613.000s` | block-001 | 3613.0 | 61.0 |
| `test/driving_B_2019.12.16_11.49.59.csv#block-000@time-0.000s` | block-000 | 0.0 | 392.0 |
| `test/driving_B_2019.12.16_11.49.59.csv#block-000@time-180.000s` | block-000 | 180.0 | 212.0 |
| `test/driving_B_2019.12.16_11.49.59.csv#block-000@time-360.000s` | block-000 | 360.0 | 32.0 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-000@time-0.000s` | block-000 | 0.0 | 1868.0 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-000@time-1440.000s` | block-000 | 1440.0 | 428.0 |
| `test/driving_B_2019.12.16_14.23.32.csv#block-004@time-2913.000s` | block-004 | 2913.0 | 34.0 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-0.000s` | block-000 | 0.0 | 2270.0 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-1080.000s` | block-000 | 1080.0 | 1190.0 |
| `test/driving_B_2020.02.14_07.29.00.csv#block-000@time-2220.000s` | block-000 | 2220.0 | 50.0 |
| `test/static_B_2020.02.13_13.57.29.csv#block-000@time-0.000s` | block-000 | 0.0 | 3497.0 |
| `test/static_B_2020.02.13_13.57.29.csv#block-000@time-1740.000s` | block-000 | 1740.0 | 1757.0 |
| `test/static_B_2020.02.13_13.57.29.csv#block-000@time-3480.000s` | block-000 | 3480.0 | 17.0 |

## Windows with insufficient observed support

| Window | Block | Global start (s) | Measured tail (s) | Reason | Failed cases |
|---|---|---:|---:|---|---:|
| `train/driving_B_2019.12.17_07.32.39.csv#block-000@time-1500.000s` | block-000 | 1500.0 | 3.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-001@time-2054.000s` | block-001 | 2054.0 | 1.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-002@time-2063.000s` | block-002 | 2063.0 | 6.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/driving_B_2020.02.13_13.03.24.csv#block-006@time-2258.000s` | block-006 | 2258.0 | 26.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/driving_B_2020.02.27_17.30.15.csv#block-000@time-0.000s` | block-000 | 0.0 | 2.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/static_B_2019.12.16_13.40.04.csv#block-000@time-540.000s` | block-000 | 540.0 | 48.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/static_B_2019.12.16_13.40.04.csv#block-003@time-1271.000s` | block-003 | 1271.0 | 18.0 | vlow_requires_unobserved_post_block_time | 48 |
| `train/static_B_2020.02.27_18.39.27.csv#block-000@time-0.000s` | block-000 | 0.0 | 1.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-000@time-0.000s` | block-000 | 0.0 | 3.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-1034.000s` | block-002 | 1034.0 | 128.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-1094.000s` | block-002 | 1094.0 | 68.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv#block-002@time-1154.000s` | block-002 | 1154.0 | 8.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/driving_B_2019.12.16_07.22.43.csv#block-001@time-1492.000s` | block-001 | 1492.0 | 1.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/static_B_2020.01.16_10.43.34.csv#block-000@time-1620.000s` | block-000 | 1620.0 | 5.0 | vlow_requires_unobserved_post_block_time | 48 |
| `validation/static_B_2020.01.16_10.43.34.csv#block-001@time-3673.000s` | block-001 | 3673.0 | 1.0 | vlow_requires_unobserved_post_block_time | 48 |

## Old 120-row rule: strict comparison

| Window | Start (s) | Measured tail (s) | Status | Failed cases |
|---|---:|---:|---|---:|
| `legacy-validation/driving_B_2019.12.14_10.16.30.csv@sample-0` | 0.0 | 3.0 | ineligible | 48 |
| `legacy-validation/driving_B_2019.12.14_10.16.30.csv@sample-494` | 528.0 | 199.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.14_10.16.30.csv@sample-988` | 1033.0 | 129.0 | ineligible | 48 |
| `legacy-validation/driving_B_2019.12.16_07.22.43.csv@sample-0` | 0.0 | 1048.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.16_07.22.43.csv@sample-1179` | 1250.0 | 243.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.16_07.22.43.csv@sample-2358` | 2470.0 | 126.0 | eligible | 0 |
| `legacy-validation/driving_B_2020.02.27_20.35.57.csv@sample-0` | 0.0 | 324.0 | eligible | 0 |
| `legacy-validation/driving_B_2020.02.27_20.35.57.csv@sample-360` | 380.0 | 468.0 | eligible | 0 |
| `legacy-validation/driving_B_2020.02.27_20.35.57.csv@sample-720` | 726.0 | 122.0 | eligible | 0 |
| `legacy-validation/static_B_2020.01.16_10.43.34.csv@sample-0` | 0.0 | 1625.0 | eligible | 0 |
| `legacy-validation/static_B_2020.01.16_10.43.34.csv@sample-1554` | 1766.0 | 1908.0 | eligible | 0 |
| `legacy-validation/static_B_2020.01.16_10.43.34.csv@sample-3109` | 3542.0 | 132.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_11.49.59.csv@sample-0` | 0.0 | 392.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_11.49.59.csv@sample-138` | 136.0 | 256.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_11.49.59.csv@sample-277` | 274.0 | 118.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_14.23.32.csv@sample-0` | 0.0 | 1868.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_14.23.32.csv@sample-1308` | 1392.0 | 476.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_14.23.32.csv@sample-2616` | 2821.0 | 126.0 | eligible | 0 |
| `legacy-test/driving_B_2020.02.14_07.29.00.csv@sample-0` | 0.0 | 2270.0 | eligible | 0 |
| `legacy-test/driving_B_2020.02.14_07.29.00.csv@sample-985` | 1062.0 | 1208.0 | eligible | 0 |
| `legacy-test/driving_B_2020.02.14_07.29.00.csv@sample-1970` | 2132.5 | 137.5 | eligible | 0 |
| `legacy-test/static_B_2020.02.13_13.57.29.csv@sample-0` | 0.0 | 3497.0 | eligible | 0 |
| `legacy-test/static_B_2020.02.13_13.57.29.csv@sample-1472` | 1682.0 | 1815.0 | eligible | 0 |
| `legacy-test/static_B_2020.02.13_13.57.29.csv@sample-2945` | 3360.0 | 137.0 | eligible | 0 |

Ineligible windows are reported separately and must not be used to rank ABR algorithms. The registry is not consumed by training/evaluation until the audit is reviewed and a follow-up pipeline change is approved. When activated, results must be macro-averaged per trace so traces with more eligible windows do not receive extra weight.
Training must first sample parent traces uniformly, then sample an eligible block/window inside that parent; otherwise fragmented or long traces would be overrepresented.

Static Very-low eligibility certifies that a window has a feasible action, not that every ABR action will finish. The follow-up runtime guard must record a policy that exhausts an eligible trace as `policy_trace_exhausted`; it must never clamp or silently remove that policy's case.
