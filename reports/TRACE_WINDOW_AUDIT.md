# Trace-window feasibility audit

This is a read-only preview. It does not modify trace CSVs, protocol splits, training episodes, checkpoints, or evaluation code.

## Decision rule

- Policy: Static Very-low G-PCC tier.
- Required content: longdress, loot, redandblack, soldier.
- Candidate starts: a 60-second grid on each trace's real timestamp axis; no 120-row tail assumption is used.
- Feasibility is the intersection of segment sizes [5, 8, 10, 15] and jitter seeds [42, 43, 44].
- A window is eligible only when every required case finishes all frames before measured capacity ends.
- Stalls do not cause exclusion when the session still finishes inside the measured window.
- A trace is retained when at least one candidate window is eligible. No source trace is deleted by this audit.

## Summary

- Traces: 21 total; 21 retained; 0 classified as outages.
- Candidate windows: 773 total; 769 eligible; 4 outages.
- Old 120-row rule: 1/24 registered validation/test windows are outages under the strict test.
- Mobility: {'Static': 5, 'Driving': 16, 'Unknown': 0}.
- Duration-weighted RAT composition: 5G 70.8%, HSDPA 0.1%, HSPA+ 8.2%, HSUPA 0.0%, LTE 20.9%.
- Timestamp-gap warning: 13 traces contain a gap >5 s; the largest is 627.0 s.

> **Audit caveat:** the current simulator holds the previous measured capacity between timestamps. This report prevents all capacity use after the final timestamp, but it does not resolve long internal gaps. A maximum-gap/splitting rule must be approved before the registry is activated.

## Per-trace results

| Split | Trace | Mobility | Duration (s) | RAT mix | Eligible | Selected | Decision |
|---|---|---:|---:|---|---:|---:|---|
| train | `driving_B_2019.12.16_12.27.05.csv` | Driving | 1348.0 | 5G 73.5%, HSPA+ 25.1%, LTE 1.3% | 23/23 | 0 | eligible |
| train | `driving_B_2019.12.17_07.32.39.csv` | Driving | 2623.0 | 5G 62.1%, LTE 37.9% | 44/44 | 0 | eligible |
| train | `driving_B_2020.01.16_07.26.43.csv` | Driving | 2088.0 | 5G 45.6%, LTE 54.4% | 35/35 | 0 | eligible |
| train | `driving_B_2020.01.16_09.56.56.csv` | Driving | 1824.0 | 5G 71.9%, LTE 28.1% | 31/31 | 0 | eligible |
| train | `driving_B_2020.01.16_12.10.03.csv` | Driving | 381.0 | 5G 100.0% | 7/7 | 0 | eligible |
| train | `driving_B_2020.02.13_13.03.24.csv` | Driving | 2666.0 | 5G 35.0%, HSPA+ 22.7%, LTE 42.3% | 45/45 | 0 | eligible |
| train | `driving_B_2020.02.13_15.02.01.csv` | Driving | 3770.0 | 5G 75.9%, HSPA+ 0.1%, LTE 24.0% | 63/63 | 0 | eligible |
| train | `driving_B_2020.02.14_09.38.22.csv` | Driving | 1712.0 | 5G 31.0%, LTE 69.0% | 29/29 | 0 | eligible |
| train | `driving_B_2020.02.14_12.58.17.csv` | Driving | 994.0 | 5G 15.5%, HSPA+ 68.1%, HSUPA 1.1%, LTE 15.3% | 17/17 | 0 | eligible |
| train | `driving_B_2020.02.27_17.30.15.csv` | Driving | 885.0 | 5G 15.7%, HSPA+ 72.4%, LTE 11.9% | 15/15 | 0 | eligible |
| train | `static_B_2019.12.16_13.40.04.csv` | Static | 2281.0 | 5G 100.0% | 38/39 | 0 | eligible |
| train | `static_B_2020.02.14_13.21.26.csv` | Static | 1101.0 | 5G 100.0% | 19/19 | 0 | eligible |
| train | `static_B_2020.02.27_18.39.27.csv` | Static | 6653.0 | 5G 100.0% | 111/111 | 0 | eligible |
| validation | `driving_B_2019.12.14_10.16.30.csv` | Driving | 1162.0 | 5G 45.7%, HSPA+ 36.8%, LTE 17.4% | 17/20 | 3 | eligible |
| validation | `driving_B_2019.12.16_07.22.43.csv` | Driving | 2596.0 | 5G 59.1%, LTE 40.9% | 44/44 | 3 | eligible |
| validation | `driving_B_2020.02.27_20.35.57.csv` | Driving | 848.0 | 5G 39.3%, HSPA+ 60.7% | 15/15 | 3 | eligible |
| validation | `static_B_2020.01.16_10.43.34.csv` | Static | 3674.0 | 5G 100.0% | 62/62 | 3 | eligible |
| test | `driving_B_2019.12.16_11.49.59.csv` | Driving | 392.0 | HSDPA 11.0%, HSPA+ 89.0% | 7/7 | 3 | eligible |
| test | `driving_B_2019.12.16_14.23.32.csv` | Driving | 2947.0 | 5G 69.9%, LTE 30.1% | 50/50 | 3 | eligible |
| test | `driving_B_2020.02.14_07.29.00.csv` | Driving | 2270.0 | 5G 35.9%, HSPA+ 8.9%, LTE 55.2% | 38/38 | 3 | eligible |
| test | `static_B_2020.02.13_13.57.29.csv` | Static | 3497.0 | 5G 100.0% | 59/59 | 3 | eligible |

## Selected evaluation windows

| Window | Start (s) | Measured tail (s) |
|---|---:|---:|
| `validation/driving_B_2019.12.14_10.16.30.csv@time-0.000s` | 0.0 | 1162.0 |
| `validation/driving_B_2019.12.14_10.16.30.csv@time-480.000s` | 480.0 | 682.0 |
| `validation/driving_B_2019.12.14_10.16.30.csv@time-960.000s` | 960.0 | 202.0 |
| `validation/driving_B_2019.12.16_07.22.43.csv@time-0.000s` | 0.0 | 2596.0 |
| `validation/driving_B_2019.12.16_07.22.43.csv@time-1260.000s` | 1260.0 | 1336.0 |
| `validation/driving_B_2019.12.16_07.22.43.csv@time-2580.000s` | 2580.0 | 16.0 |
| `validation/driving_B_2020.02.27_20.35.57.csv@time-0.000s` | 0.0 | 848.0 |
| `validation/driving_B_2020.02.27_20.35.57.csv@time-420.000s` | 420.0 | 428.0 |
| `validation/driving_B_2020.02.27_20.35.57.csv@time-840.000s` | 840.0 | 8.0 |
| `validation/static_B_2020.01.16_10.43.34.csv@time-0.000s` | 0.0 | 3674.0 |
| `validation/static_B_2020.01.16_10.43.34.csv@time-1800.000s` | 1800.0 | 1874.0 |
| `validation/static_B_2020.01.16_10.43.34.csv@time-3660.000s` | 3660.0 | 14.0 |
| `test/driving_B_2019.12.16_11.49.59.csv@time-0.000s` | 0.0 | 392.0 |
| `test/driving_B_2019.12.16_11.49.59.csv@time-180.000s` | 180.0 | 212.0 |
| `test/driving_B_2019.12.16_11.49.59.csv@time-360.000s` | 360.0 | 32.0 |
| `test/driving_B_2019.12.16_14.23.32.csv@time-0.000s` | 0.0 | 2947.0 |
| `test/driving_B_2019.12.16_14.23.32.csv@time-1440.000s` | 1440.0 | 1507.0 |
| `test/driving_B_2019.12.16_14.23.32.csv@time-2940.000s` | 2940.0 | 7.0 |
| `test/driving_B_2020.02.14_07.29.00.csv@time-0.000s` | 0.0 | 2270.0 |
| `test/driving_B_2020.02.14_07.29.00.csv@time-1080.000s` | 1080.0 | 1190.0 |
| `test/driving_B_2020.02.14_07.29.00.csv@time-2220.000s` | 2220.0 | 50.0 |
| `test/static_B_2020.02.13_13.57.29.csv@time-0.000s` | 0.0 | 3497.0 |
| `test/static_B_2020.02.13_13.57.29.csv@time-1740.000s` | 1740.0 | 1757.0 |
| `test/static_B_2020.02.13_13.57.29.csv@time-3480.000s` | 3480.0 | 17.0 |

## Outage windows

| Window | Start (s) | Measured tail (s) | Reason | Failed cases |
|---|---:|---:|---|---:|
| `train/static_B_2019.12.16_13.40.04.csv@time-2280.000s` | 2280.0 | 1.0 | vlow_exceeds_trace_end | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv@time-1020.000s` | 1020.0 | 142.0 | vlow_exceeds_trace_end | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv@time-1080.000s` | 1080.0 | 82.0 | vlow_exceeds_trace_end | 48 |
| `validation/driving_B_2019.12.14_10.16.30.csv@time-1140.000s` | 1140.0 | 22.0 | vlow_exceeds_trace_end | 48 |

## Old 120-row rule: strict comparison

| Window | Start (s) | Measured tail (s) | Status | Failed cases |
|---|---:|---:|---|---:|
| `legacy-validation/driving_B_2019.12.14_10.16.30.csv@sample-0` | 0.0 | 1162.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.14_10.16.30.csv@sample-494` | 528.0 | 634.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.14_10.16.30.csv@sample-988` | 1033.0 | 129.0 | outage | 48 |
| `legacy-validation/driving_B_2019.12.16_07.22.43.csv@sample-0` | 0.0 | 2596.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.16_07.22.43.csv@sample-1179` | 1250.0 | 1346.0 | eligible | 0 |
| `legacy-validation/driving_B_2019.12.16_07.22.43.csv@sample-2358` | 2470.0 | 126.0 | eligible | 0 |
| `legacy-validation/driving_B_2020.02.27_20.35.57.csv@sample-0` | 0.0 | 848.0 | eligible | 0 |
| `legacy-validation/driving_B_2020.02.27_20.35.57.csv@sample-360` | 380.0 | 468.0 | eligible | 0 |
| `legacy-validation/driving_B_2020.02.27_20.35.57.csv@sample-720` | 726.0 | 122.0 | eligible | 0 |
| `legacy-validation/static_B_2020.01.16_10.43.34.csv@sample-0` | 0.0 | 3674.0 | eligible | 0 |
| `legacy-validation/static_B_2020.01.16_10.43.34.csv@sample-1554` | 1766.0 | 1908.0 | eligible | 0 |
| `legacy-validation/static_B_2020.01.16_10.43.34.csv@sample-3109` | 3542.0 | 132.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_11.49.59.csv@sample-0` | 0.0 | 392.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_11.49.59.csv@sample-138` | 136.0 | 256.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_11.49.59.csv@sample-277` | 274.0 | 118.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_14.23.32.csv@sample-0` | 0.0 | 2947.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_14.23.32.csv@sample-1308` | 1392.0 | 1555.0 | eligible | 0 |
| `legacy-test/driving_B_2019.12.16_14.23.32.csv@sample-2616` | 2821.0 | 126.0 | eligible | 0 |
| `legacy-test/driving_B_2020.02.14_07.29.00.csv@sample-0` | 0.0 | 2270.0 | eligible | 0 |
| `legacy-test/driving_B_2020.02.14_07.29.00.csv@sample-985` | 1062.0 | 1208.0 | eligible | 0 |
| `legacy-test/driving_B_2020.02.14_07.29.00.csv@sample-1970` | 2132.5 | 138.0 | eligible | 0 |
| `legacy-test/static_B_2020.02.13_13.57.29.csv@sample-0` | 0.0 | 3497.0 | eligible | 0 |
| `legacy-test/static_B_2020.02.13_13.57.29.csv@sample-1472` | 1682.0 | 1815.0 | eligible | 0 |
| `legacy-test/static_B_2020.02.13_13.57.29.csv@sample-2945` | 3360.0 | 137.0 | eligible | 0 |

Outage windows are reported separately and must not be used to rank ABR algorithms. The registry is not consumed by training/evaluation until the audit is reviewed and a follow-up pipeline change is approved. When activated, results must be macro-averaged per trace so traces with more eligible windows do not receive extra weight.

Static Very-low eligibility certifies that a window has a feasible action, not that every ABR action will finish. The follow-up runtime guard must record a policy that exhausts an eligible trace as `policy_trace_exhausted`; it must never clamp or silently remove that policy's case.
