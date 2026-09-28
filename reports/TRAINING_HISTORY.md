# Training History and Artifact Index

This report is the entry point for reviewing model development from the available evidence on the Mamtech and Nima PCs. The generated machine inventory is in `training_run_inventory.csv` and `training_run_inventory.json`; compact diagnostic records are under `experiments/runs/`.

## Evidence policy

- Raw logs remain on the machine that ran the experiment and are not committed to Git.
- Each discovered raw directory has a compact summary, file/checksum manifest, and sampled training curves.
- Model checkpoints and selected large result artifacts belong in Git LFS; summaries and compact curves remain normal Git files.
- `unknown_or_partial` means the automatic scanner could not prove completion from the retained files. It does not necessarily mean the run failed.
- No metric, configuration, or outcome is reconstructed when its source artifact is missing.

## Development timeline

| Period / evidence | Main purpose | Outcome or status |
|---|---|---|
| 2026-07-10 to 2026-07-14 raw runs | Early smoke/full training iterations | Seven directories retained on Nima; legacy runs lack the modern `RUN.log` and `training.config.json`, so their exact intent and completion require manual reconciliation with Git history. |
| `TRAINING_SUMMARY.md` / `20260718_122736_full` | Registered clean retraining, 8-frame segments | DQN QoE 26.020; Buffer-based 36.165; MPC 38.819. This motivated reward and trace-feasibility investigation. |
| `TRAINING_SUMMARY_qoe_v2_wide.md` | Wide segment search over 5, 8, 10, and 15 | Selected 8 frames; DQN 32.570, Buffer-based 33.360, MPC 37.660. Referenced raw folder `qoe_v2_wide` was not found on either scanned PC. |
| Trace-window feasibility and request-pacing changes | Remove terminal trace clamping, exclude infeasible outages, and replace frame-drop accounting with request pacing | Protocol/code change; later metrics must not be compared to older QoE values without noting formula differences. |
| `TRAINING_SUMMARY_qoe_v4_request_pacing.md` | QoE-v4 wide search with request pacing | Selected 10 frames; DQN 49.946, Buffer-based 45.692, MPC 45.797. Referenced raw folder `20260720_005851_wide` was not found on either scanned PC. |
| `20260721_203138_wide` / startup weight 2.0 | Startup-delay sensitivity at 10 frames | Completed training and registered evaluation. DQN 46.940, Buffer-based 43.261, MPC 43.366 under the weight-2 formula. |
| `20260721_222523_wide` / startup weight 1.5 | Compare startup-delay weight 1.5 using the same registered split and 10-frame setup | Training artifacts and compact curves exist; retained summary reports validation QoE 45.534 ? 0.287. Final baseline/test comparison is not present in that run. |

## Raw-run inventory

| Machine | Run ID | Mode | Files | Size (MiB) | Scanner status |
|---|---|---:|---:|---:|---|
| Nima | 20260710_083903_smoke | smoke | 6 | 0.73 | unknown or partial |
| Nima | 20260710_083943_full | full | 52 | 37.76 | unknown or partial |
| Nima | 20260711_073759_smoke | smoke | 10 | 1.05 | unknown or partial |
| Nima | 20260711_073843_full | full | 100 | 46.73 | unknown or partial |
| Nima | 20260711_130607_full | full | 68 | 30.22 | unknown or partial |
| Nima | 20260714_065805_full | full | 2 | 0.58 | unknown or partial; likely interrupted/partial |
| Nima | 20260714_131016_full | full | 100 | 54.00 | unknown or partial |
| Nima | 20260718_122700_smoke | smoke | 8 | 0.89 | unknown or partial |
| Nima | 20260718_122736_full | full | 76 | 56.13 | unknown or partial; linked to committed summary |
| Mamtech | 20260721_193635_smoke | smoke | 4 | 0.21 | unknown or partial |
| Mamtech | 20260721_193854_smoke | smoke | 7 | 0.22 | failed or interrupted |
| Mamtech | 20260721_195929_smoke | smoke | 29 | 3.10 | completed with evaluation |
| Nima | 20260721_203138_wide | wide | 127 | 61.38 | completed with evaluation |
| Mamtech | 20260721_221816_smoke | smoke | 29 | 3.11 | completed with evaluation |
| Nima | 20260721_222523_wide | wide | 126 | 58.79 | training completed or partial; no final evaluation log |

Total discovered raw evidence: 15 directories, 355.0 MiB. Smoke runs are retained in the index because they explain failures and protocol validation, but they should not be reported as paper results.

## Known gaps and follow-up

1. Locate or formally mark unavailable the raw `qoe_v2_wide` and `20260720_005851_wide` directories.
2. Manually associate the July 10?14 legacy directories with the corresponding Git commits and research questions. Their old layout does not contain enough metadata for safe automatic attribution.
3. Run the registered final evaluation for startup weight 1.5 if it is still a candidate; validation QoE alone is not enough for comparison with weight 2.0.
4. Archive raw directories outside Git with an archive checksum before deleting anything from either training PC.
5. For every future run, retain `RUN.log`, the resolved config/protocol/split, compact curves, final metrics, environment metadata, and the raw-archive checksum using `experiments/SUMMARY_TEMPLATE.md`.

## Comparing runs safely

Compare runs only when the trace registry/split, segment size, reward/QoE definition, evaluation protocol, and baseline configuration match. Startup-weight experiments optimize different numerical objectives, so a lower raw QoE after increasing the startup penalty does not by itself mean the policy is worse. Use the component metrics and registered baseline comparison alongside QoE.
