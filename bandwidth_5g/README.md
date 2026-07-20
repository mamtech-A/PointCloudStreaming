# bandwidth_5g — cleaned Irish 5G Download traces (STATIC only)

Cleaned access-link capacity traces for the point-cloud streaming simulator.
One CSV row ≈ one second; the simulator's loader reads `DL_bitrate` (kbps) and
`Timestamp`, and consumes the trace on a **wall-clock time axis** (capacity is
piecewise-constant over the timestamp-derived sample windows; see
`src/network_model/trace.py`).

## Provenance

- Source: **"Beyond Throughput, The Next Generation: a 5G Dataset with Channel
  and Context Metrics"** — D. Raca, D. Leahy, C.J. Sreenan, J. Quinlan,
  ACM MMSys 2020. https://github.com/uccmisl/5Gdataset
- License: GPL-3.0 (see `LICENSE` in this folder, copied from the dataset repo).
- Collected on a production Irish 5G (NSA) network with G-NetTrack Pro, 2019–2020.
- Only the **Download** application traces are used (a saturating file download
  makes `DL_bitrate` ≈ link capacity). Netflix / Amazon_Prime traces are
  app-limited (capped by video bitrate, not the link) and excluded.
- **Only the Static mobility traces are used** (5 files, all 100% 5G
  NetworkMode). The Driving traces were removed 2026-07 as unsuitable for the
  point-cloud streaming scenario (handover churn + HSPA+/4G fallback periods);
  recover them from git history or the dataset repo if ever needed.

## Cleaning policy (applied by `prepare_5g_traces.py`, one-time)

1. Keep only rows with `State == 'D'` (downloading). Idle rows are app-idle,
   not zero capacity — and 0-bps samples would trigger the zero-capacity
   fallback in `tcp_protocol.py` (0 ⇒ treated as unconstrained).
2. Drop remaining rows with `DL_bitrate <= 0` or non-numeric (`-`).
3. **Kept**: the deep-fade tail (samples < 1 Mbps — real measurements; low
   ladder tiers remain fetchable through them).
4. Row order preserved. Dropped idle rows create wall-clock gaps in `Timestamp`;
   the loader handles them as **hold-previous** windows (a sample stays in
   effect until the next one starts). Duplicate timestamps (sub-second rows)
   split their shared second evenly.
5. Files with < 300 cleaned samples would be dropped (none were).
   No concatenation or chunking — either would leak data across the
   file-level train/test split.
6. All original G-NetTrack columns are kept for future multivariate work
   (RSRP/RSRQ/SNR/CQI, PING*, cell IDs, GPS).

## The 5 static traces

| file | rows | wall-clock span |
|---|---|---|
| static_B_2019.12.16_13.40.04.csv | 2024 | ~2281 s |
| static_B_2020.01.16_10.43.34.csv | 3229 | ~3674 s |
| static_B_2020.02.13_13.57.29.csv | 3065 | ~3497 s |
| static_B_2020.02.14_13.21.26.csv |  970 | ~1101 s |
| static_B_2020.02.27_18.39.27.csv | 5862 | ~6653 s |

Timestamp delta counts across all 5: 0 s ×965, 1 s ×11478, 2 s ×2445,
3 s ×224, 4–11 s ×32 (dominant cadence 1 Hz).

Canonical file-level split (seed 42, test_size 0.2 ⇒ 4 train / 1 test):
**held-out test = `static_B_2020.01.16_10.43.34.csv`** (all run/compare/eval
defaults point at it).

- `PINGAVG` (ms), 5G mode, not load-inflated: median 72, p10 66, p90 83
  → simulator RTT config **72 ms ± 8 jitter** (dataset-derived).

## Regeneration

```
python prepare_5g_traces.py
```
(Requires the raw zip extracted at `data/5g_raw/5G-production-dataset/`;
download from the dataset repo above. The script now processes the Static
folder only.)
