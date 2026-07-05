# bandwidth_5g — cleaned Irish 5G Download traces

Cleaned access-link capacity traces for the point-cloud streaming simulator.
One CSV row ≈ one second; the simulator's loader reads only `DL_bitrate` (kbps)
and consumes one sample per frame, positionally.

## Provenance

- Source: **"Beyond Throughput, The Next Generation: a 5G Dataset with Channel
  and Context Metrics"** — D. Raca, D. Leahy, C.J. Sreenan, J. Quinlan,
  ACM MMSys 2020. https://github.com/uccmisl/5Gdataset
- License: GPL-3.0 (see `LICENSE` in this folder, copied from the dataset repo).
- Collected on a production Irish 5G (NSA) network with G-NetTrack Pro, 2019–2020.
- Only the **Download** application traces are used (a saturating file download
  makes `DL_bitrate` ≈ link capacity). Netflix / Amazon_Prime traces are
  app-limited (capped by video bitrate, not the link) and excluded.
- File naming: `driving_*` / `static_*` prefix = mobility pattern of the source
  folder; the rest is the original filename.

## Cleaning policy (applied by `prepare_5g_traces.py`, one-time)

1. Keep only rows with `State == 'D'` (downloading). Idle rows are app-idle,
   not zero capacity — and 0-bps samples would trigger the zero-capacity
   fallback in `tcp_protocol.py` (0 ⇒ treated as unconstrained).
2. Drop remaining rows with `DL_bitrate <= 0` or non-numeric (`-`).
3. **Kept**: NetworkMode 4G-fallback periods (realistic NSA behavior) and the
   deep-fade tail (~13% of samples < 1 Mbps — handovers/fades while driving;
   real measurements, and low ladder tiers remain fetchable through them).
4. Row order preserved; dropped idle gaps create wall-clock discontinuities,
   acceptable because samples are consumed positionally.
5. Files with < 300 cleaned samples would be dropped (none were; min is 364).
   No concatenation or chunking — either would leak data across the
   file-level train/test split.
6. All original G-NetTrack columns are kept for future multivariate work
   (RSRP/RSRQ/SNR/CQI, PING*, cell IDs, GPS).

## Key statistics (21 files, 40,963 samples)

- `DL_bitrate` (Mbps): mean 44.0, median 13.6, p90 161.4, p99.5 307.4, max 532.9
  → `features.DEFAULT_NORM['bw_mbps'] = 300`.
- Rolling-5 std (Mbps): p90 66.4, p99 123.4
  → `features.DEFAULT_NORM['tput_std_mbps'] = 125`.
- `PINGAVG` (ms), 5G mode, not load-inflated: median 72, p10 66, p90 83
  → simulator RTT config **72 ms ± 8 jitter** (dataset-derived; replaces the
  old modeled 50 ms ± 10).

## Regeneration

```
python prepare_5g_traces.py
```
(Requires the raw zip extracted at `data/5g_raw/5G-production-dataset/`;
download from the dataset repo above.)
