"""One-time offline preparation of the Irish 5G Download traces.

Source: Raca, Leahy, Sreenan, Quinlan — "Beyond Throughput, The Next Generation:
a 5G Dataset with Channel and Context Metrics" (ACM MMSys 2020),
https://github.com/uccmisl/5Gdataset (GPL-3.0). Raw zip extracted under
data/5g_raw/ (gitignored); only the cleaned traces in bandwidth_5g/ are tracked.

Only the Download application traces are used: a file download saturates the
link, so DL_bitrate approximates the access-link capacity. The Netflix /
Amazon_Prime traces are application-limited (throughput is capped by the video
bitrate, not the link) and are excluded.

Cleaning policy (per file):
- Keep only rows with State == 'D' (downloading). Idle rows mean "nothing was
  being fetched", NOT zero capacity — and a 0-bps sample would hit the
  zero-capacity fallback in tcp_protocol.py (0 => unconstrained send).
- Drop remaining rows with DL_bitrate <= 0 or non-numeric ('-') for the same
  reason; assert the cleaned minimum is > 0.
- KEEP NetworkMode 4G-fallback periods: realistic NSA behavior; the per-file
  %5G is printed so the mix is visible.
- Row order is preserved; dropping idle gaps creates time discontinuities,
  which is acceptable because the simulator consumes samples positionally
  (one per frame), not by wall clock.
- STATIC mobility traces only (2026-07): the Driving traces were removed from
  the corpus as unsuitable for the point-cloud streaming scenario (handover
  churn + HSPA+/4G fallback periods). Recover via git history if needed.
- Drop files with fewer than MIN_SAMPLES cleaned rows (matches the 300-frame
  episode length; avoids flat clamped tails). No concatenation of short files
  and no chunking of long ones — either would leak data across the file-level
  train/test split.
- ALL original G-NetTrack columns are kept (RSRP/SNR/CQI/PING* enable future
  multivariate work). The simulator's loader only reads DL_bitrate (kbps).

Regenerate with:  python prepare_5g_traces.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import shutil

import numpy as np
import pandas as pd

RAW_DIR = os.path.join('data', '5g_raw', '5G-production-dataset', 'Download')
OUT_DIR = 'bandwidth_5g'
LICENSE_SRC = os.path.join('data', '5g_raw', 'LICENSE')
MIN_SAMPLES = 300          # one cleaned sample per frame of the 300-frame episode
UNIT_SANITY_MBPS = (100.0, 1000.0)  # dataset-wide max must land here (kbps misread guard)


def clean_file(path):
    """Return (cleaned_df, stats) for one raw Download CSV."""
    df = pd.read_csv(path, low_memory=False)
    dl = pd.to_numeric(df['DL_bitrate'], errors='coerce')
    keep = (df['State'].astype(str) == 'D') & (dl > 0)
    out = df[keep].copy()
    kept_dl = dl[keep]
    nm = df.loc[keep, 'NetworkMode'].astype(str)
    stats = {
        'raw_rows': len(df),
        'kept_rows': len(out),
        'min_mbps': kept_dl.min() / 1000.0 if len(out) else 0.0,
        'mean_mbps': kept_dl.mean() / 1000.0 if len(out) else 0.0,
        'max_mbps': kept_dl.max() / 1000.0 if len(out) else 0.0,
        'pct_5g': 100.0 * (nm == '5G').mean() if len(out) else 0.0,
    }
    return out, stats


def main():
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(
            f"{RAW_DIR} not found - download and extract 5G-production-dataset.zip "
            f"from https://github.com/uccmisl/5Gdataset first.")

    os.makedirs(OUT_DIR, exist_ok=True)

    rows, all_kept_mbps, dropped = [], [], []
    for mob in ('Static',):  # Driving excluded (see module docstring)
        src_dir = os.path.join(RAW_DIR, mob)
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith('.csv'):
                continue
            cleaned, st = clean_file(os.path.join(src_dir, fname))
            out_name = f"{mob.lower()}_{fname}"
            if st['kept_rows'] < MIN_SAMPLES:
                dropped.append((out_name, st['kept_rows']))
                continue
            assert st['min_mbps'] > 0, f"{out_name}: zero-bps sample survived cleaning"
            cleaned.to_csv(os.path.join(OUT_DIR, out_name), index=False)
            all_kept_mbps.append(pd.to_numeric(cleaned['DL_bitrate']) / 1000.0)
            rows.append((out_name, st))

    print(f"{'file':44}{'raw':>6}{'kept':>6}{'minMbps':>9}{'meanMbps':>10}{'maxMbps':>9}{'%5G':>7}")
    for name, st in rows:
        print(f"{name:44}{st['raw_rows']:>6}{st['kept_rows']:>6}"
              f"{st['min_mbps']:>9.3f}{st['mean_mbps']:>10.1f}{st['max_mbps']:>9.1f}{st['pct_5g']:>7.1f}")
    for name, n in dropped:
        print(f"{name:44} DROPPED ({n} < {MIN_SAMPLES} cleaned samples)")

    allv = pd.concat(all_kept_mbps)
    ds_max = allv.max()
    lo, hi = UNIT_SANITY_MBPS
    assert lo <= ds_max <= hi, (
        f"dataset max {ds_max:.1f} Mbps outside sanity band [{lo}, {hi}] - "
        f"check the DL_bitrate unit (expected kbps)")

    # Rolling 5-sample std informs the tput_std_mbps normalization constant.
    roll_std = pd.concat([s.rolling(5).std().dropna() for s in all_kept_mbps])
    print(f"\nfiles written: {len(rows)} (dropped {len(dropped)}) | total samples: {int(allv.size)}")
    print(f"DL_bitrate (Mbps): mean={allv.mean():.1f} median={allv.median():.1f} "
          f"p90={allv.quantile(0.9):.1f} p99={allv.quantile(0.99):.1f} "
          f"p99.5={allv.quantile(0.995):.1f} max={ds_max:.1f}")
    print(f"rolling5 std (Mbps): p90={np.percentile(roll_std, 90):.1f} p99={np.percentile(roll_std, 99):.1f}")
    print("norm guidance: features.DEFAULT_NORM bw_mbps ~ p99.5, tput_std_mbps ~ std p99")

    if os.path.exists(LICENSE_SRC):
        shutil.copyfile(LICENSE_SRC, os.path.join(OUT_DIR, 'LICENSE'))
        print(f"copied dataset LICENSE -> {OUT_DIR}/LICENSE")
    else:
        print(f"WARNING: {LICENSE_SRC} missing - fetch it from the dataset repo "
              f"so {OUT_DIR}/LICENSE ships with the traces")


if __name__ == '__main__':
    main()
