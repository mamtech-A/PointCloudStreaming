#!/usr/bin/env python3
"""Per-tier PSNR of the G-PCC ladder vs the original frames.

For each tier's decoded frame (*_dec.ply) computes, against the original PLY:
  - D1 geometry PSNR (MPEG pc_error convention: symmetric point-to-point —
    max of the two directional MSEs — peak = 2^vox - 1 = 1023 for vox10;
    lossless reported as the --psnr-cap value),
  - color Y-PSNR (BT.709 luma, nearest-neighbor color transfer, symmetric,
    peak 255),
plus decoded point count and .bin size. Writes logs/psnr_ladder.csv and prints
a per-tier summary table (mean ± std) for the paper.

NOTE: the TMC13-decoded PLYs list color properties as green,blue,red and end
with an "element face 0" that makes Open3D's RPly reader ABORT mid-vertex
(red channel silently reads as 0!). We therefore parse the ASCII PLYs
ourselves, mapping color channels by property NAME.

Usage:
    python tools/psnr_ladder.py                 # every 10th frame (30 frames)
    python tools/psnr_ladder.py --frames all    # all 300 frames
    python tools/psnr_ladder.py --frames 20     # every 20th frame
"""

import os
import sys
import csv
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TIERS = ['vlow', 'low', 'medlow', 'med', 'medhigh', 'high']
DEFAULT_DATASET = r"F:\Education\Tarbiat Modares\Holography dataset\longdress"
DEFAULT_ORIG = r"F:\Education\Tarbiat Modares\Holography dataset\8iVFBv2\longdress\Ply"
GEOM_PEAK = 1023.0  # vox10
COLOR_PEAK = 255.0


def read_ascii_ply(path):
    """Read an ASCII PLY -> (points Nx3 float64, colors Nx3 uint8 RGB).

    Maps color channels by property NAME (the decoded files are g,b,r order);
    tolerates trailing non-vertex elements (e.g. "element face 0").
    """
    with open(path, 'r', encoding='ascii', errors='replace') as f:
        assert f.readline().strip() == 'ply', f"not a PLY: {path}"
        n_vertex, props, in_vertex = None, [], False
        header_lines = 1  # the 'ply' line
        for line in f:
            header_lines += 1
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'element':
                in_vertex = (tok[1] == 'vertex')
                if in_vertex:
                    n_vertex = int(tok[2])
            elif tok[0] == 'property' and in_vertex and tok[1] != 'list':
                props.append(tok[2])
            elif tok[0] == 'end_header':
                break
    df = pd.read_csv(path, sep=r'\s+', skiprows=header_lines,
                     nrows=n_vertex, header=None, names=props,
                     engine='c', dtype=np.float64)
    assert len(df) == n_vertex, f"{path}: read {len(df)} of {n_vertex} vertices"
    pts = df[['x', 'y', 'z']].to_numpy()
    cols = df[['red', 'green', 'blue']].to_numpy().clip(0, 255).astype(np.uint8)
    return pts, cols


def rgb_to_y(rgb):
    """BT.709 luma from uint8 RGB (float64 out, 0..255 scale)."""
    r, g, b = rgb[:, 0].astype(np.float64), rgb[:, 1].astype(np.float64), rgb[:, 2].astype(np.float64)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def psnr(mse, peak, cap):
    if mse <= 0:
        return cap
    return min(cap, 10.0 * np.log10(peak * peak / mse))


def frame_metrics(orig_pts, orig_y, dec_pts, dec_y, cap):
    """Symmetric D1 geometry PSNR + symmetric Y-PSNR (pc_error style)."""
    tree_o = cKDTree(orig_pts)
    tree_d = cKDTree(dec_pts)
    d_do, i_do = tree_o.query(dec_pts, k=1)   # dec -> orig
    d_od, i_od = tree_d.query(orig_pts, k=1)  # orig -> dec
    geom_mse = max(np.mean(d_do ** 2), np.mean(d_od ** 2))
    y_mse = max(np.mean((dec_y - orig_y[i_do]) ** 2),
                np.mean((orig_y - dec_y[i_od]) ** 2))
    return psnr(geom_mse, GEOM_PEAK, cap), psnr(y_mse, COLOR_PEAK, cap)


def main():
    p = argparse.ArgumentParser(description="G-PCC ladder PSNR vs originals")
    p.add_argument('--dataset-dir', default=DEFAULT_DATASET)
    p.add_argument('--orig-dir', default=DEFAULT_ORIG)
    p.add_argument('--frames', default='10',
                   help="frame stride (e.g. 10 = every 10th) or 'all'")
    p.add_argument('--tiers', default=','.join(TIERS))
    p.add_argument('--psnr-cap', type=float, default=100.0,
                   help='dB value reported for lossless (mse=0)')
    p.add_argument('--out', default=os.path.join('logs', 'psnr_ladder.csv'))
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    tiers = [t.strip() for t in args.tiers.split(',') if t.strip()]
    frame_ids = sorted(int(f.split('_')[2].split('.')[0])
                       for f in os.listdir(args.orig_dir) if f.endswith('.ply'))
    stride = 1 if args.frames == 'all' else int(args.frames)
    frame_ids = frame_ids[::stride]
    print(f"{len(frame_ids)} frames x {len(tiers)} tiers "
          f"(stride {stride}, cap {args.psnr_cap} dB)")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    rows = []
    for k, fid in enumerate(frame_ids):
        orig_pts, orig_rgb = read_ascii_ply(
            os.path.join(args.orig_dir, f"longdress_vox10_{fid}.ply"))
        orig_y = rgb_to_y(orig_rgb)
        for tier in tiers:
            base = os.path.join(args.dataset_dir, f"longdress_vox10_{fid}_{tier}")
            dec_pts, dec_rgb = read_ascii_ply(base + "_dec.ply")
            g_db, y_db = frame_metrics(orig_pts, orig_y, dec_pts,
                                       rgb_to_y(dec_rgb), args.psnr_cap)
            rows.append(dict(frame=fid, tier=tier,
                             d1_psnr_db=round(g_db, 3), y_psnr_db=round(y_db, 3),
                             points=len(dec_pts),
                             bin_bytes=os.path.getsize(base + ".bin")))
        print(f"  [{k + 1}/{len(frame_ids)}] frame {fid} done")

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    print(f"\n{'tier':<9}{'D1 PSNR (dB)':>16}{'Y PSNR (dB)':>16}"
          f"{'points':>12}{'bin size':>12}")
    print('-' * 65)
    for tier in tiers:
        t = df[df.tier == tier]
        print(f"{tier:<9}"
              f"{t.d1_psnr_db.mean():>10.2f} ±{t.d1_psnr_db.std():>4.2f}"
              f"{t.y_psnr_db.mean():>10.2f} ±{t.y_psnr_db.std():>4.2f}"
              f"{t.points.mean():>12,.0f}"
              f"{t.bin_bytes.mean() / 1e3:>10.1f}kB")
    print(f"\n(D1 = symmetric point-to-point geometry PSNR, peak {GEOM_PEAK:.0f}; "
          f"Y = BT.709 luma PSNR, peak 255; {args.psnr_cap:.0f} dB = lossless)")


if __name__ == '__main__':
    main()
