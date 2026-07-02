#!/usr/bin/env python3
"""Encode point-cloud frames with real MPEG G-PCC (TMC13) into a coded manifest.

For each input .ply frame, encodes 3 representations (a rate ladder) with the
locally-built tmc3, decodes each to measure reconstructed point count (fidelity),
and writes:
  - gpcc/coded_frames.json  (detailed per-frame/per-rep sizes + densities)
  - config/mpd_gpcc.xml     (simulator-compatible manifest with REAL coded sizes)

The simulator then streams these real coded bitstream sizes (see
network_model.manifest.coded_size_bytes, which prefers the manifest values).

Features for large runs (300 frames x 3 tiers = 900 encodes):
  - cross-platform tmc3 discovery (tmc3.exe / tmc3) + --tmc3 override
  - --dir FOLDER input (all *.ply sorted = playback order)
  - --jobs N parallel encoding/decoding (multiprocessing)
  - resume: existing non-empty outputs are reused unless --force

Usage:
    python gpcc/encode_frames.py --dir "F:/path/to/longdress/Ply" --jobs 8
    python gpcc/encode_frames.py PATH1.ply PATH2.ply ...
    python gpcc/encode_frames.py --list frames.txt          # one .ply path per line
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import multiprocessing
import xml.etree.ElementTree as ET
from xml.dom import minidom

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

FPS = 30.0
# rep_id, label, positionQuantizationScale, attribute qp  (chosen to span the 4G traces:
# @30fps ladder ~ high 102 / med 35.5 / low 3.4 Mbps on longdress_vox10)
TIERS = [
    (0, 'high', 1.0, 22),
    (1, 'med', 0.75, 34),
    (2, 'low', 0.25, 46),
]


def find_tmc3(override=None):
    """Locate the tmc3 binary (cross-platform), or exit with build instructions."""
    if override:
        if os.path.isfile(override):
            return os.path.abspath(override)
        sys.exit(f"--tmc3 path not found: {override}")
    names = ['tmc3.exe', 'tmc3'] if os.name == 'nt' else ['tmc3', 'tmc3.exe']
    roots = [
        os.path.join(HERE, 'mpeg-pcc-tmc13', 'build', 'tmc3'),
        os.path.join(HERE, 'mpeg-pcc-tmc13', 'build'),
    ]
    for root in roots:
        for name in names:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    for root in roots:
        if os.path.isdir(root):
            hits = glob.glob(os.path.join(root, '**', 'tmc3*'), recursive=True)
            hits = [h for h in hits if os.path.basename(h) in names]
            if hits:
                return hits[0]
    sys.exit("tmc3 not found under gpcc/mpeg-pcc-tmc13/build/ — build it first "
             "(see gpcc/README.md) or pass --tmc3 PATH.")


def ply_vertex_count(path):
    with open(path, 'rb') as f:
        for _ in range(60):
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.lower().startswith('element vertex'):
                return int(line.split()[2])
            if line == 'end_header':
                break
    return 0


def encode_one(task):
    """Worker: encode (and optionally decode) ONE (frame, tier). Picklable.

    task keys: tmc3, ply, out_bin, dec_ply, scale, qp, decode, force, key
    Returns (key, coded_bytes, density_or_None, skipped_encode, error_or_None).
    """
    tmc3 = task['tmc3']
    try:
        skipped = False
        if (not task['force'] and os.path.isfile(task['out_bin'])
                and os.path.getsize(task['out_bin']) > 0):
            skipped = True  # resume: bitstream already encoded
        else:
            subprocess.run([
                tmc3, '--mode=0',
                f"--uncompressedDataPath={task['ply']}",
                f"--compressedStreamPath={task['out_bin']}",
                '--mergeDuplicatedPoints=1',
                f"--positionQuantizationScale={task['scale']}",
                '--convertPlyColourspace=1', '--transformType=0',
                f"--qp={task['qp']}", '--bitdepth=8', '--attribute=color',
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        coded_bytes = os.path.getsize(task['out_bin'])

        density = None
        if task['decode']:
            if (not task['force'] and os.path.isfile(task['dec_ply'])
                    and os.path.getsize(task['dec_ply']) > 0):
                density = ply_vertex_count(task['dec_ply'])
            if not density:
                subprocess.run([
                    tmc3, '--mode=1',
                    f"--compressedStreamPath={task['out_bin']}",
                    f"--reconstructedDataPath={task['dec_ply']}",
                    '--convertPlyColourspace=1',
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                density = ply_vertex_count(task['dec_ply'])
        return (task['key'], coded_bytes, density, skipped, None)
    except Exception as e:  # report, don't kill the pool
        return (task['key'], 0, None, False, f"{type(e).__name__}: {e}")


def collect_plys(args):
    plys = list(args.plys)
    if args.list:
        with open(args.list) as f:
            plys += [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    if args.dir:
        found = sorted(glob.glob(os.path.join(args.dir, '*.ply')))
        if not found:
            sys.exit(f"no .ply files in {args.dir}")
        plys += found
    return plys


def main():
    p = argparse.ArgumentParser(description="Encode .ply frames with G-PCC into a coded manifest")
    p.add_argument('plys', nargs='*', help='.ply frame paths (in playback order)')
    p.add_argument('--list', help='file with one .ply path per line')
    p.add_argument('--dir', help='folder: all *.ply sorted by name = playback order')
    p.add_argument('--jobs', type=int, default=1, help='parallel encode workers (default 1)')
    p.add_argument('--force', action='store_true', help='re-encode even if outputs exist')
    p.add_argument('--no-decode', action='store_true', help='skip decode (density = source point count)')
    p.add_argument('--tmc3', help='explicit path to the tmc3 binary')
    p.add_argument('--out-dir', default=os.path.join(HERE, 'encoded'))
    p.add_argument('--json-out', default=os.path.join(HERE, 'coded_frames.json'))
    p.add_argument('--mpd-out', default=os.path.join(PROJECT, 'config', 'mpd_gpcc.xml'))
    args = p.parse_args()

    tmc3 = find_tmc3(args.tmc3)
    plys = collect_plys(args)
    if not plys:
        sys.exit("No .ply frames given (positional paths, --list, or --dir).")
    for ply in plys:
        if not os.path.exists(ply):
            sys.exit(f"missing: {ply}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"tmc3: {tmc3}")
    print(f"frames: {len(plys)} x {len(TIERS)} tiers = {len(plys) * len(TIERS)} encodes "
          f"(jobs={args.jobs}, decode={'no' if args.no_decode else 'yes'}, "
          f"resume={'off (--force)' if args.force else 'on'})")

    # Build the full task list.
    tasks = []
    for idx, ply in enumerate(plys):
        base = os.path.splitext(os.path.basename(ply))[0]
        for rep_id, label, scale, qp in TIERS:
            tasks.append({
                'key': (idx, rep_id), 'tmc3': tmc3, 'ply': ply,
                'out_bin': os.path.join(args.out_dir, f"{base}_{label}.bin"),
                'dec_ply': os.path.join(args.out_dir, f"{base}_{label}_dec.ply"),
                'scale': scale, 'qp': qp,
                'decode': not args.no_decode, 'force': args.force,
            })

    # Run (parallel or sequential), collecting results by key.
    results = {}
    errors = []
    done = 0

    def take(res):
        nonlocal done
        key, coded_bytes, density, skipped, err = res
        done += 1
        if err:
            errors.append((key, err))
            print(f"  [{done}/{len(tasks)}] frame {key[0]} rep {key[1]} FAILED: {err}")
        else:
            results[key] = (coded_bytes, density)
            if done % 25 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] done{' (resumed some)' if skipped else ''}")

    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            for res in pool.imap_unordered(encode_one, tasks):
                take(res)
    else:
        for t in tasks:
            take(encode_one(t))

    if errors:
        sys.exit(f"{len(errors)} encode(s) failed — rerun to resume (already-done frames are skipped).")

    # Assemble manifest structures.
    frames = []
    for idx, ply in enumerate(plys):
        src_pts = ply_vertex_count(ply)
        reps = []
        for rep_id, label, scale, qp in TIERS:
            coded_bytes, density = results[(idx, rep_id)]
            reps.append({
                'id': rep_id, 'quality': label, 'scale': scale, 'qp': qp,
                'coded_bytes': coded_bytes, 'bitrate_bps': coded_bytes * 8 * FPS,
                'density': density if density else src_pts,
                'src': os.path.basename(ply),
            })
        frames.append({'id': idx, 'src': os.path.basename(ply),
                       'src_points': src_pts, 'representations': reps})

    with open(args.json_out, 'w') as f:
        json.dump({'fps': FPS, 'tiers': TIERS, 'frames': frames}, f, indent=2)

    # Simulator-compatible MPD with REAL coded sizes. BaseURL is just the source
    # frame's filename (machine-independent; nothing streams from it).
    mpd = ET.Element('MPD', {'format': 'pointcloud/gpcc', 'type': 'static', 'encoding': 'G-PCC'})
    ET.SubElement(mpd, 'BaseURL').text = '.'
    for fr in frames:
        fe = ET.SubElement(mpd, 'Frame', {'id': str(fr['id'])})
        aset = ET.SubElement(fe, 'AdaptationSet')
        for r in fr['representations']:
            re = ET.SubElement(aset, 'Representation', {
                'id': str(r['id']),
                'density': str(r['density']),
                'size': str(int(r['coded_bytes'])),
                'codedBytes': str(int(r['coded_bytes'])),
                'bandwidth': str(int(r['bitrate_bps'])),
                'quality': r['quality'],
                'attrQP': str(r['qp']),
                'geomQP': '0',
            })
            ET.SubElement(re, 'BaseURL').text = r['src']
    xml = minidom.parseString(ET.tostring(mpd)).toprettyxml(indent='  ')
    os.makedirs(os.path.dirname(args.mpd_out), exist_ok=True)
    with open(args.mpd_out, 'w', encoding='utf-8') as f:
        f.write(xml)

    print(f"\nEncoded {len(frames)} frame(s) x {len(TIERS)} reps.")
    print(f"  JSON: {args.json_out}")
    print(f"  MPD : {args.mpd_out}")
    print("Commit back to git: config/mpd_gpcc.xml + gpcc/coded_frames.json "
          "(the .bin bitstreams stay out of git).")


if __name__ == "__main__":
    main()
