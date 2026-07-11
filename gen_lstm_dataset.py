#!/usr/bin/env python3
"""Generate ACHIEVED-throughput series for LSTM training (signal-mismatch fix).

The LSTM used to be trained on trace capacity (`DL_bitrate`) but is FED the
session's achieved per-frame throughput at inference — a different signal
(RTT-bound on small frames, serialization-bound in fades). This script closes
that gap: it runs quiet simulations (representative policies x static traces x
several start offsets) under the CURRENT transport model and records each
SEGMENT's achieved throughput (one sample per fetch, as at inference), producing a derived dataset the LSTM can train on
that matches what it will see at inference.

Outputs (default data/lstm_achieved/):
  <source-trace-stem>__<policy>__off<N>.csv   State,DL_bitrate (kbps) rows
  split.json                                  train/test file lists derived from
                                              the canonical bandwidth_5g split
                                              (all series from one source trace
                                              stay on one side — leak-free)

Train the LSTM on it with:
  python train_model.py --bandwidth-dir data/lstm_achieved --split-from data/lstm_achieved/split.json

Usage:
    python gen_lstm_dataset.py                 # full: 4 policies x 4 offsets
    python gen_lstm_dataset.py --offsets 1 --policies fixed3 --max-frames 40  # smoke
"""

import os
import sys
import json
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model import (
    Server, EdgeNode, User, Topology, BandwidthABR, DEFAULT_TCP_PARAMS,
)
from src.network_model.abr import ABRStrategy
from src.network_model.manifest import parse_mpd_xml
from src.network_model.trace import BandwidthTrace
from src.lstm_model import split_bandwidth_files


class FixedABR(ABRStrategy):
    """Constant-arm policy: always the a-th rep (sorted by id)."""
    name = 'fixed'

    def __init__(self, arm):
        self.arm = arm

    def select(self, state):
        reps = sorted(state.reps, key=lambda r: r['id'])
        return reps[max(0, min(self.arm, len(reps) - 1))]['id']


POLICIES = {
    'fixed1': lambda: FixedABR(1),      # medhigh: long downloads, fade-sensitive
    'fixed3': lambda: FixedABR(3),      # medlow: the mid regime
    'fixed5': lambda: FixedABR(5),      # vlow: RTT-bound tiny frames
    'bandwidth': lambda: BandwidthABR(),  # the adaptive rule baseline
}


def run_series(frames, trace, abr_factory, segment_frames=1):
    """One quiet episode; returns the achieved throughput series (bps) — one
    sample per SEGMENT (matches what the LSTM is fed at inference)."""
    server = Server("gen://origin")
    server.manifest.frames = frames
    from src.network_model.manifest import PointCloud
    for fr in frames:
        for rep in fr['representations']:
            server.add_pointcloud(fr['id'], rep['id'], PointCloud(points=None))
    topo = Topology(server)
    edge = EdgeNode("edge", server=server,
                    tcp_params={**DEFAULT_TCP_PARAMS, 'log_packets': False},
                    abr_factory=abr_factory)
    topo.add_edge(edge)
    user = User("gen-user")
    session = topo.add_user(user, edge, trace=trace)
    session.start()
    seg = max(1, int(segment_frames))
    for i in range(0, len(frames), seg):
        session.step_segment(frames[i:i + seg], i)
    topo.close_all()
    return list(session.observed_throughput_history)


def main():
    p = argparse.ArgumentParser(description="Generate achieved-throughput LSTM dataset")
    p.add_argument('--trace-dir', default=os.path.join(project_root, 'bandwidth_5g'))
    p.add_argument('--mpd', default=os.path.join(project_root, 'config', 'mpd_gpcc_longdress.xml'))
    p.add_argument('--out-dir', default=os.path.join(project_root, 'data', 'lstm_achieved'))
    p.add_argument('--policies', nargs='*', default=list(POLICIES),
                   choices=list(POLICIES), help='which policies generate series')
    p.add_argument('--offsets', type=int, default=4,
                   help='episode start offsets per trace (spread evenly)')
    p.add_argument('--segment-frames', type=int, default=10,
                   help='frames per fetched segment; MUST match the DQN eval/'
                        'inference setting so the LSTM trains on the same signal')
    p.add_argument('--max-frames', type=int, default=0, help='0 = full manifest')
    p.add_argument('--seed', type=int, default=42,
                   help='canonical split seed (must match LSTM/DQN training)')
    args = p.parse_args()

    frames = parse_mpd_xml(args.mpd)
    if args.max_frames:
        frames = frames[:args.max_frames]

    train_src, test_src = split_bandwidth_files(args.trace_dir, test_size=0.2,
                                                random_state=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    # Derived dataset: wipe stale series so split.json and the dir never disagree
    # (e.g. leftovers generated under a different segment size).
    for old in os.listdir(args.out_dir):
        if old.endswith('.csv'):
            os.remove(os.path.join(args.out_dir, old))

    split = {'train_files': [], 'test_files': []}
    n_series = 0
    for src in train_src + test_src:
        side = 'train_files' if src in train_src else 'test_files'
        stem = os.path.splitext(src)[0]
        full = BandwidthTrace.from_file(os.path.join(args.trace_dir, src))
        n = len(full)
        offs = [int(round(i * n / float(args.offsets))) for i in range(args.offsets)]
        offs = sorted({min(o, max(0, n - 120)) for o in offs})
        for off in offs:
            trace = full.slice_from(off) if off else full
            for pol in args.policies:
                series = run_series(frames, trace, POLICIES[pol], args.segment_frames)
                name = f"{stem}__{pol}__off{off}.csv"
                with open(os.path.join(args.out_dir, name), 'w', encoding='utf-8',
                          newline='') as f:
                    f.write('State,DL_bitrate\n')
                    for bps in series:
                        f.write(f"D,{bps / 1000.0:.3f}\n")
                split[side].append(name)
                n_series += 1
                print(f"  {name}: {len(series)} samples "
                      f"(mean {sum(series)/max(1,len(series))/1e6:.2f} Mbps)")

    split['source_split'] = {'train': train_src, 'test': test_src, 'seed': args.seed}
    split['policies'] = args.policies
    split['mpd'] = os.path.basename(args.mpd)
    split['segment_frames'] = args.segment_frames
    with open(os.path.join(args.out_dir, 'split.json'), 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)
    print(f"\n{n_series} series -> {args.out_dir} "
          f"({len(split['train_files'])} train / {len(split['test_files'])} test files)")
    print(f"split: {os.path.join(args.out_dir, 'split.json')}")
    print("Train with: python train_model.py --bandwidth-dir data/lstm_achieved "
          "--split-from data/lstm_achieved/split.json")


if __name__ == '__main__':
    main()
