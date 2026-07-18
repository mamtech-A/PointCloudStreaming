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
  split.json                                  train/validation file lists from
                                              the registered experiment protocol
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
import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from src.experiment_protocol import load_protocol, protocol_digest


class FixedABR(ABRStrategy):
    """Constant-arm policy: always the a-th rep (sorted by id)."""
    name = 'fixed'

    def __init__(self, arm):
        self.arm = arm

    def select(self, state):
        reps = sorted(state.reps, key=lambda r: r['id'])
        return reps[max(0, min(self.arm, len(reps) - 1))]['id']


class CyclingABR(ABRStrategy):
    """Deterministic exploratory policy that exposes every tier payload size."""
    name = 'cycle'

    def __init__(self):
        self.index = 0

    def reset(self):
        self.index = 0

    def select(self, state):
        reps = sorted(state.reps, key=lambda rep: rep['id'])
        selected = reps[self.index % len(reps)]['id']
        self.index += 1
        return selected


POLICIES = {
    'fixed1': lambda: FixedABR(1),      # medhigh: long downloads, fade-sensitive
    'fixed3': lambda: FixedABR(3),      # medlow: the mid regime
    'fixed5': lambda: FixedABR(5),      # vlow: RTT-bound tiny frames
    'cycle': lambda: CyclingABR(),      # exploratory coverage of all six tiers
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


def sequence_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len('mpd_gpcc_'):] if stem.startswith('mpd_gpcc_') else stem


def load_content_pool(pattern, max_frames=0):
    paths = sorted(glob.glob(pattern)) if any(ch in pattern for ch in '*?[') else [pattern]
    if not paths:
        raise FileNotFoundError(f"no manifest matches {pattern}")
    pool = {}
    for path in paths:
        frames = parse_mpd_xml(path)
        pool[sequence_name(path)] = frames[:max_frames] if max_frames else frames
    return pool, paths


def main():
    p = argparse.ArgumentParser(description="Generate achieved-throughput LSTM dataset")
    p.add_argument('--trace-dir', default=os.path.join(project_root, 'bandwidth_5g'))
    p.add_argument('--protocol', default=os.path.join(project_root, 'configs',
                                                       'experiment_protocol.json'),
                   help='registered trace split; final-test traces are never generated')
    p.add_argument('--mpd', default=os.path.join(project_root, 'manifests', 'mpd_gpcc*.xml'),
                   help='manifest path or glob; default includes all four 8i sequences')
    p.add_argument('--out-dir', default=os.path.join(project_root, 'data', 'lstm_achieved'))
    p.add_argument('--policies', nargs='*', default=list(POLICIES),
                   choices=list(POLICIES), help='which policies generate series')
    p.add_argument('--offsets', type=int, default=4,
                   help='episode start offsets per trace (spread evenly)')
    p.add_argument('--segment-frames', type=int, default=10,
                   help='frames per fetched segment; MUST match the DQN eval/'
                        'inference setting so the LSTM trains on the same signal')
    p.add_argument('--max-frames', type=int, default=0, help='0 = full manifest')
    args = p.parse_args()

    content_pool, manifest_paths = load_content_pool(args.mpd, args.max_frames)

    protocol_path = (args.protocol if os.path.isabs(args.protocol)
                     else os.path.join(project_root, args.protocol))
    protocol = load_protocol(protocol_path, args.trace_dir)
    train_src = list(protocol['trace_split']['train'])
    validation_src = list(protocol['trace_split']['validation'])
    final_test_src = list(protocol['trace_split']['test'])
    os.makedirs(args.out_dir, exist_ok=True)
    # Derived dataset: wipe stale series so split.json and the dir never disagree
    # (e.g. leftovers generated under a different segment size).
    for old in os.listdir(args.out_dir):
        if old.endswith('.csv'):
            os.remove(os.path.join(args.out_dir, old))

    split = {'train_files': [], 'validation_files': []}
    n_series = 0
    traces = {
        src: BandwidthTrace.from_file(os.path.join(args.trace_dir, src))
        for src in train_src + validation_src
    }
    for sequence, frames in sorted(content_pool.items()):
        for src in train_src + validation_src:
            side = 'train_files' if src in train_src else 'validation_files'
            stem = os.path.splitext(src)[0]
            full = traces[src]
            n = len(full)
            offs = [int(round(i * n / float(args.offsets))) for i in range(args.offsets)]
            offs = sorted({min(o, max(0, n - 120)) for o in offs})
            for off in offs:
                trace = full.slice_from(off) if off else full
                for pol in args.policies:
                    series = run_series(
                        frames, trace, POLICIES[pol], args.segment_frames)
                    name = f"{sequence}__{stem}__{pol}__off{off}.csv"
                    with open(os.path.join(args.out_dir, name), 'w', encoding='utf-8',
                              newline='') as f:
                        f.write('State,DL_bitrate\n')
                        for bps in series:
                            f.write(f"D,{bps / 1000.0:.3f}\n")
                    split[side].append(name)
                    n_series += 1
                    print(f"  {name}: {len(series)} samples "
                          f"(mean {sum(series)/max(1,len(series))/1e6:.2f} Mbps)")

    # ``test_files`` is retained as an API alias for the existing LSTM trainer;
    # scientifically these are validation files used for early stopping/model
    # selection. Registered final-test traces are deliberately absent here.
    split['test_files'] = list(split['validation_files'])
    split['source_split'] = {
        'train': train_src,
        'validation': validation_src,
        'registered_test_not_generated': final_test_src,
    }
    split['protocol'] = os.path.relpath(protocol_path, project_root)
    split['protocol_digest'] = protocol_digest(protocol)
    split['policies'] = args.policies
    split['mpd'] = [os.path.basename(path) for path in manifest_paths]
    split['content_sequences'] = sorted(content_pool)
    split['segment_frames'] = args.segment_frames
    with open(os.path.join(args.out_dir, 'split.json'), 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)
    print(f"\n{n_series} series -> {args.out_dir} "
          f"({len(split['train_files'])} train / "
          f"{len(split['validation_files'])} validation files; final test excluded)")
    print(f"split: {os.path.join(args.out_dir, 'split.json')}")
    print("Train with: python train_model.py --bandwidth-dir data/lstm_achieved "
          "--split-from data/lstm_achieved/split.json")


if __name__ == '__main__':
    main()
