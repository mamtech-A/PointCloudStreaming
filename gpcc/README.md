# G-PCC (MPEG TMC13) encoding pipeline

This directory holds the real MPEG G-PCC codec toolchain used to encode the
point-cloud `.ply` frames into compressed bitstreams. The simulator streams the
**real coded sizes** produced here (written into `config/mpd_gpcc.xml`), not the
raw `.ply` file sizes.

Contents:
- `encode_frames.py` — encodes frames at 3 rate tiers, measures decoded fidelity,
  writes `config/mpd_gpcc.xml` + `gpcc/coded_frames.json`.
- `mpeg-pcc-tmc13/` — the MPEG reference codec (cloned + built locally, **not in git**).
- `encoded/` — generated bitstreams / decoded plys (**not in git**, regenerable).

---

## 1. Build TMC13 (once per machine)

Requirements: `git`, CMake ≥ 3.10, a C++11 compiler (MinGW gcc, MSVC, or g++ on Linux).

```bash
# from the repo root
git clone --depth 1 --branch master-v12.x https://github.com/MPEGGroup/mpeg-pcc-tmc13.git gpcc/mpeg-pcc-tmc13
```

### Windows (MinGW gcc + GNU make — verified with gcc 6.3)
```bash
cmake -S gpcc/mpeg-pcc-tmc13 -B gpcc/mpeg-pcc-tmc13/build -G "Unix Makefiles" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_COMPILER=<path>/gcc.exe -DCMAKE_CXX_COMPILER=<path>/g++.exe \
  -DCMAKE_MAKE_PROGRAM=<path>/make.exe -DCMAKE_BUILD_TYPE=Release
make -C gpcc/mpeg-pcc-tmc13/build -j4
```
(`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required with CMake ≥ 4.)

### Windows (Visual Studio)
```bat
cmake -S gpcc/mpeg-pcc-tmc13 -B gpcc/mpeg-pcc-tmc13/build -G "Visual Studio 17 2022" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build gpcc/mpeg-pcc-tmc13/build --config Release
```

### Linux
```bash
cmake -S gpcc/mpeg-pcc-tmc13 -B gpcc/mpeg-pcc-tmc13/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release
make -C gpcc/mpeg-pcc-tmc13/build -j$(nproc)
```

Result: `gpcc/mpeg-pcc-tmc13/build/tmc3/tmc3(.exe)`. `encode_frames.py` finds it
automatically (or pass `--tmc3 PATH`).

---

## 2. Encode a sequence

```bash
# all *.ply in a folder, sorted by name = playback order; 8 parallel workers
python gpcc/encode_frames.py --dir "F:/Education/Tarbiat Modares/Holography dataset/8iVFBv2/longdress/Ply" --jobs 8
```

- ~4–10 s per encode → 300 frames × 3 tiers sequentially ≈ 1–2.5 h; `--jobs 8` ≈ 15–25 min.
- **Resume:** re-running skips every (frame, tier) whose `.bin` already exists — a
  crashed/interrupted run continues where it stopped. `--force` re-encodes.
- `--no-decode` skips the fidelity decode (density then = source point count).

Outputs:
- `config/mpd_gpcc.xml` — the coded manifest the simulator uses (real
  `codedBytes`, `bandwidth`, decoded `density` per representation).
- `gpcc/coded_frames.json` — detailed per-frame record.

**Commit back to git:** `config/mpd_gpcc.xml` + `gpcc/coded_frames.json` only.
The bitstreams in `gpcc/encoded/` (~180 MB for 300 frames) stay untracked.

---

## 3. The rate ladder (and how G-PCC rate control works)

G-PCC encodes each frame directly in 3D: geometry as an **octree** (occupancy
entropy coding), colors with the **RAHT** transform (`--transformType=0`) +
quantization. There is **no target-bitrate mode** — rate is set indirectly:

| knob | effect |
|---|---|
| `--positionQuantizationScale` (0–1) | scales coordinates before the octree → fewer occupied voxels → fewer points & geometry bits (this is why decoded density drops per tier) |
| `--qp` (4–51) | color quantization (video-style); higher = coarser = far fewer attribute bits |
| `--qpChromaOffset`, `--bitdepth` | chroma/precision details |

**Ordering gotcha:** attribute parameters (`--qp`, `--bitdepth`, `--transformType`)
must appear **before** `--attribute=color` — TMC13 binds them when it sees
`--attribute`.

The 6 tiers used (`TIERS` in `encode_frames.py`) are the full MPEG CTC rate points,
measured on longdress_vox10 @30 fps:

| tier | CTC | scale | qp | ≈ bitrate |
|---|---|---|---|---|
| high (rep 0) | r06 | 1.0 | 22 | 102 Mbps |
| medhigh (rep 1) | r05 | 0.875 | 28 | 59.6 Mbps |
| med (rep 2) | r04 | 0.75 | 34 | 35.5 Mbps |
| medlow (rep 3) | r03 | 0.5 | 40 | 14.1 Mbps |
| low (rep 4) | r02 | 0.25 | 46 | 3.4 Mbps |
| vlow (rep 5) | r01 | 0.125 | 51 | 0.9 Mbps |

Re-running the encode after adding tiers only encodes the NEW tiers — the
existing high/med/low bitstreams are reused (resume by filename).

To hit an exact target bitrate, encode a sample frame at several (scale, qp)
pairs and pick the closest (bisection on qp at fixed scale) — then reuse those
parameters for the whole sequence.
