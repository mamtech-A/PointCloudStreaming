# DQN ABR Agent — Complete Technical Report

*Point-cloud streaming simulator, branch `feat/oo-topology-dqn-gpcc`. All numbers in this report were measured
in this repository on the real MPEG G-PCC (TMC13) encoded longdress sequence (300 frames @ 30 fps, 10 s).*

> **⚠️ LEGACY-4G NUMBERS.** Sections below report results on the original 40-trace 4G dataset
> (Ghent 2016) at a modeled 50 ms RTT — that dataset and those checkpoints were removed in the
> 5G migration (PLAN.md §12). Headline legacy numbers for reference: best fixed arm
> always-medlow **+144.0**, best DQN eval **+132.5**. Rewards are NOT comparable across
> datasets/RTT configs. §8's 5G re-baseline numbers were in turn superseded by the
> **2026-07-10 realism overhaul** (time-varying capacity + serialization delay, static-only
> traces, 4 content sequences, reward/QoE reform, full sweep) — **current results are in §9.**

---

## 1. What the DQN does

The DQN is an **ABR (adaptive bitrate) controller**: at every video frame it selects which of the encoded
**representations** (quality tiers) to download, based on what the client can observe — predicted/measured
bandwidth, buffer state, and the cost/benefit of each tier. It replaces hand-written rules
(`BandwidthABR`, `LSTMABR`) with a learned policy, and **keeps the LSTM bandwidth predictor as a feature
provider** — the LSTM's prediction is one input to the DQN's state, not the decision-maker.

| component | file |
|---|---|
| Inference strategy `DQNABR` | `src/network_model/abr.py` |
| Q-network / replay / agent | `src/rl/dqn.py` |
| State features + reward quality | `src/rl/features.py` |
| Training environment `StreamingEnv` | `src/rl/env.py` |
| Training script | `train_dqn.py` |
| Evaluation / comparison | `run_dqn.py`, `compare.py` |
| Trained model (6-action) | `models/abr_dqn.pkl` (retrained on the 5G dataset — see §8) |

---

## 2. Problem formulation (MDP)

One **episode = one bandwidth trace** driving one streaming session over the 300-frame manifest.
One **step = one frame**: the agent picks a representation → the frame's real coded bitstream is transferred
over the simulated TCP link → the playback buffer is updated → reward is computed.

### 2.1 Action space

`action ∈ {0..5}` = the 6-tier MPEG CTC ladder (real G-PCC encodes, `config/mpd_gpcc.xml`):

| action / rep | CTC | (posQuantScale, colorQP) | bitrate @30fps | decoded points (frame 0) |
|---|---|---|---|---|
| 0 high | r06 | (1.0, 22) | 102.4 Mbps | 765,821 |
| 1 medhigh | r05 | (0.875, 28) | 59.6 Mbps | 602,139 |
| 2 med | r04 | (0.75, 34) | 35.5 Mbps | 453,698 |
| 3 medlow | r03 | (0.5, 40) | 14.1 Mbps | 212,105 |
| 4 low | r02 | (0.25, 46) | 3.4 Mbps | 55,374 |
| 5 vlow | r01 | (0.125, 51) | 0.9 Mbps | 14,057 |

The action count and all feature dimensions follow the manifest automatically (`num_reps` is inferred), so the
same code ran the earlier 3-tier ladder.

### 2.2 State vector (23-dim for 6 reps)

Built by the **single shared builder** `build_state()` in `src/rl/features.py` — used identically by the
training env and by inference (`DQNABR`), with the `feature_spec` and normalization constants **persisted
inside the model checkpoint**, so train-time and inference-time states can never diverge. Every block is
individually toggleable via `feature_spec` (for ablation studies).

| feature | dim | meaning | normalization |
|---|---|---|---|
| `lstm_pred` | 1 | LSTM-predicted bandwidth (Mbps); harmonic-mean fallback during warm-up | ÷100, clip [0, 1.5] |
| `buffer` | 1 | current buffer level (s) | ÷5 (buffer capacity) |
| `last_rep_onehot` | 6 | previous frame's chosen rep | one-hot |
| `tput_last` | 1 | last achieved throughput (Mbps) | ÷100 |
| `tput_mean` | 1 | mean of last 5 throughput samples | ÷100 |
| `tput_std` | 1 | std of last 5 throughput samples | ÷50, clip ≤1 |
| `rep_bitrates` | 6 | each rep's real coded bitrate (Mbps) | ÷100, clip ≤3 |
| `rep_densities` | 6 | each rep's decoded point count | log10(d)/7 |

Observability discipline: the agent sees only **achieved** throughput of completed downloads and the LSTM's
prediction — never the true link capacity (no oracle).

### 2.3 Reward function (what is rewarded and punished)

Pensieve-style per-frame reward (`src/rl/env.py`):

```
r_t = quality(rep_t)  −  μ · stall_t  −  λ · | quality(rep_t) − quality(rep_{t−1}) |
        └── reward ──┘   └── punish stalls ──┘  └── punish quality switching ──┘

μ = 4.3   (rebuffer penalty per stalled second)
λ = 1.0   (smoothness penalty per unit quality jump)
```

- `stall_t` = rebuffering seconds caused this step, returned by `ClientBuffer.add_frame()`.
- `quality(rep) ∈ [0, 1]` is a **log-density utility** with endpoints derived from the manifest itself
  (`quality_endpoints()`), so the lowest tier maps to ≈0 and the highest to ≈1 on any ladder:

```
quality(rep) = clip( (log10(density) − log10(d_min)) / (log10(d_max) − log10(d_min)), 0, 1 )
d_min = 13,449 (vlow), d_max = 916,250 (high)   → per-tier ≈ {high 1.0, medhigh 0.93, med 0.80,
                                                    medlow 0.66, low 0.35, vlow 0.0}
```

Interpretation of μ=4.3: one second of rebuffering erases the quality gain of ~4.3 frames at max quality —
a strong stall-avoidance prior. λ=1.0 makes a full-ladder jump cost one max-quality frame.

### 2.4 The project QoE formula (reported, NOT optimized)

```
QoE = max(0, 100 − 10·rebuffer_events − 5·total_stall_seconds − 2·dropped_frames)
```

**Important caveat:** this QoE counts only stalls/drops — it contains **no quality term**, so a policy that
hides at the lowest tier trivially maximizes it. We therefore train on the quality-aware reward and report
**both** metrics (`compare.py` prints QoE *and* mean-quality *and* Pensieve reward). This tension is visible in
every results table below.

---

## 3. Algorithm and hyperparameters

**Double DQN** with experience replay and a periodically-synced target network (`src/rl/dqn.py`).

### 3.1 Q-network

```
MLP:  Linear(23 → 128) → ReLU → Linear(128 → 128) → ReLU → Linear(128 → 6)
```
A feed-forward net suffices because temporal bandwidth structure is already summarized by the LSTM prediction
+ throughput-history statistics in the state.

### 3.2 Learning rule

Double-DQN target (reduces Q-value overestimation):

```
y = r + γ · (1 − done) · Q_target(s′, argmax_a Q_online(s′, a))
loss = Huber(Q_online(s, a), y)
```

### 3.3 All hyperparameters (final, stabilized configuration)

| parameter | value | notes |
|---|---|---|
| discount γ | 0.99 | 300-step episodes |
| optimizer | Adam, **lr = 1e-4** | default 5e-4 caused eval oscillation (−97…+37); 1e-4 converged |
| loss | Huber (SmoothL1) | robust to large stall-penalty outliers |
| batch size | 64 | |
| replay buffer | 100,000 transitions | ~333 episodes of history |
| target-net sync | every **3,000** learn-steps | default 500 (≈1.7 episodes) was the main instability source |
| gradient clip | max-norm 10 | |
| ε-greedy | linear 1.0 → 0.05 over 60% of total steps, then hold 0.05 | |
| μ, λ (reward) | 4.3, 1.0 | |
| episodes | 40 epochs × 36 traces = 1,440 (~432k steps) | ~70 min wall-clock (CPU) |
| eval protocol | greedy (ε=0) on the 4 held-out traces, fixed RNG seed, every 50 episodes | RNG state saved/restored so eval never perturbs training randomness |
| checkpointing | best-by-mean-eval-reward → `models/abr_dqn.pkl` | final weights do not overwrite a better checkpoint |
| seed | 42 (random / numpy / torch; and the trace split) | |

### 3.4 Training data protocol

- **File-level trace split identical to the LSTM's** (no leakage, comparable results):
  36 train / 4 test traces (`split_bandwidth_files`, seed 42);
  test = `report_car_0003/0004/0007.log`, `report_foot_0006.log`.
- Train-trace order shuffled every epoch; transitions from all traces mix in the replay buffer.
- The LSTM predictor (`models/bandwidth_lstm.pkl`) is **frozen** — used purely as a feature.

### 3.5 Simulation environment constants

| parameter | value |
|---|---|
| playback | 30 fps, 300 frames (10 s of content) |
| client buffer | capacity 5 s, min 1 s to start/resume |
| TCP | RTT 50 ms ± 10 ms jitter, loss 0, init cwnd 10, MSS 1460 B, Jacobson/Karels RTO, slow start + AIMD, persistent per-user connection |
| link capacity | one 4G trace sample per frame (mean ≈ 30 Mbps, peak ≈ 110) |
| transferred bytes | the **real G-PCC coded bitstream size** of the chosen rep |

---

## 4. Results

### 4.1 Fixed-policy bars (mean reward over the 4 test traces, fixed-seed)

| always-… | reward | QoE |
|---|---|---|
| high | −132.9 | 0.0 |
| medhigh | +40.9 | 0.0 |
| med | +125.1 | 3.4 |
| **medlow** | **+144.0** | 31.7 |
| low | +68.0 | 46.5 |
| vlow | −24.1 | 48.5 |

Key fact: **no single tier is best on every trace** — `med` wins the fast car traces, `medlow` the variable ones.

### 4.2 Trained DQN (best checkpoint, eval +132.5)

Per-trace behavior — the policy is genuinely adaptive (rep counts hi/mh/med/ml/lo/vlo):

| test trace | DQN reward | best fixed arm | DQN rep mix |
|---|---|---|---|
| car_0003 | **170.7** | med 169.9 — **DQN wins** | 0/43/236/21/0/0 |
| car_0004 | 165.3 | med 176.9 | 0/73/207/20/0/0 |
| car_0007 | 115.3 | medlow 128.2 | 0/152/52/96/0/0 |
| foot_0006 | 78.8 | medlow 132.4 | 0/129/134/37/0/0 |
| **mean** | **132.5** | (oracle-static 151.9; best single arm 144.0) | |

The DQN beats **5 of 6** fixed arms, matches the correct *regime* per trace, and beats the per-trace best
static arm on car_0003. Its shortfall vs always-medlow comes from over-committing to `medhigh` on foot_0006.
Training converged at episode 100 and did not improve for the remaining 1,340 episodes.

### 4.3 Final comparison vs rule-based ABR (foot_0006, 300 frames)

| strategy | QoE | rebuf | stall_s | mean quality | Pensieve reward |
|---|---|---|---|---|---|
| bandwidth rule | 46.9 | 3 | 4.6 | 0.030 | −30.7 |
| LSTM rule | 47.6 | 3 | 4.5 | 0.030 | −23.3 |
| **DQN** | 0.0 | 5 | 21.6 | **0.864** | **+71.9** |

With 6 tiers the conservative rules sink to the bottom rung (their achieved-throughput estimate is
RTT-bound on tiny files — see §5.3), maximizing the stall-only QoE but delivering ~3% quality utility.
The DQN delivers **~29× the quality** at the cost of stalls the QoE formula punishes heavily.

### 4.4 Ladder ablation (3-rep vs 6-rep)

On the earlier 3-tier ladder (3.4 / 35.5 / 102 Mbps) the learned policy degenerated to *always-MED with
1-frame deviations* (+108.3, barely above the always-MED bar +103.7): the 10× quality gap between LOW and MED
made dips never worth taking. The 6-tier ladder gave the agent usable intermediate actions, and the learned
policy became **truly adaptive** (per-trace regime switching, 18 switches on foot_0006). Note: reward scales
are ladder-relative (quality endpoints derive from each manifest), so compare policies across ladders by
quality/stall behavior, not absolute reward.

---

## 5. Known limitations (honest accounting)

1. **Below the best static arm on mean** (+132.5 vs +144.0) — converged early; the gap is algorithmic, not
   a matter of more epochs.
2. **QoE-metric mismatch** — the stall-only QoE anti-correlates with quality; reporting both is necessary
   but a unified quality-aware QoE would be better science (§6.5).
3. **RTT-bound sequential fetching** — at 30 fps the 33 ms frame budget < 50 ms RTT, so *every* policy stalls
   sometimes; also achieved throughput measured on tiny files (~1–10 Mbps on a 40 Mbps link) traps the
   rule-based ABRs at the bottom tier. This is a transport-model realism limit (§6.4).
4. **Single content sequence** (longdress) and 4 test traces — generalization untested (§6.6).
5. **Default hypers are unstable** — lr 5e-4 / target-sync 500 oscillated between −97 and +37; the stabilized
   configuration was found manually, not swept.

---

## 6. Proposed improvements (ranked by expected value / effort)

### 6.1 Algorithmic upgrades to the DQN (highest expected gain)
- **Prioritized Experience Replay (PER)** — stall events are rare but carry the learning signal; uniform
  sampling dilutes them. Expected to specifically fix the foot_0006 over-commitment. *(medium effort)*
- **Dueling network** (separate V(s) and A(s,a) heads) — better value estimation when most actions are
  similar (adjacent tiers). *(low effort)*
- **n-step returns (n≈5)** — a stall's cause is the choice made several frames earlier; n-step propagates
  the penalty back faster. *(low effort)*
- **Soft (Polyak) target updates** (τ≈0.005 per step) instead of hard syncs — removes the sync-period
  hyperparameter that caused the instability. *(trivial)*
- **Demonstration warm-start**: pre-fill the replay buffer with fixed-policy episodes (always-med/medlow)
  so the agent starts near the good arms instead of random. *(low effort)*
- Together these are most of "Rainbow-lite" — the standard recipe for exactly this plateau.

### 6.2 Better exploration & training schedule
- Longer ε floor exploration or ε per-episode annealing; try 80–120 epochs with early stopping on eval plateau.
- **Systematic hyperparameter sweep** (lr × target-update × γ × μ) — the current config was hand-found;
  a 20–30 run sweep on `--epochs 10` would likely find a better basin. *(cheap: each 10-epoch run ≈ 18 min)*

### 6.3 Richer state (Pensieve parity and beyond)
- **Raw throughput sequence** (last k=8 samples, not just mean/std) — Pensieve's most informative input.
- **Download time of the last chunk** and **time-since-last-stall**.
- **LSTM prediction error/uncertainty** (|pred − actual| history) — lets the agent learn when to trust the LSTM.
- **Future frame sizes** for the next few frames per rep (available in the manifest — content-aware lookahead).

### 6.4 Environment realism (fixes the RTT bound)
- **Pipelined/batched fetching**: request the next frame while the current one downloads (or fetch groups of
  frames per request, like DASH segments). Removes the artificial 1-RTT-per-frame floor, which currently
  (a) forces stalls at every tier and (b) starves rule ABRs via tiny-file throughput underestimates. This is
  the single most impactful realism improvement and will change all absolute numbers.
- Trace augmentation (scale ×[0.5, 2], time-warp) for robustness; longer episodes by concatenating traces.

### 6.5 A quality-aware QoE metric (report-level)
Adopt an ITU-P.1203-flavored score alongside the legacy one, e.g.
`QoE' = 100 · mean_quality − 4.3·stall_s − 1.0·Σ|Δq| (clip ≥ 0)` — consistent with the training reward and
comparable across strategies. Keep the legacy stall-only QoE for continuity.

### 6.6 Data breadth
- Encode **loot / redandblack / soldier** (same `encode_frames.py --dir` runbook, resume-safe) and train on a
  mixed-content manifest pool — tests content generalization, strengthens any publication claim.
- More traces (e.g., 5G datasets) — also the prerequisite for the deferred **multi-user shared-bottleneck**
  extension (PLAN.md §2.9), where RL-vs-rules gaps typically widen.

### 6.7 Engineering
- CSV/TensorBoard learning curves; multi-seed (≥3) eval averaging with std; a `sweep.py` harness;
  optional GPU (currently CPU-bound in the simulator, so PER/dueling cost almost nothing extra).

---

## 7. Reproduce

```bash
# train (stabilized config used for the shipped model)
python train_dqn.py --epochs 40 --target-update 3000 --lr 1e-4

# evaluate / compare on the unseen test trace
python run_dqn.py
python compare.py
```

---

## 8. 5G re-baseline (Irish 5G dataset, dataset-derived 72 ms RTT)

Everything in this section was measured after the migration to the **Raca et al. (MMSys 2020)
Irish 5G dataset** (PLAN.md §12): 21 Download traces (17 train / 4 test, file-level split seed 42,
`test_size=0.2`), RTT **72 ms ± 8** derived from the dataset''s own `PINGAVG` (5G-mode median 72 ms,
not load-inflated), norm constants `bw_mbps=300` / `tput_std_mbps=125`.

### 8.1 LSTM retrain (feature provider)

16-combo grid + seq-length/transform follow-ups; picked by held-out test MAE:

| variant | test MAE (Mbps) | MAPE | RMSE |
|---|---|---|---|
| seq10 / z-score | 12.81 | 147% | 26.03 |
| **seq20 / z-score (installed)** | **12.53** | **119%** | **25.87** |
| seq10 / log1p | 13.07 | **79%** | 28.89 |
| persistence baseline | 13.84 | 170% | 33.75 |

Winner: `hidden=64, layers=2, dropout=0.2, lr=5e-4, seq_len=20` — beats persistence on all three
metrics (acceptance gate PASS). Note the log1p variant halves MAPE (much better *relative*
accuracy in the deep-fade tail) but loses on absolute error; worth revisiting if the ABR ever
keys on low-tier boundaries.

### 8.2 Fixed-arm baselines (`eval_fixed.py`, `models/fixed_arm_baseline.json`)

At the dataset-faithful 72 ms RTT **every fixed arm is deeply negative** — sequential per-frame
fetching costs ≥ 1 RTT/frame, i.e. ≥ ~21.6 s of transport time for a 10 s clip at ANY tier:

| arm | tier | mean reward | mean stall |
|---|---|---|---|
| 0 | 102.4 Mbps | −4876.05 | 1202 s |
| 1 | 59.6 Mbps | −2638.48 | 678 s |
| 2 | 35.5 Mbps | −1485.09 | 405 s |
| 3 | 14.1 Mbps | −518.20 | 167 s |
| 4 | 3.4 Mbps | −116.09 | 52 s |
| **5** | **0.9 Mbps** | **−102.15** | **26 s** |

### 8.3 DQN retrain

`train_dqn.py --epochs 40 --target-update 3000 --lr 1e-4 --random-offset --reward-scale 0.1`.
Two runs: raw rewards (scale 1.0) **diverged** (evals −160 → −1483 as ε decayed; stall spikes of
−40..−90 per step put Q-targets in the thousands against grad-clip 1.0). With learner-side reward
scale 0.1 (argmax-invariant) training was stable early and the best held-out eval reached
**−101.84 at episode 150** — just above the best fixed arm (−102.15). Late-training evals still
drift optimistic (−700..−1000); the best-checkpoint-by-eval logic preserves the good policy.

`compare.py` (full 300 frames, unseen test traces):

| trace | baseline | LSTM-rule | DQN |
|---|---|---|---|
| driving_B_2020.02.27_20.35.57 | −94.9 | −95.6 | **−94.9** |
| static_B_2020.01.16_10.43.34 | −101.4 | −100.6 | **−100.5** |

### 8.4 The headline finding

**At real 5G NSA latency (72 ms), the sequential frame-per-request transport model cannot sustain
30 fps point-cloud streaming at any quality tier** — all policies stall ~12–13 s per 10 s clip and
the reward-optimal policy collapses to bottom-tier hiding; the learned DQN can only match, not
meaningfully beat, the always-lowest arm (−101.84 vs −102.15). This quantifies the §5/§6.4
transport-realism limit with dataset-derived numbers instead of an assumed RTT, and motivates the
obvious next step: **pipelined / batched frame fetching (GoP-style segments) or an edge cache that
cuts effective RTT below the 33 ms frame budget.**

---

## 9. Realism overhaul + full sweep re-baseline (2026-07-10)

Everything in this section was produced by the one-command pipeline
(`run_training.py`, config `configs/training.json`) on the second PC; artifacts:
`models/TRAINING_SUMMARY.md`, `models/dqn_sweep_results.json`,
`logs/train_runs/20260710_083943_full/`. **§8's numbers are superseded** — the
environment itself changed.

### 9.1 What changed vs §8

1. **Time-varying capacity**: the trace is consumed on its CSV-timestamp wall-clock
   axis (`capacity_at_time`); capacity is re-queried every TCP round, so a long
   download traverses the trace instead of freezing one sample.
2. **Serialization delay**: each TCP round costs `max(RTT, bytes·8/capacity)` — deep
   fades are no longer floored at 1 MSS/RTT (the §5.3 optimism is gone).
3. **Static-only traces** (driving deleted): 5 files, 4 train / 1 held-out
   (`static_B_2020.01.16_10.43.34.csv`); coverage-tiled epochs (episodes ∝ trace length).
4. **4 content sequences** (longdress/loot/redandblack/soldier, same 6-tier QP ladder)
   rotate in training; **eval stays longdress-only** for comparability.
5. **Reward reform**: pluggable spec (`src/rl/reward.py`); learner-side running-std
   reward normalization replaced the `--reward-scale 0.1` hack (no divergence in any
   of the 24 trials).
6. **Quality-aware QoE′** `= 100·mean_q − 4.3·stall_s − 1.0·Σ|Δq|` reported everywhere.
   The legacy stall-only QoE saturates at 0 for *every* policy in this regime
   (all stall ≥ 20 s) — QoE′ is the only discriminating metric.
7. **LSTM retrained on achieved throughput** (the signal it is fed at inference),
   winner = log1p transform: MAE 0.68 vs persistence 0.80, low-bw MAE 0.190 vs 0.195,
   directional accuracy 0.703 (persistence: none).

### 9.2 Fixed-arm baselines (held-out trace, 300 frames, new transport)

| arm | tier | reward | QoE′ | mean quality | stall |
|---|---|---|---|---|---|
| 0 | 102.4 Mbps | +45.2 | −62.9 | 0.978 | 57.6 s |
| 1 | 59.6 Mbps | +26.0 | −35.0 | 0.921 | 58.1 s |
| 2 | 35.5 Mbps | +25.4 | **−31.0** | 0.854 | 53.6 s |
| 3 | 14.1 Mbps | **+63.6** | −31.2 | 0.674 | 32.1 s |
| 4 | 3.4 Mbps | −40.0 | −66.5 | 0.356 | 34.0 s |
| 5 | 0.9 Mbps | −128.6 | −94.4 | 0.030 | 31.9 s |

**The regime inverted: bottom-tier hiding is dead.** Under §8's transport, always-vlow
was optimal (−102.15) because tiny frames escaped serialization. Now serialization
charges every fade regardless of tier, and the static traces' sustained capacity makes
quality affordable — vlow is the *worst* arm and the sweet spot moved to med/medlow.
All arms still stall 32–58 s per 10 s clip: the 72 ms-RTT sequential-fetch floor stands.

### 9.3 DQN sweep (24 configs = 3 μ × 2 λ × 2 reward shapes × lstm_pred ablation)

`sweep.py`, 8 epochs × ~200 episodes each, jobs=2, 5.9 h, selection by held-out QoE′.

- **Winner config**: μ=2.0, λ=1.0, **bounded stall** (cap 2 s/step + 2.0/event), with
  lstm_pred. Best-checkpoint eval QoE′ **+11.96** (mean_q 0.951, stall 18.4 s) — but
  the eval trajectory shows this was a **single spike at ep 500** (neighbors −31…−49);
  the reproducible plateau of the top configs is **QoE′ ≈ −29…−36 at mean_q ≈ 0.95,
  stall ≈ 28 s**. Checkpoint-by-max-eval overfits eval-jitter noise; use n-seed evals
  before trusting a single number.
- **Bounded stall dominates**: 7 of the top 8 trials use the bounded spec. Capping the
  per-step spike (while stall still accumulates) is the single most effective reward
  change.
- **Low μ wins**: with ~28 s of stall unavoidable, μ=8 merely suppresses quality
  (bounded μ=8 trials collapse to mean_q 0.674) without buying stall reduction;
  μ=2 rides near the top tier.
- **lstm_pred ablation verdict: no consistent benefit.** Matched-pair mean ΔQoE′
  (with − without) = **+0.41** over 12 pairs, 8 of 12 pairs negative; the +46.6
  outlier pair is the winner's lucky checkpoint. The feature is kept optional
  (`--no-lstm-pred`); nothing justifies requiring it.

### 9.4 Final comparison (compare.py, held-out trace, 300 frames)

| strategy | QoE (legacy) | QoE′ | mean quality | mean rep | switches | stall | Pensieve reward |
|---|---|---|---|---|---|---|---|
| bandwidth rule | 0.0 | −95.1 | 0.030 | 5.00 | 0 | 22.7 s | −129.6 |
| LSTM rule | 0.0 | −100.2 | 0.030 | 5.00 | 0 | 23.9 s | −139.0 |
| **DQN** | 0.0 | **−34.0** | **0.955** | 0.37 | 48 | 29.0 s | **+34.0** |

### 9.5 The headline finding (v2)

**Under the realistic transport, learned ABR finally has headroom — and uses it.**
The DQN rides near the top tier (mean quality 0.955 vs the rules' 0.030 — a 32×
quality gap) at QoE′ −34, **on the fixed-arm quality-stall Pareto frontier** (beats
arm 0/1 at comparable quality; ≈ arm 2/3's QoE′ at +0.1–0.28 higher quality). The
rule baselines stay trapped at the bottom tier by the RTT-bound achieved-throughput
underestimate. What remains transport-limited is the absolute level: every policy
stalls 20–60 s per 10 s clip, so QoE′ stays negative and legacy QoE stays 0 —
pipelined/segment fetching (§6.4) is still the binding next step.

### 9.6 Reproduce

```
python run_training.py --jobs 2          # full pipeline (see TRAINING.md)
python compare.py                        # table 9.4
python eval_fixed.py                     # table 9.2
```

---

## 10. Segment-based fetching + adaptive playback (round 2, 2026-07-11)

§9 diagnosed the per-frame RTT floor as the binding constraint. Round 2 fixes it
by **fetching S frames per request** (one ABR decision + one TCP transfer, DASH
segments) — the §6.4 recommendation — plus **adaptive playback** (0.9x floor) and
**multi-seed evals**. Artifacts: `models/TRAINING_SUMMARY.md`,
`models/dqn_sweep_results.json`, `logs/train_runs/20260711_073843_full/`. The
48-trial sweep swept **S ∈ {1,5,8,10,15,30}** × μ{2,4.3} × the lstm_pred ablation,
2 training seeds each, reward shape fixed to the §9 winner (bounded stall).

### 10.1 The segment-size curve (the headline)

Mean over μ × lstm-ablation × 2 seeds per S, held-out longdress:

| S (frames/req) | QoE′ | mean quality | stall (s) | legacy QoE |
|---|---|---|---|---|
| 1 (per-frame = §9) | −9.8 | 0.865 | 19.1 | 1.5 |
| **5** | **69.4** | 0.850 | **2.9** | **71.7** |
| 8 | 65.4 | 0.875 | 4.3 | 57.4 |
| 10 | 53.9 | 0.893 | 7.5 | 53.1 |
| 15 | 54.9 | 0.916 | 8.2 | 40.6 |
| 30 | 7.5 | 0.958 | 20.4 | 12.5 |

**A clean inverted-U, peaking at S=5** — both extremes are bad for *opposite*
reasons, and the data rejects each with numbers rather than assumption:
- **S=1** pays one 72 ms RTT per 33 ms frame → 19 s of stall (the §9 regime).
- **S=30** amortizes the RTT fully but commits to a ~1 s, multi-MB block per
  decision (only 10 decisions per clip): it can't react to a fade mid-segment, so
  stall climbs back to 20 s **despite the highest mean quality (0.958)**. This is
  exactly the "coarse adaptation" downside we included S=30 to measure — confirmed
  and rejected.
- **S=5–8** is the sweet spot: enough RTT amortization to nearly eliminate stall,
  still 38–60 decisions per clip for fine adaptation.

### 10.2 Winner + fixed arms

- **Winner** (`models/abr_dqn.pkl`): **S=5, μ=4.3, +lstm_pred** — QoE′ **77.3**
  (2-seed mean of 91.6 and 62.9), **legacy QoE 92.2**, mean quality 0.809, stall
  **0.56 s**, reward 238.7. Legacy stall-only QoE has finally lifted off 0 — the
  RTT floor is broken.
- **Fixed arms** (S=10 + AMP, `eval_fixed.py`): best arm **arm 1 (59.6 Mbps)** now
  streams with **zero stall**, legacy QoE 100, QoE′ 90.4 — vs §9 where every arm
  stalled ≥ 32 s. arm 0 (top tier) still stalls 8 s; the cheap arms (4,5) drop to
  QoE′ 35 / 2.5 as quality collapses. The quality-stall frontier is now favorable.

### 10.3 Final comparison (compare.py, S=10 + AMP, held-out trace)

| strategy | legacy QoE | QoE′ | mean quality | mean rep | stall | dropped |
|---|---|---|---|---|---|---|
| bandwidth rule | 0.0 | 2.5 | 0.030 | 5.00 | 0.0 s | 84 |
| LSTM rule | 0.0 | 41.1 | 0.426 | 3.33 | 0.0 s | 56 |
| **DQN** | **100.0** | **91.6** | **0.921** | 1.00 | 0.0 s | 0 |

The DQN streams the second-highest tier stall-free at 0.921 quality (QoE 100). Note
the **LSTM rule jumped from 0.030 → 0.426 quality** vs §9: with segments the client
finally *measures real bandwidth* (§ below), so even the rule-based ABR climbs off
the bottom tier. The DQN still wins by learning to avoid buffer overflow (0 dropped
vs the rules' 56–84 — see §10.5).

### 10.4 LSTM + lstm_pred ablation

- **LSTM** retrained on the S=10 achieved-throughput signal (now ~60–160 Mbps with
  real variance, not the flat ~7 Mbps RTT artifact of §9). Winner = log1p, seq_len 8:
  MAE 5.80, RMSE 12.88 (beats persistence 14.02), low-bw MAE 1.368 (beats 1.539) —
  **but loses persistence on *aggregate* MAE (5.80 vs 5.36)**; the `none` transform
  actually beat persistence on both (MAE 5.10) and was arguably the safer pick. The
  selection metric (`low_bw_mae`) favored log1p; with stalls now rare, aggregate MAE
  deserves more weight next round.
- **lstm_pred ablation**: matched-pair mean ΔQoE′ (with − without) = **−1.78** over
  12 pairs (range −39…+11) — i.e. the feature is, if anything, marginally *negative*
  on average, dominated by noise. Consistent with §9's "no consistent benefit." It is
  +1.3 at the winner config, so kept, but the state does not need it.

### 10.5 Caveats & residual levers

- **Training-seed variance is still large**: at the winner config seed 42 plateaus at
  QoE′ ~82 while seed 43 sits at ~62. The multi-seed averaging means the reported
  77.3 is honest (not a spike), but run-to-run stability wants more seeds or epochs.
- **Eager fetching overflows the buffer**: the rule ABRs drop 56–84 frames because
  there is **no request pacing** — a fast policy fetches faster than playback drains
  the 5 s buffer. The DQN avoids it by observing the buffer; adding a "fetch only when
  the buffer has room" gate (ABR-agnostic) is the clean next fix and would also lift
  the rule baselines.
- **Final-eval S mismatch**: `compare.py`/`eval_fixed.py` ran at S=10 while the winner
  trained at S=5 (it generalizes — QoE′ 91.6 at S=10 — but the final eval S should
  track the winning S for a strict apples-to-apples).

### 10.6 Bottom line

Segmentation was the missing piece §9 pointed to: at S=5–8 the point-cloud stream
plays essentially stall-free at high quality (legacy QoE 92–100), the learned policy
sits well above every fixed arm and rule baseline on the quality-stall frontier, and
the S-curve gives a principled operating point (S≈5). The remaining work is
engineering polish (request pacing, seed count, S-matched eval), not a fundamental
transport limit.

### 10.7 Reproduce

```
python run_training.py --jobs 2                        # full round-2 pipeline
python eval_fixed.py --segment-frames 5 --playback-rate-min 0.9   # table 10.2
python compare.py   --segment-frames 5 --playback-rate-min 0.9    # table 10.3 at the winning S
```

### 10.8 Robust winner (round 3, 8 seeds, S-matched eval) — 2026-07-11

Round 2's winner (QoE′ 77.3) was a 2-seed average of a wide spread. Round 3 narrows
to the winning neighborhood (S∈{5,8} × μ=4.3 × lstm-ablation = 4 configs) and runs
**8 training seeds** each (32 trials, 76 min), reporting mean ± std of the selection
metric; the final eval now runs at the **winning S** (auto-read from the sweep result).
Artifacts: `logs/train_runs/20260711_130607_full/`.

**Robust ranking (held-out longdress, QoE′ mean ± std over 8 seeds):**

| trial | config | QoE′ | ±std | mean q | stall |
|---|---|---|---|---|---|
| 3 | S=8, μ=4.3, **lstm off** | **74.60** | 19.79 | 0.909 | 3.2 s |
| 0 | S=5, μ=4.3, lstm on | 74.23 | 20.26 | 0.909 | 3.3 s |
| 2 | S=5, μ=4.3, lstm off | 73.77 | 20.63 | 0.905 | 3.2 s |
| 1 | S=8, μ=4.3, lstm on | 72.60 | 19.03 | 0.893 | 3.3 s |

**Three findings, all honest:**

1. **All four configs are statistically tied** (74.6…72.6, every ±std ≈ 20). S=5 vs S=8
   and lstm-on vs lstm-off are indistinguishable given the seed variance — the round-2
   "S=5 wins" was within noise. The robust operating point is simply **S≈5–8, μ=4.3**.
2. **The lstm_pred ablation verdict is now settled: the feature is irrelevant.** The
   robust winner *drops* it (lstm off) and is tied with the lstm-on config. Consistent
   with rounds 1–2; across 8 seeds it neither helps nor hurts.
3. **The binding constraint is now training variance, not transport.** Even with 8
   seeds the winner's std is ~20 QoE′ (per-seed range **38 → 96**). The *good* seeds
   (90–96) match or beat the best fixed arm; the bad ones (~38) drag the mean to 74.6.
   Round-2's 77.3 was the top of this band; **74.6 ± 19.8 is the honest level.**

**S-matched final comparison (S=8, AMP, single held-out trace):**

| strategy | legacy QoE | QoE′ | mean q | stall |
|---|---|---|---|---|
| best fixed arm (arm 1, always 59.6 Mbps) | 100 | **90.3** | 0.921 | 0.0 s |
| bandwidth / LSTM rule | 0 | 2.5 | 0.030 | 0.0 s |
| DQN (installed winner, single-seed draw) | 75.1 | 82.0 | 0.968 | 3.0 s |

**The classic "no-headroom" result returns.** Once segmentation makes the link
comfortable, a *static* always-medhigh arm streams at zero stall and QoE′ 90.3 — and
the DQN does **not** clearly beat it on the robust mean (74.6 ± 19.8), because its good
seeds (~90+) match arm 1 but ~40% of seeds underperform. This mirrors the original 4G
finding (§4–5): when adaptation headroom is small, a well-chosen fixed arm is a strong
baseline and the value of learned ABR is gated by **RL stability**, not the
environment. On this comfortable 5-static-trace regime the honest takeaway is: the
learned policy *can* match the best static arm but training is unreliable; the next
lever is variance reduction (n-step returns / prioritized replay / dueling, or seed
ensembling) — not more transport realism.

### 10.9 Where the project stands

The two-round transport overhaul achieved its goal: point-cloud streaming now plays at
high quality with little/no stall (legacy QoE off 0, up to 100 for the best fixed arm
and the good DQN seeds). The learned policy reaches that frontier but does not yet
dominate a strong static baseline on this comfortable regime, and its run-to-run
variance is the honest headline. Highest-value next steps: (a) **RL variance reduction**
(§6.1) to make the learned policy reliably beat the best arm; (b) a **harder/more
variable network regime** (more traces, mid-session bandwidth cliffs) where adaptation
has real headroom to exploit; (c) **request pacing** to stop the eager-fetch buffer
overflow that still cripples the rule baselines (§10.5).

### 10.10 Transport fidelity: integral serialization (2026-07-12)

The RTT-round TCP model used to sample the trace capacity once per round and hold
it, so a transfer that STARTED inside a fade was charged the fade rate for its whole
first round — the demo's segment 0 took 7.8 s because one packet was billed 5.84 s
at the trace's idle-attach 2 kbps even though the link recovered at t=3 s. Real
links deliver bits at the instantaneous rate, so serialization is now the exact
integral of the piecewise-constant capacity (`BandwidthTrace.time_to_transmit`,
threaded through `_TraceCapacity` → `TCPConnection.send`): a round finishes as soon
as the accumulated capacity·dt covers its bytes. Scalars/bare callables keep the
legacy formula; flat-trace timing is bit-identical (unit-tested, 17/17).

Effect: startup on the held-out trace drops 7.8 s → 4.9 s (fade integral ~3 s + the
genuine TCP slow-start ramp). Because the download clock shifts ~3 s relative to
the trace, per-run numbers move: the demo run now clears the mid-session dip
without stalling (QoE 100 / QoE′ 96.2 / quality 0.967; compare.py DQN row 96.1 vs
rules 2.5). Steady-state segment timings are unchanged, and all §10 comparisons
were made under one consistent model, so the S-curve and robust-winner conclusions
stand; only absolute startup timings tightened. (The remaining known
over-pessimism: the trace's opening 2 kbps samples are idle-device THROUGHPUT, not
capacity — an attach-time artifact of the dataset, kept as-is for honesty.)

## 11. Round-4 prep: mixed regime, QoE″ v2, reward ablation (2026-07-14)

Setup changes ahead of the round-4 retrain (all committed; run executes on the
fast PC via `run_training.py --jobs 2`):

**Dataset.** The 16 driving 5G traces return (removed in §9 as "unsuitable" —
a verdict issued under the pre-§10.10 transport model that over-charged fades).
With integral serialization they are realistic, and they supply what the static
regime lacked: capacity crosses the 6-tier ladder 57–76 % of the time, so
adaptation has genuine headroom and fixed arms can actually be beaten. Pool:
21 traces (5 static + 16 driving), ~11.4 h, split seed 42 → 17 train / 4
held-out (3 driving + 1 static; one held-out driving trace contains a genuine
137 s coverage blackout — reported per-trace, not hidden). Traces keep their
near-zero idle-attach openings (restored): under the integral transport these
are a legitimate cold-start scenario, not a pathology. The train/test split is
now DECOUPLED from the RL seed (`--split-seed`, fixed 42): previously each seed
evaluated on a different held-out set, which on a heterogeneous pool would have
conflated split luck with policy variance (part of round-3's ±19.8).

**QoE″ v2.** Two tracked-but-free perceptual costs join the quality-aware QoE
(raw network parameters stay out by design — QoE is what the user perceives):

    QoE″ = 100·mean_q − 4.3·stall_s − 1.0·Σ|dq| − 10·slowdown_integral
           − 1.0·startup_delay_s − (100/N)·frames_dropped

Startup (1.0/s ≈ ¼ of the stall weight; waiting at t=0 annoys less than
mid-stream freezing) is binding now that cold starts are real; stalls only
accrue after playback begins, so there is no double-count. Drops charge the
rule baselines honestly (56–84 dropped frames were previously free). The
simulator summary prints the per-term breakdown; v1 = v2 minus the two new
terms, so §10 numbers remain interpretable. Absolute QoE′ comparability with
§9–10 is intentionally broken (new regime anyway).

**Reward ablation (the round-4 second axis).** The reward spec gains optional
`startup_weight` / `drop_weight` (defaults 0.0 = old shape, unit-tested
bit-identical). Sweep axes: reward v1 (settled bounded shape) vs v2
(startup 1.0 + drop 1.0) × lstm on/off × 12 seeds = 48 trials, S=8 and μ=4.3
fixed. Open questions the run answers: does charging cold-start waiting during
training improve startup behavior; is lstm_pred still irrelevant on the
low-autocorrelation driving traces; how much of round-3's seed variance
survives a fixed split + 4-trace held-out eval + non-trivial task.

**Introspection.** `run_dqn.py` now logs every decision of the trained policy
(state seen, Q-values for all 6 tiers, chosen tier + margin, per-segment
reward from the winner's own training objective) to the console and
`logs/dqn/decisions.csv`, and the summary prints the accumulated RL reward —
directly comparable to `dqn_sweep_results.json`. Tests: 17 transport + 7 new
QoE/reward = 24.

## 12. Round-4 results: the variance problem is solved (2026-07-15)

48 trials (4 configs × 12 seeds), 20 epochs each on the 17-trace mixed train
split, evaluated on the FIXED 4-trace held-out (3 driving + 1 static) × 3 eval
seeds. All numbers are QoE″ v2 (startup + drop terms included).

| config | QoE″ (12-seed) | q | stall |
|---|---|---|---|
| **reward-v2 · lstm-ON (winner)** | **54.50 ± 0.95** | 0.775 | 1.8 s |
| reward-v1 · lstm-OFF | 54.33 ± 0.86 | 0.782 | 1.8 s |
| reward-v2 · lstm-OFF | 53.56 ± 1.61 | 0.759 | 1.7 s |
| reward-v1 · lstm-ON | 53.43 ± 1.43 | 0.777 | 1.9 s |

**1. Seed variance collapsed: ±19.8 → ±0.95.** Round-3's binding constraint is
gone. Per-seed spread is 51.9–55.3 (was 38–96). The fix was NOT algorithmic
(no dueling/n-step needed — Phase 3 cancelled): it was (a) a non-trivial
regime where decisions matter, (b) the split decoupled from the RL seed,
(c) 4-trace held-out eval, (d) 12 seeds, (e) 20 saturated epochs.

**2. The DQN now clearly beats every fixed arm: 54.5 vs 40.6 (+13.9).** In the
driving regime no static tier works — arm_0 stalls 76 s, arm_5 wastes 97% of
quality; the best compromise (arm_3, always-med) reaches only 40.6. Per-trace
shows WHERE the DQN wins: it ~ties the best arm on the 137 s-blackout trace
(33.0 vs 32.2 — nothing can stream through a blackout) but nearly doubles it
on the static trace (85.1 vs 41.1) because it rides high tiers when capacity
allows while the fixed arm is stuck at med. That asymmetry — match the safe
policy in the worst case, crush it in the good case — is the adaptation story
round 1–3 could never show.

**3. Both ablations are settled null results.** lstm_pred: irrelevant even on
low-autocorrelation driving traces (Δ < 1.1, within noise) — the measured
tput last/mean/std features carry all usable signal. Reward v2 vs v1: no
outcome difference (the policy already avoids slow startups/drops because
they cost quality-time); winner keeps v2 since it aligns objective and metric
at zero cost.

**4. 20 epochs over-saturated (good).** Winner-config best checkpoints landed
at mean position 0.10 of training (0/12 in the final quarter) — convergence is
fast on this task and the run definitively brackets it. No further training
warranted: this is the final model.

Installed winner: trial 1 (reward-v2, lstm-ON), models/abr_dqn.pkl.
Caveat for the paper: winner vs runner-ups is within noise — the honest claim
is "all four configs are equivalent at ≈54 ± 1.5"; we ship trial 1's median
behavior. Per-trace chart data: dqn_sweep_results.json
(qoe_quality_per_trace), per-trial history eval entries, fixed-arm per_trace.
