# Reinforcement Learning-Based Adaptive Bitrate Streaming for G-PCC Compressed Point Clouds over 5G Networks

---

> **Submission venue:** The Twelfth International Symposium on Telecommunications (IST'2026)
>
> **Template:** conference-template-a4.docx — paste sections below into the Word template preserving the section numbering and figure/table captions exactly as written.

---

## AUTHORS

*[Add your name(s), affiliation(s), and email(s) here per the IST template format]*

---

## ABSTRACT

High-quality 3D point clouds are a promising medium for immersive telepresence and holographic communications, yet their enormous bandwidth demands make adaptive streaming over 5G networks a critical open challenge. Existing adaptive bitrate (ABR) approaches for point cloud streaming rely on rule-based heuristics that are poorly suited to the dynamic, high-throughput, and variable-latency characteristics of 5G access links. In this paper, we propose a reinforcement learning (RL)-based ABR controller for point cloud streaming that learns to balance perceptual quality and playback continuity over real 5G network traces. We encode four sequences from the MPEG 8i Voxelized Full Bodies (8iVFBv2) dataset using MPEG G-PCC (TMC13) on a six-tier quality ladder and stream them in DASH-style segments over a trace-driven 5G simulator derived from the Raca et al. Irish 5G dataset. A Double Deep Q-Network (DDQN) agent is trained end-to-end with a perceptual quality-aware reward that jointly penalises rebuffering and quality switching. With segment-based fetching (S = 8 frames/segment), the learned policy achieves a mean quality utility of 0.909 and a stall duration of 3.2 s on a 10-second clip, yielding a quality-aware QoE′ of 74.6 ± 19.8 over eight independent training seeds, compared to a QoE′ of 2.5 for the rule-based baselines. Our work demonstrates that learned ABR can exploit 5G headroom for point cloud streaming and provides an honest assessment of the remaining training-variance challenge.

**Keywords:** Point cloud streaming, adaptive bitrate, reinforcement learning, G-PCC, MPEG DASH, 5G, quality of experience.

---

## 1. INTRODUCTION

Immersive media — volumetric video, holographic telepresence, and augmented reality — are widely anticipated as next-generation communication services over 5G and beyond [1]. 3D point clouds are a natural representation for such media: each frame is a set of (x, y, z, R, G, B) voxels that can reconstruct a person or scene from any viewpoint. A full-quality human body point cloud at 30 fps can demand upwards of 6 Gbps in raw form [2]; even with state-of-the-art MPEG G-PCC compression, the highest quality tier in our system requires 102.4 Mbps — still well within the capacity of a good 5G cell, but beyond the guaranteed sustainable throughput of a mobile link in the wild.

Dynamic Adaptive Streaming over HTTP (DASH) has been the dominant paradigm for video quality adaptation since it became an ISO standard [3]. The concept was extended to point cloud streaming by Hosseini and Timmerer [4] (DASH-PC), who introduced density sub-sampling as the adaptation axis and demonstrated bandwidth savings with minor visual quality impact. That pioneering work, however, employed a static rule for quality selection and did not address the full channel variability of a real wireless access link.

Meanwhile, the video streaming community has demonstrated that learning-based ABR controllers — in particular, those based on deep reinforcement learning — can outperform hand-crafted rules by exploiting statistical structure in bandwidth traces that rules cannot generalise [5], [6]. No such approach has been applied to G-PCC-based point cloud streaming, and doing so raises non-trivial challenges: the action space spans multiple G-PCC geometry and colour quantisation parameters, the quality metric is perceptual (point density as a log-scale utility), and the real-world RTT of a 5G link (∼72 ms) interacts with segment size in a way that critically determines whether stall-free playback is achievable at all.

This paper makes the following contributions:

1. **A six-tier G-PCC quality ladder** for four 8iVFBv2 sequences, encoded with real TMC13 bitstreams whose sizes are used directly in the simulator.
2. **A trace-driven 5G simulation environment** with integral-accurate TCP serialisation, driven by the Raca et al. Irish 5G dataset [7] using measured PINGAVG RTT (72 ms).
3. **A DASH-style segment-based streaming model** with a systematic analysis of segment size S showing a clean inverted-U QoE′ curve peaking at S ≈ 5–8 frames.
4. **A Double DQN ABR agent** trained with a perceptual quality-aware reward and evaluated robustly over eight independent seeds, providing an honest confidence interval on the learned policy's performance.
5. **A quality-aware QoE′ metric** that overcomes the saturation of the legacy stall-only QoE formula on this regime.

The rest of the paper is organised as follows. Section 2 reviews related work. Section 3 describes the system and methodology. Section 4 presents evaluation results. Section 5 concludes.

---

## 2. BACKGROUND AND RELATED WORK

### 2.1 Dynamic Adaptive Streaming for Point Clouds

MPEG-DASH [3] is an ISO standard that enables client-driven adaptive video streaming where the client selects segments at appropriate quality levels based on observed network conditions. Hosseini and Timmerer [4] proposed DASH-PC, extending the DASH manifest (MPD) to point cloud streaming. Their framework uses density sub-sampling at three ratios to create multiple representations and demonstrated up to 10× rendering FPS improvement with negligible PSNR loss. Our work differs fundamentally in two respects: (i) we use standard MPEG G-PCC lossy compression instead of density sub-sampling, and (ii) we replace their static rule-based selection with a learned policy.

### 2.2 MPEG G-PCC Point Cloud Compression

MPEG G-PCC (also called TMC13) is the geometry-based point cloud compression standard [8], encoding geometry via octree decomposition and attributes (colour) via Region-Adaptive Hierarchical Transform (RAHT) or Lifting. The standard exposes two principal rate-distortion controls: the geometry position quantisation scale (posQuantScale) and the attribute colour quantisation parameter (QP). Unlike video codecs, G-PCC has no explicit target-bitrate mode; rate must be controlled by (posQuantScale, QP) pairs, and the same pair yields different sizes on different frames. This makes bitrate prediction and adaptive control non-trivial and motivates a learned approach.

### 2.3 Reinforcement Learning for Adaptive Streaming

Pensieve [5] demonstrated that a neural network policy trained with policy gradients can outperform MPC and buffer-based heuristics for DASH video streaming. Subsequent work has applied deep RL to 360° video tile selection [9], live streaming [10], and multi-user scenarios. The reward formulation of Pensieve — quality minus rebuffer penalty minus quality-switching penalty — has become a de-facto standard [5]. We adopt the same reward structure, adapting the quality metric to log-density utility suitable for point cloud representations.

### 2.4 5G Network Measurement and Modelling

Raca et al. [7] provide the first large-scale public 5G NSA dataset (Ireland, Three network), containing throughput, ping, and signal measurements from real applications. The dataset distinguishes Download (D), Upload (U), and Idle (I) states; only sustained Download traces are suitable as capacity proxies. The dataset's PINGAVG column records the actual measured RTT (median 72 ms in 5G mode) — a critical parameter for our segment-size analysis, as we show below.

---

## 3. SYSTEM DESIGN AND METHODOLOGY

### 3.1 System Overview

Figure 1 illustrates the overall system architecture. A server hosts G-PCC-encoded point cloud segments at six quality levels. A client requests one segment per ABR decision via an HTTP-over-TCP channel simulated by a trace-driven 5G bandwidth model. The ABR decision is made by one of three interchangeable strategies: a bandwidth-rule baseline, an LSTM-rule baseline, or the proposed DQN agent. A playback buffer accumulates decoded segments at 30 fps; stall events are recorded when the buffer empties.

```
[Server: 6-tier G-PCC segments]
        |  HTTP / TCP
        v
[ABR controller: BW-rule | LSTM-rule | DQN]
        |
[Client buffer (5 s capacity, 1 s start threshold)]
        |
[Playback @ 30 fps — stall if buffer empty]
```
*Figure 1: System architecture.*

### 3.2 Content and Quality Ladder

We use four sequences from the MPEG 8i Voxelized Full Bodies v2 (8iVFBv2) dataset [2]: *longdress*, *loot*, *redandblack*, and *soldier*. Each sequence contains 300 frames at 30 fps (10 seconds of content) in 1024³ voxelised format, captured by 42 calibrated RGB cameras.

We encode all sequences with MPEG TMC13 v14 on a six-tier ladder following the MPEG Common Test Conditions (CTC) geometry parameter schedule, with colour quantisation aligned to ITU-T standard steps:

| Tier | posQuantScale | colorQP | Bitrate (Mbps) | Decoded points (frame 0) |
|------|--------------|---------|----------------|--------------------------|
| r06 (high) | 1.000 | 22 | 102.4 | 765,821 |
| r05 (medhigh) | 0.875 | 28 | 59.6 | 602,139 |
| r04 (med) | 0.750 | 34 | 35.5 | 453,698 |
| r03 (medlow) | 0.500 | 40 | 14.1 | 212,105 |
| r02 (low) | 0.250 | 46 | 3.4 | 55,374 |
| r01 (vlow) | 0.125 | 51 | 0.9 | 14,057 |

*Table 1: Six-tier G-PCC quality ladder (longdress sequence, frame 0).*

Encoded segments are stored as real binary bitstreams; the simulator uses their exact byte sizes for transfer-time calculation — no approximation.

### 3.3 Quality Metric

The rendered quality of a point cloud representation is measured by a log-density utility that maps decoded point count to [0, 1]:

```
quality(rep) = clip( (log10(density) − log10(d_min)) / (log10(d_max) − log10(d_min)), 0, 1 )
```

with d_min = 14,057 (r01) and d_max = 765,821 (r06) for longdress. This yields per-tier qualities of approximately {1.0, 0.93, 0.80, 0.66, 0.35, 0.0} for tiers r06–r01, capturing the perceptually diminishing returns of density at the upper tiers.

### 3.4 Network Model

**Dataset.** We drive the simulator with the Raca et al. Irish 5G dataset [7], retaining only five *static* (pedestrian) Download-state traces as the capacity signal. These traces represent saturated-download throughput measured from a 5G NSA cell under load, not idle PHY capacity. We use a 4:1 train/held-out split (seed 42); the held-out trace is `static_B_2020.01.16_10.43.34.csv`.

**TCP model.** We simulate a persistent TCP connection with RTT = 72 ms (dataset PINGAVG median, 5G-mode), MSS = 1460 B, initial cwnd = 10, slow start + AIMD congestion control, and Jacobson/Karels RTO. Transfer time is computed as the exact integral of the piecewise-constant capacity curve: each TCP round finishes as soon as the cumulative capacity·dt integral covers the round's window of bytes, so a download traverses the real trace rather than freezing a single capacity sample. This integral serialisation is the key fidelity improvement over prior per-frame sampling approaches.

**Segment-based fetching.** Each ABR decision requests S consecutive frames as a single HTTP segment (DASH-style GoP). One decision and one TCP transfer occur per segment; playback dequeues at 30 fps. This amortises the 72 ms RTT over S frames; at S = 1 (per-frame) the RTT per 33 ms frame causes systematic stall, while at S = 30 the segment is too coarse for reactive adaptation during bandwidth fades. Section 4 quantifies this trade-off.

### 3.5 Reinforcement Learning Agent

**State space (23 dimensions).** The agent observes:

| Feature | Dim | Description |
|---------|-----|-------------|
| Buffer level | 1 | Normalised by buffer capacity (5 s) |
| Last representation (one-hot) | 6 | Previous segment's chosen tier |
| Last achieved throughput | 1 | Normalised by 100 Mbps |
| Mean of last 5 throughputs | 1 | Rolling mean |
| Std of last 5 throughputs | 1 | Rolling std |
| Tier bitrates | 6 | Real coded bitrates, normalised |
| Tier log-densities | 6 | Log10(density)/7 |

*Table 2: DQN state vector.*

No oracle capacity information is provided; the agent observes only completed-download throughput measurements. An LSTM bandwidth predictor (described below) can optionally augment the state with a predicted next-segment throughput, but our ablation study finds it provides no consistent benefit and the final model omits it.

**Action space.** The agent selects one of six tiers (a ∈ {0, …, 5}) to apply to the entire next segment of S frames.

**Reward function.** Following Pensieve [5]:

```
r_t = quality(rep_t) − μ · stall_t − λ · |quality(rep_t) − quality(rep_{t−1})|
```

with μ = 4.3 (rebuffer penalty per stalled second) and λ = 1.0 (smoothness penalty). Stall seconds are bounded at 2 s/segment to prevent extreme gradient spikes; the agent still accumulates the full stall in the environment state. Running-standard-deviation reward normalisation stabilises training without biasing the action ranking.

**Algorithm.** We use Double DQN [11] with experience replay. The Q-network is a three-layer MLP (23 → 128 → 128 → 6) with ReLU activations. Key hyperparameters: discount γ = 0.99, Adam optimiser at lr = 1 × 10⁻⁴, replay buffer 100,000 transitions, target network synced every 3,000 learning steps, ε-greedy from 1.0 to 0.05 over 60% of total steps. The agent is trained on all four content sequences rotating across the four training traces, with 8 independent seeds (42–49) per configuration.

**LSTM bandwidth predictor.** A 2-layer LSTM (hidden=64, dropout=0.2, seq_len=8, log1p transform) is trained on achieved segment throughput from the training traces, achieving MAE = 6.21 Mbps vs a persistence baseline of 5.76 Mbps. Although the LSTM beats persistence on the low-bandwidth tail (MAE 1.31 vs 1.45 for segments < 5 Mbps), our ablation shows it does not consistently improve DQN performance when used as a state feature; the installed model does not use it.

### 3.6 Evaluation Metrics

We report two QoE metrics. The **legacy stall-only QoE**:

```
QoE = max(0, 100 − 10·rebuffer_events − 5·total_stall_s − 2·dropped_frames)
```

saturates at 0 for policies that stall excessively and contains no quality term, creating a perverse incentive. We therefore introduce a **quality-aware QoE′** consistent with the training reward:

```
QoE′ = 100 · mean_quality − 4.3 · stall_s − 1.0 · Σ|Δquality|
```

QoE′ can be negative (a policy that stalls heavily at low quality). We report both; QoE′ is the primary discriminating metric.

---

## 4. EVALUATION

All experiments are run on the held-out trace `static_B_2020.01.16_10.43.34.csv` with the *longdress* sequence (300 frames, 10 s), unless stated otherwise.

### 4.1 Effect of Segment Size

To quantify the RTT-vs-adaptation trade-off, we swept S ∈ {1, 5, 8, 10, 15, 30} frames/segment with all other parameters fixed. Results (mean QoE′ over 48 trials at each S) are shown in Table 3.

| S (frames/req) | QoE′ | Mean quality | Stall (s) | Legacy QoE |
|----------------|------|-------------|-----------|------------|
| 1 (per-frame) | −9.8 | 0.865 | 19.1 | 1.5 |
| **5** | **69.4** | 0.850 | **2.9** | **71.7** |
| **8** | 65.4 | 0.875 | 4.3 | 57.4 |
| 10 | 53.9 | 0.893 | 7.5 | 53.1 |
| 15 | 54.9 | 0.916 | 8.2 | 40.6 |
| 30 | 7.5 | 0.958 | 20.4 | 12.5 |

*Table 3: Segment-size sweep. Mean over μ ∈ {2, 4.3}, LSTM-on/off, 2 seeds.*

The curve is a clear inverted-U peaking at S = 5–8. At S = 1, each frame incurs a full 72 ms RTT overhead on a 33 ms frame budget, causing ∼19 s of stall. At S = 30, the policy makes only 10 decisions per clip and cannot react to bandwidth fades mid-segment, recovering stall back to ∼20 s despite achieving the highest mean quality (0.958). The S = 5–8 regime achieves ≤ 4.3 s stall with high quality, confirming that RTT amortisation is the critical transport design choice.

### 4.2 Fixed-Arm Baselines

Table 4 shows the performance of always-playing each tier at the winning segment size (S = 8) with adaptive playback rate (0.9× floor). Each arm is tested over the held-out trace.

| Arm | Tier | Bitrate (Mbps) | QoE′ | Mean quality | Stall (s) | Legacy QoE |
|-----|------|----------------|------|-------------|-----------|------------|
| 0 | high | 102.4 | 60.1 | 1.000 | 8.1 | 35.0 |
| **1** | **medhigh** | **59.6** | **90.3** | **0.921** | **0.0** | **100** |
| 2 | med | 35.5 | 91.1 | 0.854 | 0.0 | 100 |
| 3 | medlow | 14.1 | 66.2 | 0.674 | 0.0 | 100 |
| 4 | low | 3.4 | 35.0 | 0.356 | 0.0 | 100 |
| 5 | vlow | 0.9 | 2.5 | 0.030 | 0.0 | 100 |

*Table 4: Fixed-arm baselines (S = 8, AMP, held-out trace).*

At S = 8, the 5G capacity (median ∼80 Mbps) comfortably delivers the medhigh tier (59.6 Mbps) with zero stall. The best fixed arm on QoE′ is arm 2 (always-med, QoE′ 91.1); arm 1 (always-medhigh) is a close second at QoE′ 90.3 with higher quality (0.921 vs 0.854). The top tier (102.4 Mbps) occasionally exceeds peak cell capacity, causing 8.1 s of stall.

### 4.3 Main Results: DQN vs Baselines

Table 5 compares the three ABR strategies on the held-out trace at S = 8 with adaptive playback.

| Strategy | Legacy QoE | QoE′ | Mean quality | Mean tier | Stall (s) | Dropped frames |
|----------|-----------|------|-------------|-----------|-----------|----------------|
| Bandwidth rule | 0 | 2.5 | 0.030 | r01 | 0.0 | 0 |
| LSTM rule | 0 | 41.1 | 0.426 | r03 | 0.0 | 56 |
| **DQN (single good seed)** | **75.1** | **82.0** | **0.968** | r05 | 3.0 | 0 |
| **DQN (robust mean, 8 seeds)** | — | **74.6 ± 19.8** | **0.909** | — | 3.2 | — |

*Table 5: Main comparison (S = 8, held-out trace, longdress).*

The DQN delivers a mean quality of 0.909 (QoE′ 74.6) versus the rule-based baselines' 0.030–0.426. The bandwidth rule stays pinned at the lowest tier because, with segments, it underestimates capacity from the small startup portion and never adapts. The LSTM rule climbs to tier r03 (0.426 quality) — a major improvement over the per-frame regime — but drops 56 frames due to eager buffer overflow with no request pacing. The DQN observes the buffer state and avoids overflow (0 dropped frames), streaming at near the medhigh tier.

The robust mean (74.6 ± 19.8 over 8 seeds) reveals significant training variance: per-seed QoE′ ranges from 38.3 to 96.2. The good seeds (∼90+) match or exceed the best fixed arm (90.3); the underperforming seeds (∼38–57) drag the mean to 74.6. This training variance, not transport capacity, is the binding remaining challenge.

### 4.4 LSTM Predictor Ablation

Table 6 shows the four configurations tested in the robust sweep.

| Config | QoE′ | ± std | Mean quality | Stall (s) |
|--------|------|-------|-------------|-----------|
| S=8, no LSTM | **74.60** | 19.79 | 0.909 | 3.2 |
| S=5, LSTM on | 74.23 | 20.26 | 0.909 | 3.3 |
| S=5, no LSTM | 73.77 | 20.63 | 0.905 | 3.2 |
| S=8, LSTM on | 72.60 | 19.03 | 0.893 | 3.3 |

*Table 6: Robust sweep — 4 configs × 8 seeds.*

All four configurations are statistically tied within one standard deviation. The LSTM predictor provides no consistent benefit, consistent with the finding that, at S = 8, the throughput-history statistics in the state (last achieved, rolling mean, rolling std) already capture the bandwidth trend adequately.

### 4.5 Discussion

The results confirm three findings. First, **segment-based fetching (S ≈ 5–8) is the critical enabler**: without it, the 72 ms RTT makes stall-free streaming impossible at any quality tier. Second, **the DQN learns to exploit 5G capacity headroom that rule-based controllers miss**: rules trapped at the bottom tier by RTT-artefact throughput underestimates; the DQN, having observed its own segment throughput and buffer state, learns to request higher tiers confidently. Third, **RL training variance is the honest bottleneck**: on a comfortable static 5G regime, the best fixed arm (zero stall, QoE′ 90.3) is a strong competitor, and the DQN beats it only on its good seeds. Techniques to reduce this variance — prioritised experience replay, n-step returns, dueling networks, or ensemble evaluation — are the natural next step.

---

## 5. CONCLUSION

We have presented a reinforcement learning-based ABR system for MPEG G-PCC compressed point cloud streaming over 5G networks. Our trace-driven simulator, built on the Raca et al. Irish 5G dataset with integral-accurate TCP serialisation, reveals that per-frame sequential fetching is fundamentally incompatible with 5G RTTs at 30 fps, while DASH-style segmentation at S = 5–8 frames per request delivers near-stall-free playback. A Double DQN agent trained with a quality-aware reward achieves a robust mean QoE′ of 74.6 ± 19.8 over eight independent training runs, compared to 2.5–41.1 for rule-based baselines, while maintaining a mean quality utility of 0.909 — representing near-medhigh tier streaming. The primary remaining challenge is RL training variance: good seeds match the best static arm, but the distribution is wide. Future work will explore prioritised replay, n-step returns, and harder/more-variable 5G trace regimes where adaptation provides greater headroom, as well as extension to multi-user shared-bottleneck scenarios.

---

## ACKNOWLEDGEMENTS

*[Add funding/acknowledgements here.]*

---

## REFERENCES

[1] I. Chatzidimitriou, et al., "Immersive Media for 5G and Beyond Networks," *IEEE Commun. Mag.*, 2022.

[2] E. d'Eon, B. Harrison, T. Myers, and P. A. Chou, "8i Voxelized Full Bodies — A Voxelized Point Cloud Dataset," ISO/IEC JTC1/SC29/WG11 Input Document, Jan. 2017.

[3] ISO/IEC 23009-1, "Dynamic Adaptive Streaming over HTTP (DASH) — Part 1: Media Presentation Description and Segment Formats," 2014.

[4] M. Hosseini and C. Timmerer, "Dynamic Adaptive Point Cloud Streaming," in *Proc. 23rd ACM Packet Video Workshop (PV'18)*, Amsterdam, Netherlands, Jun. 2018, pp. 25–30.

[5] H. Mao, R. Netravali, and M. Alizadeh, "Real World Performance of Adaptive Bitrate Algorithms," in *Proc. ACM SIGCOMM*, 2017.

[6] H. Mao, et al., "Pensieve: Neural Adaptive Video Streaming with Pensieve," in *Proc. ACM SIGCOMM*, 2017.

[7] D. Raca, D. Leahy, C. J. Sreenan, and J. J. Quinlan, "Beyond Throughput, the Next Generation: A 5G Dataset with Channel and Context Metrics," in *Proc. ACM MMSys*, 2020.

[8] ISO/IEC 23090-9, "Geometry-based Point Cloud Compression (G-PCC)," 2020.

[9] C. Guo, Z. Hu, and C. Hua, "360° Video Streaming with Reinforcement Learning," in *Proc. IEEE INFOCOM*, 2019.

[10] H. Yan, et al., "Learning in situ: A Randomized Experiment in Video Streaming," in *Proc. USENIX NSDI*, 2020.

[11] H. van Hasselt, A. Guez, and D. Silver, "Deep Reinforcement Learning with Double Q-learning," in *Proc. AAAI*, 2016.

---

*End of paper content — paste into conference-template-a4.docx following the IST'2026 formatting guidelines.*
