# Deep Reinforcement Learning-Based Adaptive Point Cloud Streaming over 5G Networks



**Authors:** *[Given Name Surname, dept., organization, City, Country, bura@panto.org — fill in]*

---

## Abstract

Abstract—High-quality point clouds enable immersive volumetric media, but their extreme bandwidth demands make adaptive streaming over mobile networks an open challenge. This paper presents a learning-based adaptive bitrate (ABR) controller for point cloud streaming. Four sequences of the 8i Voxelized Full Bodies dataset are encoded with the MPEG Geometry-based Point Cloud Compression (G-PCC) reference codec into a six-tier quality ladder and streamed as DASH-style segments over a trace-driven 5G simulator built from a public Irish 5G dataset with its measured 72 ms round-trip time. Per-frame fetching cannot sustain 30 frames per second at this latency; a segment-size sweep shows a quality-of-experience (QoE) peak at ten frames per request, which the final system adopts together with request pacing that admits a segment only when it fits the client buffer. A Double Deep Q-Network agent is trained with the same five-term objective used for evaluation—normalized quality minus stall duration, rebuffering-event frequency, segment-quality changes, and startup delay—so that its discounted return equals the reported QoE exactly. On a registered, once-only test split of four held-out traces evaluated across all four sequences, the learned policy reaches a QoE of 49.9, improving on the buffer-based and model-predictive-control (MPC) baselines by about 9 percent and on the best fixed-quality arm by 59 percent, and it wins 83 percent of matched scenarios against MPC. Across twelve training seeds the test QoE holds at 49.9 ± 0.95.

**Keywords**—point cloud streaming, adaptive bitrate, deep reinforcement learning, G-PCC, MPEG-DASH, 5G, quality of experience

---

## I. Introduction

Immersive volumetric media, such as holographic telepresence and augmented reality, are among the flagship services envisioned for 5G networks and beyond. Dynamic 3D point clouds are a natural representation for such media: each frame is a set of colored points that can be rendered from any viewpoint. Uncompressed, a single high-quality human-body sequence at 30 frames per second (fps) can exceed 6 Gbit/s [2]. Even after compression with the MPEG Geometry-based Point Cloud Compression standard (G-PCC) [4], [5], the highest quality tier used in this work requires 102.4 Mbit/s, which a mobile 5G link can deliver on average but not at every instant.

Dynamic Adaptive Streaming over HTTP (DASH) [1] addresses throughput variability by encoding content at multiple quality levels and letting the client select a level per segment. Hosseini and Timmerer extended this concept to point clouds with DASH-PC [3], using spatial sub-sampling to generate density representations. Their adaptation logic, like most deployed ABR schemes, is rule-based. For video, learning-based controllers such as Pensieve [6] have shown that a policy trained with reinforcement learning (RL) can outperform hand-crafted rules. This idea is now being carried into volumetric media: recent work applies deep RL to real-time volumetric video streaming through rolling prediction-and-optimization [13] and to perceptual-quality-aware volumetric adaptation [14], while the broader shift toward learning-based adaptive streaming is charted in a recent tutorial and survey [15]. Applying it specifically to standard G-PCC point cloud streaming, however, raises challenges these works do not jointly address: the rate ladder spans two orders of magnitude, perceived quality is governed by point density and coding rate rather than pixel fidelity, and the interaction between the 5G round-trip time (RTT) and the request granularity determines whether stall-free playback is possible at all.

This paper makes four contributions. First, a complete open simulation pipeline that streams real G-PCC bitstreams (six-tier ladder, four 8i Voxelized Full Bodies sequences [2]) over a trace-driven 5G channel with a TCP model whose serialization time is the exact integral of the measured time-varying capacity. Second, a quantitative analysis of segment size: per-frame fetching is round-trip-time (RTT) bound, and the measured QoE peaks at ten frames per request before larger segments react too slowly to fades. Third, a Double Deep Q-Network (DQN) [10] ABR agent trained on a quality-aware reward whose discounted return is, by construction, identical to the reported QoE metric. Fourth, a robust 12-seed evaluation on a registered, once-only test split in which the learned policy beats a buffer-based controller and an MPC controller by about 9 percent QoE and the best fixed arm by 59 percent, with a 12-seed standard deviation below 1 and an 83 percent per-scenario win rate against MPC.

## II. Related Work

### A. Point cloud compression and streaming

MPEG standardized two point cloud codecs: V-PCC, which projects the cloud onto video planes, and G-PCC, which codes geometry directly with octrees and attributes with hierarchical transforms [4], [5]. G-PCC exposes rate control through the geometry position quantization scale and the attribute quantization parameter (QP); it has no target-bitrate mode, so a bitrate ladder must be built from parameter pairs. DASH-PC [3] pioneered manifest-driven adaptive point cloud streaming with density sub-sampling. Van der Hooft et al. proposed PCC-DASH [8], rate-adapting V-PCC streams with heuristic policies. More recent volumetric streaming systems have begun to adopt learned adaptation [13], [14], though typically for V-PCC or mesh-based content and often with viewport or field-of-view assumptions. Our work differs by using standard G-PCC bitstreams, by replacing heuristic adaptation with a learned policy, and by making the training reward identical to the reported QoE.

### B. Learning-based bitrate adaptation

Rule-based ABR uses throughput estimates [7] or buffer occupancy; Pensieve [6] showed that an RL policy trained in simulation generalizes across network conditions and outperforms such rules for 2D video. Its reward, quality minus rebuffering and quality switching, has become standard, and recent surveys track the rapid extension of learning-based adaptation to sustainable, energy-aware, and volumetric settings [15]. We adopt this structure, distinguish rebuffering duration from event frequency, replace the quality term with a log-bitrate utility suited to the two-orders-of-magnitude G-PCC ladder, and add a startup-delay cost. We compare the learned policy against two strong causal baselines used across the ABR literature: a buffer-occupancy controller in the style of BBA [12] and a short-horizon MPC controller in the style of RobustMPC [7].

### C. 5G measurement datasets

Raca et al. published a 5G dataset with throughput, latency, and context metrics collected on a commercial Irish network [9]. We use its saturated-download traces as the access-link capacity signal and its measured ping (median 72 ms in 5G mode) as the RTT, rather than assuming idealized values.

## III. System Model

### A. Architecture

Fig. 1 shows the system. Four 8iVFBv2 sequences (longdress, loot, redandblack, soldier; 300 frames, 30 fps) are encoded offline by the G-PCC reference encoder (TMC13) into six representations and stored on a content server together with a DASH-style media presentation description (MPD). An edge node fetches segments from the server across an unconstrained backhaul and serves the client over the bottleneck 5G access link. The client runs the ABR controller: it observes only its playback buffer and the achieved throughput of completed downloads (never the true link capacity) and requests one segment of S consecutive frames per decision.

<!-- LAYOUT: Fig. 1 — page 1, right column, top -->
![Fig. 1 — System architecture](figures/Fig1/fig1_render.png)

*Fig. 1. System architecture: G-PCC encoding, content server, unconstrained backhaul, edge node, and the 5G access link to the DQN-driven point cloud client.*

### B. Content and quality ladder

Table I lists the ladder. Geometry follows the MPEG common test condition rate points (position quantization scale 1.0 down to 0.125) paired with attribute QP 22 to 51. The simulator transfers the exact coded size of every frame, so all results reflect real G-PCC rate characteristics.

**TABLE I. Six-Tier G-PCC Quality Ladder (Longdress)**

| Tier | Pos. scale | Attr. QP | Bitrate (Mbit/s) | Points (frame 0) | Utility q |
|---|---|---|---|---|---|
| r06 (high) | 1.000 | 22 | 102.4 | 765,821 | 1.000 |
| r05 | 0.875 | 28 | 59.6 | 602,139 | 0.898 |
| r04 | 0.750 | 34 | 35.5 | 453,698 | 0.803 |
| r03 | 0.500 | 40 | 14.1 | 212,105 | 0.613 |
| r02 | 0.250 | 46 | 3.4 | 55,374 | 0.305 |
| r01 (very low) | 0.125 | 51 | 0.9 | 14,057 | 0.000 |

Perceived quality is modeled as a normalized log-bitrate utility, reflecting the diminishing perceptual return of higher rate points on a ladder that spans two orders of magnitude:

> q(r) = [log b(r) − log b_min] / [log b_max − log b_min]    (1)

where b(r) is the coded bitrate of representation r and b_min, b_max are the ladder endpoints, so q ranges from 0 (r01) to 1 (r06). This content-independent tier utility (last column of Table I) is fixed across sequences, so the reward and the reported metric never depend on per-clip point counts.

### C. Network and transport model

The access link is driven by 21 saturated-download traces of the Irish 5G dataset [9] (5 static, 16 driving), split by a registered protocol into 13 training, 4 validation, and 4 held-out test traces (the test split, sampled once with a fixed seed after removing the validation traces, contains three driving traces and one static trace). Model, reward, checkpoint, and baseline settings are frozen before the test split is ever evaluated. Every trace is cut into feasibility-checked windows on the measured timestamp grid, so no download crosses a coverage gap. A TCP connection with slow start, additive-increase multiplicative-decrease, and the dataset-derived 72 ms RTT carries every transfer. Capacity is re-queried every RTT round, and serialization time is computed as the exact integral of the piecewise-constant capacity, so a download traverses the trace instead of freezing a single sample. The client buffer holds 5 s, playback starts after 1 s has been buffered, and playback then runs at a fixed 30 fps; an empty buffer causes rebuffering. A request-pacing rule admits the next segment only when it fits the buffer—the client waits w_t = max(0, B_t + S/f − B_max) seconds when the buffer is nearly full—so buffer overflow never discards frames and the pacing wait is excluded from throughput measurements.

### D. Segment-based fetching

Each ABR decision requests S consecutive frames as one HTTP transfer, amortizing the RTT. At S = 1 every 33 ms frame pays a 72 ms round trip, so no tier can sustain 30 fps; at large S the controller cannot react to mid-segment fades. Section V-C sweeps S over {5, 8, 10, 15}; QoE peaks at S = 10, which the final system adopts.

## IV. DQN-Based Bitrate Adaptation

### A. State, action, and decision flow

Fig. 2 shows the runtime decision flow. The 23-dimensional state contains an LSTM one-step bandwidth prediction (1), the buffer level (1), the previous tier as a one-hot vector (6), the last, mean, and standard deviation of the five most recent achieved-throughput samples (3), and the per-tier bitrates (6) and tier utilities (6) taken from the manifest. All features are normalized with constants stored in the model checkpoint, so training and inference can never diverge. The action selects one of the six tiers for the next 10-frame segment.

<!-- LAYOUT: Fig. 2 — page 2, right column, top -->
![Fig. 2 — Runtime decision flow](figures/Fig2/fig2_render.png)

*Fig. 2. Runtime decision flow of the client-side DQN controller: the 23-feature state is mapped by the Q-network to a tier for the next 10-frame segment; the download outcome updates the state.*

### B. Reward

Training and evaluation use exactly the same additive objective. Let \(S_t\) contain the \(n_t\) frames requested in segment \(t\), let \(N\) be the episode's total frame count, and let \(q_{t,i}\in[0,1]\) and \(\bar q_t\) denote the frame utility and segment-mean utility, respectively. The per-segment reward has five terms:

> r_t = (100/N)·Σ_{i∈S_t}q_{t,i} − 4.3·ΔT_stall,t − 2·ΔN_rebuf,t − |q̄_t−q̄_{t−1}| − ΔT_start,t    (2)

The first segment has zero quality-change cost. Stall duration is linear and uncapped. With discount factor \(\gamma=1\) and no learner-side reward scaling, the episodic return is therefore identical to the reported QoE:

> QoE = Σ_t r_t = 100·q̄ − 4.3·T_stall − 2·N_rebuf − Σ_{t=2}^{K}|q̄_t−q̄_{t−1}| − T_start    (3)

Here \(T_start\) is measured from session time zero until playback begins for the first time. It then freezes permanently. In contrast, \(T_stall\) and \(N_rebuf\) accumulate only after playback has begun, respectively measuring the duration and number of buffer-underflow interruptions. Initial buffering is thus charged only as startup delay and can never also be a stall or rebuffering event. There is no frame-drop term: request pacing (Section III-C) guarantees a segment is fetched only when it fits, so buffer overflow cannot discard frames. Raw network parameters deliberately do not appear: QoE measures only user-observable outcomes. This exact reward–metric identity holds for every policy, so a learned return is directly comparable to a baseline's QoE.

### C. Learning algorithm

The agent is a Double DQN [10] with experience replay. The Q-network is a multilayer perceptron (23-256-256-6, ReLU). A staged search screened 20 configurations on three seeds, confirmed the four best on twelve seeds, and ranked the finalists on a disjoint seed set; the selected configuration uses Adam with learning rate 3×10⁻⁴, batch size 64, target-network synchronization every 6,000 learning steps, and an epsilon-greedy schedule from 1.0 to 0.05 over 50% of training, with discount 1.0 and a 100,000-transition replay buffer. Each run is capped at eight epochs—one epoch is a single parent-uniform pass of 468 windowed episodes over the 13 training traces with all four sequences rotating—but stops early when validation QoE stops improving (patience of three evaluations, one every 400 episodes). In practice each seed converges in about 2,500 episodes, roughly 76,000 gradient updates, and the best validation checkpoint occurs near episode 1,300; as Fig. 7 shows, the raw evaluation curve peaks and then declines, so additional training does not help. The best checkpoint is selected by validation QoE. Training a single seed takes about one hour on a desktop CPU.

## V. Evaluation

### A. Setup

All results are measured on the four registered held-out test traces, never seen in training, across all four content sequences (longdress, loot, redandblack, soldier), with three timestamp offsets and three jitter seeds per trace, giving 144 evaluation cases per policy; the learned policy is additionally run for all twelve training seeds (1,728 cases). The metric is the QoE of (3). We compare the DQN against two strong causal baselines that observe only the same buffer and throughput history: a buffer-occupancy controller (buffer-based [12]) and a short-horizon model-predictive controller (MPC [7]) whose lookahead models the request-pacing admission rule. For context we also report the single best fixed-tier arm. All results are macro-averaged over the four parent traces.

### B. Main results

Absolute scores must be read against the ceiling: a QoE of 100 corresponds to stall-free playback at the highest tier with no switching, an operating point the driving traces physically cannot support, because their measured capacity routinely falls below the upper tiers, so even an oracle must ride the middle tiers through fades. The meaningful quantity is the margin over the strongest baseline on identical scenarios.

Table II reports the comparison. No fixed tier works across the mixed regime: the best single arm (always r03) reaches only 31.4, stalling 5.1 s per clip on the driving traces. The two adaptive baselines are far stronger and nearly tied at 45.7–45.8. The DQN reaches **49.9 ± 0.95** across twelve seeds, improving on MPC and the buffer-based controller by about **9 percent** and on the best fixed arm by **59 percent**. The mechanism is visible in the component columns: the DQN earns the highest mean quality (0.615) by selecting higher tiers, and pays for it with modestly more stall and startup (1.11 s and 3.8 s) than the conservative baselines—so the honest claim is that it wins QoE by trading a little stall for more quality, not that it minimizes stalling. The buffer-based controller is competitive on QoE but thrashes, switching tier 3.4 times per clip versus 1.5–1.6 for MPC and the DQN.

**TABLE II. Held-Out Test Results: Baselines vs. Learned Policy**

| Policy | QoE | Mean quality | Stall (s) | Switches |
|---|---|---|---|---|
| Best fixed arm (always r03) | 31.4 | 0.613 | 5.07 | 0.0 |
| Buffer-based [12] | 45.7 | 0.531 | 0.24 | 3.40 |
| MPC [7] | 45.8 | 0.517 | 0.28 | 1.54 |
| **DQN (12-seed mean ± std)** | **49.9 ± 0.95** | **0.615** | **1.11** | **1.59** |

The per-trace breakdown (Fig. 3) shows where the learned policy wins and where it does not. The DQN leads clearly on the two moderate driving traces (Driving B, 63.8; Driving C, 67.6) and on the static trace among the adaptive baselines (51.8). On the hardest low-capacity trace (Driving A) the conservative MPC edges it (19.8 vs. 16.6), because the DQN occasionally reaches for a higher tier and absorbs a stall where the cautious baseline does not. No fixed arm is robust: always-r03 tops the static trace (55.8) yet collapses to −27 on the scarcity trace, whereas the DQN stays first or a close second everywhere. Aggregated over the 144 matched scenarios, the DQN beats MPC in 83 percent and the buffer-based controller in 78 percent, by a median of about +7 QoE.

<!-- LAYOUT: Fig. 3 — page 3, left column, top (pair with Fig. 4) -->
![Fig. 3 — Per-trace QoE](figures/Fig3/per_trace_qoe.png)

*Fig. 3. QoE per held-out test trace (plus the four-trace mean): the DQN (12-seed mean) against the buffer-based and MPC baselines. The DQN leads on Driving B/C and Static and overall; MPC edges it only on the hardest low-capacity trace.*

Fig. 4 illustrates the learned behavior on a driving trace whose capacity rises from a low attach through the middle of the ladder. The policy starts at a low tier while the first throughput samples arrive, then climbs the ladder step by step (r02→r06) as measured capacity opens up, holding the selected bitrate just under the link capacity and building the buffer without a single stall—the concrete form of the quality-versus-safety trade quantified in Table II.

<!-- LAYOUT: Fig. 4 — page 3, right column, top (pair with Fig. 3) -->
![Fig. 4 — Policy adaptation over time](figures/Fig4/policy_timeseries.png)

*Fig. 4. DQN adaptation on a driving trace: link capacity and selected tier bitrate as per-segment bars (top), client buffer level (bottom). The policy climbs the ladder as measured throughput allows, tracking just below capacity and streaming stall-free.*

### C. Effect of segment size

Fig. 5 reports the segment-size sweep over S ∈ {5, 8, 10, 15} (best configuration per size, validation QoE). Per-frame fetching is infeasible—at S = 1 every 33 ms frame pays the 72 ms RTT—so the sweep begins at S = 5. QoE rises from 45.8 at S = 5 to a peak of 49.5 at S = 10, then eases to 49.1 at S = 15 as larger segments react more slowly to fades; stall time broadly decreases as the RTT is amortized over more frames. S = 10 (shaded) gives the best QoE and is adopted for the final system.

<!-- LAYOUT: Fig. 5 — page 4, top (pair with Fig. 6) -->
![Fig. 5 — Segment-size sweep](figures/Fig5/segment_size_curve.png)

*Fig. 5. Effect of segment size S: QoE (left axis, solid) and stall time (right axis, dashed). QoE peaks at S = 10 (shaded); smaller segments pay more RTT overhead and larger ones react too slowly.*

### D. Distributional view

Fig. 6 shows the empirical CDF of per-case QoE. The DQN curve lies to the right of both baselines across the upper two-thirds of the distribution—higher median (52.0 vs. 43.8–45.4) and a fatter right tail (90th percentile 85 vs. 76–77)—so the typical and best-case sessions are clearly better. The curves cross in the lower-left: the DQN has a slightly heavier failure tail, 8.4 percent of cases below zero versus 2–4 percent for the baselines, concentrated on the low-capacity trace where its higher-tier choices occasionally cost a stall. The distribution therefore makes the trade explicit rather than hiding it in a mean: broadly higher QoE in exchange for a few more bad-case failures.

<!-- LAYOUT: Fig. 6 — page 4, top (pair with Fig. 5) -->
![Fig. 6 — QoE CDF](figures/Fig6/qoe_cdf.png)

*Fig. 6. Empirical CDF of per-case QoE. The DQN dominates the upper two-thirds of the distribution (higher median and best cases) at the cost of a slightly heavier left tail on the low-capacity trace.*

### E. Training convergence and robustness

Fig. 7 examines training convergence across the twelve seeds. The best-checkpoint QoE rises from the baseline level and clears the MPC reference within roughly the first fifth of training, then locks onto a plateau near 48.8; the mean best checkpoint occurs about two-thirds of the way through training. The raw evaluation curve peaks earlier and then drifts downward—the well-known DQN over-training effect—which the best-checkpoint selection used throughout this work renders harmless. The final spread is tight: a 12-seed test standard deviation of 0.95 (per-seed range 47.9 to 51.1), with every seed beating both adaptive baselines, so the result is a stable property of the method rather than a lucky checkpoint. The shipped policy uses a one-step LSTM bandwidth prediction as one of its 23 state features (Section IV-A), paired with the segment-matched predictor for the winning segment size.

<!-- LAYOUT: Fig. 7 — page 5 (last page), left column, top -->
![Fig. 7 — Training convergence](figures/Fig7/training_convergence.png)

*Fig. 7. Test-proxy (validation) QoE during training (winner configuration, 12 seeds). The best-checkpoint curve clears the MPC baseline early and plateaus; raw evaluations later degrade, motivating best-checkpoint selection.*

## VI. Conclusion

This paper demonstrated end-to-end learned bitrate adaptation for G-PCC point cloud streaming over measured 5G conditions. Segment-based fetching, with a QoE optimum at ten frames per request, resolves the RTT bound that makes per-frame streaming infeasible at 72 ms latency, and a Double DQN whose return equals the reported QoE outperforms a buffer-based controller and an MPC controller by about 9 percent and the best fixed arm by 59 percent on a registered, once-only test split, with a 12-seed standard deviation below 1 QoE point and an 83 percent per-scenario win rate against MPC. The learned advantage comes from selecting higher-quality tiers when capacity allows while staying robust on scarce links; its one weakness is a slightly heavier failure tail on the hardest low-capacity trace, where the conservative baselines stall less. Future work includes multi-user streaming over a shared bottleneck, viewport-aware tiling, and a cold-start prior to close that low-capacity gap.

## Acknowledgment

The authors thank the maintainers of the MPEG G-PCC reference software and the providers of the 8iVFBv2 and Irish 5G datasets.

## References

1. Information technology—Dynamic adaptive streaming over HTTP (DASH)—Part 1: Media presentation description and segment formats, ISO/IEC 23009-1, 2014.
2. E. d'Eon, B. Harrison, T. Myers, and P. A. Chou, "8i voxelized full bodies—a voxelized point cloud dataset," ISO/IEC JTC1/SC29 (MPEG/JPEG) input document m40059, Geneva, Jan. 2017.
3. M. Hosseini and C. Timmerer, "Dynamic adaptive point cloud streaming," in Proc. 23rd Packet Video Workshop, Amsterdam, Netherlands, 2018, pp. 25–30.
4. S. Schwarz et al., "Emerging MPEG standards for point cloud compression," IEEE J. Emerg. Sel. Topics Circuits Syst., vol. 9, no. 1, pp. 133–148, Mar. 2019.
5. Information technology—Coded representation of immersive media—Part 9: Geometry-based point cloud compression, ISO/IEC 23090-9, 2023.
6. H. Mao, R. Netravali, and M. Alizadeh, "Neural adaptive video streaming with Pensieve," in Proc. ACM SIGCOMM, Los Angeles, CA, USA, 2017, pp. 197–210.
7. X. Yin, A. Jindal, V. Sekar, and B. Sinopoli, "A control-theoretic approach for dynamic adaptive video streaming over HTTP," in Proc. ACM SIGCOMM, London, U.K., 2015, pp. 325–338.
8. J. van der Hooft, T. Wauters, F. De Turck, C. Timmerer, and H. Hellwagner, "Towards 6DoF HTTP adaptive streaming through point cloud compression," in Proc. 27th ACM Int. Conf. Multimedia, Nice, France, 2019, pp. 2405–2413.
9. D. Raca, D. Leahy, C. J. Sreenan, and J. J. Quinlan, "Beyond throughput, the next generation: a 5G dataset with channel and context metrics," in Proc. 11th ACM Multimedia Syst. Conf. (MMSys), Istanbul, Turkey, 2020, pp. 303–308.
10. H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proc. 30th AAAI Conf. Artif. Intell., Phoenix, AZ, USA, 2016, pp. 2094–2100.
11. S. S. Krishnan and R. K. Sitaraman, "Video stream quality impacts viewer behavior: inferring causality using quasi-experimental designs," IEEE/ACM Trans. Netw., vol. 21, no. 6, pp. 2001–2014, Dec. 2013.
12. T.-Y. Huang, R. Johari, N. McKeown, M. Trunnell, and M. Watson, "A buffer-based approach to rate adaptation: evidence from a large video streaming service," in Proc. ACM SIGCOMM, Chicago, IL, USA, 2014, pp. 187–198.
13. J. Li, H. Wang, Z. Liu, P. Zhou, X. Chen, Q. Li, and R. Hong, "Toward optimal real-time volumetric video streaming: a rolling optimization and deep reinforcement learning based approach," IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 12, pp. 7870–7883, Dec. 2023.
14. X. Wang, W. Liu, H. Liu, and P. Yang, "Spatial perceptual quality aware adaptive volumetric video streaming," in Proc. IEEE Global Commun. Conf. (GLOBECOM), 2023.
15. R. Farahani, Z. Azimi, C. Timmerer, and R. Prodan, "Towards AI-assisted sustainable adaptive video streaming systems: tutorial and survey," arXiv:2406.02302, 2024.
