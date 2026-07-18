# Plan: Align the DQN training reward with the reported QoE

> This file lives in the repo so you can comment on it directly (same review flow as `docs/PLAN.md`).
> It is a **plan only** — no code has been changed yet. Nothing here touches `exp/clean-eval-retrain`.
>
> - **Branch:** `exp/reward-aligned-dqn` (created off `feat/oo-topology-dqn-gpcc`).
> - **Worktree:** `C:\Users\Mamtech\PointCloudStreaming-reward-aligned` (separate dir so your uncommitted
>   paper edits on `exp/clean-eval-retrain` stay untouched).

---

## 1. Context & goal

The clean run (`exp/clean-eval-retrain`, commit `e9bac7f`) produced a DQN that **loses to MPC and buffer on the
reported QoE metric** (DQN 26.02 vs MPC 38.82 vs Buffer 36.17) despite delivering the **highest quality**
(mean_q 0.651 vs MPC 0.496). A ChatGPT review attributed this to a training-reward vs. reported-QoE mismatch and
proposed a fix. This plan records what we **verified**, where the ChatGPT diagnosis is **right / wrong / incomplete**,
and the concrete change we will make on this branch.

Key fact that shapes everything: **the reward misalignment is not new.** `src/rl/reward.py` and
`src/network_model/session.py` are **byte-identical** on `feat/oo-topology-dqn-gpcc` and `exp/clean-eval-retrain`.
So this is a fix to the core reward, developed on a branch rooted in `feat`.

---

## 2. What we verified (independent re-derivation from code + artifacts)

The mismatch is **real and reproduces to the digit**:

| Policy | RL reward (trained objective) | QoE_quality (paper metric) | mean_q | stall_s |
|---|---:|---:|---:|---:|
| **DQN** | **149.31** ← wins | 26.02 | 0.651 | 5.905 |
| MPC | 139.26 | **38.82** ← wins | 0.496 | 0.480 |
| Buffer | 123.74 | 36.17 | 0.442 | 0.007 |

- **Training reward** sums per-frame quality: `step_segment()` in `src/rl/reward.py:127-136` returns
  `q_sum − penalty − switch`; over a 300-frame episode the quality contribution ≈ `Σq`, max ≈ **N = 300**
  (verified: every manifest has exactly 300 frames).
- **Reported QoE** uses `100·mean_q` (`src/network_model/session.py:205`), max **100**.
- So quality is weighted **N/100 = 3.0×** more heavily *relative to every cost term* in training than in eval
  (the cost weights `mu=4.3`, `lam=1.0`, `startup=1.0` are identical in both). The "3×" is exact *because* N=300,
  not a structural constant.
- **Confirmed extra mismatches** (ChatGPT named these): training caps stall per-segment at 2 s
  (`stall_mode:'bounded'`, `stall_cap_s:2.0`) vs. eval's full stall; training has **no slowdown term** vs. eval's
  `w_slow=10` (the *largest* QoE coefficient); training adds a `event_penalty=2.0` per stall-event that QoE has no
  term for.
- **The smoking gun:** DQN is **#1 on the objective it was trained on** (149.31) and **#2 on the metric the paper
  reports** (26.02). It optimized exactly what we asked.

### Mismatches ChatGPT *missed* (found by adversarial verification)

1. **Slowdown ↔ `playback-rate-min=0.9` coupling — the most important one.** Slowing playback delays buffer
   depletion, which *reduces* the stall the training reward penalizes, and slowdown itself is **invisible to the
   reward** (`grep slowdown` finds nothing in `src/rl/env.py` or `src/rl/reward.py`). Eval charges the same slow-play
   at weight **10**. **Training rewards the exact behavior eval punishes hardest.** This — not the quality scale — is
   the most likely driver of the persistent −11…−19 "other" penalty per seed.
2. **Drop weight is 3× too heavy in training** (`drop_weight=1.0`) vs. eval (`100/N ≈ 0.333`). Small magnitude.
3. **Switch is aggregated differently**: training penalizes `|Δ(segment-mean q)|` (~38 comparisons); eval penalizes
   per-frame `Σ|Δq|` (~299 comparisons). They diverge even at a constant tier (measured 0.286 vs 0.473 on longdress)
   because density drifts frame-to-frame. Small magnitude.
4. **Apples-to-oranges head-to-head:** DQN aggregates `n_cases=1728` (12 seeds) vs. every baseline's `144`.

---

## 3. Where the ChatGPT diagnosis is right vs. wrong

**Right:** a genuine reward↔QoE mismatch exists; the arithmetic (the 149.31>139.26 inversion, the QoE
decomposition, the headroom numbers) is all correct; aligning the training reward to the reported QoE and
validating on all four sequences are sound.

**Wrong / mis-located mechanism — this changes what we prioritize:**

- **The quality rescale (100/N) is largely cosmetic.** The DQN trains with `--reward-norm=True` (on by default,
  `src/rl/dqn.py:112-124`) + Huber loss + grad-clip 10. Normalization divides by a running std, so choosing to scale
  quality *down* vs. scale costs *up* gives the **identical policy**. What matters is the **ratio** of quality to
  each cost, not the absolute scale.
- **Quality is not the lever.** Across the 12 seeds: `corr(QoE, mean_q) ≈ −0.02` (≈ zero), `corr(QoE, stall) ≈ −0.67`,
  `corr(mean_q, stall) ≈ +0.73`. QoE is dominated by **stall/slowdown**, which are *positively coupled* to quality.
- **The headroom premise "cut stall while holding quality constant" is physically false.** Bigger tiers ⇒ bigger
  downloads ⇒ buffer drains ⇒ more stall. Reweighting toward stall will push DQN to **lower tiers** (lower quality),
  moving it along the Pareto frontier — not holding quality at 0.651. The honest question is whether DQN's
  *achievable frontier* beats MPC's operating point (QoE 38.82 @ mean_q 0.496, stall 0.48 s).
- **The deficit is a robustness failure, not just a scale bug.** DQN loses to MPC **even on longdress alone**
  (28.78 vs 37.75 — the sequence it was selected on), and its **best single seed (~35.5) still loses** to MPC. The
  whole gap concentrates in **one hard driving trace** (QoE ≈ −25, ~12 s stall) where the policy fails to back off.
  So fixing validation content (ChatGPT's #2) will **not by itself** flip the ranking.

**Bottom line:** the diagnosis is materially correct that a mismatch exists; the prescription over-weights the
cosmetic part (quality scale) and under-weights the parts that actually move the policy: **adding the slowdown
penalty, closing the playback-floor stall-dodge, and giving training the full (uncapped) stall signal so it learns
to avoid catastrophic hard-trace stalls.**

---

## 4. Design decisions

- **Stall: uncapped / real (`stall_mode='linear'`).** _(confirmed with user)_ Rationale: the cap makes training
  unable to distinguish a 5 s stall from a 12 s stall, so the policy never learns the hard-trace stall is
  catastrophic — the cap is plausibly *part of the cause* of the failure. It also exactly matches eval. The
  instability the cap guarded against is already handled by Huber + grad-clip + reward-norm.
  **Caveat to watch in the dev run:** one catastrophic trace can inflate the reward-norm running std and dilute the
  learning signal on normal traces. If that bites, the fix is to soften/adjust reward-norm — *not* to re-cap stall.
- **Fix is additive & config-driven & back-compatible.** New spec keys default to current behavior, so the old
  reward stays bit-identical and reproducible for the control arm.
- **Isolate the mechanism with an A/B/C ablation** rather than changing everything at once (see §6), because the
  verification says the quality rescale and the stall/slowdown fixes have *different* expected impact.

---

## 5. Implementation plan (code changes, on this branch only)

All changes are small and localized:

1. **`src/rl/reward.py`** — add two spec keys to `DEFAULT_REWARD_SPEC`, both defaulting to no-op:
   - `q_weight` (default `1.0`): multiply the quality contribution. In `step()` return `q_weight*q − penalty − switch`;
     in `step_segment()` return `base − q_weight*q_mean + q_weight*q_sum`. Set `q_weight = 100/N ≈ 0.333` to match the
     eval `100·mean_q` scale.
   - `slow_weight` (default `0.0` = off): add `penalty += slow_weight * slowdown_s` in `step()`, with a new
     `slowdown_s` argument (per-step slowdown-integral delta).
   - Update `describe()` accordingly.
2. **`src/rl/env.py`** — plumb per-step slowdown. Mirror the existing per-step **startup/drop delta** pattern
   (the env already tracks `_prev_startup_s` / `_prev_dropped` and feeds deltas to the reward): track
   `_prev_slowdown`, compute `slowdown_s = max(0, stats['slowdown_integral'] − _prev_slowdown)`, pass it into
   `step_segment(...)`. (`slowdown_integral` is already produced in `src/network_model/buffer.py` and consumed by
   `session.qoe_quality_terms`; we just also feed it to the reward.)
3. **`configs/`** — add the reward-spec presets for the ablation arms (§6) and set `eval-sequences` to all four
   sequences for validation; keep `select-by: qoe_quality`.
4. **Optional (recommended) — tail-risk checkpoint selection.** Add an option to select the checkpoint by a
   worst-trace / CVaR criterion instead of the plain validation mean, to directly attack the hard-trace failure.
5. **Bring in the clean-eval harness.** Cherry-pick the infra commit `f94005a` from `exp/clean-eval-retrain`
   (protocol split, validation-only selection, MPC/buffer baselines, locked test) onto this branch. Skip the
   artifacts commit `e9bac7f`. This is needed so the dev runs are judged the same way as the baselines.

No training is run here — you run that on the second PC (commit + push, pull there).

---

## 6. Reward ablation arms (recommended default — open for your review)

All arms use **uncapped stall**. The point is to see *which* fix actually moves the ranking:

| Arm | q_weight | stall | event_penalty | drop_weight | slow_weight | Tests |
|---|---|---|---|---|---|---|
| **0 — Control** | 1.0 | bounded, cap 2.0 | 2.0 | 1.0 | 0.0 | Reproduces the current 26-QoE result |
| **1 — Stall-real + slowdown** | 1.0 | **linear** | 0.0 | 0.333 | **10.0** | Fixes the dominant terms, quality scale untouched |
| **2 — Full QoE-aligned** | **0.333** | **linear** | 0.0 | 0.333 | 10.0 | Arm 1 + quality down-weighted to QoE scale (ChatGPT's C11, uncapped) |

- **0 → 1** isolates the effect of the stall/slowdown/event/drop alignment (the levers the analysis says matter).
- **1 → 2** isolates the quality-weight change (the lever the analysis says is mostly cosmetic under reward-norm) —
  a clean test of whether that belief holds empirically.

---

## 7. Dev experiment protocol (cheap, train/validation only)

- **3 seeds** per arm (not 12) — this is a development screen, not the final run.
- Train/validation **only**. **Do not touch the test set.**
- Validation on **all four sequences**; `select-by: qoe_quality`.
- **Primary readout:** does each arm's *validation* quality-stall frontier clear MPC's operating point
  (QoE 38.82 @ mean_q 0.496, stall 0.48 s)? **Look specifically at the hard driving trace** — that is where the
  ranking is decided.
- **Secondary:** per-seed spread; does uncapped stall cause reward-norm signal dilution (watch training curves on
  normal traces)?
- **Decision rule:** if an arm beats buffer *and* MPC on validation (especially on the hard trace), freeze it and
  move to a **clean** final study. If none does, the problem is robustness/architecture, not reward accounting, and
  we reconsider before spending a full run.

---

## 8. Test-set integrity

The current test set has been **opened** (results reported in commit `e9bac7f`), so it can no longer serve as an
untouched final test after tuning. For the final study, either **grouped/outer cross-validation** over trace groups,
or **acquire new independent test traces**. This must be settled before the final campaign — flagged here, decided
after the dev screen.

---

## 9. Explicitly NOT doing (yet)

- No changes to `exp/clean-eval-retrain` or to `docs/PLAN.md`.
- No full 12-seed run, no final-test evaluation, no paper/figure edits — those come after the dev screen picks an arm.
- Not "just train longer" — the artifacts show later iterates often collapse; that is not the fix.

---

## 10. Open questions for you

1. **Ablation arms (§6):** run all three (0/1/2), or go straight to Arm 1 (stall-real + slowdown) only?
2. **Tail-risk checkpoint selection (§5.4):** add it now, or keep plain validation-mean selection for the dev screen
   and add it only for the final run?
3. **Seeds for the dev screen:** 3 (fast) or 5 (steadier signal)?
4. **Cherry-pick the infra commit `f94005a` onto this branch** (§5.5), or do you want the reward change kept minimal
   and the harness merged separately?
