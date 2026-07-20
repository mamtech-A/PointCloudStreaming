# IST-2026 clean experiment and post-training revision plan

## What is frozen before final test

- Segment size: 8 frames/request.
- Reward = QoE: 100/N-scaled quality, linear stall duration, rebuffer-event
  count, segment-quality change, 100/N-scaled frame drops, and startup delay.
- Startup ends at first playback; stall duration and rebuffer events begin only
  afterward, so the temporal terms are disjoint.
- Discount factor 1.0 and no learner-side reward scaling.
- Double-DQN architecture and optimizer settings from Round 4.
- Twelve training seeds.
- One preregistered feature ablation: DQN with versus without LSTM prediction.
- Trace partitions and final-test cases in `configs/experiment_protocol.json`.
- The global fixed tier and adaptive-baseline parameters selected on validation
  by `scripts/tune_baselines.py` and
  frozen in `models/baseline_config.json` before final test.

No reward or hyperparameter grid is run against the final-test traces.

The locked test traces came from the former Round-4 training pool, so they were
not previously used for held-out reporting but are not a newly acquired external
dataset. The clean run excludes them from every fitting/selection stage. If the
venue requires a test set never present anywhere in method development, acquire
additional traces or use a preregistered nested grouped cross-validation study.

## Outputs required before paper result editing

- `models/dqn_sweep_results.json`: validation selection and seed variation.
- `models/final_test_results.json`: locked clean-run test case-level observations.
- `models/TRAINING_SUMMARY.md`: compact run summary.
- The full run directory and frozen copies of both JSON configuration files.

## Figure decisions after results return

1. Keep the approved system model.
2. Keep the approved DQN decision-flow figure only if the LSTM-feature model is
   retained. If validation favors the simpler model, remove LSTM prediction
   from the figure/state description.
3. Rebuild the primary result figure from `final_test_results.json`, including
   DQN uncertainty, fair adaptive baselines, best global fixed, and the clearly
   marked per-trace oracle.
4. Rebuild the segment-size figure only from a final-protocol experiment
   artifact; do not retain hard-coded chart values.
5. Use a difficult driving case for the policy time series and state the
   representative-run selection rule.
6. Drop the current four-trace CDF and running-best convergence figures.

## Paper edits that wait for final results

- Abstract result numbers and superiority language.
- Main evaluation table and per-trace discussion.
- Whether LSTM is part of the final model or only an ablation.
- Any percentage improvement over adaptive baselines.
- Final conclusion claims.

## Paper edits that are safe independently of results

- Position novelty around actual G-PCC sizes, measured 5G traces, segment-level
  client adaptation, and RTT/segment interaction—not “first RL point-cloud ABR.”
- Replace “perceptual quality” with “normalized log-density utility/proxy” unless
  a validated objective/perceptual metric is added.
- State explicitly that the undiscounted reward sum equals QoE and define startup
  separately from post-start stall duration and rebuffer-event count.
- Describe the complete observable client state and say that no oracle future
  capacity is exposed.
- Qualify RTT conclusions as applying to the implemented sequential request
  model.
- Add direct related work on RL-based point-cloud streaming.
- Renumber figures in order of first citation and reduce the final figure set.

The working `paper_IST2026.md` was already modified before this branch and is
therefore deliberately not included in the experiment-preparation commit. It
should be cleaned after the registered result has been pulled back.
