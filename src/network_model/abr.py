"""Pluggable ABR (adaptive bitrate) selection strategies.

`ABRState` is the immutable per-decision observation a strategy is allowed to see
(it preserves the legacy no-oracle discipline: the strategy never receives the true
link capacity, only observed history + buffer + the current frame's reps).

`BandwidthABR` and `LSTMABR` port the legacy `EdgeNode._bandwidth_selection` and
`EdgeNodeLSTM.select_representation_lstm` bodies verbatim, so single-user runs match
the pre-refactor behavior exactly. `DQNABR` is the learned policy (filled in during
the RL phase); it composes an `LSTMABR` so the LSTM prediction can feed the RL state.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .manifest import coded_bitrate_bps


@dataclass(frozen=True)
class ABRState:
    """Everything a strategy may observe when choosing a representation.

    All bandwidth/throughput values are in bps. `observed_throughput_history` holds
    the achieved throughput of *completed* downloads for this user only (frames
    0..i-1 when deciding frame i) — never the true link capacity.
    """
    reps: Sequence[dict]
    observed_throughput_history: Sequence[float] = field(default_factory=tuple)
    buffer_level_s: float = 0.0
    last_rep_id: Optional[int] = None
    predicted_bandwidth_bps: Optional[float] = None
    frame_id: Optional[int] = None

    def harmonic_mean_throughput(self):
        """Harmonic mean of positive observed throughput (legacy fallback estimate)."""
        positive = [bw for bw in self.observed_throughput_history if bw and bw > 0]
        if not positive:
            return 0
        return len(positive) / sum(1.0 / bw for bw in positive)


class ABRStrategy:
    """Abstract pluggable policy: choose a representation id from an ABRState."""

    name = 'abr'

    def select(self, state: ABRState):
        raise NotImplementedError

    def reset(self):
        """Clear any per-session internal state (called on session start)."""
        pass

    def report(self, observed_throughput_bps=None):
        """Optional end-of-run report dict (e.g. LSTM prediction stats). None if N/A."""
        return None


class BandwidthABR(ABRStrategy):
    """Baseline: highest rep whose bitrate <= safety_factor x estimated throughput.

    Ports `EdgeNode._bandwidth_selection` + `_estimate_bandwidth_from_history`.
    """

    name = 'bandwidth'

    def __init__(self, safety_factor=0.8):
        self.safety_factor = safety_factor

    def select(self, state: ABRState):
        reps = state.reps
        if not reps:
            return None
        bandwidth_bps = state.harmonic_mean_throughput()
        # Rank/threshold by the bitrate of what is ACTUALLY streamed: the real
        # manifest bandwidth when present, else the coded-size-derived rate.
        # (The legacy rep.get('bandwidth', fallback) never fell back — the key
        # always exists, 0 when absent — which degenerated selection to rep 0.)
        reps_sorted = sorted(reps, key=coded_bitrate_bps, reverse=True)
        for rep in reps_sorted:
            if coded_bitrate_bps(rep) <= bandwidth_bps * self.safety_factor:
                return rep['id']
        return reps_sorted[-1]['id']


class LSTMABR(ABRStrategy):
    """LSTM-prediction-driven selection mapped to 3 quality tiers.

    Ports `EdgeNodeLSTM.predict_bandwidth` / `select_representation_lstm` /
    `_select_quality_level`. Also exposes `predict(state) -> bps` so it can act as a
    pure feature provider for `DQNABR`.

    `predictor` is a SimpleLSTM/LSTMPredictor-like object exposing
    `sequence_length`, `history`, `update(mbps)`, `predict_next()`.
    """

    name = 'lstm'

    def __init__(self, predictor, use_prediction=True,
                 high_mbps=20.0, medium_mbps=5.0, safety_factor=0.8):
        self.predictor = predictor
        self.use_prediction = use_prediction and predictor is not None
        # Legacy fixed thresholds — used ONLY when the manifest carries no
        # bitrate information (see _select_quality_level).
        self.high_mbps = high_mbps
        self.medium_mbps = medium_mbps
        self.safety_factor = safety_factor
        self.prediction_history = []  # list of {'predicted_mbps': ...}

    def reset(self):
        self.prediction_history = []

    def _predict_bps(self, state: ABRState):
        """Replicate EdgeNodeLSTM.predict_bandwidth (returns bps or None)."""
        if not self.use_prediction or self.predictor is None:
            return None
        seq_len = self.predictor.sequence_length
        history = state.observed_throughput_history
        if len(history) < seq_len:
            return None
        recent_bw = [bw / 1e6 for bw in history[-seq_len:]]
        # Rebuild the predictor's window from this user's latest samples only,
        # avoiding any carry-over across calls/sessions.
        if hasattr(self.predictor, 'history'):
            self.predictor.history.clear()
            inner = getattr(self.predictor, '_predictor', None)
            if inner is not None and hasattr(inner, 'history'):
                inner.history.clear()
        for bw in recent_bw:
            self.predictor.update(bw)
        predicted_mbps = self.predictor.predict_next()
        self.prediction_history.append({'predicted_mbps': predicted_mbps})
        return predicted_mbps * 1e6

    def get_prediction_stats(self, observed_throughput_bps):
        """Replicate EdgeNodeLSTM.get_prediction_stats for the run summary."""
        if not self.prediction_history:
            return None
        predicted_mbps = [p['predicted_mbps'] for p in self.prediction_history]
        actual_mbps = [bw / 1e6 for bw in observed_throughput_bps[-len(predicted_mbps):]]
        if actual_mbps and predicted_mbps:
            n = min(len(actual_mbps), len(predicted_mbps))
            errors = [abs(predicted_mbps[i] - actual_mbps[i]) for i in range(n)]
            mae = sum(errors) / len(errors) if errors else 0
            mape = (sum(e / max(a, 0.1) * 100 for e, a in zip(errors, actual_mbps[:n])) / n
                    if errors else 0)
        else:
            mae = mape = 0
        return {
            'total_predictions': len(predicted_mbps),
            'mean_predicted_mbps': sum(predicted_mbps) / len(predicted_mbps) if predicted_mbps else 0,
            'mean_actual_mbps': sum(actual_mbps) / len(actual_mbps) if actual_mbps else 0,
            'mae_mbps': mae,
            'mape_percent': mape,
        }

    def report(self, observed_throughput_bps=None):
        """Polymorphic end-of-run report; None if not applicable."""
        if observed_throughput_bps is None:
            return None
        return self.get_prediction_stats(observed_throughput_bps)

    def predict(self, state: ABRState):
        """Bandwidth (bps) the policy would act on: LSTM prediction when warmed up,
        else harmonic mean of observed throughput, else 0. Used as the DQN feature."""
        predicted_bps = self._predict_bps(state)
        if predicted_bps is not None and self.use_prediction:
            return predicted_bps
        return state.harmonic_mean_throughput()

    def _select_quality_level(self, reps, bandwidth_bps):
        reps_sorted = sorted(reps, key=coded_bitrate_bps, reverse=True)
        # Manifest-aware selection: when reps carry real bitrate info, choose the
        # highest rep whose streamed bitrate fits under predicted * safety (same
        # rule as BandwidthABR). Fixed Mbps tier thresholds would be wrong for an
        # arbitrary ladder (e.g. the real G-PCC high tier needs ~102 Mbps).
        if any(r.get('bandwidth') or r.get('coded_bytes') for r in reps):
            for rep in reps_sorted:
                if coded_bitrate_bps(rep) <= bandwidth_bps * self.safety_factor:
                    return rep['id']
            return reps_sorted[-1]['id']
        # Legacy fallback: no bitrate info at all — fixed Mbps tiers.
        bandwidth_mbps = bandwidth_bps / 1e6
        if bandwidth_mbps >= self.high_mbps:
            return reps_sorted[0]['id'] if len(reps_sorted) > 0 else 0
        elif bandwidth_mbps >= self.medium_mbps:
            return reps_sorted[1]['id'] if len(reps_sorted) > 1 else reps_sorted[0]['id']
        else:
            return reps_sorted[2]['id'] if len(reps_sorted) > 2 else reps_sorted[-1]['id']

    def select(self, state: ABRState):
        reps = state.reps
        if not reps:
            return None
        predicted_bps = self._predict_bps(state)
        if predicted_bps is not None and self.use_prediction:
            bandwidth_for_selection = predicted_bps
        elif state.observed_throughput_history:
            bandwidth_for_selection = state.harmonic_mean_throughput()
        else:
            bandwidth_for_selection = 0
        return self._select_quality_level(reps, bandwidth_for_selection)


class DQNABR(ABRStrategy):
    """Learned RL policy (filled in during the RL phase).

    Composes an `LSTMABR` so the LSTM prediction becomes one feature of the RL state.
    The DQN policy + shared feature builder live in `src/rl/`; they are imported
    lazily here (function-local) to keep `network_model` free of an import-time
    dependency on `rl`.
    """

    name = 'dqn'

    def __init__(self, policy=None, lstm_provider=None, epsilon=0.0,
                 log_decisions=False):
        self.policy = policy            # path or a loaded DQNAgent (resolved lazily)
        self.lstm_provider = lstm_provider
        self.epsilon = epsilon
        self._agent = None
        # When log_decisions is on, every select() stores what the policy saw
        # and thought in `last_decision` (state summary, per-action Q-values,
        # chosen action) — run_dqn.py reads it for the decision log + CSV.
        self.log_decisions = log_decisions
        self.last_decision = None

    def _ensure_agent(self):
        if self._agent is None:
            from ..rl.dqn import DQNAgent  # lazy: avoids network_model -> rl cycle
            if isinstance(self.policy, str):
                self._agent = DQNAgent.load(self.policy)
            else:
                self._agent = self.policy
            if self._agent is None:
                raise NotImplementedError(
                    "DQNABR requires a trained policy (path or DQNAgent). "
                    "Train one with train_dqn.py first."
                )
        return self._agent

    def select(self, state: ABRState):
        from ..rl.features import build_state  # lazy
        agent = self._ensure_agent()
        predicted_bps = self.lstm_provider.predict(state) if self.lstm_provider else None
        features = build_state(
            agent.feature_spec, state, predicted_bandwidth_bps=predicted_bps,
            norm=agent.norm_constants,
        )
        action = agent.act(features, epsilon=self.epsilon)
        if self.log_decisions:
            q = agent.q_values(features)
            hist = [bw / 1e6 for bw in state.observed_throughput_history[-5:]]
            self.last_decision = {
                'q_values': [float(v) for v in q],
                'action': int(action),
                'greedy': int(action) == int(q.argmax()),
                'buffer_s': float(state.buffer_level_s),
                'tput_last_mbps': hist[-1] if hist else 0.0,
                'tput_mean_mbps': (sum(hist) / len(hist)) if hist else 0.0,
                'lstm_pred_mbps': (predicted_bps / 1e6) if predicted_bps else None,
                'features': [float(v) for v in features],
            }
        # The Q-net outputs an ACTION INDEX; map it to a representation id
        # (same convention as the training env's _ManualABR: reps sorted by id).
        reps_sorted = sorted(state.reps, key=lambda r: r['id'])
        if 0 <= action < len(reps_sorted):
            return reps_sorted[action]['id']
        return reps_sorted[-1]['id']
