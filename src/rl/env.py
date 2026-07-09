"""StreamingEnv: gym-like environment wrapping a one-session Topology.

One (sequence, trace) pair = one episode. The env drives a single
StreamingSession whose ABR is a `_ManualABR` placeholder — the DQN agent
supplies the action, the env injects it, and `session.step()` runs the same
transfer+buffer+feature code path used at inference (no train/inference skew).
The LSTM is used here only as a feature provider (its prediction is one input
to the state vector).

Multi-sequence: pass `manifest_frames` as a dict {sequence_name: frames} and
pick the content per episode via `reset(trace, sequence=...)`. Reward quality
endpoints are recomputed per episode from that sequence's ladder, so any
manifest maps its lowest rep to ~0.0 and highest to ~1.0.

Reward: pluggable via `reward_spec` (src/rl/reward.py). The default spec is the
legacy Pensieve form quality - mu*stall_seconds - lam*|quality change|.
"""

import numpy as np

from ..network_model import Server, EdgeNode, User, Topology, LSTMABR, PointCloud
from ..network_model.abr import ABRStrategy
from .features import (build_state, state_dim, quality_endpoints,
                       DEFAULT_FEATURE_SPEC, DEFAULT_NORM)
from .reward import RewardFunction


class _ManualABR(ABRStrategy):
    """Placeholder strategy: returns the action injected by the env each step."""
    name = 'manual'

    def __init__(self):
        self.next_action = 0

    def select(self, state):
        reps = sorted(state.reps, key=lambda r: r['id'])
        a = max(0, min(self.next_action, len(reps) - 1))
        return reps[a]['id']


class StreamingEnv:
    def __init__(self, manifest_frames, lstm_predictor=None, tcp_params=None,
                 feature_spec=None, mu=4.3, lam=1.0, hist_len=5,
                 target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0,
                 num_reps=None, norm=None, bits_per_point=None, reward_spec=None):
        # Normalize content to a pool {sequence_name: frames}; a plain frame list
        # (the historical single-manifest API) becomes {'default': frames}.
        if isinstance(manifest_frames, dict):
            if not manifest_frames:
                raise ValueError("manifest_frames dict is empty")
            self.manifest_pool = dict(manifest_frames)
        else:
            self.manifest_pool = {'default': manifest_frames}
        self.default_sequence = next(iter(self.manifest_pool))
        self.frames = self.manifest_pool[self.default_sequence]
        self.sequence = self.default_sequence

        self.lstm_predictor = lstm_predictor
        self.tcp_params = dict(tcp_params or {})
        self.tcp_params.setdefault('log_packets', False)  # training never reads packet logs
        self.feature_spec = list(feature_spec or DEFAULT_FEATURE_SPEC)
        self.hist_len = hist_len
        self.target_fps = target_fps
        self.buffer_capacity_s = buffer_capacity_s
        self.min_buffer_s = min_buffer_s
        self.bits_per_point = bits_per_point

        # Every sequence in the pool must share one ladder size (= action space).
        sizes = {name: len(fr[0]['representations']) if fr else 0
                 for name, fr in self.manifest_pool.items()}
        distinct = set(sizes.values())
        if len(distinct) != 1:
            raise ValueError(f"manifest pool has mixed ladder sizes: {sizes}")
        n_reps_manifest = distinct.pop()
        # num_reps follows the manifest (any ladder size); an explicit value must match.
        self.num_reps = num_reps if num_reps is not None else n_reps_manifest
        if self.num_reps != n_reps_manifest:
            raise ValueError(f"Manifest has {n_reps_manifest} representations per frame "
                             f"but the env/agent is configured for num_reps={num_reps}")

        # Pluggable reward. Back-compat: bare mu/lam args seed the spec unless a
        # full reward_spec overrides them.
        spec = {'mu': mu, 'lam': lam}
        spec.update(reward_spec or {})
        self.reward_fn = RewardFunction(spec)
        self.mu = self.reward_fn.spec['mu']
        self.lam = self.reward_fn.spec['lam']
        # Reward quality endpoints for the CURRENT sequence (reset per episode).
        self.q_low_density, self.q_high_density = quality_endpoints(self.frames)
        self.reward_fn.set_endpoints(self.q_low_density, self.q_high_density)

        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.norm['num_reps'] = self.num_reps
        self.norm['hist_len'] = hist_len
        self.norm['fps'] = target_fps
        if bits_per_point is not None:
            self.norm['bits_per_point'] = bits_per_point
        self.num_actions = self.num_reps
        self.state_dim = state_dim(self.feature_spec, self.num_reps, hist_len)

        # Build the origin once. Frame/rep ids overlap across sequences (0..N-1 x
        # 0..5) and the stored PointClouds are dummies, so registering the union
        # covers every sequence in the pool.
        self.server = Server("rl://origin")
        self.server.manifest.frames = self.frames
        seen = set()
        for fr_list in self.manifest_pool.values():
            for fr in fr_list:
                for rep in fr['representations']:
                    key = (fr['id'], rep['id'])
                    if key not in seen:
                        seen.add(key)
                        self.server.add_pointcloud(fr['id'], rep['id'], PointCloud(points=None))

        self.session = None

    def reset(self, trace, sequence=None):
        """Start an episode on `trace`, optionally switching content sequence."""
        if sequence is not None:
            if sequence not in self.manifest_pool:
                raise KeyError(f"unknown sequence '{sequence}' "
                               f"(pool: {sorted(self.manifest_pool)})")
            self.sequence = sequence
            self.frames = self.manifest_pool[sequence]
        self.server.manifest.frames = self.frames
        self.q_low_density, self.q_high_density = quality_endpoints(self.frames)
        self.reward_fn.set_endpoints(self.q_low_density, self.q_high_density)

        # The Server is shared across episodes; drop the previous episode's
        # backhaul registration so backhaul_links doesn't grow unboundedly.
        self.server.backhaul_links.clear()
        self.edge = EdgeNode("edge", server=self.server, tcp_params=self.tcp_params,
                             abr_factory=lambda: _ManualABR(), bits_per_point=self.bits_per_point)
        self.topo = Topology(self.server)
        self.topo.add_edge(self.edge)
        self.user = User("rl-user", self.target_fps, self.buffer_capacity_s, self.min_buffer_s)
        self.session = self.topo.add_user(self.user, self.edge, trace=trace)
        self.session.start()
        self.manual = self.session.abr
        self.lstm_provider = (LSTMABR(self.lstm_predictor, use_prediction=True)
                              if self.lstm_predictor is not None else None)
        if self.lstm_provider:
            self.lstm_provider.reset()
        self.frame_idx = 0
        self.prev_quality = None
        return self._observe()

    def _observe(self):
        if self.frame_idx >= len(self.frames):
            return np.zeros(self.state_dim, dtype=np.float32)
        frame = self.frames[self.frame_idx]
        st = self.session._build_state(frame['representations'], frame['id'])
        pred = self.lstm_provider.predict(st) if self.lstm_provider else None
        return build_state(self.feature_spec, st, predicted_bandwidth_bps=pred, norm=self.norm)

    def step(self, action):
        frame = self.frames[self.frame_idx]
        self.manual.next_action = int(action)
        record = self.session.step(frame, self.frame_idx)
        rep = record['rep']
        q = self.reward_fn.quality(rep)
        buffer_result = record['buffer_result']
        stall = buffer_result.get('stall_time_s', 0.0) or 0.0
        new_event = buffer_result.get('event') == 'rebuffering_start'
        reward = self.reward_fn.step(q, self.prev_quality, stall, new_event)
        self.prev_quality = q

        self.frame_idx += 1
        done = self.frame_idx >= len(self.frames)
        info = {
            'frame_id': record['frame_id'], 'rep_id': record['rep_id'], 'quality': q,
            'stall_s': stall, 'buffer_s': self.user.buffer.buffer_level_s, 'reward': reward,
            'sequence': self.sequence,
        }
        return self._observe(), reward, done, info

    def qoe(self):
        """Legacy stall-only QoE of the current episode's session."""
        return self.session.qoe() if self.session else 0

    def qoe_quality(self, w_stall=4.3, w_switch=1.0):
        """Quality-aware QoE' (raw, unclipped) of the current episode's session."""
        return self.session.qoe_quality(w_stall, w_switch) if self.session else 0.0

    def mean_quality(self):
        return self.session.mean_quality() if self.session else 0.0

    def total_stall_s(self):
        if not self.session:
            return 0.0
        return self.user.get_buffer_stats()['total_stall_time_s']
