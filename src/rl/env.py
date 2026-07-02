"""StreamingEnv: gym-like environment wrapping a one-session Topology.

One trace = one episode. The env drives a single StreamingSession whose ABR is a
`_ManualABR` placeholder — the DQN agent supplies the action, the env injects it,
and `session.step()` runs the same transfer+buffer+feature code path used at
inference (no train/inference skew). The LSTM is used here only as a feature
provider (its prediction is one input to the state vector).

Reward (Pensieve-style): quality(rep) - mu*stall_seconds - lam*|quality change|.
"""

import numpy as np

from ..network_model import Server, EdgeNode, User, Topology, LSTMABR, PointCloud
from ..network_model.abr import ABRStrategy
from .features import (build_state, state_dim, quality, quality_endpoints,
                       DEFAULT_FEATURE_SPEC, DEFAULT_NORM)


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
                 num_reps=3, norm=None, bits_per_point=None):
        self.frames = manifest_frames
        self.lstm_predictor = lstm_predictor
        self.tcp_params = dict(tcp_params or {})
        self.tcp_params.setdefault('log_packets', False)  # training never reads packet logs
        self.feature_spec = list(feature_spec or DEFAULT_FEATURE_SPEC)
        self.mu = mu
        self.lam = lam
        self.hist_len = hist_len
        self.target_fps = target_fps
        self.buffer_capacity_s = buffer_capacity_s
        self.min_buffer_s = min_buffer_s
        self.bits_per_point = bits_per_point
        self.num_reps = num_reps
        n_reps_manifest = len(manifest_frames[0]['representations']) if manifest_frames else 0
        if n_reps_manifest != num_reps:
            raise ValueError(f"Manifest has {n_reps_manifest} representations per frame "
                             f"but the env/agent is configured for num_reps={num_reps}")
        # Reward quality endpoints derived from THIS manifest's density range, so
        # the lowest rep maps to ~0.0 and the highest to ~1.0 on any ladder.
        self.q_low_density, self.q_high_density = quality_endpoints(manifest_frames)
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.norm['num_reps'] = num_reps
        self.norm['hist_len'] = hist_len
        self.norm['fps'] = target_fps
        if bits_per_point is not None:
            self.norm['bits_per_point'] = bits_per_point
        self.num_actions = num_reps
        self.state_dim = state_dim(self.feature_spec, num_reps, hist_len)

        # Build the origin once (point clouds reused across episodes).
        self.server = Server("rl://origin")
        self.server.manifest.frames = self.frames
        for fr in self.frames:
            for rep in fr['representations']:
                self.server.add_pointcloud(fr['id'], rep['id'], PointCloud(points=None))

        self.session = None

    def reset(self, trace):
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
        q = quality(rep, self.q_low_density, self.q_high_density)
        stall = record['buffer_result'].get('stall_time_s', 0.0) or 0.0
        switch = abs(q - self.prev_quality) if self.prev_quality is not None else 0.0
        reward = q - self.mu * stall - self.lam * switch
        self.prev_quality = q

        self.frame_idx += 1
        done = self.frame_idx >= len(self.frames)
        info = {
            'frame_id': record['frame_id'], 'rep_id': record['rep_id'], 'quality': q,
            'stall_s': stall, 'buffer_s': self.user.buffer.buffer_level_s, 'reward': reward,
        }
        return self._observe(), reward, done, info

    def qoe(self):
        return self.session.qoe() if self.session else 0
