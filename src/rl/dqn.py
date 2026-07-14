"""Double-DQN agent for ABR representation selection.

Q-network is a small MLP over the state snapshot (the LSTM already encodes temporal
bandwidth). Save format mirrors LSTMPredictor.save (a pickled dict) and persists the
`feature_spec` + `norm_constants` + `lstm_model_path` so inference rebuilds the exact
state vector.
"""

import os
import pickle
import random as _random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32), np.asarray(s2, dtype=np.float32),
                np.asarray(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim, num_actions, feature_spec, norm_constants,
                 hidden=128, lr=5e-4, gamma=0.99, batch_size=64, buffer_capacity=100_000,
                 target_update_freq=500, double_dqn=True, device=None,
                 mu=4.3, lam=1.0, lstm_model_path=None, sequence_length=10,
                 reward_norm=False):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.feature_spec = list(feature_spec)
        self.norm_constants = dict(norm_constants)
        self.hidden = hidden
        self.gamma = gamma
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.target_update_freq = target_update_freq
        self.mu = mu
        self.lam = lam
        self.lstm_model_path = lstm_model_path
        self.sequence_length = sequence_length
        # Learner-side reward normalization (argmax-invariant): rewards are
        # divided by a RUNNING std (Welford) at learn() time, replacing the
        # hand-tuned --reward-scale hack. Stall spikes of -40..-90 stop blowing
        # up Q-targets against the grad clip; reported rewards stay raw.
        self.reward_norm = bool(reward_norm)
        self._r_count = 0
        self._r_mean = 0.0
        self._r_m2 = 0.0

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.policy = QNetwork(state_dim, num_actions, hidden).to(self.device)
        self.target = QNetwork(state_dim, num_actions, hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(buffer_capacity)
        self.learn_steps = 0
        self.training_history = {'loss': [], 'eval': []}

    def act(self, features, epsilon=0.0):
        if epsilon > 0 and _random.random() < epsilon:
            return _random.randrange(self.num_actions)
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy(x).argmax(dim=1).item())

    def q_values(self, features):
        """Q-value per action for one state (inference-time introspection —
        run_dqn.py logs these per decision; not part of the training path)."""
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(x).squeeze(0).cpu().numpy()

    def push(self, s, a, r, s2, done):
        if self.reward_norm:
            self._r_count += 1
            delta = r - self._r_mean
            self._r_mean += delta / self._r_count
            self._r_m2 += delta * (r - self._r_mean)
        self.replay.push(s, a, r, s2, done)

    def reward_scale(self):
        """Current 1/std normalizer (>= 1e-8 count guard, std floored at 0.1)."""
        if not self.reward_norm or self._r_count < 2:
            return 1.0
        std = (self._r_m2 / self._r_count) ** 0.5
        return 1.0 / max(std, 0.1)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        if self.reward_norm:
            r = r * self.reward_scale()
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_a = self.policy(s2).argmax(dim=1)
                next_q = self.target(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target(s2).max(dim=1)[0]
            target = r + self.gamma * (1.0 - d) * next_q

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())

        l = float(loss.item())
        self.training_history['loss'].append(l)
        return l

    def save(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        save = {
            'model_state_dict': self.policy.state_dict(),
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'feature_spec': self.feature_spec,
            'norm_constants': self.norm_constants,
            'hidden': self.hidden,
            'gamma': self.gamma,
            'double_dqn': self.double_dqn,
            'mu': self.mu,
            'lam': self.lam,
            'lstm_model_path': self.lstm_model_path,
            'sequence_length': self.sequence_length,
            'training_history': self.training_history,
            'reward_norm': self.reward_norm,
            'reward_norm_stats': (self._r_count, self._r_mean, self._r_m2),
        }
        with open(path, 'wb') as f:
            pickle.dump(save, f)

    @classmethod
    def load(cls, path, device=None):
        with open(path, 'rb') as f:
            save = pickle.load(f)
        # Migrate checkpoints saved before the 'rep_sizes' -> 'rep_bitrates' rename
        # (same dimensionality, so weights stay valid).
        feature_spec = ['rep_bitrates' if f == 'rep_sizes' else f
                        for f in save['feature_spec']]
        agent = cls(
            save['state_dim'], save['num_actions'], feature_spec, save['norm_constants'],
            hidden=save['hidden'], gamma=save['gamma'], double_dqn=save['double_dqn'],
            mu=save.get('mu', 4.3), lam=save.get('lam', 1.0),
            lstm_model_path=save.get('lstm_model_path'),
            sequence_length=save.get('sequence_length', 10), device=device,
        )
        agent.policy.load_state_dict(save['model_state_dict'])
        agent.target.load_state_dict(save['model_state_dict'])
        agent.training_history = save.get('training_history', {'loss': [], 'eval': []})
        agent.reward_norm = save.get('reward_norm', False)
        stats = save.get('reward_norm_stats')
        if stats:
            agent._r_count, agent._r_mean, agent._r_m2 = stats
        return agent
