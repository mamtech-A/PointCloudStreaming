"""One canonical six-term objective for both DQN reward and reported QoE.

For an episode containing ``N`` frames and segment ``t`` containing ``S_t``::

    r_t = (100/N) * sum(q_i for i in S_t)
          - mu * delta_stall_seconds
          - rebuffer_weight * delta_rebuffer_events
          - lam * abs(segment_quality_t - segment_quality_t_minus_1)
          - (100/N) * delta_dropped_frames
          - startup_weight * delta_startup_seconds

Summing the segment rewards gives the reported episode QoE exactly. Startup is
pre-playback waiting; stall time and rebuffer events begin only after playback
has started. The quality and default drop coefficients share the fixed episode
scale ``100/N``.
"""

from ..network_model.manifest import tier_quality

OBJECTIVE_VERSION = 'six_term_equal_reward_qoe_v2_fixed_log_bitrate'

DEFAULT_REWARD_SPEC = {
    'utility': 'fixed_log_bitrate',
    'mu': 4.3,
    'rebuffer_weight': 2.0,
    'lam': 1.0,
    'startup_weight': 1.0,
    # None means the same 100/N coefficient used for per-frame quality.
    'drop_weight': None,
}

_UTILITIES = ('fixed_log_bitrate',)


class RewardFunction:
    """Canonical reward/QoE with one fixed six-tier quality table."""

    def __init__(self, spec=None, q_low_density=None, q_high_density=None):
        unknown = set(spec or {}) - set(DEFAULT_REWARD_SPEC)
        if unknown:
            raise ValueError(
                f"unsupported reward keys {sorted(unknown)}; the reward and QoE "
                "must use only the canonical six-term objective"
            )
        self.spec = {**DEFAULT_REWARD_SPEC, **(spec or {})}
        if self.spec['utility'] not in _UTILITIES:
            raise ValueError(f"unknown utility '{self.spec['utility']}' (want {_UTILITIES})")
        self.q_lo = q_low_density
        self.q_hi = q_high_density

    def set_endpoints(self, q_low_density, q_high_density):
        """Compatibility no-op; fixed tier utility has no content endpoints."""
        self.q_lo = q_low_density
        self.q_hi = q_high_density

    def quality(self, rep):
        """Fixed normalized log-bitrate utility for the chosen tier."""
        return tier_quality(rep)

    @staticmethod
    def quality_scale(episode_frames):
        n = int(episode_frames)
        if n <= 0:
            raise ValueError("episode_frames must be positive")
        return 100.0 / n

    def drop_weight(self, episode_frames):
        configured = self.spec.get('drop_weight')
        return (self.quality_scale(episode_frames) if configured is None
                else float(configured))

    def step(self, q, prev_q, stall_s, rebuffer_events=0,
             startup_s=0.0, dropped=0, episode_frames=None):
        """One-frame adapter for :meth:`step_segment`."""
        return self.step_segment(
            q, q, prev_q, stall_s, rebuffer_events,
            startup_s=startup_s, dropped=dropped,
            episode_frames=episode_frames,
        )

    def step_segment(self, q_sum, q_mean, prev_q_mean, stall_s,
                     rebuffer_events=0, startup_s=0.0, dropped=0,
                     episode_frames=None):
        """Canonical per-segment reward whose episode sum equals QoE."""
        scale = self.quality_scale(episode_frames)
        quality_change = (0.0 if prev_q_mean is None
                          else abs(float(q_mean) - float(prev_q_mean)))
        return (
            scale * float(q_sum)
            - float(self.spec['mu']) * float(stall_s or 0.0)
            - float(self.spec['rebuffer_weight']) * int(rebuffer_events or 0)
            - float(self.spec['lam']) * quality_change
            - self.drop_weight(episode_frames) * int(dropped or 0)
            - float(self.spec['startup_weight']) * float(startup_s or 0.0)
        )

    def episode_terms(self, q_sum, quality_change_sum, stall_s,
                      rebuffer_events, dropped, startup_s, episode_frames):
        """Signed episode terms computed independently from cumulative outcomes."""
        terms = {
            'quality': self.quality_scale(episode_frames) * float(q_sum),
            'stall_duration': -float(self.spec['mu']) * float(stall_s or 0.0),
            'rebuffering': (-float(self.spec['rebuffer_weight'])
                            * int(rebuffer_events or 0)),
            'quality_change': (-float(self.spec['lam'])
                               * float(quality_change_sum or 0.0)),
            'frame_drops': (-self.drop_weight(episode_frames)
                            * int(dropped or 0)),
            'startup_delay': (-float(self.spec['startup_weight'])
                              * float(startup_s or 0.0)),
        }
        terms['total'] = sum(terms.values())
        return terms

    def describe(self):
        sp = self.spec
        drop = '(100/N)' if sp.get('drop_weight') is None else sp['drop_weight']
        return (
            f"(100/N)*sum(q[{sp['utility']}]) - {sp['mu']}*stall_duration "
            f"- {sp['rebuffer_weight']}*rebuffer_events "
            f"- {sp['lam']}*sum|delta_segment_q| - {drop}*dropped_frames "
            f"- {sp['startup_weight']}*startup_delay"
        )
