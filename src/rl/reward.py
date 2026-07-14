"""Pluggable, config-driven reward formulation for the streaming RL env.

The reward spec is a plain dict (JSON-serializable so sweep.py can grid over it
and checkpoints can persist it). All keys optional; the defaults reproduce the
legacy Pensieve form exactly:

    reward = quality - mu * stall_s - lam * |dq|

Spec keys:
  utility:       'log_density' (default) | 'linear_density' | 'coded_psnr'
                 quality signal in [0, 1]. 'coded_psnr' reads `psnr_key` from the
                 manifest rep (falls back to log_density when the key is absent,
                 so it is safe to enable before the PSNR-enriched manifests land).
  psnr_key:      manifest rep key for 'coded_psnr' (default 'd1_psnr')
  psnr_lo/psnr_hi: dB endpoints mapped to 0/1 (defaults 30 / 75)
  mu:            stall weight (default 4.3)
  stall_mode:    'linear' (default) | 'bounded'
  stall_cap_s:   per-STEP stall cap in seconds for 'bounded' (default 2.0) —
                 bounds the per-frame penalty spike so Q-targets stay sane while
                 total stall still accumulates across steps
  event_penalty: extra one-off penalty each time a NEW stall event begins
                 (default 0.0; use with 'bounded' to keep event count costly)
  lam:           switch weight (default 1.0)
  switch_mode:   'l1' (default) | 'l2'
  lam_down:      optional separate weight for DOWNWARD switches (asymmetric;
                 None => symmetric lam for both directions)
  startup_weight: penalty per second of PRE-playback waiting (default 0.0 = off,
                 old shape bit-identical). Mirrors the QoE″ startup term so the
                 policy is charged for slow cold starts (binding now that traces
                 keep their near-zero idle-attach openings).
  drop_weight:   penalty per frame dropped this step from buffer overflow
                 (default 0.0 = off). Guardrail — a buffer-watching policy
                 rarely drops, but the harder driving regime may tempt it.
"""

from ..network_model.manifest import density_quality

DEFAULT_REWARD_SPEC = {
    'utility': 'log_density',
    'psnr_key': 'd1_psnr',
    'psnr_lo': 30.0,
    'psnr_hi': 75.0,
    'mu': 4.3,
    'stall_mode': 'linear',
    'stall_cap_s': 2.0,
    'event_penalty': 0.0,
    'lam': 1.0,
    'switch_mode': 'l1',
    'lam_down': None,
    'startup_weight': 0.0,
    'drop_weight': 0.0,
}

_UTILITIES = ('log_density', 'linear_density', 'coded_psnr')
_STALL_MODES = ('linear', 'bounded')
_SWITCH_MODES = ('l1', 'l2')


class RewardFunction:
    """Quality/stall/switch reward with per-manifest quality endpoints.

    Endpoints are (re)set per episode via `set_endpoints` so any sequence's
    ladder maps its lowest rep to ~0.0 and highest to ~1.0.
    """

    def __init__(self, spec=None, q_low_density=None, q_high_density=None):
        self.spec = {**DEFAULT_REWARD_SPEC, **(spec or {})}
        if self.spec['utility'] not in _UTILITIES:
            raise ValueError(f"unknown utility '{self.spec['utility']}' (want {_UTILITIES})")
        if self.spec['stall_mode'] not in _STALL_MODES:
            raise ValueError(f"unknown stall_mode '{self.spec['stall_mode']}' (want {_STALL_MODES})")
        if self.spec['switch_mode'] not in _SWITCH_MODES:
            raise ValueError(f"unknown switch_mode '{self.spec['switch_mode']}' (want {_SWITCH_MODES})")
        self.q_lo = q_low_density
        self.q_hi = q_high_density

    def set_endpoints(self, q_low_density, q_high_density):
        self.q_lo = q_low_density
        self.q_hi = q_high_density

    def quality(self, rep):
        """Quality utility in [0, 1] for a chosen representation."""
        u = self.spec['utility']
        if u == 'coded_psnr':
            v = rep.get(self.spec['psnr_key'])
            if v is not None:
                lo, hi = float(self.spec['psnr_lo']), float(self.spec['psnr_hi'])
                if hi > lo:
                    return max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
            # PSNR not in this manifest -> fall through to log_density.
        elif u == 'linear_density':
            lo = float(self.q_lo or 1.0)
            hi = float(self.q_hi or 1.0)
            if hi > lo:
                d = float(rep.get('density') or 0.0)
                return max(0.0, min(1.0, (d - lo) / (hi - lo)))
            return 0.0
        return density_quality(rep.get('density'), self.q_lo, self.q_hi)

    def step(self, q, prev_q, stall_s, new_stall_event=False,
             startup_s=0.0, dropped=0):
        """Per-step reward from this step's quality, stall and the previous q.

        startup_s: seconds of pre-playback waiting elapsed THIS step (charged
        at startup_weight; 0.0 weight by default = old shape). dropped: frames
        dropped to buffer overflow this step (charged at drop_weight)."""
        s = float(stall_s or 0.0)
        if self.spec['stall_mode'] == 'bounded':
            s = min(s, float(self.spec['stall_cap_s']))
        penalty = float(self.spec['mu']) * s
        if new_stall_event and self.spec['event_penalty']:
            penalty += float(self.spec['event_penalty'])
        if startup_s and self.spec.get('startup_weight'):
            penalty += float(self.spec['startup_weight']) * float(startup_s)
        if dropped and self.spec.get('drop_weight'):
            penalty += float(self.spec['drop_weight']) * float(dropped)
        switch = 0.0
        if prev_q is not None:
            dq = q - prev_q
            mag = dq * dq if self.spec['switch_mode'] == 'l2' else abs(dq)
            w = float(self.spec['lam'])
            if dq < 0 and self.spec.get('lam_down') is not None:
                w = float(self.spec['lam_down'])
            switch = w * mag
        return q - penalty - switch

    def step_segment(self, q_sum, q_mean, prev_q_mean, stall_s, new_stall_event=False,
                     startup_s=0.0, dropped=0):
        """Per-SEGMENT reward: per-frame qualities SUMMED (so episode reward
        magnitudes stay comparable across segment sizes), ONE stall penalty
        (the bounded cap applies to the whole segment's stall) and ONE switch
        penalty on mean quality vs the previous segment. With a 1-frame segment
        this equals step() exactly."""
        base = self.step(q_mean, prev_q_mean, stall_s, new_stall_event,
                         startup_s=startup_s, dropped=dropped)
        return base - q_mean + q_sum

    def describe(self):
        sp = self.spec
        stall = f"{sp['mu']}*stall"
        if sp['stall_mode'] == 'bounded':
            stall = f"{sp['mu']}*min(stall,{sp['stall_cap_s']})"
        if sp['event_penalty']:
            stall += f" + {sp['event_penalty']}*new_event"
        dq = '|dq|' if sp['switch_mode'] == 'l1' else 'dq^2'
        lam = (f"{sp['lam']}(up)/{sp['lam_down']}(down)"
               if sp.get('lam_down') is not None else f"{sp['lam']}")
        extra = ''
        if sp.get('startup_weight'):
            extra += f" - {sp['startup_weight']}*startup"
        if sp.get('drop_weight'):
            extra += f" - {sp['drop_weight']}*drops"
        return f"q[{sp['utility']}] - {stall} - {lam}*{dq}{extra}"
