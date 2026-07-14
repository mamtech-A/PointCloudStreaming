"""Tests for QoE″ v2 (startup + drop terms) and the matching reward-spec keys.

Run directly: python tests/test_qoe_reward.py
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.reward import RewardFunction, DEFAULT_REWARD_SPEC
from src.network_model import Server, EdgeNode, User, Topology
from src.network_model.manifest import PointCloud
from src.network_model.trace import BandwidthTrace


# ---------------------------------------------------------------- reward spec

def test_reward_old_specs_bit_identical():
    """Round-1/2/3 specs (no startup/drop keys) are unaffected — even when the
    env passes nonzero startup_s/dropped, the 0.0 default weights ignore them."""
    rf = RewardFunction({'stall_mode': 'bounded', 'stall_cap_s': 2.0,
                         'event_penalty': 2.0})
    rf.set_endpoints(1e4, 1e6)
    base = rf.step(0.8, 0.6, 1.5, new_stall_event=True)
    with_extras = rf.step(0.8, 0.6, 1.5, new_stall_event=True,
                          startup_s=7.0, dropped=42)
    assert base == with_extras, (base, with_extras)
    seg = rf.step_segment(6.4, 0.8, 0.6, 1.5, True)
    seg2 = rf.step_segment(6.4, 0.8, 0.6, 1.5, True, startup_s=7.0, dropped=42)
    assert seg == seg2


def test_reward_startup_and_drop_weights_exact():
    """startup_weight/drop_weight change the reward by EXACTLY w*value."""
    spec = {'stall_mode': 'bounded', 'stall_cap_s': 2.0}
    off = RewardFunction(spec)
    on = RewardFunction({**spec, 'startup_weight': 1.0, 'drop_weight': 0.5})
    for q, prev, stall, su, dr in [(0.9, 0.5, 0.0, 3.0, 0), (0.3, None, 2.5, 0.0, 4),
                                   (0.7, 0.7, 1.0, 1.25, 2)]:
        d = (off.step(q, prev, stall, startup_s=su, dropped=dr)
             - on.step(q, prev, stall, startup_s=su, dropped=dr))
        assert abs(d - (1.0 * su + 0.5 * dr)) < 1e-12, d


def test_reward_describe_mentions_new_terms():
    rf = RewardFunction({'startup_weight': 1.0, 'drop_weight': 1.0})
    d = rf.describe()
    assert 'startup' in d and 'drops' in d, d
    assert 'startup' not in RewardFunction().describe()


def test_default_spec_has_new_keys_off():
    assert DEFAULT_REWARD_SPEC['startup_weight'] == 0.0
    assert DEFAULT_REWARD_SPEC['drop_weight'] == 0.0


# ------------------------------------------------------------- session / QoE″

def _mk_session(trace, n_frames=30, buffer_capacity_s=5.0):
    """Tiny synthetic 2-rep session driven by a fixed mid-tier ABR."""
    frames = [{'id': i, 'representations': [
        {'id': 0, 'density': 2e4, 'size': 4000, 'quality': 'lo'},
        {'id': 1, 'density': 8e5, 'size': 160000, 'quality': 'hi'},
    ]} for i in range(n_frames)]
    server = Server("t://origin")
    server.manifest.frames = frames
    for fr in frames:
        for rep in fr['representations']:
            server.add_pointcloud(fr['id'], rep['id'], PointCloud(points=None))
    topo = Topology(server)
    edge = EdgeNode("e", server=server,
                    tcp_params=dict(rtt_ms=40.0, rtt_jitter_ms=0.0, loss_prob=0.0,
                                    cwnd_packets=10, mss_bytes=1460, log_packets=False))
    topo.add_edge(edge)
    user = User("u", buffer_capacity_s=buffer_capacity_s)
    session = topo.add_user(user, edge, trace=trace)
    session.start()

    class _Fixed:
        name = 'fixed'
        def reset(self): pass
        def select(self, state): return 1
        def report(self, *a): return None
    session.abr = _Fixed()
    return session, frames


def test_qoe_terms_total_and_startup():
    """Breakdown sums to the total; startup term = -1.0 * startup_delay_s; a
    dead 2 s trace opening surfaces as startup (NOT as stall — no double count)."""
    random.seed(0)
    tr = BandwidthTrace([2e3, 2e3, 50e6, 50e6, 50e6, 50e6],
                        sample_times_s=[0, 1, 2, 3, 4, 5])
    session, frames = _mk_session(tr)
    for i in range(0, len(frames), 5):
        session.step_segment(frames[i:i + 5], i)
    stats = session.user.get_buffer_stats()
    terms = session.qoe_quality_terms()
    assert abs(terms['total'] - sum(v for k, v in terms.items() if k != 'total')) < 1e-9
    assert abs(session.qoe_quality() - terms['total']) < 1e-9
    # Cold start crossed the 2 s dead zone -> startup delay > 2 s, charged at 1/s.
    assert stats['startup_delay_s'] > 2.0, stats['startup_delay_s']
    assert abs(terms['startup'] + 1.0 * stats['startup_delay_s']) < 1e-9
    # No double count: the pre-playback wait is NOT stall time.
    assert stats['total_stall_time_s'] < 0.5, stats['total_stall_time_s']
    # v1 is recoverable: total minus the two new terms.
    v1 = terms['total'] - terms['startup'] - terms['drops']
    assert abs(v1 - (terms['quality'] + terms['stall'] + terms['switch']
                     + terms['slowdown'])) < 1e-9


def test_qoe_drop_term_charges_100_over_N():
    """Overflowing the buffer charges exactly (100/N) per dropped frame."""
    random.seed(0)
    # Fast flat link + tiny buffer -> eager fetch must overflow.
    tr = BandwidthTrace([200e6] * 30)
    session, frames = _mk_session(tr, n_frames=90, buffer_capacity_s=1.0)
    for i in range(0, len(frames), 5):
        session.step_segment(frames[i:i + 5], i)
    stats = session.user.get_buffer_stats()
    assert stats['frames_dropped'] > 0, "expected buffer-overflow drops"
    terms = session.qoe_quality_terms()
    n = len(session.chosen_densities)
    assert abs(terms['drops'] + (100.0 / n) * stats['frames_dropped']) < 1e-9
    # Weight override works.
    t0 = session.qoe_quality_terms(w_drop=0.0)
    assert t0['drops'] == 0.0


def test_startup_delay_frozen_after_playback_starts():
    """startup_delay_s stops growing once playback begins."""
    random.seed(0)
    tr = BandwidthTrace([50e6] * 60)
    session, frames = _mk_session(tr, n_frames=60)
    # min_buffer_s=1.0 s = 30 frames at 30 fps; 35 frames clears it even with
    # the 1/30-summation float error.
    for i in range(0, 35, 5):
        session.step_segment(frames[i:i + 5], i)
    s1 = session.user.get_buffer_stats()
    assert s1['playback_started']
    d1 = s1['startup_delay_s']
    for i in range(35, 60, 5):
        session.step_segment(frames[i:i + 5], i)
    s2 = session.user.get_buffer_stats()
    assert s2['startup_delay_s'] == d1, (d1, s2['startup_delay_s'])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")


if __name__ == '__main__':
    _run_all()
