"""Tests for the one canonical five-term reward/QoE objective.

Run directly: python tests/test_qoe_reward.py
"""

import os
import sys
import random
import inspect
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.reward import RewardFunction, DEFAULT_REWARD_SPEC
from src.rl.env import StreamingEnv
from src.rl.dqn import DQNAgent
from src.network_model import Server, EdgeNode, User, Topology
from src.network_model.buffer import ClientBuffer
from src.network_model.finite_trace import FiniteTraceWindow, TraceWindowExhausted
from src.network_model.manifest import (
    PointCloud, TIER_AVERAGE_BITRATE_BPS, TIER_QUALITY_UTILITY, tier_quality,
    parse_mpd_xml,
)
from src.network_model.trace import BandwidthTrace


# ---------------------------------------------------------------- reward spec

def test_segment_reward_has_exact_five_terms():
    rf = RewardFunction()
    got = rf.step_segment(
        q_sum=6.4, q_mean=0.8, prev_q_mean=0.6,
        stall_s=1.5, rebuffer_events=1, startup_s=0.7,
        episode_frames=300,
    )
    expected = ((100.0 / 300) * 6.4 - 4.3 * 1.5 - 2.0
                - abs(0.8 - 0.6) - 0.7)
    assert abs(got - expected) < 1e-12, (got, expected)


def test_stall_duration_is_not_capped():
    rf = RewardFunction()
    got = rf.step_segment(
        q_sum=0.0, q_mean=0.0, prev_q_mean=None,
        stall_s=9.0, rebuffer_events=0, episode_frames=300,
    )
    assert got == -4.3 * 9.0


def test_quality_change_uses_fixed_tier_gap():
    rf = RewardFunction()
    high = rf.quality({'quality': 'high'})
    medium = rf.quality({'quality': 'med'})
    got = rf.step_segment(
        q_sum=medium, q_mean=medium, prev_q_mean=high,
        stall_s=0.0, rebuffer_events=0, episode_frames=100,
    )
    expected = medium - abs(medium - high)
    assert abs(got - expected) < 1e-12


def test_episode_terms_match_sum_of_segment_rewards():
    rf = RewardFunction()
    r1 = rf.step_segment(4.0, 0.5, None, 0.0, 0,
                         startup_s=2.0, episode_frames=16)
    r2 = rf.step_segment(6.4, 0.8, 0.5, 1.25, 1,
                         startup_s=0.0, episode_frames=16)
    terms = rf.episode_terms(10.4, 0.3, 1.25, 1, 2.0, 16)
    assert abs((r1 + r2) - terms['total']) < 1e-12
    assert set(terms) == {
        'quality', 'stall_duration', 'rebuffering', 'quality_change',
        'startup_delay', 'total',
    }


def test_superseded_reward_shapes_are_rejected():
    for key in ('stall_mode', 'stall_cap_s', 'event_penalty',
                'switch_mode', 'lam_down', 'drop_weight'):
        try:
            RewardFunction({key: 1})
        except ValueError as exc:
            assert 'canonical five-term' in str(exc)
        else:
            raise AssertionError(f"superseded key {key!r} was accepted")


def test_default_spec_enables_all_requested_costs():
    assert DEFAULT_REWARD_SPEC['utility'] == 'fixed_log_bitrate'
    assert DEFAULT_REWARD_SPEC['mu'] == 4.3
    assert DEFAULT_REWARD_SPEC['rebuffer_weight'] == 2.0
    assert DEFAULT_REWARD_SPEC['lam'] == 1.0
    assert DEFAULT_REWARD_SPEC['startup_weight'] == 1.0
    description = RewardFunction().describe()
    for text in ('stall_duration', 'rebuffer_events', 'delta_segment_q',
                 'startup_delay'):
        assert text in description, description
    assert 'drop' not in description


def test_fixed_tier_quality_is_log_bitrate_normalized():
    lo = TIER_AVERAGE_BITRATE_BPS['vlow']
    hi = TIER_AVERAGE_BITRATE_BPS['high']
    denominator = math.log(hi) - math.log(lo)
    assert TIER_QUALITY_UTILITY['high'] == 1.0
    assert TIER_QUALITY_UTILITY['vlow'] == 0.0
    previous = 2.0
    for tier, bitrate in TIER_AVERAGE_BITRATE_BPS.items():
        expected = (math.log(bitrate) - math.log(lo)) / denominator
        assert abs(tier_quality(tier) - expected) < 1e-12
        assert tier_quality(tier) < previous
        previous = tier_quality(tier)


def test_fixed_tier_quality_does_not_change_with_frame_content():
    a = {'quality': 'med', 'density': 100, 'bandwidth': 1}
    b = {'quality': 'medium', 'density': 9_999_999, 'bandwidth': 9_999_999_999}
    assert tier_quality(a) == tier_quality(b) == TIER_QUALITY_UTILITY['med']


def test_fixed_tier_bitrates_match_all_four_manifests():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sequences = ('longdress', 'loot', 'redandblack', 'soldier')
    per_sequence = []
    for sequence in sequences:
        frames = parse_mpd_xml(os.path.join(
            project_root, 'manifests', f'mpd_gpcc_{sequence}.xml'))
        sums = {tier: 0.0 for tier in TIER_AVERAGE_BITRATE_BPS}
        counts = {tier: 0 for tier in TIER_AVERAGE_BITRATE_BPS}
        for frame in frames:
            for rep in frame['representations']:
                tier = rep['quality']
                sums[tier] += rep['bandwidth']
                counts[tier] += 1
        per_sequence.append({tier: sums[tier] / counts[tier] for tier in sums})
    for tier, frozen_bitrate in TIER_AVERAGE_BITRATE_BPS.items():
        equal_weight_mean = sum(row[tier] for row in per_sequence) / len(per_sequence)
        assert abs(equal_weight_mean - frozen_bitrate) < 1e-6


def test_dqn_uses_undiscounted_unscaled_reward():
    parameters = inspect.signature(DQNAgent.__init__).parameters
    assert parameters['gamma'].default == 1.0
    assert 'reward_norm' not in parameters


# ------------------------------------------------------------- session / QoE″

def _synthetic_frames(n_frames):
    return [{'id': i, 'representations': [
        {'id': 0, 'density': 8e5, 'size': 160000, 'quality': 'high'},
        {'id': 1, 'density': 2e4, 'size': 4000, 'quality': 'vlow'},
    ]} for i in range(n_frames)]


def _mk_session(trace, n_frames=30, buffer_capacity_s=5.0):
    """Tiny synthetic 2-rep session driven by a fixed mid-tier ABR."""
    frames = _synthetic_frames(n_frames)
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
    assert abs(terms['startup_delay'] + stats['startup_delay_s']) < 1e-9
    # No double count: the pre-playback wait is NOT stall time.
    assert stats['total_stall_time_s'] < 0.5, stats['total_stall_time_s']
    assert stats['rebuffer_count'] == 0
    assert terms['rebuffering'] == 0.0


def test_request_pacing_prevents_buffer_overflow():
    """A fast link waits before requests instead of discarding received media."""
    random.seed(0)
    # Fast flat link + tiny buffer forces intentional inter-request waiting.
    tr = BandwidthTrace([200e6] * 30)
    session, frames = _mk_session(tr, n_frames=90, buffer_capacity_s=1.0)
    for i in range(0, len(frames), 5):
        session.step_segment(frames[i:i + 5], i)
    stats = session.user.get_buffer_stats()
    assert 'frames_dropped' not in stats
    assert stats['frames_received'] == len(frames)
    assert session.total_request_pacing_s > 0.0
    assert stats['buffer_level_s'] <= stats['buffer_capacity_s'] + 1e-9
    terms = session.qoe_quality_terms()
    assert 'frame_drops' not in terms


def test_request_pacing_advances_trace_clock_but_not_qoe_or_throughput_sample():
    """Idle pacing is visible to the next decision but is not network time."""
    random.seed(0)
    session, frames = _mk_session(
        BandwidthTrace([200e6] * 60), n_frames=40, buffer_capacity_s=1.0
    )
    # Fill the one-second buffer and cross the one-second startup threshold.
    for i in range(0, 30, 5):
        session.step_segment(frames[i:i + 5], i)
    before = session.user.get_buffer_stats()
    assert before['playback_started'] and before['is_playing']
    clock_before = session.cumulative_time_s
    history_before = list(session.observed_throughput_history)

    paced_s = session.prepare_request(5, 30)
    paced_clock = session.cumulative_time_s
    assert session.prepare_request(5, 30) == paced_s
    assert session.cumulative_time_s == paced_clock
    try:
        session.prepare_request(5, 35)
    except RuntimeError as exc:
        assert 'does not match' in str(exc)
    else:
        raise AssertionError('mismatched prepared request was accepted')
    after_pacing = session.user.get_buffer_stats()
    assert paced_s > 0.0
    assert abs(session.cumulative_time_s - clock_before - paced_s) < 1e-9
    assert session.observed_throughput_history == history_before
    assert after_pacing['startup_delay_s'] == before['startup_delay_s']
    assert after_pacing['total_stall_time_s'] == before['total_stall_time_s']
    assert after_pacing['rebuffer_count'] == before['rebuffer_count']
    assert abs(after_pacing['buffer_level_s'] - (1.0 - 5.0 / 30.0)) < 1e-9

    class _RecordingFixed:
        name = 'recording-fixed'
        def __init__(self): self.buffer_levels = []
        def reset(self): pass
        def select(self, state):
            self.buffer_levels.append(state.buffer_level_s)
            return 1
        def report(self, *args): return None

    policy = _RecordingFixed()
    session.abr = policy
    record = session.step_segment(frames[30:35], 30)
    assert abs(record['request_pacing_s'] - paced_s) < 1e-9
    assert abs(record['request_start_s'] - (clock_before + paced_s)) < 1e-9
    assert abs(policy.buffer_levels[-1] - after_pacing['buffer_level_s']) < 1e-9
    transfer_s = record['metrics']['time_s']
    expected_bps = record['metrics']['sent_bytes'] * 8.0 / transfer_s
    assert abs(session.observed_throughput_history[-1] - expected_bps) < 1e-6
    assert abs(session.observed_throughput_history[-1]
               - record['data_bytes'] * 8.0 / (transfer_s + paced_s)) > 1.0


def test_fractional_playback_waits_preserve_frame_accounting():
    buffer = ClientBuffer(target_fps=30.0, buffer_capacity_s=1.0,
                          min_buffer_s=1.0 / 30.0)
    for frame_id in range(12):
        buffer.add_frame(frame_id, 0, 1, 0.0, 0.0)
    for step in range(1, 11):
        buffer.advance_playback(0.01, step * 0.01)
    stats = buffer.get_statistics()
    assert stats['frames_played'] == 3
    assert stats['buffer_level_frames'] == 9
    assert abs(stats['buffer_level_s'] - 0.3) < 1e-9


def test_exact_depletion_is_not_rebuffer_until_positive_unplayed_time():
    buffer = ClientBuffer(target_fps=10.0, buffer_capacity_s=1.0,
                          min_buffer_s=0.1)
    for frame_id in range(5):
        buffer.add_frame(frame_id, 0, 1, 0.0, 0.0)
    buffer.advance_playback(0.5, 0.5)
    assert buffer.buffer_level_s == 0.0
    assert buffer.rebuffer_count == 0

    # Zero-time arrival at the depletion boundary keeps playback continuous.
    buffer.add_frame(5, 0, 1, 0.0, 0.5)
    assert buffer.rebuffer_count == 0
    assert buffer.total_stall_time == 0.0

    # After that frame is played, a positive 0.2 s gap is one true event.
    buffer.advance_playback(0.1, 0.6)
    result = buffer.add_frame(6, 0, 1, 0.2, 0.8)
    assert result['event'] == 'playback_resumed'
    assert buffer.rebuffer_count == 1
    assert abs(buffer.total_stall_time - 0.2) < 1e-9


def test_pacing_cannot_move_a_request_beyond_a_finite_trace():
    session, frames = _mk_session(
        BandwidthTrace([200e6] * 60), n_frames=40, buffer_capacity_s=1.0
    )
    for i in range(0, 30, 5):
        session.step_segment(frames[i:i + 5], i)
    required_wait = max(0.0, session.user.buffer.buffer_level_s
                        - (1.0 - 5.0 / 30.0))
    assert required_wait > 0.0
    measured_end = session.cumulative_time_s + required_wait / 2.0
    session.access_link.trace = FiniteTraceWindow(BandwidthTrace(
        [200e6, 200e6], name='short-window',
        sample_times_s=[0.0, measured_end],
    ))
    clock_before = session.cumulative_time_s
    buffer_before = session.user.buffer.buffer_level_s
    try:
        session.prepare_request(5, 30)
    except TraceWindowExhausted as exc:
        assert exc.trace_end_s == measured_end
    else:
        raise AssertionError('pacing silently crossed the measured trace end')
    assert session.cumulative_time_s == clock_before
    assert session.user.buffer.buffer_level_s == buffer_before


def test_atomic_segment_geometry_is_rejected_before_an_episode_starts():
    try:
        StreamingEnv(
            _synthetic_frames(30), segment_frames=9, target_fps=30.0,
            buffer_capacity_s=1.0, min_buffer_s=0.95,
        )
    except ValueError as exc:
        assert 'cannot reach' in str(exc)
    else:
        raise AssertionError('incompatible segment/buffer geometry was accepted')


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


def test_env_reward_sum_equals_qoe_for_partial_segments():
    """Equality holds independently of segment size, including a short tail."""
    frames = _synthetic_frames(37)
    for segment_frames in (1, 5, 8):
        random.seed(7)
        env = StreamingEnv(frames, segment_frames=segment_frames,
                           tcp_params=dict(rtt_ms=40.0, rtt_jitter_ms=0.0,
                                           loss_prob=0.0, cwnd_packets=10,
                                           mss_bytes=1460, log_packets=False))
        trace = BandwidthTrace([2e3, 2e3] + [80e6] * 100,
                               sample_times_s=list(range(102)))
        env.reset(trace)
        total_reward = 0.0
        stall_deltas = 0.0
        rebuffer_deltas = 0
        done = False
        step_index = 0
        while not done:
            _, reward, done, info = env.step(step_index % 2)
            total_reward += reward
            stall_deltas += info['stall_s']
            rebuffer_deltas += info['rebuffer_events']
            step_index += 1
        terms = env.qoe_terms()
        stats = env.user.get_buffer_stats()
        assert abs(total_reward - terms['total']) < 1e-9
        assert abs(total_reward - env.qoe()) < 1e-9
        assert env.qoe_quality() == env.qoe()
        assert abs(stall_deltas - stats['total_stall_time_s']) < 1e-9
        assert rebuffer_deltas == stats['rebuffer_count']


def test_env_reward_qoe_equality_with_request_pacing():
    frames = _synthetic_frames(90)
    env = StreamingEnv(
        frames, segment_frames=5, buffer_capacity_s=1.0,
        tcp_params=dict(rtt_ms=1.0, rtt_jitter_ms=0.0, loss_prob=0.0,
                        cwnd_packets=1000, mss_bytes=1460, log_packets=False),
    )
    env.reset(BandwidthTrace([200e6] * 200))
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, _ = env.step(1)
        total_reward += reward
    stats = env.user.get_buffer_stats()
    assert 'frames_dropped' not in stats
    assert env.session.total_request_pacing_s > 0.0
    assert abs(total_reward - env.qoe()) < 1e-9
    assert 'frame_drops' not in env.qoe_terms()


def test_reward_qoe_equality_with_fixed_tier_utility():
    frames = _synthetic_frames(13)
    env = StreamingEnv(
        frames, segment_frames=5,
        tcp_params=dict(rtt_ms=1.0, rtt_jitter_ms=0.0, loss_prob=0.0,
                        cwnd_packets=1000, mss_bytes=1460,
                        log_packets=False),
    )
    env.reset(BandwidthTrace([200e6] * 30))
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, _ = env.step(1)
        total_reward += reward
    assert abs(total_reward - env.qoe()) < 1e-9


def test_superseded_quality_utilities_are_rejected():
    for utility in ('log_density', 'linear_density', 'coded_psnr'):
        try:
            RewardFunction({'utility': utility})
        except ValueError as exc:
            assert 'unknown utility' in str(exc)
        else:
            raise AssertionError(f"superseded utility {utility!r} was accepted")


def test_startup_and_rebuffering_are_disjoint_and_stall_deltas_do_not_repeat():
    """Initial waiting is startup only; a later underflow is one rebuffer event."""
    buffer = ClientBuffer(min_buffer_s=0.1)
    frame_id = 0
    while not buffer.playback_started:
        buffer.add_frame(frame_id, 0, 1, 0.0, 0.01)
        frame_id += 1
    startup = buffer.get_statistics()['startup_delay_s']
    assert startup == 0.01
    assert buffer.total_stall_time == 0.0
    assert buffer.rebuffer_count == 0

    records = [buffer.add_frame(frame_id, 0, 1, 1.0, 1.01)]
    frame_id += 1
    arrival = 1.11
    while buffer.is_rebuffering:
        records.append(buffer.add_frame(frame_id, 0, 1, 0.1, arrival))
        frame_id += 1
        arrival += 0.1

    stats = buffer.get_statistics()
    stall_delta_sum = sum(r.get('stall_time_s', 0.0) for r in records)
    resumed = records[-1]
    assert resumed['event'] == 'playback_resumed'
    assert resumed['stall_time_s'] < resumed['rebuffer_duration_s']
    assert abs(stall_delta_sum - stats['total_stall_time_s']) < 1e-9
    assert stats['rebuffer_count'] == 1
    assert stats['startup_delay_s'] == startup


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")


if __name__ == '__main__':
    _run_all()
