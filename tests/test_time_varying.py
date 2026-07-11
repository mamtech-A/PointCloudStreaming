"""Unit sanity tests for the time-varying capacity model.

Runs standalone (`python tests/test_time_varying.py`) or under pytest.
Covers PLAN verification items 1-2: trace time axis (duplicate seconds, gaps,
clamping, slicing) and TCP serialization delay / callable capacity.
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network_model.trace import BandwidthTrace
from src.network_model.tcp_protocol import TCPConnection


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_trace_uniform_no_timestamps():
    """No timestamps -> uniform 1 s cadence."""
    tr = BandwidthTrace([1e6, 2e6, 3e6])
    assert tr.capacity_at_time(0.0) == 1e6
    assert tr.capacity_at_time(0.99) == 1e6
    assert tr.capacity_at_time(1.0) == 2e6
    assert tr.capacity_at_time(2.5) == 3e6
    assert tr.capacity_at_time(999.0) == 3e6      # clamp past end
    assert tr.capacity_at_time(-5.0) == 1e6       # clamp before start
    assert approx(tr.duration_s, 3.0)


def test_trace_duplicate_seconds_split_evenly():
    """A run of k rows sharing timestamp T splits [T, T_next) evenly."""
    # times: 100, 100, 102 -> two rows share t=0 (rebased), next distinct at 2 s.
    tr = BandwidthTrace([1e6, 2e6, 3e6], sample_times_s=[100, 100, 102])
    # starts: 0.0, 1.0, 2.0 (width (2-0)/2 = 1.0 each)
    assert tr.capacity_at_time(0.0) == 1e6
    assert tr.capacity_at_time(0.5) == 1e6
    assert tr.capacity_at_time(1.0) == 2e6
    assert tr.capacity_at_time(1.9) == 2e6
    assert tr.capacity_at_time(2.0) == 3e6
    # start times strictly increasing
    starts = [tr.start_time_of_sample(i) for i in range(3)]
    assert all(starts[i] < starts[i + 1] for i in range(2)), starts


def test_trace_gap_holds_previous_sample():
    """Gap (t=1 -> t=3): the t=1 sample HOLDS through the missing t=2 second."""
    tr = BandwidthTrace([5e6, 2e6, 3e6], sample_times_s=[200, 201, 203])
    assert tr.capacity_at_time(0.5) == 5e6
    assert tr.capacity_at_time(1.0) == 2e6
    assert tr.capacity_at_time(2.0) == 2e6   # inside the gap -> previous holds
    assert tr.capacity_at_time(2.99) == 2e6
    assert tr.capacity_at_time(3.0) == 3e6
    assert approx(tr.duration_s, 4.0)        # last sample: default 1 s width


def test_trace_slice_from_rebases_time():
    tr = BandwidthTrace([1e6, 2e6, 3e6, 4e6], sample_times_s=[10, 11, 13, 14])
    sl = tr.slice_from(1)
    assert sl.samples == [2e6, 3e6, 4e6]
    assert sl.capacity_at_time(0.0) == 2e6
    assert sl.capacity_at_time(1.5) == 2e6   # 2 s gap (11->13) preserved: holds
    assert sl.capacity_at_time(2.0) == 3e6
    assert sl.capacity_at_time(3.0) == 4e6


def test_trace_legacy_positional_lookup_unchanged():
    tr = BandwidthTrace([1e6, 2e6], sample_times_s=[0, 1])
    assert tr.capacity_bps(0) == 1e6
    assert tr.capacity_bps(1) == 2e6
    assert tr.capacity_bps(99) == 2e6
    assert len(tr) == 2


def _mk_conn(**kw):
    params = dict(rtt_ms=72.0, rtt_jitter_ms=0.0, loss_prob=0.0,
                  cwnd_packets=10, mss_bytes=1460, log_packets=False)
    params.update(kw)
    c = TCPConnection(**params)
    c.establish()
    return c


def test_tcp_serialization_dominates_in_deep_fade():
    """1460 B at 1 kbps: serialization ~11.68 s (legacy model gave ~1 RTT)."""
    random.seed(0)
    c = _mk_conn()
    m = c.send(1460, capacity_bps=1000.0)
    # first send includes the handshake RTT (0.072) + serialization round
    expected = 0.072 + (1460 * 8) / 1000.0
    assert abs(m['time_s'] - expected) < 1e-6, m['time_s']


def test_tcp_fast_link_stays_rtt_bound():
    """High capacity: serialization < RTT, so timing matches the legacy model."""
    random.seed(0)
    c = _mk_conn()
    m = c.send(3 * 1460, capacity_bps=100e6)  # 3 packets, one cwnd-10 round
    expected = 0.072 + 0.072                  # handshake + 1 RTT round
    assert abs(m['time_s'] - expected) < 1e-6, m['time_s']


def test_tcp_callable_capacity_traverses_trace():
    """Capacity provider is re-queried per round with elapsed time: a transfer
    that starts in a fade finishes fast once the trace recovers."""
    random.seed(0)
    # 10 kbps for the first 2 s of the transfer, then 100 Mbps.
    fade_then_fast = lambda t_rel: 10e3 if t_rel < 2.0 else 100e6
    c = _mk_conn()
    m = c.send(20 * 1460, capacity_bps=fade_then_fast)
    # Round 1 (starts at handshake end, t_rel=0.072 < 2.0): fade -> 1 pkt,
    # serialization 1460*8/10e3 = 1.168 s -> t_rel jumps past 2.0.
    # Round 2+: fast link -> RTT-bound rounds drain the rest quickly.
    assert m['time_s'] < 3.0, m['time_s']
    # Same transfer under constant fade would take ~ 20 * 1.168 s.
    c2 = _mk_conn()
    m2 = c2.send(20 * 1460, capacity_bps=10e3)
    assert m2['time_s'] > 20.0, m2['time_s']


def test_tcp_scalar_none_capacity_unconstrained():
    random.seed(0)
    c = _mk_conn()
    m = c.send(10 * 1460, capacity_bps=None)  # unconstrained: one cwnd round
    assert abs(m['time_s'] - (0.072 + 0.072)) < 1e-6, m['time_s']


# ---------------------------------------------------------------------------
# Round 2: segment-based fetching + adaptive playback rate
# ---------------------------------------------------------------------------

from src.network_model.buffer import ClientBuffer


def _mk_session(trace, segment_frames_hint=None, playback_rate_min=1.0, n_frames=20):
    """Tiny session over synthetic 6-tier frames (1460 B per rep for exact math)."""
    from src.network_model import Server, EdgeNode, User, Topology
    from src.network_model.manifest import PointCloud
    frames = [{'id': i, 'representations': [
        {'id': r, 'density': 1000 * (6 - r), 'size': '1460', 'coded_bytes': 1460,
         'bandwidth': 1460 * 8 * 30}
        for r in range(6)]} for i in range(n_frames)]
    server = Server("t://origin")
    server.manifest.frames = frames
    for fr in frames:
        for rep in fr['representations']:
            server.add_pointcloud(fr['id'], rep['id'], PointCloud(points=None))
    topo = Topology(server)
    edge = EdgeNode("e", server=server,
                    tcp_params=dict(rtt_ms=72.0, rtt_jitter_ms=0.0, loss_prob=0.0,
                                    cwnd_packets=10, mss_bytes=1460, log_packets=False))
    topo.add_edge(edge)
    user = User("u", playback_rate_min=playback_rate_min)
    session = topo.add_user(user, edge, trace=trace)
    session.start()
    return session, frames


def test_segment_amortizes_rtt():
    """10 one-packet frames: S=1 pays ~10 RTTs, S=10 pays ~1 RTT + cwnd rounds."""
    random.seed(0)
    tr = BandwidthTrace([100e6] * 200)  # fast flat link: serialization negligible
    s1, frames = _mk_session(tr)
    for i in range(10):
        s1.step(frames[i], i)
    t_perframe = s1.cumulative_time_s          # handshake + 10 rounds ~ 11 * 0.072

    random.seed(0)
    s2, frames2 = _mk_session(tr)
    s2.step_segment(frames2[:10], 0)
    t_segment = s2.cumulative_time_s           # handshake + 1 cwnd-10 round ~ 2 * 0.072
    assert abs(t_perframe - 11 * 0.072) < 1e-6, t_perframe
    assert abs(t_segment - 2 * 0.072) < 1e-6, t_segment
    assert t_segment < t_perframe / 5


def test_segment_batch_buffer_accounting():
    """S frames land at one arrival time; buffer level and stall math stay exact."""
    random.seed(0)
    tr = BandwidthTrace([100e6] * 200)
    s, frames = _mk_session(tr)
    rec = s.step_segment(frames[:10], 0)
    assert len(rec['frames']) == 10
    # 10 frames * 1/30 s buffered, none consumed (playback not started: 10/30 < 1 s min)
    assert abs(s.user.buffer.buffer_level_s - 10 / 30.0) < 1e-9
    assert s.user.buffer.playback_started is False
    # throughput history: ONE sample per segment
    assert len(s.observed_throughput_history) == 1


def test_step_is_one_frame_segment():
    """step() must be exactly step_segment([frame]) — same record content."""
    random.seed(0)
    tr = BandwidthTrace([50e6] * 200)
    s1, frames = _mk_session(tr)
    r1 = s1.step(frames[0], 0)
    random.seed(0)
    s2, frames2 = _mk_session(tr)
    r2 = s2.step_segment([frames2[0]], 0)
    assert r1['frame_time_s'] == r2['segment_time_s']
    assert r1['rep_id'] == r2['rep_id']
    assert r1['buffer_result'] == r2['frames'][0]['buffer_result']


def test_amp_rate_floors_and_tracks_slowdown():
    b = ClientBuffer(target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0,
                     playback_rate_min=0.9)
    # healthy buffer -> full rate
    b.buffer_level_s = 2.0
    assert b._playback_rate() == 1.0
    # low buffer -> floored at 0.9 (0.3/1.0 = 0.3 would be below the floor)
    b.buffer_level_s = 0.3
    assert abs(b._playback_rate() - 0.9) < 1e-12
    # near-full min buffer -> linear ramp region
    b.buffer_level_s = 0.95
    assert abs(b._playback_rate() - 0.95) < 1e-12
    # consumption at the floor: 1 wall second drains 0.9 content-seconds
    b.buffer_level_s = 0.5
    b.is_playing = True
    b.frames_in_buffer = [{'frame_id': i} for i in range(15)]
    played_wall = b._consume_buffer(0.2)
    assert abs(played_wall - 0.2) < 1e-12
    assert abs(b.buffer_level_s - (0.5 - 0.2 * 0.9)) < 1e-12
    assert abs(b.slowdown_time_s - 0.2) < 1e-12
    assert abs(b.slowdown_integral - 0.1 * 0.2) < 1e-12


def test_amp_off_is_legacy_exact():
    """playback_rate_min=1.0 must reproduce the legacy consumption exactly."""
    for level, elapsed in [(2.0, 0.5), (0.4, 0.5), (0.0, 0.3)]:
        legacy = ClientBuffer(playback_rate_min=1.0)
        legacy.buffer_level_s = level
        legacy.is_playing = True
        legacy.frames_in_buffer = [{'frame_id': i} for i in range(int(level * 30))]
        got = legacy._consume_buffer(elapsed)
        expect = elapsed if level >= elapsed else level  # old semantics
        assert abs(got - expect) < 1e-12, (level, elapsed, got)
        assert legacy.slowdown_time_s == 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")


if __name__ == '__main__':
    _run_all()
