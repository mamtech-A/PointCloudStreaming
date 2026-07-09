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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")


if __name__ == '__main__':
    _run_all()
