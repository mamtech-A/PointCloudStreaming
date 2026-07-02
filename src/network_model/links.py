"""Physical link objects.

`BackhaulLink` (Server <-> EdgeNode) is unconstrained today (`capacity_bps=None` =>
infinite, transfer time 0); it is the single seam where a server<->edge capacity
limit would later be added (PLAN.md §2.5). `AccessLink` (EdgeNode <-> User) is the
real bottleneck: it owns the persistent per-user TCPConnection and the per-user
bandwidth trace, and is where `TCPConnection.send()` is reused verbatim.
"""

from .tcp_protocol import TCPConnection


class BackhaulLink:
    """Server <-> EdgeNode link. Unconstrained by default."""

    def __init__(self, server, edge, capacity_bps=None, propagation_delay_s=0.0):
        self.server = server
        self.edge = edge
        self.capacity_bps = capacity_bps
        self.propagation_delay_s = propagation_delay_s

    def is_constrained(self):
        return self.capacity_bps is not None

    def transfer(self, data_bytes):
        """Return transfer metrics for moving `data_bytes` edge<-server.

        Unconstrained (today): time_s = propagation_delay_s (0.0). When a capacity
        is set later: analytical delay = propagation + data_bits / capacity_bps.
        """
        if not self.is_constrained():
            return {'time_s': self.propagation_delay_s, 'bytes': data_bytes}
        time_s = self.propagation_delay_s + (data_bytes * 8) / self.capacity_bps
        return {'time_s': time_s, 'bytes': data_bytes}


class AccessLink:
    """EdgeNode <-> User bottleneck link: owns the per-user TCPConnection + trace.

    Replicates the legacy `EdgeNode.serve_to_user` transport bookkeeping (lazy
    establish, persistent connection reused across frames, per-frame capacity from
    the trace, packet-log slicing per transfer).
    """

    def __init__(self, edge, user, tcp_params=None, trace=None):
        self.edge = edge
        self.user = user
        self.tcp_params = tcp_params or {}
        self.trace = trace
        self.tcp = None

    def establish(self):
        if self.tcp is None or self.tcp.closed:
            src = getattr(self.edge, 'edge_id', 'EdgeNode')
            self.tcp = TCPConnection(src=src, dst=self.user.user_id, **self.tcp_params)
            self.tcp.establish()
        return self.tcp

    def capacity_bps(self, frame_idx):
        """This link's capacity for `frame_idx` (per-user trace sample)."""
        if self.trace is None:
            return None
        return self.trace.capacity_bps(frame_idx)

    def transfer(self, data_bytes, frame_idx):
        """Send `data_bytes` over the link at the trace capacity; return TCP metrics
        with a `packet_log` slice for just this transfer."""
        if self.tcp is None or self.tcp.closed:
            self.establish()
            packet_log_start = 0
        else:
            packet_log_start = len(self.tcp.get_packet_log())
        capacity = self.capacity_bps(frame_idx)
        metrics = self.tcp.send(int(data_bytes) if data_bytes else 0, capacity_bps=capacity)
        metrics['packet_log'] = self.tcp.get_packet_log()[packet_log_start:]
        return metrics

    def close(self):
        if self.tcp and not self.tcp.closed:
            self.tcp.close()
