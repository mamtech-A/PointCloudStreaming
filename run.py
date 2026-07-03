import os
import sys

# Project root on sys.path so `from src.network_model import ...` resolves.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Make emoji-rich status prints safe under non-UTF-8 consoles (e.g. cp1256 on redirect).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model import Server, EdgeNode, User, Topology, Simulator, BandwidthABR
from src.network_model.trace import BandwidthTrace

# Default paths: real G-PCC coded manifest + an UNSEEN test-split trace.
# (Use config/mpd.xml + data/bandwidth/report_foot_0001.log for legacy parity runs.)
mpd_path = os.path.join(project_root, "config", "mpd_gpcc.xml")
bandwidth_log_path = os.path.join(project_root, "bandwidth", "report_foot_0006.log")

# TCP / buffer defaults
tcp_params = {
    'rtt_ms': 50.0,
    'rtt_jitter_ms': 10.0,
    'loss_prob': 0.0,
    'cwnd_packets': 10.0,
    'mss_bytes': 1460,
    'rto_formula': 'jacobson',
    'rto_fixed_s': 1.0,
}

target_fps = 30.0  # 300 frames @ 30 fps = 10 s clip; each frame = 1/30 s of playback
buffer_capacity_s = 5.0
min_buffer_s = 1.0

# Build the topology: Server -> EdgeNode (baseline ABR) -> one User.
# Frames are streamed as coded (G-PCC-modeled) bitstreams — see manifest.coded_size_bytes.
server = Server(base_url="http://localhost/")
topo = Topology(server)
edge = EdgeNode("edge-1", server=server, tcp_params=tcp_params,
                abr_factory=lambda: BandwidthABR())
topo.add_edge(edge)

user = User("User", target_fps=target_fps, buffer_capacity_s=buffer_capacity_s,
            min_buffer_s=min_buffer_s)
topo.add_user(user, edge, trace=BandwidthTrace.from_log(bandwidth_log_path))

sim = Simulator(topo)
sim.run(mpd_path=mpd_path)
