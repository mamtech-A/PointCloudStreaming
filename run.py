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

from src.network_model import (
    Server, EdgeNode, User, Topology, Simulator, BandwidthABR, DEFAULT_TCP_PARAMS,
)
from src.network_model.trace import BandwidthTrace

# Default paths: real G-PCC coded manifest + an UNSEEN test-split 5G trace.
mpd_path = os.path.join(project_root, "config", "mpd_gpcc_longdress.xml")
bandwidth_log_path = os.path.join(project_root, "bandwidth_5g",
                                  "static_B_2020.01.16_10.43.34.csv")

# TCP / buffer defaults (RTT is dataset-derived — see DEFAULT_TCP_PARAMS)
tcp_params = dict(DEFAULT_TCP_PARAMS)

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
topo.add_user(user, edge, trace=BandwidthTrace.from_file(bandwidth_log_path))

sim = Simulator(topo)
sim.run(mpd_path=mpd_path)
