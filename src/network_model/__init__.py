"""network_model package: point-cloud streaming simulation + transport core.

Re-exports the public API so `from src.network_model import EdgeNode, Server, ...`
keeps working as a one-liner. Includes back-compat aliases/shims for the legacy
names (`PointCloudServer`, `PointCloudClient`, `EdgeNodeLSTM`).
"""

from .manifest import (
    parse_mpd_xml, size_to_bytes, size_to_bits,
    PointCloud, DASHPCManifest, Server, PointCloudServer,
)

# Single source of truth for the transport config (entry scripts copy this).
# RTT is DATASET-DERIVED: the Irish 5G Download traces' PINGAVG in 5G mode is
# median 72 ms (p10 66 / p90 83, not load-inflated) — see bandwidth_5g/README.md.
# Note 72 ms > the 33 ms frame budget at 30 fps, so sequential per-frame fetching
# still stalls structurally (kept as a finding, not tuned away).
DEFAULT_TCP_PARAMS = {
    'rtt_ms': 72.0,
    'rtt_jitter_ms': 8.0,
    'loss_prob': 0.0,
    'cwnd_packets': 10.0,
    'mss_bytes': 1460,
    'rto_formula': 'jacobson',
    'rto_fixed_s': 1.0,
}
from .trace import load_5g_trace, BandwidthTrace
from .finite_trace import FiniteTraceWindow, TraceWindowExhausted
from .tcp_protocol import TCPConnection
from .buffer import ClientBuffer, PointCloudClient
from .abr import (
    ABRState, ABRStrategy, BandwidthABR, BufferBasedABR, MPCABR, LSTMABR, DQNABR,
)
from .links import BackhaulLink, AccessLink
from .user import User
from .session import StreamingSession
from .edge_node import EdgeNode
from .topology import Topology
from .simulator import Simulator


def EdgeNodeLSTM(server, bandwidth_limit_bps=None, tcp_params=None,
                 lstm_model_path=None, use_prediction=True, edge_id="edge"):
    """Back-compat shim: returns an EdgeNode whose ABR policy is an LSTMABR.

    Reproduces the legacy EdgeNodeLSTM constructor's model-load + messages. The
    edge's abr_factory mints an LSTMABR (sharing the loaded predictor; predictions
    rebuild their window each call, so this is safe single-threaded).
    """
    predictor = None
    if lstm_model_path and use_prediction:
        import os
        if not os.path.exists(lstm_model_path):
            print(f"Warning: LSTM model not found at {lstm_model_path}, using actual bandwidth")
        else:
            try:
                from ..lstm_model import SimpleLSTM
                predictor = SimpleLSTM()
                predictor.load(lstm_model_path)
                print(f"✅ LSTM model loaded from {lstm_model_path}")
            except Exception as e:  # pragma: no cover
                print(f"Warning: Failed to load LSTM model: {e}")
                predictor = None

    enabled = use_prediction and predictor is not None

    def factory():
        return LSTMABR(predictor, use_prediction=enabled)

    return EdgeNode(edge_id=edge_id, server=server, tcp_params=tcp_params,
                    abr_factory=factory, bandwidth_limit_bps=bandwidth_limit_bps)


__all__ = [
    'DEFAULT_TCP_PARAMS',
    'parse_mpd_xml', 'size_to_bytes', 'size_to_bits', 'PointCloud', 'DASHPCManifest',
    'Server', 'PointCloudServer', 'load_5g_trace', 'BandwidthTrace',
    'FiniteTraceWindow', 'TraceWindowExhausted', 'TCPConnection',
    'ClientBuffer', 'PointCloudClient', 'ABRState', 'ABRStrategy', 'BandwidthABR',
    'BufferBasedABR', 'MPCABR', 'LSTMABR', 'DQNABR', 'BackhaulLink', 'AccessLink',
    'User', 'StreamingSession', 'EdgeNode',
    'Topology', 'Simulator', 'EdgeNodeLSTM',
]
