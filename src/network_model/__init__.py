"""network_model package: point-cloud streaming simulation + transport core.

Re-exports the public API so `from src.network_model import EdgeNode, Server, ...`
keeps working as a one-liner. Includes back-compat aliases/shims for the legacy
names (`PointCloudServer`, `PointCloudClient`, `EdgeNodeLSTM`).
"""

from .manifest import (
    parse_mpd_xml, size_to_bytes, size_to_bits,
    PointCloud, DASHPCManifest, Server, PointCloudServer,
)
from .trace import load_bandwidth_trace, BandwidthTrace
from .tcp_protocol import TCPConnection
from .buffer import ClientBuffer, PointCloudClient
from .abr import ABRState, ABRStrategy, BandwidthABR, LSTMABR, DQNABR
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
    'parse_mpd_xml', 'size_to_bytes', 'size_to_bits', 'PointCloud', 'DASHPCManifest',
    'Server', 'PointCloudServer', 'load_bandwidth_trace', 'BandwidthTrace', 'TCPConnection',
    'ClientBuffer', 'PointCloudClient', 'ABRState', 'ABRStrategy', 'BandwidthABR', 'LSTMABR',
    'DQNABR', 'BackhaulLink', 'AccessLink', 'User', 'StreamingSession', 'EdgeNode',
    'Topology', 'Simulator', 'EdgeNodeLSTM',
]
