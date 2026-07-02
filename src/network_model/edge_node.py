"""EdgeNode: the CDN edge.

Connects to the Server via a BackhaulLink, serves many users, and is the factory
for StreamingSessions. It is configured with ONE ABR policy for all its users (via
`abr_factory`, which mints a fresh strategy instance per session so per-user state
stays isolated). No per-algorithm subclasses — the policy is composition, not type.
"""

from .links import BackhaulLink, AccessLink
from .session import StreamingSession
from .abr import BandwidthABR


class EdgeNode:
    def __init__(self, edge_id="edge", server=None, tcp_params=None,
                 abr_factory=None, bandwidth_limit_bps=None, bits_per_point=None):
        self.edge_id = edge_id
        self.server = server
        self.tcp_params = tcp_params or {}
        # One policy for the whole edge; called once per session to mint an instance.
        self.abr_factory = abr_factory or (lambda: BandwidthABR())
        self.bandwidth_limit = bandwidth_limit_bps
        # Codec rate for the streamed bitstream (None => package default).
        self.bits_per_point = bits_per_point
        self.backhaul = None
        self.sessions = {}  # user_id -> StreamingSession

    def attach_backhaul(self, link):
        self.backhaul = link

    def _ensure_backhaul(self):
        if self.backhaul is None and self.server is not None:
            self.backhaul = BackhaulLink(self.server, self)
            self.server.register_backhaul(self.backhaul)
        return self.backhaul

    def connect_user(self, user, trace=None, tcp_params=None, abr=None):
        """Connect a user to this edge; returns its StreamingSession.

        `abr` is normally left None so the user inherits the edge's policy
        (`abr_factory()`); each session gets its own instance for isolated state.
        """
        self._ensure_backhaul()
        link = AccessLink(self, user, tcp_params or self.tcp_params, trace)
        strategy = abr if abr is not None else self.abr_factory()
        session = StreamingSession(user, self, self.server, strategy, link, self.backhaul,
                                   bits_per_point=self.bits_per_point)
        self.sessions[user.user_id] = session
        user.session = session
        return session

    def fetch_from_server(self, frame_id, rep_id):
        """Pull a representation from the origin (across the backhaul)."""
        return self.server.serve_pointcloud(frame_id, rep_id)

    def close_connections(self):
        for session in self.sessions.values():
            session.access_link.close()
