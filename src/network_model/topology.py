"""Topology: the root container + builder.

The single place that knows every node/link/session. Exposes the first-class
connect operations and iterates sessions for the Simulator. Supports many users
per edge and (cheaply) multiple edges per server; generic multi-server routing is
deliberately out of scope (PLAN.md §10.6).
"""

from .links import BackhaulLink


class Topology:
    def __init__(self, server):
        self.server = server
        self.edges = {}      # edge_id -> EdgeNode
        self.users = {}      # user_id -> User
        self.sessions = []   # in connection order

    def add_edge(self, edge):
        if edge.server is None:
            edge.server = self.server
        if edge.backhaul is None:
            link = BackhaulLink(self.server, edge)
            edge.attach_backhaul(link)
            self.server.register_backhaul(link)
        self.edges[edge.edge_id] = edge
        return edge

    def add_user(self, user, edge, trace=None, tcp_params=None):
        """Connect `user` to `edge` (policy comes from the edge); register + return the session."""
        session = edge.connect_user(user, trace=trace, tcp_params=tcp_params)
        self.users[user.user_id] = user
        self.sessions.append(session)
        return session

    def all_sessions(self):
        return list(self.sessions)

    def close_all(self):
        for edge in self.edges.values():
            edge.close_connections()
