"""First-class client (User): identity + playback buffer.

Replaces the role of the legacy PointCloudClient. The User owns a ClientBuffer and
its QoS config; it does not talk to the network — its StreamingSession drives it.
"""

from .buffer import ClientBuffer


class User:
    def __init__(self, user_id="User", target_fps=30.0, buffer_capacity_s=5.0,
                 min_buffer_s=1.0):
        self.user_id = user_id
        self.target_fps = target_fps
        self.buffer_capacity_s = buffer_capacity_s
        self.min_buffer_s = min_buffer_s
        self.buffer = ClientBuffer(
            target_fps=target_fps,
            buffer_capacity_s=buffer_capacity_s,
            min_buffer_s=min_buffer_s,
        )
        self.session = None  # back-ref, set on connect

    def receive_frame(self, frame_id, rep_id, size_bytes, download_time_s, arrival_time_s):
        return self.buffer.add_frame(frame_id, rep_id, size_bytes, download_time_s, arrival_time_s)

    def get_buffer_stats(self):
        return self.buffer.get_statistics()

    def get_stall_info(self):
        return self.buffer.get_stall_info()

    @property
    def buffer_level_s(self):
        return self.buffer.buffer_level_s
