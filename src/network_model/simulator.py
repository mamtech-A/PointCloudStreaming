"""Topology-driven Simulator.

Replaces the legacy monolithic per-client loop. Frame-outer / session-inner: for
each frame, step every active session (true concurrent multi-user; single-user is
the N=1 case). Logs carry a leading `user_id` column. Files are opened once and
closed after all sessions (fixing the legacy close-inside-loop bug). ABR reporting
is polymorphic via `session.abr.report(...)` — no per-algorithm branching.
"""

import os
import csv

from .manifest import parse_mpd_xml, PointCloud


_BUFFER_EMOJI = {'critical': '🔴', 'low': '🟡', 'normal': '🟢', 'high': '🔵'}


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Simulator:
    def __init__(self, topology, duration=10.0):
        self.topology = topology
        self.duration = duration

    def run(self, mpd_path, run_label=None, return_summary=False, max_frames=None,
            verbose=True):
        self._verbose = verbose
        print("--- DASH-PC Point Cloud Streaming Simulation ---")
        if not mpd_path or not mpd_path.endswith('.xml') or not os.path.exists(mpd_path):
            raise FileNotFoundError("mpd.xml not found or invalid path.")

        server = self.topology.server
        frames = parse_mpd_xml(mpd_path)
        if max_frames:
            frames = frames[:max_frames]
        server.manifest.frames = frames
        for frame in frames:
            for rep in frame['representations']:
                server.add_pointcloud(
                    frame['id'], rep['id'],
                    PointCloud(points=None, meta={'density': rep['density']})
                )
        total_frames = len(frames)

        sessions = self.topology.all_sessions()
        for session in sessions:
            session.start()

        # --- Output files (opened once; one row per (frame, session)) ---
        logs_dir = os.path.join(_project_root(), 'logs')
        if run_label:
            logs_dir = os.path.join(logs_dir, run_label)
        os.makedirs(logs_dir, exist_ok=True)

        csv_path = os.path.join(logs_dir, 'results.csv')
        packet_log_path = os.path.join(logs_dir, 'packets.log')
        buffer_log_path = os.path.join(logs_dir, 'buffer.log')

        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'user_id', 'frame_id', 'rep_id', 'density', 'size', 'bandwidth_mbps', 'send_time_s',
            'retransmissions', 'cwnd_start', 'cwnd_end', 'srtt_s', 'rto_s',
            'buffer_level_s', 'buffer_health', 'stall', 'stall_duration_s', 'cumulative_time_s'
        ])
        packet_log_file = open(packet_log_path, 'w', encoding='utf-8')
        packet_log_file.write('user_id,frame_id,time_s,round,packet_num,size_bytes,event,src,dst,cwnd,seq_num,ack_num,rtt_ms\n')
        buffer_log_file = open(buffer_log_path, 'w', encoding='utf-8')
        buffer_log_file.write('user_id,time_s,frame_id,status,buffer_level_s,buffer_health,stall_duration_s,event_type\n')

        for session in sessions:
            u = session.user
            print(f"\n[Session] {u.user_id} via {session.edge.edge_id} - Buffer: "
                  f"{u.buffer_capacity_s}s capacity, {u.min_buffer_s}s min, {u.target_fps} FPS target")
        print("=" * 100)

        # --- Frame-outer / session-inner main loop ---
        for frame_idx, frame in enumerate(frames):
            for session in sessions:
                record = session.step(frame, frame_idx)
                self._log_record(session, record, csv_writer, packet_log_file, buffer_log_file)

        csv_file.close()
        packet_log_file.close()
        buffer_log_file.close()

        # --- Per-session summary ---
        summaries = []
        for session in sessions:
            summaries.append(self._report_session(session, total_frames))

        print(f"\n📁 Output Files:")
        print(f"   Per-frame results: {csv_path}")
        print(f"   Packet log: {packet_log_path}")
        print(f"   Buffer log: {buffer_log_path}")

        self.topology.close_all()

        if return_summary:
            return summaries

    def _log_record(self, session, record, csv_writer, packet_log_file, buffer_log_file):
        u = session.user
        rep = record['rep']
        m = record['metrics']
        br = record['buffer_result']
        capacity = record['capacity_bps'] or 0
        frame_time = record['frame_time_s']
        cumulative = record['cumulative_time_s']

        bstats = u.get_buffer_stats()
        buffer_health = bstats['buffer_health']
        buffer_level = bstats['buffer_level_s']
        stall_time = br.get('stall_time_s', 0)
        is_rebuffering = br.get('is_rebuffering', False)
        is_playing = br.get('is_playing', False)
        event_type = br.get('event', 'buffered')
        status = br.get('status', 'buffered')

        if getattr(self, '_verbose', True):
            if event_type == 'playback_started':
                state = " ▶️ PLAYBACK STARTED"
            elif event_type == 'playback_resumed':
                state = f" ▶️ RESUMED (rebuffer={stall_time:.2f}s)"
            elif is_rebuffering:
                state = " ⏸️ REBUFFERING..."
            elif is_playing:
                state = " ▶️ PLAYING"
            else:
                state = " ⏳ INITIAL BUFFER"
            emoji = _BUFFER_EMOJI.get(buffer_health, '⚪')
            icon = "❌" if status == 'dropped' else "✅"
            print(f"[{u.user_id}] Frame {record['frame_id']:3d}: {icon} Rep {record['rep_id']} "
                  f"(density={rep['density']}, size={rep['size']:>5}) | DL={frame_time:.2f}s | "
                  f"BW={capacity/1e6:.1f}Mbps | Buffer: {emoji} {buffer_level:.2f}s ({buffer_health}){state}")

        buffer_log_file.write(
            f"{u.user_id},{cumulative:.6f},{record['frame_id']},{status},{buffer_level:.4f},"
            f"{buffer_health},{stall_time:.4f},{event_type}\n")

        csv_writer.writerow([
            u.user_id, record['frame_id'], record['rep_id'], rep['density'], rep['size'],
            f"{capacity/1e6:.3f}", f"{frame_time:.6f}",
            m.get('retransmissions', 0), m.get('cwnd_start', ''), m.get('cwnd_end', ''),
            m.get('srtt_s', ''), m.get('rto_s', ''),
            f"{buffer_level:.4f}", buffer_health,
            'yes' if is_rebuffering else 'no', f"{stall_time:.4f}", f"{cumulative:.6f}"
        ])

        for pkt in m.get('packet_log', []):
            packet_log_file.write(
                f"{u.user_id},{record['frame_id']},{pkt['time_s']:.6f},{pkt['round']},"
                f"{pkt['packet_num']},{pkt['size_bytes']},{pkt['event']},{pkt['src']},{pkt['dst']},"
                f"{pkt['cwnd']},{pkt.get('seq_num',0)},{pkt.get('ack_num',0)},{pkt.get('rtt_ms',0):.2f}\n")

    def _report_session(self, session, total_frames):
        u = session.user
        stats = u.get_buffer_stats()
        total_time = session.cumulative_time_s
        fps_real = total_frames / total_time if total_time > 0 else 0
        qoe = session.qoe()

        print(f"\n{'='*100}")
        print(f"--- Session Finished: {u.user_id} ---")
        print(f"\n📊 Playback Statistics:")
        print(f"   Total Frames: {total_frames}")
        print(f"   Total Download Time: {total_time:.2f}s")
        print(f"   Real Video FPS: {fps_real:.2f} frames per second")
        print(f"\n📦 Buffer Statistics:")
        print(f"   Final Buffer Level: {stats['buffer_level_s']:.2f}s ({stats['buffer_level_frames']} frames)")
        print(f"   Buffer Health: {stats['buffer_health']}")
        print(f"   Frames Played: {stats['frames_played']}")
        print(f"   Frames Dropped: {stats['frames_dropped']}")
        print(f"\n⚠️  Stall Statistics:")
        print(f"   Rebuffer Events: {stats['rebuffer_count']}")
        print(f"   Total Stall Time: {stats['total_stall_time_s']:.2f}s")
        if stats['rebuffer_count'] > 0:
            print(f"   Average Rebuffer Duration: {stats['total_stall_time_s']/stats['rebuffer_count']:.3f}s")
        print(f"\n🎯 QoE Score: {qoe:.1f}/100")

        pred = session.abr.report(session.observed_throughput_history)
        if pred:
            print(f"\n🤖 LSTM Prediction Statistics:")
            print(f"   Total Predictions: {pred['total_predictions']}")
            print(f"   Mean Predicted BW: {pred['mean_predicted_mbps']:.2f} Mbps")
            print(f"   Mean Actual BW: {pred['mean_actual_mbps']:.2f} Mbps")
            print(f"   Mean Absolute Error: {pred['mae_mbps']:.2f} Mbps")
            print(f"   Mean Absolute % Error: {pred['mape_percent']:.1f}%")

        qh = session.quality_history
        switches = sum(1 for i in range(1, len(qh)) if qh[i] != qh[i-1])
        return {
            'user_id': u.user_id,
            'abr': session.abr.name,
            'qoe': qoe,
            'total_time_s': total_time,
            'fps': fps_real,
            'frames_played': stats['frames_played'],
            'frames_dropped': stats['frames_dropped'],
            'rebuffer_count': stats['rebuffer_count'],
            'total_stall_time_s': stats['total_stall_time_s'],
            'mean_rep_id': (sum(qh) / len(qh)) if qh else 0,
            'quality_switches': switches,
            'prediction_stats': pred,
        }
