"""Topology-driven Simulator.

Replaces the legacy monolithic per-client loop. Frame-outer / session-inner: for
each frame, step every active session (true concurrent multi-user; single-user is
the N=1 case). Logs carry a leading `user_id` column. Files are opened once and
closed after all sessions (fixing the legacy close-inside-loop bug). ABR reporting
is polymorphic via `session.abr.report(...)` — no per-algorithm branching.
"""

import os
import csv

from .manifest import (parse_mpd_xml, PointCloud, density_quality,
                       manifest_quality_endpoints)


_BUFFER_EMOJI = {'critical': '🔴', 'low': '🟡', 'normal': '🟢', 'high': '🔵'}


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Simulator:
    def __init__(self, topology, duration=10.0):
        self.topology = topology
        self.duration = duration

    def run(self, mpd_path, run_label=None, return_summary=False, max_frames=None,
            verbose=True, segment_frames=1, reward_fn=None):
        """`segment_frames` = frames fetched per request (DASH-style segment).
        1 (default) = legacy per-frame fetching; S>1 amortizes the per-request
        RTT over S frames (ONE ABR decision + ONE TCP transfer per segment).

        `reward_fn`: optional rl.reward.RewardFunction — when given, the run
        accumulates the RL training objective per segment (same numbers the
        sweep reports) and, for ABRs that expose `last_decision` (DQNABR with
        log_decisions=True), logs each decision (state, Q-values, reward) to
        the console and to <logs>/decisions.csv."""
        self._verbose = verbose
        # Per-session RL-reward bookkeeping (active only when reward_fn given).
        self._reward_fn = reward_fn
        self._reward_total = {}
        self._reward_prev_qmean = {}
        self._prev_startup = {}
        self._prev_dropped = {}
        self._decision_rows = []
        if verbose:
            print("--- DASH-PC Point Cloud Streaming Simulation ---")
        if not mpd_path or not mpd_path.endswith('.xml') or not os.path.exists(mpd_path):
            raise FileNotFoundError("mpd.xml not found or invalid path.")

        server = self.topology.server
        frames = parse_mpd_xml(mpd_path)
        if max_frames:
            frames = frames[:max_frames]
        server.manifest.frames = frames
        # Human-readable log helpers: manifest quality endpoints (lowest rep -> ~0,
        # highest -> ~1) and a rep_id -> tier-name map, computed once per run.
        self._q_lo, self._q_hi = manifest_quality_endpoints(frames)
        self._tier_name = {r['id']: r.get('quality', f"rep{r['id']}")
                           for r in frames[0]['representations']} if frames else {}
        if self._reward_fn is not None:
            self._reward_fn.set_endpoints(self._q_lo, self._q_hi)
        for frame in frames:
            for rep in frame['representations']:
                server.add_pointcloud(
                    frame['id'], rep['id'],
                    PointCloud(points=None, meta={'density': rep['density']})
                )
        total_frames = len(frames)
        self._total_frames = total_frames

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
        # One row per FRAME (segment_frames > 1: `send_time_s` carries the full
        # segment transfer time on the segment's FIRST frame, 0.0 on the rest;
        # `segment_id` groups the rows).
        csv_writer.writerow([
            'user_id', 'frame_id', 'rep_id', 'density', 'size', 'bandwidth_mbps', 'send_time_s',
            'retransmissions', 'cwnd_start', 'cwnd_end', 'srtt_s', 'rto_s',
            'buffer_level_s', 'buffer_health', 'stall', 'stall_duration_s', 'cumulative_time_s',
            'segment_id'
        ])
        packet_log_file = open(packet_log_path, 'w', encoding='utf-8')
        packet_log_file.write('user_id,frame_id,time_s,round,packet_num,size_bytes,event,src,dst,cwnd,seq_num,ack_num,rtt_ms\n')
        buffer_log_file = open(buffer_log_path, 'w', encoding='utf-8')
        buffer_log_file.write('user_id,time_s,frame_id,status,buffer_level_s,buffer_health,stall_duration_s,event_type\n')

        if verbose:
            for session in sessions:
                u = session.user
                print(f"\n[Session] {u.user_id} via {session.edge.edge_id} - Buffer: "
                      f"{u.buffer_capacity_s}s capacity, {u.min_buffer_s}s min, {u.target_fps} FPS target")
            print("=" * 100)

        # --- Segment-outer / session-inner main loop (S=1 == per-frame) ---
        seg = max(1, int(segment_frames or 1))
        for seg_idx in range(0, len(frames), seg):
            segment = frames[seg_idx:seg_idx + seg]
            for session in sessions:
                record = session.step_segment(segment, seg_idx)
                self._log_segment(session, seg_idx // seg, record,
                                  csv_writer, packet_log_file, buffer_log_file)

        csv_file.close()
        packet_log_file.close()
        buffer_log_file.close()

        # Per-decision CSV (one row per DQN decision) when reward_fn was given.
        decisions_path = None
        if self._decision_rows:
            decisions_path = os.path.join(logs_dir, 'decisions.csv')
            with open(decisions_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(self._decision_rows[0].keys()))
                w.writeheader()
                w.writerows(self._decision_rows)

        # --- Per-session summary ---
        summaries = []
        for session in sessions:
            summaries.append(self._report_session(session, total_frames))

        if verbose:
            print(f"\n📁 Output Files:")
            print(f"   Per-frame results: {csv_path}")
            print(f"   Packet log: {packet_log_path}")
            print(f"   Buffer log: {buffer_log_path}")
            if decisions_path:
                print(f"   DQN decisions: {decisions_path}")

        self.topology.close_all()

        if return_summary:
            return summaries

    def _log_segment(self, session, segment_id, record, csv_writer, packet_log_file,
                     buffer_log_file):
        """Log one segment: ONE console line + per-FRAME csv/buffer rows.
        `send_time_s` carries the segment transfer time on the first frame only."""
        u = session.user
        m = record['metrics']
        capacity = record['capacity_bps'] or 0
        seg_time = record['segment_time_s']
        cumulative = record['cumulative_time_s']
        frames = record['frames']

        # Per-frame rows (buffer state evolves within the batch as frames land).
        seg_stall = 0.0
        for i, fr in enumerate(frames):
            rep = fr['rep'] or {}
            br = fr['buffer_result']
            stall_time = br.get('stall_time_s', 0) or 0
            seg_stall += stall_time
            status = br.get('status', 'buffered')
            event_type = br.get('event', 'buffered')
            buffer_level = br.get('buffer_level_s', u.buffer.buffer_level_s)
            buffer_health = u.buffer.get_buffer_health()

            buffer_log_file.write(
                f"{u.user_id},{cumulative:.6f},{fr['frame_id']},{status},{buffer_level:.4f},"
                f"{buffer_health},{stall_time:.4f},{event_type}\n")

            csv_writer.writerow([
                u.user_id, fr['frame_id'], record['rep_id'], rep.get('density', ''),
                rep.get('size', ''),
                f"{capacity/1e6:.3f}", f"{seg_time:.6f}" if i == 0 else "0.000000",
                m.get('retransmissions', 0), m.get('cwnd_start', ''), m.get('cwnd_end', ''),
                m.get('srtt_s', ''), m.get('rto_s', ''),
                f"{buffer_level:.4f}", buffer_health,
                'yes' if br.get('is_rebuffering', False) else 'no',
                f"{stall_time:.4f}", f"{cumulative:.6f}", segment_id
            ])

        for pkt in m.get('packet_log', []):
            packet_log_file.write(
                f"{u.user_id},{record['first_frame_id']},{pkt['time_s']:.6f},{pkt['round']},"
                f"{pkt['packet_num']},{pkt['size_bytes']},{pkt['event']},{pkt['src']},{pkt['dst']},"
                f"{pkt['cwnd']},{pkt.get('seq_num',0)},{pkt.get('ack_num',0)},{pkt.get('rtt_ms',0):.2f}\n")

        # --- RL reward accumulation + per-decision log (reward_fn runs only) ---
        seg_reward = None
        decision = getattr(session.abr, 'last_decision', None)
        if decision is not None:
            session.abr.last_decision = None  # consume (never reuse a stale one)
        if self._reward_fn is not None:
            rf = self._reward_fn
            uid = u.user_id
            qs = [rf.quality(f['rep']) if f['rep'] else 0.0 for f in frames]
            q_sum, q_mean = sum(qs), sum(qs) / len(qs)
            new_event = any(f['buffer_result'].get('event') == 'rebuffering_start'
                            for f in frames)
            bstats0 = u.get_buffer_stats()
            startup_s = max(0.0, bstats0.get('startup_delay_s', 0.0)
                            - self._prev_startup.get(uid, 0.0))
            dropped_d = max(0, bstats0.get('frames_dropped', 0)
                            - self._prev_dropped.get(uid, 0))
            self._prev_startup[uid] = bstats0.get('startup_delay_s', 0.0)
            self._prev_dropped[uid] = bstats0.get('frames_dropped', 0)
            seg_reward = rf.step_segment(q_sum, q_mean,
                                         self._reward_prev_qmean.get(uid), seg_stall,
                                         new_event, startup_s=startup_s,
                                         dropped=dropped_d)
            self._reward_prev_qmean[uid] = q_mean
            self._reward_total[uid] = self._reward_total.get(uid, 0.0) + seg_reward
            if decision:
                self._decision_rows.append({
                    'user_id': uid, 'segment_id': segment_id,
                    'first_frame_id': frames[0]['frame_id'],
                    'buffer_s': round(decision['buffer_s'], 3),
                    'tput_last_mbps': round(decision['tput_last_mbps'], 2),
                    'tput_mean_mbps': round(decision['tput_mean_mbps'], 2),
                    'lstm_pred_mbps': (round(decision['lstm_pred_mbps'], 2)
                                       if decision['lstm_pred_mbps'] is not None else ''),
                    **{f'q{i}': round(v, 3) for i, v in enumerate(decision['q_values'])},
                    'action': decision['action'],
                    'chosen_tier': self._tier_name.get(record['rep_id'], record['rep_id']),
                    'greedy': decision['greedy'],
                    'reward': round(seg_reward, 4),
                })

        if not getattr(self, '_verbose', True):
            return
        # ONE readable sentence per segment (narrative log). The bandwidth shown is
        # ACHIEVED throughput (bytes/time) — the number that matters — NOT the
        # trace's capacity-at-start (that stays in results.csv).
        bstats = u.get_buffer_stats()
        buf_level = bstats['buffer_level_s']
        emoji = _BUFFER_EMOJI.get(bstats['buffer_health'], '⚪')
        data_bytes = record['data_bytes']
        thr_mbps = (data_bytes * 8 / seg_time / 1e6) if seg_time > 0 else 0.0
        rep0 = frames[0]['rep'] or {}
        tier = self._tier_name.get(record['rep_id'], f"rep{record['rep_id']}").upper()
        q = density_quality(rep0.get('density'), self._q_lo, self._q_hi)
        first_id, last_id = frames[0]['frame_id'], frames[-1]['frame_id']

        events = [f['buffer_result'].get('event', 'buffered') for f in frames]
        last_br = frames[-1]['buffer_result']
        dropped = sum(1 for f in frames if f['buffer_result'].get('status') == 'dropped')
        is_playing = last_br.get('is_playing', False)
        rate = u.buffer._playback_rate()
        # Slowdown is only meaningful while actually playing back.
        rate_note = f" ⏩{rate:.2f}x" if (is_playing and rate < 1.0) else ""
        if 'playback_started' in events:
            state = "▶ PLAYBACK START"
        elif 'playback_resumed' in events:
            state = f"▶ RESUMED (stalled {seg_stall:.1f}s)"
        elif last_br.get('is_rebuffering', False):
            state = f"⏸ REBUFFERING ({seg_stall:.1f}s)"
        elif is_playing:
            state = "▶ playing"
        else:
            state = "⏳ buffering…"
        if dropped:
            state += f"  ❌{dropped} dropped (buffer full)"

        played = bstats['frames_played']
        total = getattr(self, '_total_frames', 0)
        prog = f"{played}/{total}" if total else f"{played}"
        print(f"[{u.user_id}] [t={cumulative:5.1f}s] seg {segment_id:<3d} f{first_id:03d}-{last_id:03d}: "
              f"{tier:<7} q{q:.2f} — {data_bytes/1e6:.1f}MB in {seg_time:.2f}s @{thr_mbps:.0f}Mbps — "
              f"buf {emoji}{buf_level:.2f}s · played {prog} · {state}{rate_note}")

        # Per-decision model log: what the DQN saw and thought for this segment.
        if decision:
            qv = decision['q_values']
            best2 = sorted(qv, reverse=True)[:2]
            margin = best2[0] - best2[1] if len(best2) > 1 else 0.0
            qv_str = " ".join(
                (f"[{v:.1f}]" if i == decision['action'] else f"{v:.1f}")
                for i, v in enumerate(qv))
            seen = (f"buf {decision['buffer_s']:.1f}s · tput {decision['tput_last_mbps']:.0f}"
                    f"/{decision['tput_mean_mbps']:.0f}Mbps")
            if decision['lstm_pred_mbps'] is not None:
                seen += f" · lstm {decision['lstm_pred_mbps']:.0f}Mbps"
            r_str = f" · r {seg_reward:+.2f}" if seg_reward is not None else ""
            greedy = "" if decision['greedy'] else " (explored!)"
            print(f"    🤖 saw {seen} → Q {qv_str} → {tier}{greedy} "
                  f"(margin {margin:+.2f}){r_str}")

    def _report_session(self, session, total_frames):
        u = session.user
        stats = u.get_buffer_stats()
        total_time = session.cumulative_time_s
        fps_real = total_frames / total_time if total_time > 0 else 0
        qoe = session.qoe()
        qterms = session.qoe_quality_terms()
        qoe_q = qterms['total']
        mean_q = session.mean_quality()
        rl_reward = self._reward_total.get(u.user_id) if getattr(self, '_reward_total', None) else None

        # Tier mix (share of frames streamed at each quality tier) + achieved
        # throughput distribution — the "what did the user actually get" view.
        qh = session.quality_history
        switches = sum(1 for i in range(1, len(qh)) if qh[i] != qh[i-1])
        tier_counts = {}
        for rid in qh:
            name = getattr(self, '_tier_name', {}).get(rid, f"rep{rid}")
            tier_counts[name] = tier_counts.get(name, 0) + 1
        order = list(getattr(self, '_tier_name', {}).values())
        tier_mix = " · ".join(
            f"{name.upper()} {100*tier_counts[name]/len(qh):.0f}%"
            for name in order if tier_counts.get(name))
        thr = [t / 1e6 for t in session.observed_throughput_history if t > 0]
        pred = session.abr.report(session.observed_throughput_history)

        if getattr(self, '_verbose', True):
            print(f"\n{'='*100}")
            print(f"📊 Session Summary — {u.user_id}")
            print(f"{'-'*100}")
            print(f"   QoE (quality-aware) : {max(0.0, qoe_q):5.1f} / 100   "
                  f"(raw {qoe_q:.1f})   ← headline")
            print(f"     breakdown         : quality {qterms['quality']:+.1f} · "
                  f"stall {qterms['stall']:+.1f} · switch {qterms['switch']:+.1f} · "
                  f"slowdown {qterms['slowdown']:+.1f} · startup {qterms['startup']:+.1f} · "
                  f"drops {qterms['drops']:+.1f}")
            print(f"   QoE (legacy stall)  : {qoe:5.1f} / 100")
            if rl_reward is not None:
                print(f"   Reward (RL objective): {rl_reward:.1f}   "
                      f"(comparable to dqn_sweep_results.json)")
            print(f"   Quality             : mean {mean_q:.3f}" + (f"  ·  tiers {tier_mix}" if tier_mix else ""))
            print(f"   Playback            : {stats['frames_played']}/{total_frames} frames · "
                  f"{fps_real:.1f} real fps · {total_time:.1f}s to stream "
                  f"{total_frames/30.0:.0f}s of video")
            if stats['frames_dropped']:
                print(f"   Frames dropped      : {stats['frames_dropped']} (buffer overflow — eager fetch)")
            avg = (f" · avg {stats['total_stall_time_s']/stats['rebuffer_count']:.2f}s"
                   if stats['rebuffer_count'] else "")
            print(f"   Rebuffering         : {stats['rebuffer_count']} event(s) · "
                  f"{stats['total_stall_time_s']:.2f}s total stall{avg}")
            if stats.get('playback_rate_min', 1.0) < 1.0:
                print(f"   Adaptive playback   : {stats['slowdown_time_s']:.2f}s slowed "
                      f"(floor {stats['playback_rate_min']:.2f}x)")
            if thr:
                print(f"   Throughput (achieved): mean {sum(thr)/len(thr):.0f} · "
                      f"min {min(thr):.0f} · max {max(thr):.0f} Mbps")
            print(f"   Quality switches    : {switches}")
            if pred:
                print(f"   LSTM predictor      : MAE {pred['mae_mbps']:.1f} Mbps "
                      f"(pred {pred['mean_predicted_mbps']:.0f} vs actual "
                      f"{pred['mean_actual_mbps']:.0f} Mbps, {pred['total_predictions']} preds)")
            print(f"{'='*100}")
        return {
            'user_id': u.user_id,
            'abr': session.abr.name,
            'qoe': qoe,
            'qoe_quality': qoe_q,
            'qoe_quality_clipped': max(0.0, qoe_q),
            'qoe_terms': qterms,
            'rl_reward': rl_reward,
            'mean_quality': mean_q,
            'total_time_s': total_time,
            'fps': fps_real,
            'frames_played': stats['frames_played'],
            'frames_dropped': stats['frames_dropped'],
            'rebuffer_count': stats['rebuffer_count'],
            'total_stall_time_s': stats['total_stall_time_s'],
            'slowdown_time_s': stats.get('slowdown_time_s', 0.0),
            'mean_rep_id': (sum(qh) / len(qh)) if qh else 0,
            'quality_switches': switches,
            'tier_mix': tier_mix,
            'prediction_stats': pred,
        }
