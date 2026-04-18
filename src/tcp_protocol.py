import time
import random

class TCPConnection:
    """Simple TCP connection simulator.

    Features:
    - three-way handshake simulation via `establish()`
    - `send(data)` returns a simulated acknowledgement and transfer time
    - basic RTT and loss simulation (configurable)
    - `close()` to tear down

    All timings are simulated and do not perform real network I/O.
    """

    def __init__(self, src='src', dst='dst', rtt_ms=50, rtt_jitter_ms=0, loss_prob=0.0, cwnd_packets=10, mss_bytes=1460, rto_formula='jacobson', rto_fixed_s=1.0):
        self.src = src
        self.dst = dst
        self.rtt_ms = rtt_ms
        self.rtt_jitter_ms = rtt_jitter_ms  # RTT jitter (±ms)
        self.loss_prob = loss_prob
        self.cwnd_packets = cwnd_packets
        self.mss = mss_bytes
        # RTO configuration
        # rto_formula: 'jacobson' (default) or 'fixed'
        self.rto_formula = (rto_formula or 'jacobson')
        self.rto_fixed_s = float(rto_fixed_s)
        # RTT estimator state (Jacobson/Karels)
        self.srtt = None
        self.rttvar = None
        # initialize rto depending on strategy
        if self.rto_formula == 'fixed':
            self.rto = max(0.1, float(self.rto_fixed_s))
        else:
            self.rto = max(0.2, (self.rtt_ms / 1000.0) * 2)
        self.established = False
        self.closed = False
        # Packet log: list of dicts with packet events
        self.packet_log = []
        self.handshake_time = 0.0  # track handshake end time
        self.conn_time = 0.0  # cumulative connection timeline
        self.first_data_send = True
        self.seq_num = 0  # sequence number (bytes)
        self.ack_num = 0  # acknowledgment number (bytes)

    def establish(self):
        if self.closed:
            raise RuntimeError('Connection already closed')
        # simulate three-way handshake delay
        rtt_s = self.rtt_ms / 1000.0
        
        # Initial Sequence Numbers (random in real TCP, we use 0 for simplicity)
        client_isn = 0
        server_isn = 0
        
        # Log handshake packets (SYN, SYN-ACK, ACK)
        # SYN: client -> server (t=0)
        self.packet_log.append({
            'time_s': 0.0,
            'round': -1,
            'packet_num': -1,
            'size_bytes': 40,  # TCP header only
            'event': 'SYN',
            'src': self.src,
            'dst': self.dst,
            'cwnd': 0,
            'seq_num': client_isn,
            'ack_num': 0,
            'rtt_ms': self.rtt_ms
        })
        # SYN-ACK: server -> client (t=RTT/2)
        self.packet_log.append({
            'time_s': rtt_s / 2,
            'round': -1,
            'packet_num': -1,
            'size_bytes': 40,
            'event': 'SYN-ACK',
            'src': self.dst,
            'dst': self.src,
            'cwnd': 0,
            'seq_num': server_isn,
            'ack_num': client_isn + 1,
            'rtt_ms': self.rtt_ms
        })
        # ACK: client -> server (t=RTT)
        self.packet_log.append({
            'time_s': rtt_s,
            'round': -1,
            'packet_num': -1,
            'size_bytes': 40,
            'event': 'ACK',
            'src': self.src,
            'dst': self.dst,
            'cwnd': 0,
            'seq_num': client_isn + 1,
            'ack_num': server_isn + 1,
            'rtt_ms': self.rtt_ms
        })
        
        # After handshake, seq starts at ISN+1
        self.seq_num = client_isn + 1
        self.ack_num = server_isn + 1
        
        self.handshake_time = rtt_s  # data transmission starts after handshake
        self.conn_time = rtt_s
        self.first_data_send = True
        self.established = True
        return {'event':'established', 'handshake_delay_s': rtt_s}

    def send(self, data_bytes, capacity_bps=None):
        """Simulate sending `data_bytes` bytes over the connection.

        Uses an ANALYTICAL model for speed (no per-packet loop).
        Returns a dict with simulated metrics: sent_bytes, time_s, rtt_ms, retransmissions, cwnd_start, cwnd_end, srtt_s, rto_s
        """
        if not self.established:
            raise RuntimeError('TCP connection not established')
        if self.closed:
            raise RuntimeError('Connection closed')

        if data_bytes <= 0:
            return {'sent_bytes': 0, 'time_s': 0.0, 'rtt_ms': self.rtt_ms, 'retransmissions': 0,
                    'cwnd_start': self.cwnd_packets, 'cwnd_end': self.cwnd_packets, 'srtt_s': self.srtt, 'rto_s': self.rto}

        mss = max(1, self.mss)
        total_packets = int((data_bytes + mss - 1) // mss)
        base_rtt_s = self.rtt_ms / 1000.0
        jitter_s = self.rtt_jitter_ms / 1000.0

        # Function to get RTT with jitter for this round
        def get_rtt_with_jitter():
            if jitter_s > 0:
                return base_rtt_s + random.uniform(-jitter_s, jitter_s)
            return base_rtt_s

        rtt_s = get_rtt_with_jitter()  # initial RTT sample

        # Initialize RTT estimator if needed
        if self.srtt is None:
            self.srtt = rtt_s
            self.rttvar = rtt_s / 2.0
            if self.rto_formula == 'fixed':
                self.rto = max(0.1, float(self.rto_fixed_s))
            else:
                self.rto = max(0.2, self.srtt + 4 * self.rttvar)

        cwnd_start = max(1, int(self.cwnd_packets))
        cwnd = float(cwnd_start)
        ssthresh = max(2, cwnd_start * 2)

        # --- RTT-round model with explicit retransmission rounds ---
        # Keep a pending set of packet indices and remove only ACKed packets.
        pending_packets = list(range(total_packets))
        retransmissions = 0
        rounds = 0
        max_rounds = 10000  # safety cap
        sim_time = self.conn_time
        send_start_time = 0.0 if self.first_data_send else self.conn_time

        while pending_packets and rounds < max_rounds:
            # Calculate RTT with jitter for this round
            round_rtt_s = max(0.001, base_rtt_s + random.uniform(-jitter_s, jitter_s))
            if capacity_bps and capacity_bps > 0:
                cap_packets_per_rtt = max(1, int((capacity_bps * round_rtt_s) / (mss * 8)))
            else:
                cap_packets_per_rtt = len(pending_packets)
            
            send_this_round = min(max(1, int(cwnd)), cap_packets_per_rtt, len(pending_packets))
            to_send = pending_packets[:send_this_round]
            remaining_queue = pending_packets[send_this_round:]
            lost_this_round = []
            success_count = 0
            
            # Log each packet in this round
            for pkt_num, pkt_idx in enumerate(to_send):
                pkt_seq = self.seq_num + (pkt_idx * mss)
                pkt_size = mss if pkt_idx < total_packets - 1 else (data_bytes - ((total_packets - 1) * mss))
                # Determine if this packet is lost (analytical: use loss_prob)
                is_lost = random.random() < self.loss_prob
                self.packet_log.append({
                    'time_s': sim_time,
                    'round': rounds,
                    'packet_num': pkt_num,
                    'size_bytes': pkt_size,
                    'event': 'LOST' if is_lost else 'SENT',
                    'src': self.src,
                    'dst': self.dst,
                    'cwnd': cwnd,
                    'seq_num': pkt_seq,
                    'ack_num': self.ack_num,
                    'rtt_ms': round_rtt_s * 1000
                })
                if is_lost:
                    lost_this_round.append(pkt_idx)
                    retransmissions += 1
                else:
                    success_count += 1
                    # Log ACK (cumulative: ack_num = seq + data_len)
                    self.packet_log.append({
                        'time_s': sim_time + round_rtt_s,
                        'round': rounds,
                        'packet_num': pkt_num,
                        'size_bytes': 40,  # ACK size
                        'event': 'ACK',
                        'src': self.dst,
                        'dst': self.src,
                        'cwnd': cwnd,
                        'seq_num': self.ack_num,
                        'ack_num': pkt_seq + pkt_size,
                        'rtt_ms': round_rtt_s * 1000
                    })
            
            pending_packets = lost_this_round + remaining_queue
            rounds += 1
            sim_time += round_rtt_s

            # Congestion response and growth.
            if lost_this_round:
                ssthresh = max(2, int(cwnd / 2))
                cwnd = float(ssthresh)
                if success_count == 0:
                    # Full-round loss approximates timeout behavior.
                    sim_time += self.rto
            else:
                if cwnd < ssthresh:
                    cwnd += success_count
                else:
                    cwnd += max(1.0 / max(cwnd, 1.0), success_count / max(cwnd, 1.0))

            if self.rto_formula != 'fixed':
                alpha = 1.0 / 8.0
                beta = 1.0 / 4.0
                self.rttvar = (1 - beta) * self.rttvar + beta * abs(self.srtt - round_rtt_s)
                self.srtt = (1 - alpha) * self.srtt + alpha * round_rtt_s
                self.rto = max(0.2, self.srtt + 4 * self.rttvar)

        # Duration for this send (includes initial handshake once per connection).
        time_s = sim_time - send_start_time
        self.conn_time = sim_time
        self.first_data_send = False
        self.seq_num += data_bytes

        cwnd_end = max(1, int(cwnd))
        self.cwnd_packets = cwnd_end

        return {
            'sent_bytes': data_bytes,
            'time_s': time_s,
            'rtt_ms': self.rtt_ms,
            'retransmissions': retransmissions,
            'cwnd_start': cwnd_start,
            'cwnd_end': cwnd_end,
            'srtt_s': self.srtt,
            'rto_s': self.rto
        }

    def close(self):
        self.closed = True
        self.established = False
        return {'event':'closed'}

    def get_packet_log(self):
        """Return the list of all packet events logged during this connection."""
        return self.packet_log
