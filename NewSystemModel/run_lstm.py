import os
import sys

# Assume this notebook lives at the project root
project_root = os.getcwd()
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from network_model import (
    PointCloudServer, EdgeNodeLSTM, PointCloudClient, Simulator
)

# Default paths
mpd_path = os.path.join(project_root, "config", "mpd.xml")
bandwidth_log_path = os.path.join(project_root, "data", "bandwidth", "report_foot_0001.log")
lstm_model_path = os.path.join(project_root, "models", "bandwidth_lstm.pkl")

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

target_fps = 30.0
buffer_capacity_s = 5.0
min_buffer_s = 1.0

# Build components with LSTM-enhanced EdgeNode
server = PointCloudServer(base_url="http://localhost/")

# Use EdgeNodeLSTM with LSTM prediction enabled
edge = EdgeNodeLSTM(
    server, 
    tcp_params=tcp_params,
    lstm_model_path=lstm_model_path,
    use_prediction=True
)

client = PointCloudClient(
    edge,
    target_fps=target_fps,
    buffer_capacity_s=buffer_capacity_s,
    min_buffer_s=min_buffer_s,
)

sim = Simulator(server, edge, [client])

print("="*100)
print("🤖 LSTM-Based Bandwidth Prediction for Video Quality Selection")
print("="*100)
print(f"LSTM Model: {lstm_model_path}")
print(f"Bandwidth Log: {bandwidth_log_path}")
print(f"MPD Config: {mpd_path}")
print("="*100)

sim.run(mpd_path=mpd_path, bandwidth_log_path=bandwidth_log_path)
