import os
import sys

# Project root on sys.path so `from src.network_model import ...` resolves.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.network_model import (
    Server, EdgeNode, User, Topology, Simulator, LSTMABR, DQNABR, DEFAULT_TCP_PARAMS,
)
from src.network_model.trace import BandwidthTrace
from src.lstm_model import LSTMPredictor
from src.rl.dqn import DQNAgent

# Default paths — evaluate on an UNSEEN test-split 5G trace.
mpd_path = os.path.join(project_root, "config", "mpd_gpcc_longdress.xml")
bandwidth_log_path = os.path.join(project_root, "bandwidth_5g",
                                  "static_B_2020.01.16_10.43.34.csv")
dqn_model_path = os.path.join(project_root, "models", "abr_dqn.pkl")
lstm_model_path = os.path.join(project_root, "models", "bandwidth_lstm.pkl")

tcp_params = dict(DEFAULT_TCP_PARAMS)

# Preload the policy + LSTM predictor once; the edge mints a DQNABR per session
# (sharing them read-only — safe single-threaded).
agent = DQNAgent.load(dqn_model_path)
predictor = LSTMPredictor().load(lstm_model_path)

server = Server(base_url="http://localhost/")
topo = Topology(server)
edge = EdgeNode("edge-1", server=server, tcp_params=tcp_params,
                abr_factory=lambda: DQNABR(policy=agent, lstm_provider=LSTMABR(predictor)))
topo.add_edge(edge)

user = User("User", target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0)
topo.add_user(user, edge, trace=BandwidthTrace.from_file(bandwidth_log_path))

sim = Simulator(topo)

print("=" * 100)
print("🎮 DQN-Based Representation Selection (LSTM prediction used as a state feature)")
print("=" * 100)
print(f"DQN Model:  {dqn_model_path}")
print(f"LSTM Model: {lstm_model_path}")
print(f"Bandwidth:  {bandwidth_log_path}")
print("=" * 100)

sim.run(mpd_path=mpd_path, run_label="dqn")
