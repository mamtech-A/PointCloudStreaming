import os
import sys
import json

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

# DASH-style segments: frames fetched per request (1 = legacy per-frame).
# Matches the round-3 sweep winner S=8 (S=5..8 are statistically tied;
# see DQN_REPORT §10.8). Override freely — the policy generalizes across S.
segment_frames = 8


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# --- Load the trained policy. Whether the LSTM is even used is read from the
# checkpoint (the round-3 winner is lstm-off), so we only pay for the predictor
# when the policy's state actually contains the lstm_pred feature. -----------
agent = DQNAgent.load(dqn_model_path)
lstm_used = 'lstm_pred' in agent.feature_spec
predictor = LSTMPredictor().load(lstm_model_path) if lstm_used else None


def _abr_factory():
    provider = LSTMABR(predictor) if (lstm_used and predictor is not None) else None
    return DQNABR(policy=agent, lstm_provider=provider)


server = Server(base_url="http://localhost/")
topo = Topology(server)
edge = EdgeNode("edge-1", server=server, tcp_params=tcp_params, abr_factory=_abr_factory)
topo.add_edge(edge)

user = User("User", target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0,
            playback_rate_min=0.9)
topo.add_user(user, edge, trace=BandwidthTrace.from_file(bandwidth_log_path))

sim = Simulator(topo)

# --- Provenance header: prove this is the best trained policy + give context. ---
sweep = _load_json(os.path.join(project_root, "models", "dqn_sweep_results.json"))
fixed = _load_json(os.path.join(project_root, "models", "fixed_arm_baseline.json"))

print("=" * 100)
print("🎮 DQN Point-Cloud ABR — running the best trained policy")
print("-" * 100)
print(f"Policy   : {os.path.relpath(dqn_model_path, project_root)}", end="")
if sweep and sweep.get("winner"):
    w = sweep["winner"]; wm = w.get("metrics", {}); wc = w.get("config", {})
    std = wm.get("qoe_quality_std", 0.0)
    print("  (round-3 sweep winner)")
    print(f"   config : {wc.get('segment-frames', segment_frames)} frames/segment · "
          f"μ={wc.get('mu', agent.mu)} · LSTM feature: {'ON' if lstm_used else 'OFF'}")
    ctx = ""
    if fixed and fixed.get("results", {}).get(fixed.get("best_arm", ""), {}).get("qoe_quality") is not None:
        ba = fixed["results"][fixed["best_arm"]]
        ctx = f" · best fixed arm ≈ QoE′ {ba['qoe_quality']:.0f}"
    print(f"   held-out QoE′ {wm.get('qoe_quality', float('nan')):.1f} ± {std:.1f} "
          f"(8-seed robust{ctx})")
else:
    print(f"  ·  LSTM feature: {'ON' if lstm_used else 'OFF'}")
print(f"Content  : longdress · 300 frames @30fps (10 s) · 6-tier G-PCC ladder")
print(f"Network  : {os.path.basename(bandwidth_log_path)} (unseen test) · "
      f"RTT {tcp_params['rtt_ms']:.0f}ms")
print(f"Playback : {segment_frames}-frame segments · adaptive-rate floor "
      f"{user.playback_rate_min:.2f}x · {user.buffer_capacity_s:.0f}s buffer")
print("=" * 100)

sim.run(mpd_path=mpd_path, run_label="dqn", segment_frames=segment_frames)
