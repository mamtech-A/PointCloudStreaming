"""Interactive demonstration of the validation-selected DQN policy.

Paper results must come from ``evaluate_experiment.py``.  This script resolves
the versioned wide-search artifacts and runs one Longdress/validation-trace
session for inspecting tier decisions and Q values.
"""

import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.lstm_model import LSTMPredictor
from src.network_model import (
    DEFAULT_TCP_PARAMS, DQNABR, EdgeNode, LSTMABR, Server, Simulator, Topology,
    User,
)
from src.network_model.trace import BandwidthTrace
from src.rl.dqn import DQNAgent
from src.rl.reward import RewardFunction


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def absolute(path):
    return path if os.path.isabs(path) else os.path.join(project_root, path)


training = load_json(os.path.join(project_root, "configs", "training.json"))
artifacts = training.get("artifacts", {})
sweep_path = absolute(artifacts.get(
    "sweep_results", os.path.join("models", "dqn_sweep_results.json")
))
dqn_model_path = absolute(artifacts.get(
    "dqn_model", os.path.join("models", "abr_dqn.pkl")
))
sweep = load_json(sweep_path)
winner_config = sweep["winner"]["config"]
segment_frames = int(winner_config.get(
    "segment-frames", sweep.get("base_args", {}).get("segment-frames", 8)
))
lstm_template = artifacts.get(
    "lstm_model_template",
    "models/bandwidth_lstm_request_pacing_s{segment_frames}.pkl",
)
lstm_model_path = absolute(lstm_template.format(segment_frames=segment_frames))

mpd_path = os.path.join(project_root, "manifests", "mpd_gpcc_longdress.xml")
# This is a registered validation trace, suitable for a demo but not a new test.
bandwidth_log_path = os.path.join(
    project_root, "bandwidth_5g", "static_B_2020.01.16_10.43.34.csv"
)

agent = DQNAgent.load(dqn_model_path)
lstm_used = "lstm_pred" in agent.feature_spec
predictor = LSTMPredictor().load(lstm_model_path) if lstm_used else None


def abr_factory():
    provider = LSTMABR(predictor) if predictor is not None else None
    return DQNABR(policy=agent, lstm_provider=provider, log_decisions=True)


server = Server(base_url="http://localhost/")
topology = Topology(server)
edge = EdgeNode(
    "edge-1", server=server, tcp_params=dict(DEFAULT_TCP_PARAMS),
    abr_factory=abr_factory,
)
topology.add_edge(edge)
user = User("User", target_fps=30.0, buffer_capacity_s=5.0, min_buffer_s=1.0)
topology.add_user(
    user, edge, trace=BandwidthTrace.from_file(bandwidth_log_path)
)

metrics = sweep["winner"].get("metrics", {})
select_by = sweep.get("select_by", "qoe_quality")
print("=" * 90)
print("DQN point-cloud ABR: validation-selected wide-search policy")
print(f"checkpoint : {os.path.relpath(dqn_model_path, project_root)}")
print(f"segment    : {segment_frames} frames")
print(f"LSTM       : {'on - ' + os.path.relpath(lstm_model_path, project_root) if lstm_used else 'off'}")
print(f"validation : {select_by}={metrics.get(select_by, float('nan')):.3f} "
      f"over training seeds {sweep['winner'].get('seeds', [])}")
print(f"demo trace : {os.path.basename(bandwidth_log_path)} (validation, not test)")
print("=" * 90)

reward_fn = RewardFunction(agent.reward_spec)
print(f"objective  : {reward_fn.describe()}")
simulator = Simulator(topology)
simulator.run(
    mpd_path=mpd_path,
    run_label="dqn_validation_demo",
    segment_frames=segment_frames,
    reward_fn=reward_fn,
)
