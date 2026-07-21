"""Guards for the registered wide DQN/segment search."""

import json
import os
import random
import sys
from collections import Counter

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import run_training
import sweep
from gen_lstm_dataset import select_registry_windows
from src.evaluation import _evaluate


def load(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
        return json.load(handle)


def test_wide_search_is_deterministic_and_balanced_by_segment_size():
    config = load(os.path.join("configs", "training.json"))
    search = config["dqn_sweep"]
    first = sweep.expand_trials(search)
    second = sweep.expand_trials(search)

    assert first == second
    assert len(first) == 20
    assert len({json.dumps(row, sort_keys=True) for row in first}) == 20
    assert Counter(row["segment-frames"] for row in first) == {
        5: 5, 8: 5, 10: 5, 15: 5,
    }
    paired = {}
    for row in first:
        segment = row["segment-frames"]
        paired[segment] = {
            json.dumps({k: v for k, v in config.items() if k != "segment-frames"},
                       sort_keys=True)
            for config in first if config["segment-frames"] == segment
        }
    assert paired[5] == paired[8] == paired[10] == paired[15]
    for axis, values in search["axes"].items():
        if axis != "segment-frames":
            assert {row[axis] for row in first[:5]} == set(values)


def test_confirmation_keeps_one_validation_finalist_per_segment_size():
    config = load(os.path.join("configs", "training.json"))
    search = config["dqn_sweep"]
    combos = sweep.expand_trials(search)
    # Rows are already in descending validation rank within each segment group.
    ranking = [
        {"trial": index, "config": combo, "metrics": {"qoe_quality": 100 - index}}
        for index, combo in enumerate(combos)
    ]
    selected = sweep.confirmation_trials(ranking, search)

    assert len(selected) == 4
    assert Counter(combos[index]["segment-frames"] for index in selected) == {
        5: 1, 8: 1, 10: 1, 15: 1,
    }


def test_lstm_and_dqn_search_use_the_same_requested_segment_values():
    config = load(os.path.join("configs", "training.json"))
    assert run_training.validate_search_config(config) == [5, 8, 10, 15]
    expected_contents = "longdress,loot,redandblack,soldier"
    assert config["dqn_sweep"]["base_args"]["eval-sequences"] == expected_contents
    assert config["trace_registry"] == "configs/trace_window_registry.json"
    assert config["dqn_sweep"]["base_args"]["trace-registry"] == (
        "configs/trace_window_registry.json"
    )
    assert "request_pacing" in config["artifacts"]["lstm_model_template"]
    assert "request_pacing" in config["lstm"]["data_dir_template"]
    assert config["final_eval"]["split"] == "test"
    assert config["final_eval"]["strategies"] == "dqn,fixed,buffer,mpc"
    search = config["dqn_sweep"]
    assert search["screen_overrides"]["epochs"] == 3
    assert search["base_args"]["epochs"] == 8
    assert search["confirmation_selection_seeds"] == list(range(45, 54))


def test_startup_weight_experiment_changes_only_the_intended_training_choices():
    original = load(os.path.join("configs", "training.json"))
    experiment = load(os.path.join("configs", "training_startup_w2_s10.json"))

    assert run_training.validate_search_config(experiment) == [10]
    assert experiment["protocol"] == original["protocol"]
    assert experiment["trace_registry"] == original["trace_registry"]
    assert experiment["lstm"]["segment_frames"] == [10]
    assert experiment["artifacts"]["lstm_model_template"] == (
        original["artifacts"]["lstm_model_template"]
    )

    search = experiment["dqn_sweep"]
    assert sweep.expand_trials(search) == [{
        "batch-size": 64,
        "eps-decay-frac": 0.5,
        "hidden": 256,
        "lr": 0.0003,
        "segment-frames": 10,
        "target-update": 6000,
    }]
    assert search["base_args"]["reward-spec"] == {
        "rebuffer_weight": 2.0,
        "startup_weight": 2.0,
    }
    assert search["confirmation_selection_seeds"] == list(range(45, 54))
    for key in ("split", "max_frames", "strategies"):
        assert experiment["final_eval"][key] == original["final_eval"][key]


def test_compact_validation_history_preserves_figure_metrics(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({
        "episode_reward": [1.0, 2.0],
        "validation": [{
            "episode": 400,
            "reward": 42.0,
            "qoe": 42.0,
            "qoe_quality": 42.0,
            "mean_quality": 0.6,
            "stall_s": 1.2,
            "rebuffer_events": 0.5,
            "quality_change": 1.1,
            "startup_s": 2.3,
            "request_pacing_s": 0.1,
            "per_trace": {"large": "intentionally omitted"},
        }],
    }), encoding="utf-8")

    compact = sweep.compact_validation_history({"history": str(path)})
    assert compact == [{
        "episode": 400,
        "reward": 42.0,
        "qoe": 42.0,
        "qoe_quality": 42.0,
        "mean_quality": 0.6,
        "stall_s": 1.2,
        "rebuffer_events": 0.5,
        "quality_change": 1.1,
        "startup_s": 2.3,
        "request_pacing_s": 0.1,
    }]


def test_weight_15_experiment_reuses_the_weight_2_s10_design():
    weight_15 = load(os.path.join("configs", "training_startup_w15_s10.json"))
    weight_2 = load(os.path.join("configs", "training_startup_w2_s10.json"))

    assert run_training.validate_search_config(weight_15) == [10]
    assert weight_15["protocol"] == weight_2["protocol"]
    assert weight_15["trace_registry"] == weight_2["trace_registry"]
    for key, value in weight_2["lstm"].items():
        if not key.startswith("_"):
            assert weight_15["lstm"][key] == value
    assert weight_15["dqn_sweep"]["axes"] == weight_2["dqn_sweep"]["axes"]
    assert weight_15["dqn_sweep"]["confirm_seeds"] == (
        weight_2["dqn_sweep"]["confirm_seeds"]
    )
    assert weight_15["dqn_sweep"]["base_args"]["reward-spec"] == {
        "rebuffer_weight": 2.0,
        "startup_weight": 1.5,
    }
    for key in ("split", "max_frames", "strategies"):
        assert weight_15["final_eval"][key] == weight_2["final_eval"][key]


def test_lstm_windows_are_deterministic_and_time_spread():
    rows = [
        {"id": f"w{index}", "start_time_s": float(index * 10),
         "source_sample_index": index}
        for index in range(10)
    ]
    shuffled = list(reversed(rows))
    selected = select_registry_windows(shuffled, 3)
    assert [row["id"] for row in selected] == ["w0", "w4", "w9"]


def test_registered_test_files_are_unchanged():
    protocol = load(os.path.join("configs", "experiment_protocol.json"))
    assert protocol["evaluation"]["validation"]["sequences"] == [
        "longdress", "loot", "redandblack", "soldier",
    ]
    assert protocol["trace_split"]["test"] == [
        "driving_B_2019.12.16_11.49.59.csv",
        "driving_B_2019.12.16_14.23.32.csv",
        "driving_B_2020.02.14_07.29.00.csv",
        "static_B_2020.02.13_13.57.29.csv",
    ]
    train = set(protocol["trace_split"]["train"])
    validation = set(protocol["trace_split"]["validation"])
    test = set(protocol["trace_split"]["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_validation_restores_training_rng_state():
    class Buffer:
        @staticmethod
        def get_buffer_stats():
            return {
                "rebuffer_count": 0, "startup_delay_s": 0.0,
            }

    class Env:
        user = Buffer()

        def reset(self, _trace, sequence=None):
            return 0

        def step(self, _action):
            # Evaluation is allowed to consume RNG internally; it must restore
            # the surrounding training streams on return.
            random.random()
            np.random.random()
            return 0, 0.0, True, {}

        @staticmethod
        def qoe(): return 0.0
        @staticmethod
        def mean_quality(): return 0.0
        @staticmethod
        def total_stall_s(): return 0.0
        @staticmethod
        def quality_change_sum(): return 0.0
        @staticmethod
        def qoe_terms(): return {"total": 0.0}
    class Registry:
        @staticmethod
        def validate_evaluation_count(_split, _count, trace_paths):
            filename = os.path.basename(trace_paths[0])
            return {filename: [{
                "id": "validation/example#block-000@time-0.000s",
                "block_id": "block-000",
                "start_time_s": 0.0,
                "duration_s": 10.0,
                "source_sample_index": 0,
            }]}

        @staticmethod
        def materialize(_window):
            return object()

    trace = os.path.join(
        ROOT, "bandwidth_5g", "driving_B_2019.12.16_12.27.05.csv"
    )
    random.seed(9876)
    np.random.seed(9876)
    expected = (random.random(), float(np.random.random()))
    random.seed(9876)
    np.random.seed(9876)
    _evaluate(
        Env(), [trace], [42], ["longdress"], 1,
        choose_action=lambda _observation, _env: 0,
        window_registry=Registry(), split_name="validation",
    )
    actual = (random.random(), float(np.random.random()))
    assert actual == expected


def _run_all():
    functions = [value for key, value in sorted(globals().items())
                 if key.startswith("test_")]
    for function in functions:
        function()
        print(f"  PASS {function.__name__}")
    print(f"{len(functions)} tests passed.")


if __name__ == "__main__":
    _run_all()
