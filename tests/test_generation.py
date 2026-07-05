import queue
import threading

import numpy as np

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import FisherConfig, SearchConfig, save_config
from fisher_ai.generation import RemoteEvaluator, SharedInferenceMemory, WindowGenerator
from fisher_ai.mcts import MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.self_play import SelfPlayRunner


class RecordingEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate_encoded(self, encoded_states, legal_actions=None):
        self.batch_sizes.append(len(encoded_states))
        policies = [np.zeros(len(actions), dtype=np.float32) for actions in legal_actions]
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def test_self_play_batches_multiple_sessions_into_one_inference_request():
    config = SearchConfig(
        simulations=1,
        fast_simulations=1,
        full_search_fraction=1.0,
        parallel_searches=1,
        max_game_plies=2,
    )
    evaluator = RecordingEvaluator()
    search = MCTS(evaluator, config, seed=7)
    runner = SelfPlayRunner(search, config, seed=7)
    sessions = [runner.create_session(), runner.create_session()]

    runner.advance_sessions(sessions)

    assert evaluator.batch_sizes == [2, 2]


def test_remote_evaluator_submits_one_shared_memory_request():
    shared = SharedInferenceMemory(
        actor_count=1,
        max_request_batch=64,
        max_legal_actions=4,
    )
    stop_event = threading.Event()
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    counters = ([0], [0], [0], [0])
    evaluator = RemoteEvaluator(
        0,
        request_queue,
        response_queue,
        shared.descriptor,
        stop_event,
        *counters,
    )
    result = []

    def evaluate():
        states = [np.zeros((119, 8, 8), dtype=np.float16) for _ in range(40)]
        actions = [np.asarray([0], dtype=np.uint16) for _ in range(40)]
        result.append(evaluator.evaluate_encoded(states, actions))

    thread = threading.Thread(target=evaluate)
    thread.start()

    try:
        actor_id, batch_size, request_id = request_queue.get(timeout=2)
        assert (actor_id, batch_size) == (0, 40)
        shared.policy_logits[0, :40, 0] = 1.0
        shared.values[0, :40] = 0.5
        response_queue.put((request_id, None))
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert len(result[0][0]) == 40
        assert np.allclose(result[0][1], 0.5)
    finally:
        stop_event.set()
        evaluator.close()
        shared.close()
        shared.unlink()


def test_window_generator_returns_exact_in_memory_window(tmp_path):
    config = FisherConfig()
    config.network.channels = 4
    config.network.residual_blocks = 1
    config.network.squeeze_excitation_channels = 2
    config.network.policy_channels = 4
    config.network.value_channels = 1
    config.network.value_hidden = 8
    config.search.simulations = 1
    config.search.fast_simulations = 1
    config.search.full_search_fraction = 1.0
    config.search.parallel_searches = 1
    config.search.max_game_plies = 2
    config.runtime.checkpoint_dir = str(tmp_path / "checkpoints")
    config.runtime.device = "cpu"
    config.runtime.actor_processes = 1
    config.runtime.games_per_actor = 2
    config.runtime.inference_batch_size = 2
    config.runtime.inference_max_batch_size = 2
    config.runtime.inference_batch_wait_ms = 1.0
    config.runtime.status_interval_seconds = 0.1
    config_path = tmp_path / "config.json"
    save_config(config, config_path)

    manager = CheckpointManager(config.runtime.checkpoint_dir)
    checkpoint_path = manager.save(FisherNetwork(config.network), config, 0)
    generator = WindowGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    window, metrics = generator.generate(4, timeout=60, progress=False)

    assert generator.request_capacity == 2
    assert window.position_count == 4
    assert window.game_count >= 2
    assert metrics["window_positions"] == 4
    assert metrics["evaluations"] > 0
