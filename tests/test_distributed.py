import queue
import threading

import numpy as np

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import FisherConfig, SearchConfig, save_config
from fisher_ai.distributed import (
    BATCHED_SCHEDULER,
    THREADED_SCHEDULER,
    ConcurrentRemoteEvaluator,
    DistributedSelfPlayPool,
    RemoteEvaluator,
    SelfPlayCancelled,
    SharedInferenceMemory,
)
from fisher_ai.mcts import MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import ReplayBuffer
from fisher_ai.self_play import SelfPlayRunner


class RecordingEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate_encoded(self, encoded_states, legal_actions=None):
        self.batch_sizes.append(len(encoded_states))
        policies = [np.zeros(len(actions), dtype=np.float32) for actions in legal_actions]
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def evaluator_counters():
    return [0], [0], [0], [0], [0], [0]


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


def test_pool_auto_sizes_batched_requests_for_all_sessions(tmp_path):
    config = FisherConfig()
    config.search.parallel_searches = 24
    config.runtime.self_play_scheduler = BATCHED_SCHEDULER
    config.runtime.games_per_actor = 4
    config.runtime.inference_request_batch_size = 32
    config.runtime.inference_max_batch_size = 1024
    config.runtime.replay_path = str(tmp_path / "replay.lmdb")
    config_path = tmp_path / "config.json"
    save_config(config, config_path)

    pool = DistributedSelfPlayPool(config_path=config_path, actor_count=1)
    try:
        assert pool.scheduler == BATCHED_SCHEDULER
        assert pool.slots_per_actor == 1
        assert pool.request_capacity == 96
        assert pool.shared.states.shape[:3] == (1, 1, 96)
    finally:
        pool.stop()


def test_threaded_scheduler_retains_multiple_inference_slots(tmp_path):
    config = FisherConfig()
    config.search.parallel_searches = 24
    config.runtime.self_play_scheduler = THREADED_SCHEDULER
    config.runtime.games_per_actor = 4
    config.runtime.max_inflight_requests_per_actor = 8
    config.runtime.replay_path = str(tmp_path / "replay.lmdb")
    config_path = tmp_path / "config.json"
    save_config(config, config_path)

    pool = DistributedSelfPlayPool(config_path=config_path, actor_count=1)
    try:
        assert pool.scheduler == THREADED_SCHEDULER
        assert pool.slots_per_actor == 8
        assert pool.request_capacity == 24
    finally:
        pool.stop()


def test_remote_evaluator_submits_large_batch_as_one_request():
    shared = SharedInferenceMemory(
        actor_count=1,
        slots_per_actor=1,
        max_request_batch=64,
        max_legal_actions=4,
    )
    stop_event = threading.Event()
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    counters = evaluator_counters()
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
        actor_id, slot_id, batch_size, request_id = request_queue.get(timeout=2)
        assert (actor_id, slot_id, batch_size) == (0, 0, 40)
        shared.policy_logits[0, 0, :40, 0] = 1.0
        shared.values[0, 0, :40] = 0.5
        response_queue.put((request_id, slot_id, None))
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert len(result) == 1
        assert len(result[0][0]) == 40
        assert np.allclose(result[0][1], 0.5)
    finally:
        stop_event.set()
        evaluator.close()
        shared.close()
        shared.unlink()


def test_remote_evaluator_treats_shutdown_as_cancellation():
    shared = SharedInferenceMemory(
        actor_count=1,
        slots_per_actor=1,
        max_request_batch=1,
        max_legal_actions=4,
    )
    stop_event = threading.Event()
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    evaluator = RemoteEvaluator(
        0,
        request_queue,
        response_queue,
        shared.descriptor,
        stop_event,
        *evaluator_counters(),
    )
    errors = []

    def evaluate():
        try:
            evaluator.evaluate_chunk(
                [np.zeros((119, 8, 8), dtype=np.float16)],
                [np.asarray([0], dtype=np.uint16)],
            )
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=evaluate)
    thread.start()

    try:
        request_queue.get(timeout=2)
        stop_event.set()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], SelfPlayCancelled)
    finally:
        stop_event.set()
        evaluator.close()
        shared.close()
        shared.unlink()


def test_concurrent_evaluator_treats_shutdown_as_cancellation():
    shared = SharedInferenceMemory(
        actor_count=1,
        slots_per_actor=1,
        max_request_batch=1,
        max_legal_actions=4,
    )
    stop_event = threading.Event()
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    evaluator = ConcurrentRemoteEvaluator(
        0,
        request_queue,
        response_queue,
        shared.descriptor,
        stop_event,
        *evaluator_counters(),
    )
    errors = []

    def evaluate():
        try:
            evaluator.evaluate_chunk(
                [np.zeros((119, 8, 8), dtype=np.float16)],
                [np.asarray([0], dtype=np.uint16)],
            )
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=evaluate)
    thread.start()

    try:
        request_queue.get(timeout=2)
        stop_event.set()
        evaluator.cancel_pending()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], SelfPlayCancelled)
    finally:
        stop_event.set()
        evaluator.close()
        shared.close()
        shared.unlink()


def test_distributed_batched_self_play_streams_completed_games(tmp_path):
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
    config.training.replay_max_positions = 100
    config.runtime.replay_path = str(tmp_path / "replay.lmdb")
    config.runtime.checkpoint_dir = str(tmp_path / "checkpoints")
    config.runtime.self_play_devices = ["cpu"]
    config.runtime.self_play_scheduler = BATCHED_SCHEDULER
    config.runtime.actor_processes = 1
    config.runtime.games_per_actor = 2
    config.runtime.inference_request_batch_size = 0
    config.runtime.max_inflight_requests_per_actor = 1
    config.runtime.inference_batch_size = 2
    config.runtime.inference_max_batch_size = 2
    config.runtime.inference_batch_wait_ms = 1.0
    config.runtime.status_interval_seconds = 0.1
    config.runtime.checkpoint_reload_seconds = 60.0
    config_path = tmp_path / "config.json"
    save_config(config, config_path)

    model = FisherNetwork(config.network)
    manager = CheckpointManager(config.runtime.checkpoint_dir)
    manager.save(model, config, 0)

    pool = DistributedSelfPlayPool(
        config_path=config_path,
        actor_count=1,
        games_per_actor=2,
        devices=["cpu"],
    )
    assert pool.request_capacity == 2
    assert pool.slots_per_actor == 1
    pool.run(target_games=2, timeout=60)

    assert all(process.exitcode == 0 for process in pool.actor_processes)

    replay = ReplayBuffer(config.runtime.replay_path, max_positions=100)
    assert replay.game_count >= 2
    assert replay.position_count >= 4
    replay.close()


def test_distributed_threaded_scheduler_remains_available_for_benchmark(tmp_path):
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
    config.training.replay_max_positions = 100
    config.runtime.replay_path = str(tmp_path / "replay.lmdb")
    config.runtime.checkpoint_dir = str(tmp_path / "checkpoints")
    config.runtime.self_play_devices = ["cpu"]
    config.runtime.self_play_scheduler = THREADED_SCHEDULER
    config.runtime.actor_processes = 1
    config.runtime.games_per_actor = 2
    config.runtime.inference_request_batch_size = 0
    config.runtime.max_inflight_requests_per_actor = 2
    config.runtime.inference_batch_size = 2
    config.runtime.inference_max_batch_size = 2
    config.runtime.inference_batch_wait_ms = 1.0
    config.runtime.status_interval_seconds = 0.1
    config.runtime.checkpoint_reload_seconds = 60.0
    config_path = tmp_path / "config.json"
    save_config(config, config_path)

    model = FisherNetwork(config.network)
    manager = CheckpointManager(config.runtime.checkpoint_dir)
    manager.save(model, config, 0)

    pool = DistributedSelfPlayPool(
        config_path=config_path,
        actor_count=1,
        games_per_actor=2,
        devices=["cpu"],
    )
    assert pool.request_capacity == 1
    assert pool.slots_per_actor == 2
    pool.run(target_games=2, timeout=60)

    assert all(process.exitcode == 0 for process in pool.actor_processes)

    replay = ReplayBuffer(config.runtime.replay_path, max_positions=100)
    assert replay.game_count >= 2
    replay.close()
