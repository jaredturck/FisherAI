import queue
import threading

import numpy as np

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import FisherConfig, save_config
from fisher_ai.distributed import (
    DistributedSelfPlayPool,
    RemoteEvaluator,
    SelfPlayCancelled,
    SharedInferenceMemory,
)
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import ReplayBuffer


def test_distributed_self_play_streams_completed_games(tmp_path):
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
    config.runtime.actor_processes = 1
    config.runtime.games_per_actor = 2
    config.runtime.max_inflight_requests_per_actor = 2
    config.runtime.inference_request_batch_size = 2
    config.runtime.inference_batch_size = 2
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
    pool.run(target_games=2, timeout=60)

    assert all(process.exitcode == 0 for process in pool.actor_processes)

    replay = ReplayBuffer(config.runtime.replay_path, max_positions=100)
    assert replay.game_count >= 2
    assert replay.position_count >= 4
    replay.close()


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
        [0],
        [0],
        [0],
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
