from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import FisherConfig, save_config
from fisher_ai.distributed import DistributedSelfPlayPool
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
    config.runtime.actor_processes = 2
    config.runtime.games_per_actor = 1
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
        actor_count=2,
        games_per_actor=1,
        devices=["cpu"],
    )
    pool.run(target_games=2, timeout=60)

    replay = ReplayBuffer(config.runtime.replay_path, max_positions=100)
    assert replay.game_count >= 2
    assert replay.position_count >= 4
    replay.close()
