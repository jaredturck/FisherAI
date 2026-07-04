import numpy as np

from fisher_ai.config import FisherConfig
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import GameRecord, ReplayBuffer, TrainingSample
from fisher_ai.trainer import AlphaZeroTrainer


def test_trainer_runs_one_update_with_mixed_policy_weights(tmp_path):
    config = FisherConfig()
    config.network.channels = 8
    config.network.residual_blocks = 1
    config.network.squeeze_excitation_channels = 2
    config.network.policy_channels = 8
    config.network.value_channels = 2
    config.network.value_hidden = 16
    config.training.batch_size = 8
    config.training.micro_batch_size = 4
    config.training.replay_positions_per_game = 4

    replay = ReplayBuffer(tmp_path / "replay.lmdb", max_positions=100)
    samples = []
    for index in range(8):
        state = np.zeros((119, 8, 8), dtype=np.float16)
        state[0, index, index] = 1
        samples.append(
            TrainingSample(
                state,
                [1, 2, 3],
                [4, 2, 1],
                value=1,
                policy_weight=index % 2,
            )
        )
    replay.add_game(GameRecord(samples=samples, result=1))

    model = FisherNetwork(config.network)
    trainer = AlphaZeroTrainer(model, replay, config, device="cpu")
    metrics = trainer.train_step()

    assert metrics["step"] == 1
    assert metrics["loss"] > 0
    assert metrics["policy_loss"] > 0
    assert metrics["sampled_positions"] == 8
    replay.close()
