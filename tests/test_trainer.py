import numpy as np
import torch
import torch.nn as nn

from fisher_ai.config import FisherConfig
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import GameRecord, ReplayBuffer, TrainingSample
from fisher_ai.trainer import AlphaZeroTrainer


class HalfPrecisionPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        batch_size = states.shape[0]
        policy_logits = torch.zeros((batch_size, 4), dtype=torch.float16)
        predicted_values = torch.zeros(batch_size, dtype=torch.float32)
        return policy_logits, predicted_values


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

    class Manager:
        def __init__(self):
            self.saved = 0

        def save(self, *args, **kwargs):
            self.saved += 1

    manager = Manager()
    model = FisherNetwork(config.network)
    trainer = AlphaZeroTrainer(
        model,
        replay,
        config,
        device="cpu",
        checkpoint_manager=manager,
    )
    metrics = trainer.train_step()

    assert metrics["step"] == 1
    assert metrics["loss"] > 0
    assert metrics["policy_loss"] > 0
    assert metrics["sampled_positions"] == 8
    assert manager.saved == 0
    replay.close()


def test_compute_loss_handles_half_precision_policy_logits():
    config = FisherConfig()
    trainer = AlphaZeroTrainer(HalfPrecisionPolicyModel(), None, config, device="cpu")

    states = torch.zeros((2, 1), dtype=torch.float32)
    legal_actions = torch.tensor([[0, 1, 0], [1, 2, 3]], dtype=torch.int64)
    visit_counts = torch.tensor([[2, 1, 0], [3, 2, 1]], dtype=torch.float32)
    legal_mask = torch.tensor([[True, True, False], [True, True, True]])
    target_values = torch.zeros(2, dtype=torch.float32)
    policy_weights = torch.ones(2, dtype=torch.float32)

    loss, policy_loss, value_loss = trainer.compute_loss(
        states,
        legal_actions,
        visit_counts,
        legal_mask,
        target_values,
        policy_weights,
        policy_normalizer=2.0,
        value_normalizer=2.0,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
