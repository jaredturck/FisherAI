import torch
import torch.nn as nn

from fisher_ai.dataset import InMemoryWindow
from fisher_ai.trainer import EPOCHS_PER_WINDOW, AlphaZeroTrainer
from tests.test_dataset import make_record


class TinyPolicyValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_bias = nn.Parameter(torch.zeros(4))
        self.value_bias = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        batch_size = states.shape[0]
        policy_logits = self.policy_bias.unsqueeze(0).expand(batch_size, -1)
        predicted_values = torch.tanh(self.value_bias).expand(batch_size)
        return policy_logits, predicted_values


class HalfPrecisionPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        batch_size = states.shape[0]
        policy_logits = torch.zeros((batch_size, 4), dtype=torch.float16)
        predicted_values = torch.zeros(batch_size, dtype=torch.float32)
        return policy_logits + self.weight, predicted_values + self.weight


def make_window(position_count):
    window = InMemoryWindow(position_count)
    window.add_game(make_record(position_count, action_count=3))
    return window


def test_trainer_runs_three_complete_epochs():
    trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cpu",
    )
    metrics = trainer.train_window(make_window(7))

    assert metrics["epochs"] == EPOCHS_PER_WINDOW == 3
    assert metrics["optimizer_steps"] == 9
    assert metrics["positions"] == 21
    assert trainer.step == 9


def test_compute_loss_handles_half_precision_policy_logits():
    trainer = AlphaZeroTrainer(
        HalfPrecisionPolicyModel(),
        batch_size=2,
        device="cpu",
    )

    states = torch.zeros((2, 1), dtype=torch.float32)
    legal_actions = torch.tensor([[0, 1, 0], [1, 2, 3]])
    visit_counts = torch.tensor([[2, 1, 0], [3, 2, 1]], dtype=torch.float32)
    legal_mask = torch.tensor([[True, True, False], [True, True, True]])
    target_values = torch.zeros(2)

    loss, policy_loss, value_loss = trainer.compute_loss(
        states,
        legal_actions,
        visit_counts,
        legal_mask,
        target_values,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
