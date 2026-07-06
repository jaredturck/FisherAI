import threading

import numpy as np
import torch
import torch.nn as nn

from fisher_ai.dataset import (
    InMemoryWindow,
    ReplayWindow,
    materialize_mixed_batch,
)
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


def test_trainer_samples_replay_independently_each_epoch():
    fresh = make_window(4)
    replay = make_window(8)
    calls = []

    def sample_indices(count, rng):
        calls.append(count)
        start = len(calls) - 1
        return torch.arange(start, start + count).numpy() % 8

    replay.sample_indices = sample_indices
    trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cpu",
    )
    metrics = trainer.train_window(
        fresh,
        replay_window=replay,
        replay_ratio=0.5,
    )

    assert calls == [2, 2, 2]
    assert metrics["fresh_positions_per_epoch"] == 4
    assert metrics["replay_positions_per_epoch"] == 2
    assert metrics["positions"] == 18
    assert metrics["optimizer_steps"] == 6


def test_prefetched_batches_preserve_serial_batch_contents():
    fresh = make_window(4)
    replay = ReplayWindow(8)
    replay.append_window(make_window(4), 1)
    replay.append_window(make_window(4), 2)
    serial_trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cpu",
    )
    prefetched_trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cpu",
    )
    serial_batches = []
    for source_flags, position_indices in serial_trainer.batch_indices(
        fresh.position_count,
        replay,
        replay_per_epoch=2,
    ):
        serial_batches.append(
            materialize_mixed_batch(
                fresh,
                replay,
                source_flags,
                position_indices,
            )
        )
    prefetched_batches = list(
        prefetched_trainer.prefetched_batches(
            fresh,
            replay,
            fresh.position_count,
            replay_per_epoch=2,
        )
    )

    assert len(prefetched_batches) == len(serial_batches)
    for serial_batch, prefetched_batch in zip(
        serial_batches,
        prefetched_batches,
        strict=True,
    ):
        for serial_array, prefetched_array in zip(
            serial_batch,
            prefetched_batch,
            strict=True,
        ):
            np.testing.assert_array_equal(serial_array, prefetched_array)


def test_cuda_training_prefetches_materialization_on_one_worker(monkeypatch):
    trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cpu",
    )
    trainer.device = torch.device("cuda")
    materialize_threads = []
    trained_batches = []

    def materialize_batch(
        fresh_window,
        replay_window,
        source_flags,
        position_indices,
    ):
        materialize_threads.append(threading.get_ident())
        return fresh_window.materialize_batch(position_indices)

    def train_batch(batch):
        trained_batches.append(len(batch[0]))
        return 1.0, 0.5, 0.5, 0.05

    monkeypatch.setattr(
        "fisher_ai.trainer.materialize_mixed_batch",
        materialize_batch,
    )
    trainer.train_batch = train_batch
    metrics = trainer.train_window(make_window(7))

    assert metrics["optimizer_steps"] == 9
    assert trained_batches == [3, 3, 1] * EPOCHS_PER_WINDOW
    assert len(set(materialize_threads)) == 1
    assert materialize_threads[0] != threading.get_ident()


def test_cuda_tensor_batch_reuses_pinned_staging_buffers():
    if not torch.cuda.is_available():
        return

    trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cuda:0",
    )
    batch = make_window(3).materialize_batch(torch.arange(3).numpy())
    state_buffer = trainer.pinned_states.data_ptr()
    action_buffer = trainer.pinned_legal_actions.data_ptr()

    tensors = trainer.tensor_batch(batch)
    torch.cuda.synchronize()
    trainer.tensor_batch(batch)
    torch.cuda.synchronize()

    assert trainer.pinned_states.data_ptr() == state_buffer
    assert trainer.pinned_legal_actions.data_ptr() == action_buffer
    assert tensors[0].is_contiguous(memory_format=torch.channels_last)
