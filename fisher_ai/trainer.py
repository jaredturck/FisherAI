"""Train the FisherAI network on generated self-play windows."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from fisher_ai.dataset import materialize_mixed_batch
from fisher_ai.encoding import INPUT_PLANES
from fisher_ai.mcts import MAX_LEGAL_ACTIONS

EPOCHS_PER_WINDOW = 3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LEARNING_RATES = (0.05, 0.01, 0.002, 0.0005)
LEARNING_RATE_STEPS = (0, 100000, 250000, 500000)
RANDOM_SEED = 7


class AlphaZeroTrainer:
    """Optimize the policy-value network over complete self-play windows."""

    def __init__(
        self, model, batch_size, device="cpu", checkpoint_manager=None
    ):
        self.model = model
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.checkpoint_manager = checkpoint_manager
        self.channels_last = self.device.type == "cuda"

        if self.device.type == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))

        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        self.model.train()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=LEARNING_RATES[0],
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda",
        )
        self.step = 0
        self.rng = np.random.default_rng(RANDOM_SEED)

        self.pinned_states = None
        self.pinned_legal_actions = None
        self.pinned_visit_counts = None
        self.pinned_legal_mask = None
        self.pinned_values = None
        if self.device.type == "cuda":
            self.allocate_transfer_buffers()

    def allocate_transfer_buffers(self):
        """Allocate reusable pinned staging storage for CUDA transfers."""
        action_capacity = self.batch_size * MAX_LEGAL_ACTIONS
        self.pinned_states = torch.empty(
            (self.batch_size, INPUT_PLANES, 8, 8),
            dtype=torch.float16,
            pin_memory=True,
        )
        self.pinned_legal_actions = torch.empty(
            action_capacity,
            dtype=torch.int64,
            pin_memory=True,
        )
        self.pinned_visit_counts = torch.empty(
            action_capacity,
            dtype=torch.float32,
            pin_memory=True,
        )
        self.pinned_legal_mask = torch.empty(
            action_capacity,
            dtype=torch.bool,
            pin_memory=True,
        )
        self.pinned_values = torch.empty(
            self.batch_size,
            dtype=torch.float32,
            pin_memory=True,
        )

    def learning_rate(self):
        """Return the scheduled learning rate for the current step."""
        learning_rate = LEARNING_RATES[0]
        for boundary, rate in zip(
            LEARNING_RATE_STEPS,
            LEARNING_RATES,
            strict=True,
        ):
            if self.step >= boundary:
                learning_rate = rate
        return learning_rate

    def set_learning_rate(self):
        """Apply the current scheduled rate to every optimizer group."""
        learning_rate = self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def load_checkpoint(self, path=None):
        """Restore model and optimizer state from a checkpoint."""
        if self.checkpoint_manager is None:
            return 0

        self.step = self.checkpoint_manager.load(
            self.model,
            path=path,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )
        return self.step

    def compute_loss(
        self,
        states,
        legal_actions,
        visit_counts,
        legal_mask,
        target_values,
    ):
        """Compute masked policy loss and scalar value loss."""
        policy_logits, predicted_values = self.model(states)
        legal_logits = policy_logits.gather(1, legal_actions).float()
        legal_logits = legal_logits.masked_fill(~legal_mask, -1e9)
        log_policy = F.log_softmax(legal_logits, dim=1)

        visit_counts = visit_counts * legal_mask
        target_policy = visit_counts / visit_counts.sum(dim=1, keepdim=True)
        policy_loss = -(target_policy * log_policy).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted_values, target_values)
        return policy_loss + value_loss, policy_loss, value_loss

    def cuda_tensor_batch(self, batch):
        """Stage one NumPy batch in reusable pinned buffers."""
        batch_size = len(batch[0])
        legal_width = batch[1].shape[1]
        action_count = batch_size * legal_width

        states = self.pinned_states[:batch_size]
        legal_actions = self.pinned_legal_actions[:action_count].view(
            batch_size,
            legal_width,
        )
        visit_counts = self.pinned_visit_counts[:action_count].view(
            batch_size,
            legal_width,
        )
        legal_mask = self.pinned_legal_mask[:action_count].view(
            batch_size,
            legal_width,
        )
        values = self.pinned_values[:batch_size]

        states.copy_(torch.from_numpy(batch[0]))
        legal_actions.copy_(torch.from_numpy(batch[1]))
        visit_counts.copy_(torch.from_numpy(batch[2]))
        legal_mask.copy_(torch.from_numpy(batch[3]))
        values.copy_(torch.from_numpy(batch[4]))

        return [
            states.to(
                self.device,
                non_blocking=True,
                memory_format=torch.channels_last,
            ),
            legal_actions.to(self.device, non_blocking=True),
            visit_counts.to(self.device, non_blocking=True),
            legal_mask.to(self.device, non_blocking=True),
            values.to(self.device, non_blocking=True),
        ]

    def tensor_batch(self, batch):
        """Transfer one NumPy training batch to device tensors."""
        if self.device.type == "cuda":
            return self.cuda_tensor_batch(batch)

        tensors = [torch.from_numpy(array).to(self.device) for array in batch]
        tensors[0] = tensors[0].float()
        return tensors

    def train_batch(self, batch):
        """Run one mixed-precision optimizer step for a batch."""
        tensors = self.tensor_batch(batch)
        learning_rate = self.set_learning_rate()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            loss, policy_loss, value_loss = self.compute_loss(*tensors)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return (
            float(loss.detach()),
            float(policy_loss.detach()),
            float(value_loss.detach()),
            learning_rate,
        )

    def batch_indices(self, fresh_positions, replay_window, replay_per_epoch):
        """Yield shuffled fresh and replay index slices for every epoch."""
        for _ in range(EPOCHS_PER_WINDOW):
            fresh_indices = np.arange(fresh_positions, dtype=np.int64)
            replay_indices = (
                replay_window.sample_indices(replay_per_epoch, self.rng)
                if replay_per_epoch
                else np.empty(0, dtype=np.int64)
            )
            source_flags = np.concatenate(
                (
                    np.zeros(fresh_positions, dtype=np.bool_),
                    np.ones(len(replay_indices), dtype=np.bool_),
                )
            )
            position_indices = np.concatenate((fresh_indices, replay_indices))
            order = self.rng.permutation(len(position_indices))
            source_flags = source_flags[order]
            position_indices = position_indices[order]

            for start in range(0, len(order), self.batch_size):
                end = start + self.batch_size
                yield source_flags[start:end], position_indices[start:end]

    def prefetched_batches(
        self,
        window,
        replay_window,
        fresh_positions,
        replay_per_epoch,
    ):
        """Materialize one batch ahead while CUDA trains the current batch."""
        indices = iter(
            self.batch_indices(
                fresh_positions,
                replay_window,
                replay_per_epoch,
            )
        )
        first = next(indices, None)
        if first is None:
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                materialize_mixed_batch,
                window,
                replay_window,
                first[0],
                first[1],
            )
            for source_flags, position_indices in indices:
                batch = future.result()
                future = executor.submit(
                    materialize_mixed_batch,
                    window,
                    replay_window,
                    source_flags,
                    position_indices,
                )
                yield batch
            yield future.result()

    def train_window(self, window, replay_window=None, replay_ratio=0.0):
        """Train for three epochs over fresh data and sampled replay data."""
        started = time.perf_counter()
        loss_total = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        optimizer_steps = 0
        learning_rate = self.learning_rate()
        fresh_positions = window.position_count
        desired_replay = round(fresh_positions * float(replay_ratio))
        replay_available = (
            replay_window.position_count if replay_window is not None else 0
        )
        replay_per_epoch = min(desired_replay, replay_available)

        if self.device.type == "cuda":
            batches = self.prefetched_batches(
                window,
                replay_window,
                fresh_positions,
                replay_per_epoch,
            )
            for batch in batches:
                loss, policy_loss, value_loss, learning_rate = (
                    self.train_batch(batch)
                )
                loss_total += loss
                policy_loss_total += policy_loss
                value_loss_total += value_loss
                optimizer_steps += 1
        else:
            for _ in range(EPOCHS_PER_WINDOW):
                fresh_indices = np.arange(fresh_positions, dtype=np.int64)
                replay_indices = (
                    replay_window.sample_indices(replay_per_epoch, self.rng)
                    if replay_per_epoch
                    else np.empty(0, dtype=np.int64)
                )
                source_flags = np.concatenate(
                    (
                        np.zeros(fresh_positions, dtype=np.bool_),
                        np.ones(len(replay_indices), dtype=np.bool_),
                    )
                )
                position_indices = np.concatenate(
                    (fresh_indices, replay_indices)
                )
                order = self.rng.permutation(len(position_indices))
                source_flags = source_flags[order]
                position_indices = position_indices[order]

                for start in range(0, len(order), self.batch_size):
                    end = start + self.batch_size
                    batch = materialize_mixed_batch(
                        window,
                        replay_window,
                        source_flags[start:end],
                        position_indices[start:end],
                    )
                    loss, policy_loss, value_loss, learning_rate = (
                        self.train_batch(batch)
                    )
                    loss_total += loss
                    policy_loss_total += policy_loss
                    value_loss_total += value_loss
                    optimizer_steps += 1

        elapsed = time.perf_counter() - started
        positions_per_epoch = fresh_positions + replay_per_epoch
        positions = positions_per_epoch * EPOCHS_PER_WINDOW
        return {
            "elapsed_seconds": elapsed,
            "epochs": EPOCHS_PER_WINDOW,
            "positions": positions,
            "positions_per_second": positions / max(elapsed, 1e-6),
            "optimizer_steps": optimizer_steps,
            "loss": loss_total / optimizer_steps,
            "policy_loss": policy_loss_total / optimizer_steps,
            "value_loss": value_loss_total / optimizer_steps,
            "learning_rate": learning_rate,
            "fresh_positions_per_epoch": fresh_positions,
            "replay_positions_per_epoch": replay_per_epoch,
        }

    def save_checkpoint(self, cumulative_fresh_positions=0):
        """Save the current model, optimizer, and schedule state."""
        if self.checkpoint_manager is None:
            return None
        return self.checkpoint_manager.save(
            self.model,
            self.step,
            optimizer=self.optimizer,
            scaler=self.scaler,
            cumulative_fresh_positions=cumulative_fresh_positions,
        )
