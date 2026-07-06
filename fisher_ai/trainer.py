"""Train the FisherAI network on generated self-play windows."""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

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

    def tensor_batch(self, batch):
        """Transfer one NumPy training batch to device tensors."""
        tensors = [torch.from_numpy(array) for array in batch]
        if self.device.type == "cuda":
            tensors = [tensor.pin_memory() for tensor in tensors]
        tensors = [
            tensor.to(self.device, non_blocking=True) for tensor in tensors
        ]
        if self.device.type == "cpu":
            tensors[0] = tensors[0].float()
        if self.channels_last:
            tensors[0] = tensors[0].contiguous(
                memory_format=torch.channels_last
            )
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

    def train_window(self, window):
        """Train for three shuffled epochs over the full window."""
        started = time.perf_counter()
        loss_total = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        optimizer_steps = 0
        learning_rate = self.learning_rate()

        for _ in range(EPOCHS_PER_WINDOW):
            indices = window.shuffled_indices(self.rng)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch = window.materialize_batch(batch_indices)
                loss, policy_loss, value_loss, learning_rate = (
                    self.train_batch(batch)
                )
                loss_total += loss
                policy_loss_total += policy_loss
                value_loss_total += value_loss
                optimizer_steps += 1

        elapsed = time.perf_counter() - started
        positions = window.position_count * EPOCHS_PER_WINDOW
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
        }

    def save_checkpoint(self):
        """Save the current model and optimizer state."""
        if self.checkpoint_manager is None:
            return None
        return self.checkpoint_manager.save(
            self.model,
            self.step,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )
