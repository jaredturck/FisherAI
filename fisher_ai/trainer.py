import os
import time

import numpy as np
import torch
import torch.nn.functional as F


class AlphaZeroTrainer:
    def __init__(self, model, config, device="cpu", checkpoint_manager=None):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_manager = checkpoint_manager
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.channels_last = config.runtime.channels_last and self.device.type == "cuda"
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        self.model.train()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.training.learning_rates[0],
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.step = 0
        self.positions_trained = 0
        self.rng = np.random.default_rng(config.runtime.seed)
        self.checkpoint_extra = {}

    def learning_rate(self, step):
        rates = self.config.training.learning_rates
        boundaries = self.config.training.learning_rate_steps
        learning_rate = rates[0]

        for boundary, rate in zip(boundaries, rates, strict=True):
            if step >= boundary:
                learning_rate = rate

        return learning_rate

    def set_learning_rate(self):
        learning_rate = self.learning_rate(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def load_checkpoint(self, path=None):
        if self.checkpoint_manager is None:
            return 0

        self.step, self.checkpoint_extra = self.checkpoint_manager.load(
            self.model,
            path=path,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )
        self.positions_trained = int(self.checkpoint_extra.get("positions_trained", 0))
        return self.step

    def compute_loss(
        self,
        states,
        legal_actions,
        visit_counts,
        legal_mask,
        target_values,
        policy_weights,
    ):
        policy_logits, predicted_values = self.model(states)
        legal_logits = policy_logits.gather(1, legal_actions).float()
        legal_logits = legal_logits.masked_fill(~legal_mask, -1e9)
        log_policy = F.log_softmax(legal_logits, dim=1)

        visit_counts = visit_counts * legal_mask
        target_policy = visit_counts / visit_counts.sum(dim=1, keepdim=True)
        per_sample_policy_loss = -(target_policy * log_policy).sum(dim=1)
        policy_normalizer = max(float(policy_weights.sum()), 1.0)
        policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_normalizer
        value_loss = F.mse_loss(predicted_values, target_values)
        return policy_loss + value_loss, policy_loss, value_loss

    def tensor_batch(self, batch):
        tensors = [torch.from_numpy(array) for array in batch]
        if self.device.type == "cuda":
            tensors = [tensor.pin_memory() for tensor in tensors]
        tensors = [tensor.to(self.device, non_blocking=True) for tensor in tensors]
        if self.device.type != "cuda":
            tensors[0] = tensors[0].float()
        if self.channels_last:
            tensors[0] = tensors[0].contiguous(memory_format=torch.channels_last)
        return tensors

    def train_batch(self, batch):
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
        batch_size = len(batch[0])
        self.positions_trained += batch_size

        return {
            "step": self.step,
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "learning_rate": learning_rate,
            "positions": batch_size,
        }

    def train_window(self, window, epochs=None):
        epochs = epochs or self.config.training.epochs_per_window
        batch_size = self.config.training.batch_size
        started = time.perf_counter()
        metrics = []
        seen_indices = []

        for _ in range(epochs):
            indices = window.shuffled_indices(self.rng)
            seen_indices.append(indices.copy())
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = window.materialize_batch(batch_indices)
                metrics.append(self.train_batch(batch))

        elapsed = time.perf_counter() - started
        positions = window.position_count * epochs
        return {
            "elapsed_seconds": elapsed,
            "positions": positions,
            "positions_per_second": positions / max(elapsed, 1e-6),
            "optimizer_steps": len(metrics),
            "loss": float(np.mean([metric["loss"] for metric in metrics])),
            "policy_loss": float(np.mean([metric["policy_loss"] for metric in metrics])),
            "value_loss": float(np.mean([metric["value_loss"] for metric in metrics])),
            "learning_rate": metrics[-1]["learning_rate"],
            "seen_indices": seen_indices,
        }

    def save_checkpoint(self, extra=None):
        if self.checkpoint_manager is None:
            return None

        checkpoint_extra = {
            **self.checkpoint_extra,
            **(extra or {}),
            "positions_trained": self.positions_trained,
        }
        return self.checkpoint_manager.save(
            self.model,
            self.config,
            self.step,
            optimizer=self.optimizer,
            scaler=self.scaler,
            extra=checkpoint_extra,
        )
