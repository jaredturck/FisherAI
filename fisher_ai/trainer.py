import numpy as np
import torch
import torch.nn.functional as F


class AlphaZeroTrainer:
    def __init__(self, model, replay, config, device="cpu", checkpoint_manager=None):
        self.model = model
        self.replay = replay
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_manager = checkpoint_manager
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
        self.sampled_positions = 0
        self.rng = np.random.default_rng(config.runtime.seed)

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

        self.step, extra = self.checkpoint_manager.load(
            self.model,
            path=path,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )
        self.sampled_positions = int(extra.get("sampled_positions", 0))
        return self.step

    def can_train(self):
        self.replay.refresh()
        if self.replay.position_count < self.config.training.warmup_positions:
            return False

        allowed_samples = int(
            self.replay.total_positions_added * self.config.training.max_sample_ratio
        )
        return self.sampled_positions + self.config.training.batch_size <= allowed_samples

    def collate(self, samples):
        states = np.stack([sample.state for sample in samples]).astype(np.float32)
        values = np.asarray([sample.value for sample in samples], dtype=np.float32)
        policy_weights = np.asarray(
            [sample.policy_weight for sample in samples],
            dtype=np.float32,
        )
        max_legal_moves = max(len(sample.legal_actions) for sample in samples)
        legal_actions = np.zeros((len(samples), max_legal_moves), dtype=np.int64)
        visit_counts = np.zeros((len(samples), max_legal_moves), dtype=np.float32)
        legal_mask = np.zeros((len(samples), max_legal_moves), dtype=np.bool_)

        for index, sample in enumerate(samples):
            length = len(sample.legal_actions)
            legal_actions[index, :length] = sample.legal_actions
            visit_counts[index, :length] = sample.visit_counts
            legal_mask[index, :length] = True

        return (
            torch.from_numpy(states),
            torch.from_numpy(legal_actions),
            torch.from_numpy(visit_counts),
            torch.from_numpy(legal_mask),
            torch.from_numpy(values),
            torch.from_numpy(policy_weights),
        )

    def compute_loss(
        self,
        states,
        legal_actions,
        visit_counts,
        legal_mask,
        target_values,
        policy_weights,
        policy_normalizer,
        value_normalizer,
    ):
        policy_logits, predicted_values = self.model(states)
        legal_logits = policy_logits.gather(1, legal_actions)
        legal_logits = legal_logits.masked_fill(~legal_mask, -1e9)
        log_policy = F.log_softmax(legal_logits, dim=1)

        visit_counts = visit_counts * legal_mask
        target_policy = visit_counts / visit_counts.sum(dim=1, keepdim=True)
        per_sample_policy_loss = -(target_policy * log_policy).sum(dim=1)
        policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_normalizer
        value_loss = F.mse_loss(predicted_values, target_values, reduction="sum") / value_normalizer
        return policy_loss + value_loss, policy_loss, value_loss

    def train_step(self):
        samples = self.replay.sample(
            self.config.training.batch_size,
            rng=self.rng,
            positions_per_game=self.config.training.replay_positions_per_game,
        )
        batch = self.collate(samples)
        batch_size = len(samples)
        micro_batch_size = min(self.config.training.micro_batch_size, batch_size)
        learning_rate = self.set_learning_rate()

        if self.device.type == "cuda":
            batch = tuple(tensor.pin_memory() for tensor in batch)

        policy_normalizer = max(float(batch[-1].sum()), 1.0)
        value_normalizer = float(batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro_batch = [tensor[start:end].to(self.device, non_blocking=True) for tensor in batch]
            if self.channels_last:
                micro_batch[0] = micro_batch[0].contiguous(memory_format=torch.channels_last)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                loss, policy_loss, value_loss = self.compute_loss(
                    *micro_batch,
                    policy_normalizer=policy_normalizer,
                    value_normalizer=value_normalizer,
                )

            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach())
            total_policy_loss += float(policy_loss.detach())
            total_value_loss += float(value_loss.detach())

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        self.sampled_positions += batch_size

        if (
            self.checkpoint_manager is not None
            and self.step % self.config.training.checkpoint_interval == 0
        ):
            self.save_checkpoint()

        return {
            "step": self.step,
            "loss": total_loss,
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "learning_rate": learning_rate,
            "sampled_positions": self.sampled_positions,
        }

    def save_checkpoint(self):
        if self.checkpoint_manager is None:
            return None

        return self.checkpoint_manager.save(
            self.model,
            self.config,
            self.step,
            optimizer=self.optimizer,
            scaler=self.scaler,
            extra={
                "replay_positions": self.replay.position_count,
                "total_positions_added": self.replay.total_positions_added,
                "sampled_positions": self.sampled_positions,
            },
        )

    def train(self, steps):
        metrics = []
        for _ in range(steps):
            metrics.append(self.train_step())
        return metrics
