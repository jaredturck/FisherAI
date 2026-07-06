"""Coordinate sequential self-play generation and model training."""

import gc
from pathlib import Path

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.dataset import ReplayWindow
from fisher_ai.generation import WindowGenerator
from fisher_ai.network import FisherNetwork
from fisher_ai.notifications import DiscordNotifier
from fisher_ai.trainer import AlphaZeroTrainer

FRESH_WINDOW_SCHEDULE = (
    (100_000, 10_000),
    (300_000, 20_000),
    (750_000, 30_000),
)
FINAL_WINDOW_POSITIONS = 50_000
REPLAY_CAPACITY = 200_000
EARLY_REPLAY_RATIO = 0.5
MATURE_REPLAY_RATIO = 1.0


def scheduled_window_positions(cumulative_fresh_positions):
    """Return the fresh-window size for the current training stage."""
    for boundary, positions in FRESH_WINDOW_SCHEDULE:
        if cumulative_fresh_positions < boundary:
            return positions
    return FINAL_WINDOW_POSITIONS


def scheduled_replay_ratio(cumulative_fresh_positions):
    """Return the replay ratio for the current training stage."""
    if cumulative_fresh_positions < FRESH_WINDOW_SCHEDULE[0][0]:
        return EARLY_REPLAY_RATIO
    return MATURE_REPLAY_RATIO


class TrainingPipeline:
    """Run repeated generate, replay, train, save, and notify iterations."""

    def __init__(self, config_path="fisher_config.json"):
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.device = available_device(self.config.device)
        self.manager = CheckpointManager()
        self.notifier = DiscordNotifier()
        checkpoint_path = self.manager.ensure(FisherNetwork())
        self.cumulative_fresh_positions = (
            self.manager.cumulative_fresh_positions(checkpoint_path)
        )
        self.replay = ReplayWindow(REPLAY_CAPACITY)

    def run_iteration(self, iteration):
        """Generate, train, update replay, save, and report one iteration."""
        checkpoint_path = self.manager.latest_path()
        target_positions = scheduled_window_positions(
            self.cumulative_fresh_positions
        )
        replay_ratio = scheduled_replay_ratio(self.cumulative_fresh_positions)
        print(
            f"Iteration {iteration}: generating {target_positions:,} "
            f"positions (cumulative={self.cumulative_fresh_positions:,})",
            flush=True,
        )
        generator = WindowGenerator(
            config_path=self.config_path,
            checkpoint_path=checkpoint_path,
        )
        window, generation = generator.generate(target_positions)
        del generator
        gc.collect()

        requested_replay = round(window.position_count * replay_ratio)
        sampled_replay = min(requested_replay, self.replay.position_count)
        print(
            f"Iteration {iteration}: training for three epochs with "
            f"{window.position_count:,} fresh and "
            f"{sampled_replay:,} replay positions per epoch",
            flush=True,
        )
        model = FisherNetwork()
        trainer = AlphaZeroTrainer(
            model,
            self.config.batch_size,
            device=self.device,
            checkpoint_manager=self.manager,
        )
        trainer.load_checkpoint(checkpoint_path)
        training = trainer.train_window(
            window,
            replay_window=self.replay,
            replay_ratio=replay_ratio,
        )

        self.replay.append_window(window, iteration)
        cumulative_fresh_positions = (
            self.cumulative_fresh_positions + window.position_count
        )
        trainer.save_checkpoint(cumulative_fresh_positions)
        self.cumulative_fresh_positions = cumulative_fresh_positions

        oldest_iteration = self.replay.oldest_iteration
        replay_age = (
            f"iteration {oldest_iteration}"
            if oldest_iteration is not None
            else "empty"
        )
        print(
            f"Iteration {iteration} complete: "
            f"generation={generation['elapsed_seconds']:.1f}s "
            f"({generation['positions_per_second']:.1f} positions/s), "
            f"training={training['elapsed_seconds']:.1f}s "
            f"({training['positions_per_second']:.1f} positions/s), "
            f"replay={self.replay.position_count:,}/"
            f"{self.replay.max_positions:,} positions from {replay_age}",
            flush=True,
        )
        self.notifier.send_iteration(
            iteration,
            window.position_count,
            generation,
            training,
            self.device,
            cumulative_fresh_positions=self.cumulative_fresh_positions,
            replay_positions=self.replay.position_count,
            replay_capacity=self.replay.max_positions,
        )

        del trainer
        del model
        del window
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, iterations=None):
        """Run outer training iterations until the requested limit."""
        iteration = 1
        try:
            while iterations is None or iteration <= iterations:
                self.run_iteration(iteration)
                iteration += 1
        except KeyboardInterrupt:
            print("Stopped after the last completed checkpoint", flush=True)
