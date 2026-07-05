import gc
from pathlib import Path

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.generation import WindowGenerator
from fisher_ai.network import FisherNetwork
from fisher_ai.notifications import DiscordNotifier
from fisher_ai.trainer import AlphaZeroTrainer


class TrainingPipeline:
    def __init__(self, config_path="fisher_config.json"):
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.device = available_device(self.config.device)
        self.manager = CheckpointManager()
        self.notifier = DiscordNotifier()
        self.manager.ensure(FisherNetwork())

    def run_iteration(self, iteration):
        checkpoint_path = self.manager.latest_path()
        print(
            f"Iteration {iteration}: generating "
            f"{self.config.window_positions:,} positions",
            flush=True,
        )
        generator = WindowGenerator(
            config_path=self.config_path,
            checkpoint_path=checkpoint_path,
        )
        window, generation = generator.generate(self.config.window_positions)
        del generator
        gc.collect()

        print(
            f"Iteration {iteration}: training for three epochs",
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
        training = trainer.train_window(window)
        trainer.save_checkpoint()

        print(
            f"Iteration {iteration} complete: "
            f"generation={generation['elapsed_seconds']:.1f}s "
            f"({generation['positions_per_second']:.1f} positions/s), "
            f"training={training['elapsed_seconds']:.1f}s "
            f"({training['positions_per_second']:.1f} positions/s)",
            flush=True,
        )
        self.notifier.send_iteration(
            iteration,
            window.position_count,
            generation,
            training,
            self.device,
        )

        del trainer
        del model
        del window
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, iterations=None):
        iteration = 1
        try:
            while iterations is None or iteration <= iterations:
                self.run_iteration(iteration)
                iteration += 1
        except KeyboardInterrupt:
            print("Stopped after the last completed checkpoint", flush=True)
