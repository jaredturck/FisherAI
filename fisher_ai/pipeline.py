import gc
from pathlib import Path

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.generation import WindowGenerator
from fisher_ai.network import FisherNetwork
from fisher_ai.notifications import DiscordNotifier
from fisher_ai.trainer import AlphaZeroTrainer


class PhasedTrainingPipeline:
    def __init__(self, config_path="fisher_config.json"):
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.device = available_device(self.config.runtime.device)
        self.manager = CheckpointManager(self.config.runtime.checkpoint_dir)
        self.notifier = DiscordNotifier()
        self.ensure_checkpoint()

    def ensure_checkpoint(self):
        path = self.manager.latest_path()
        if path is None:
            path = self.manager.save(FisherNetwork(self.config.network), self.config, 0)
        return path

    def run_iteration(self):
        checkpoint_path = self.ensure_checkpoint()
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_extra = checkpoint_payload.get("extra", {})
        iteration = int(checkpoint_extra.get("iteration", 0)) + 1
        generated_games = int(checkpoint_extra.get("generated_games", 0))

        print(
            f"Iteration {iteration}: generating "
            f"{self.config.training.window_positions:,} positions",
            flush=True,
        )
        generator = WindowGenerator(
            config_path=self.config_path,
            checkpoint_path=checkpoint_path,
            generated_game_offset=generated_games,
        )
        window, generation_metrics = generator.generate(
            self.config.training.window_positions
        )
        del generator
        gc.collect()

        print(
            f"Iteration {iteration}: training on every position in the window",
            flush=True,
        )
        model = FisherNetwork(self.config.network)
        trainer = AlphaZeroTrainer(
            model,
            self.config,
            device=self.device,
            checkpoint_manager=self.manager,
        )
        trainer.load_checkpoint(checkpoint_path)
        training_metrics = trainer.train_window(window)
        generated_games += generation_metrics["games"]
        next_path = trainer.save_checkpoint(
            {
                "iteration": iteration,
                "generated_games": generated_games,
                "last_window_positions": window.position_count,
                "last_generation_seconds": generation_metrics["elapsed_seconds"],
                "last_training_seconds": training_metrics["elapsed_seconds"],
            }
        )

        print(
            f"Iteration {iteration} complete: "
            f"generation={generation_metrics['elapsed_seconds']:.1f}s "
            f"({generation_metrics['positions_per_second']:.1f} positions/s), "
            f"training={training_metrics['elapsed_seconds']:.1f}s "
            f"({training_metrics['positions_per_second']:.1f} positions/s), "
            f"checkpoint={next_path.name}",
            flush=True,
        )
        self.notifier.send(
            "Fisher AI iteration complete",
            [
                ("Iteration", f"{iteration:,}"),
                ("Window positions", f"{window.position_count:,}"),
                (
                    "Generation",
                    f"{generation_metrics['elapsed_seconds']:.1f}s • "
                    f"{generation_metrics['positions_per_second']:.1f} positions/s",
                    False,
                ),
                (
                    "Training",
                    f"{training_metrics['elapsed_seconds']:.1f}s • "
                    f"{training_metrics['positions_per_second']:.1f} positions/s",
                    False,
                ),
                ("Checkpoint", next_path.name, False),
            ],
            description="A complete generate-then-train iteration finished.",
            color="green",
            include_gpu_stats=True,
        )

        result = {
            "iteration": iteration,
            "checkpoint_path": next_path,
            "generation": generation_metrics,
            "training": training_metrics,
        }

        del trainer
        del model
        del window
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def run(self, iterations=None):
        configured_iterations = self.config.training.training_iterations
        if iterations is None:
            iterations = configured_iterations
        completed = 0

        try:
            while iterations == 0 or completed < iterations:
                self.run_iteration()
                completed += 1
        except KeyboardInterrupt:
            print("Stopping after the current completed checkpoint", flush=True)
        finally:
            self.notifier.close()

        return completed
