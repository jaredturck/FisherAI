from pathlib import Path

import torch

from fisher_ai.config import FisherConfig, save_config
from fisher_ai.dataset import InMemoryWindow
from fisher_ai.pipeline import PhasedTrainingPipeline


def test_pipeline_generates_then_trains_then_saves(monkeypatch, tmp_path):
    config = FisherConfig()
    config.runtime.checkpoint_dir = str(tmp_path / "checkpoints")
    config.runtime.device = "cpu"
    config.training.window_positions = 4
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    calls = []

    class Notifier:
        def send(self, *args, **kwargs):
            calls.append("notify")

        def close(self):
            calls.append("close")

    class Generator:
        def __init__(self, **kwargs):
            calls.append("generator_init")

        def generate(self, target_positions):
            calls.append(("generate", target_positions))
            window = InMemoryWindow(target_positions)
            window.position_count = target_positions
            return window, {
                "games": 2,
                "elapsed_seconds": 1.0,
                "positions_per_second": 4.0,
            }

    class Trainer:
        def __init__(self, model, config, device, checkpoint_manager):
            calls.append(("trainer_init", device))
            self.step = 0

        def load_checkpoint(self, path):
            calls.append(("load", Path(path).name))

        def train_window(self, window):
            calls.append(("train", window.position_count))
            return {
                "elapsed_seconds": 2.0,
                "positions_per_second": 2.0,
            }

        def save_checkpoint(self, extra):
            calls.append(("save", extra["iteration"]))
            return tmp_path / "checkpoints" / "fisher_ai_000000001.pt"

    monkeypatch.setattr("fisher_ai.pipeline.DiscordNotifier", Notifier)
    monkeypatch.setattr("fisher_ai.pipeline.WindowGenerator", Generator)
    monkeypatch.setattr("fisher_ai.pipeline.AlphaZeroTrainer", Trainer)

    pipeline = PhasedTrainingPipeline(config_path)
    result = pipeline.run_iteration()

    assert calls.index(("generate", 4)) < calls.index(("train", 4))
    assert calls.index(("train", 4)) < calls.index(("save", 1))
    assert result["iteration"] == 1
    payload = torch.load(
        pipeline.manager.latest_path(),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["step"] == 0
