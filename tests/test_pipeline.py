from fisher_ai.pipeline import TrainingPipeline


def test_pipeline_generates_trains_saves_and_notifies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "device": "cpu",
          "actor_processes": 1,
          "games_per_actor": 1,
          "inference_batch_size": 1,
          "inference_max_batch_size": 1,
          "inference_batch_wait_ms": 1.0,
          "simulations": 1,
          "parallel_searches": 1,
          "window_positions": 4,
          "batch_size": 2
        }
        """
    )
    calls = []

    class Window:
        position_count = 4

    class Generator:
        def __init__(self, **kwargs):
            calls.append("generator_init")

        def generate(self, target_positions):
            calls.append(("generate", target_positions))
            return Window(), {
                "elapsed_seconds": 1.0,
                "positions_per_second": 4.0,
            }

    class Trainer:
        def __init__(self, model, batch_size, device, checkpoint_manager):
            calls.append(("trainer_init", batch_size, device))

        def load_checkpoint(self, path):
            calls.append("load")

        def train_window(self, window):
            calls.append(("train", window.position_count))
            return {
                "elapsed_seconds": 2.0,
                "positions_per_second": 6.0,
                "epochs": 3,
            }

        def save_checkpoint(self):
            calls.append("save")

    class Notifier:
        def send_iteration(self, *args):
            calls.append("notify")

    monkeypatch.setattr("fisher_ai.pipeline.WindowGenerator", Generator)
    monkeypatch.setattr("fisher_ai.pipeline.AlphaZeroTrainer", Trainer)
    monkeypatch.setattr("fisher_ai.pipeline.DiscordNotifier", Notifier)

    pipeline = TrainingPipeline(config_path)
    pipeline.run_iteration(1)

    assert calls.index(("generate", 4)) < calls.index(("train", 4))
    assert calls.index(("train", 4)) < calls.index("save")
    assert calls.index("save") < calls.index("notify")
