from fisher_ai.pipeline import (
    TrainingPipeline,
    scheduled_replay_ratio,
    scheduled_window_positions,
)


def test_training_schedule_uses_cumulative_fresh_positions():
    assert scheduled_window_positions(0) == 10_000
    assert scheduled_window_positions(99_999) == 10_000
    assert scheduled_window_positions(100_000) == 20_000
    assert scheduled_window_positions(300_000) == 30_000
    assert scheduled_window_positions(750_000) == 50_000
    assert scheduled_replay_ratio(0) == 0.5
    assert scheduled_replay_ratio(100_000) == 1.0


def test_pipeline_generates_trains_replays_saves_and_notifies(
    monkeypatch,
    tmp_path,
):
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

    class Replay:
        position_count = 0
        max_positions = 16
        oldest_iteration = 1

        def __init__(self, capacity):
            self.max_positions = capacity

        def append_window(self, window, iteration):
            calls.append(("replay", window.position_count, iteration))
            self.position_count += window.position_count

    class Trainer:
        def __init__(self, model, batch_size, device, checkpoint_manager):
            calls.append(("trainer_init", batch_size, device))

        def load_checkpoint(self, path):
            calls.append("load")

        def train_window(self, window, replay_window, replay_ratio):
            calls.append(("train", window.position_count, replay_ratio))
            return {
                "elapsed_seconds": 2.0,
                "positions_per_second": 6.0,
                "epochs": 3,
            }

        def save_checkpoint(self, cumulative_fresh_positions):
            calls.append(("save", cumulative_fresh_positions))

    class Notifier:
        def send_iteration(self, *args, **kwargs):
            calls.append("notify")

    monkeypatch.setattr("fisher_ai.pipeline.WindowGenerator", Generator)
    monkeypatch.setattr("fisher_ai.pipeline.ReplayWindow", Replay)
    monkeypatch.setattr("fisher_ai.pipeline.AlphaZeroTrainer", Trainer)
    monkeypatch.setattr("fisher_ai.pipeline.DiscordNotifier", Notifier)
    monkeypatch.setattr(
        "fisher_ai.pipeline.scheduled_window_positions",
        lambda cumulative: 4,
    )

    pipeline = TrainingPipeline(config_path)
    pipeline.run_iteration(1)

    assert calls.index(("generate", 4)) < calls.index(("train", 4, 0.5))
    assert calls.index(("train", 4, 0.5)) < calls.index(("replay", 4, 1))
    assert calls.index(("replay", 4, 1)) < calls.index(("save", 4))
    assert calls.index(("save", 4)) < calls.index("notify")
