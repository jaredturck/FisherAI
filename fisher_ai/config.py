"""Load and validate FisherAI runtime configuration."""

import json
from pathlib import Path

import torch


class FisherConfig:
    """Hold the runtime settings used by generation and training."""

    def __init__(self, **values):
        self.device = values.get("device", "cuda:0")
        self.actor_processes = values.get("actor_processes", 24)
        self.games_per_actor = values.get("games_per_actor", 6)
        self.inference_batch_size = values.get("inference_batch_size", 512)
        self.inference_max_batch_size = values.get(
            "inference_max_batch_size",
            1024,
        )
        self.inference_batch_wait_ms = values.get(
            "inference_batch_wait_ms",
            2.0,
        )
        self.simulations = values.get("simulations", 128)
        self.parallel_searches = values.get("parallel_searches", 8)
        self.batch_size = values.get("batch_size", 2048)


def available_device(preferred):
    """Resolve the requested compute device against available hardware."""
    if not preferred.startswith("cuda"):
        return preferred
    if not torch.cuda.is_available():
        return "cpu"

    index = int(preferred.split(":", 1)[1]) if ":" in preferred else 0
    if index >= torch.cuda.device_count():
        return "cuda:0"
    return preferred


def load_config(path="fisher_config.json"):
    """Load FisherAI settings from a JSON file."""
    return FisherConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
