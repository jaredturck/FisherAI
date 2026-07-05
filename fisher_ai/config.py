import json
from pathlib import Path

import torch


class NetworkConfig:
    def __init__(self, **values):
        self.input_planes = values.get("input_planes", 119)
        self.action_size = values.get("action_size", 4672)
        self.channels = values.get("channels", 128)
        self.residual_blocks = values.get("residual_blocks", 10)
        self.squeeze_excitation_channels = values.get("squeeze_excitation_channels", 32)
        self.policy_channels = values.get("policy_channels", 128)
        self.value_channels = values.get("value_channels", 8)
        self.value_hidden = values.get("value_hidden", 128)


class SearchConfig:
    def __init__(self, **values):
        self.simulations = values.get("simulations", 128)
        self.fast_simulations = values.get("fast_simulations", 32)
        self.full_search_fraction = values.get("full_search_fraction", 0.25)
        self.evaluation_simulations = values.get("evaluation_simulations", 256)
        self.parallel_searches = values.get("parallel_searches", 8)
        self.virtual_loss = values.get("virtual_loss", 1.0)
        self.c_puct = values.get("c_puct", 1.5)
        self.dirichlet_alpha = values.get("dirichlet_alpha", 0.3)
        self.dirichlet_fraction = values.get("dirichlet_fraction", 0.25)
        self.temperature = values.get("temperature", 1.0)
        self.temperature_plies = values.get("temperature_plies", 24)
        self.late_temperature = values.get("late_temperature", 0.1)
        self.max_game_plies = values.get("max_game_plies", 320)
        self.tree_capacity = values.get("tree_capacity", 131072)


class TrainingConfig:
    def __init__(self, **values):
        self.window_positions = values.get("window_positions", 50000)
        self.batch_size = values.get("batch_size", 2048)
        self.epochs_per_window = values.get("epochs_per_window", 1)
        self.training_iterations = values.get("training_iterations", 0)
        self.momentum = values.get("momentum", 0.9)
        self.weight_decay = values.get("weight_decay", 0.0001)
        self.learning_rates = values.get("learning_rates", [0.05, 0.01, 0.002, 0.0005])
        self.learning_rate_steps = values.get(
            "learning_rate_steps",
            [0, 100000, 250000, 500000],
        )
        self.resignation_enabled_after_games = values.get(
            "resignation_enabled_after_games",
            50000,
        )
        self.resignation_threshold = values.get("resignation_threshold", -0.9)
        self.resignation_consecutive_moves = values.get("resignation_consecutive_moves", 3)
        self.resignation_audit_fraction = values.get("resignation_audit_fraction", 0.1)


class RuntimeConfig:
    def __init__(self, **values):
        self.checkpoint_dir = values.get("checkpoint_dir", "checkpoints")
        self.device = values.get("device", "cuda:0")
        self.actor_processes = values.get("actor_processes", 24)
        self.games_per_actor = values.get("games_per_actor", 6)
        self.pin_actor_cpus = values.get("pin_actor_cpus", True)
        self.inference_batch_size = values.get("inference_batch_size", 512)
        self.inference_max_batch_size = values.get("inference_max_batch_size", 1024)
        self.inference_batch_wait_ms = values.get("inference_batch_wait_ms", 2.0)
        self.inference_queue_size = values.get("inference_queue_size", 4096)
        self.game_queue_size = values.get("game_queue_size", 256)
        self.max_legal_actions = values.get("max_legal_actions", 256)
        self.status_interval_seconds = values.get("status_interval_seconds", 10.0)
        self.channels_last = values.get("channels_last", True)
        self.benchmark_dir = values.get("benchmark_dir", "benchmarks")
        self.benchmark_positions = values.get("benchmark_positions", 5000)
        self.seed = values.get("seed", 7)


class FisherConfig:
    def __init__(self, network=None, search=None, training=None, runtime=None):
        self.network = network or NetworkConfig()
        self.search = search or SearchConfig()
        self.training = training or TrainingConfig()
        self.runtime = runtime or RuntimeConfig()


def available_device(preferred):
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"

    if preferred.startswith("cuda"):
        index = int(preferred.split(":", 1)[1]) if ":" in preferred else 0
        if index >= torch.cuda.device_count():
            return "cuda:0"

    return preferred


def section_to_dict(section):
    return vars(section).copy()


def config_to_dict(config):
    return {
        "network": section_to_dict(config.network),
        "search": section_to_dict(config.search),
        "training": section_to_dict(config.training),
        "runtime": section_to_dict(config.runtime),
    }


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2) + "\n")


def load_config(path=None):
    if not path:
        return FisherConfig()

    data = json.loads(Path(path).read_text())
    return FisherConfig(
        network=NetworkConfig(**data.get("network", {})),
        search=SearchConfig(**data.get("search", {})),
        training=TrainingConfig(**data.get("training", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
    )
