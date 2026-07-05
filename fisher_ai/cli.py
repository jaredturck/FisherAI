import argparse
import os
import subprocess
import sys

import torch

from fisher_ai.benchmark import run_benchmark
from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import load_config
from fisher_ai.distributed import DistributedSelfPlayPool
from fisher_ai.network import FisherNetwork
from fisher_ai.notifications import DiscordNotifier


def build_model(config):
    return FisherNetwork(config.network)


def build_checkpoint_manager(config):
    return CheckpointManager(config.runtime.checkpoint_dir)


def ensure_checkpoint(config, model, manager):
    path = manager.latest_path()
    if path is None:
        path = manager.save(model, config, 0)
    return path


def command_workstation(args):
    config = load_config(args.config)
    model = build_model(config)
    manager = build_checkpoint_manager(config)
    ensure_checkpoint(config, model, manager)

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    learner_command = [
        sys.executable,
        "-m",
        "fisher_ai.learner_worker",
        args.config,
    ]
    pool = DistributedSelfPlayPool(config_path=args.config)
    notifier = DiscordNotifier()
    learner_process = None

    try:
        pool.start()
        print("Starting continuous learning on the configured learner GPU", flush=True)
        learner_process = subprocess.Popen(learner_command)
        notifier.send(
            "Fisher AI training started",
            [
                ("Actors", f"{config.runtime.actor_processes:,}"),
                (
                    "Active games",
                    f"{config.runtime.actor_processes * config.runtime.games_per_actor:,}",
                ),
                ("Learner", config.runtime.learner_device),
                ("Checkpoint", manager.latest_path().name, False),
            ],
            description="Self-play and learning are running.",
            color="blue",
            include_gpu_stats=True,
        )
        pool.monitor(external_processes=[learner_process])
    except KeyboardInterrupt:
        print("Stopping Fisher AI workstation training", flush=True)
        latest_path = manager.latest_path()
        notifier.send(
            "Fisher AI training stopped",
            [("Latest checkpoint", latest_path.name if latest_path else "Unavailable")],
            description="The workstation training process was stopped.",
            color="orange",
        )
    except Exception as error:
        latest_path = manager.latest_path()
        notifier.send(
            "Fisher AI training failed",
            [
                ("Error", str(error), False),
                ("Latest checkpoint", latest_path.name if latest_path else "Unavailable"),
            ],
            description="The workstation training process exited unexpectedly.",
            color="red",
            include_gpu_stats=True,
        )
        raise
    finally:
        pool.stop()
        if learner_process is not None and learner_process.poll() is None:
            learner_process.terminate()
            learner_process.wait()
        notifier.close()


def command_benchmark(args):
    results, csv_path, markdown_path = run_benchmark(config_path=args.config)
    sweep_count = sum(result["run_stage"] == "sweep" for result in results)
    confirmation_count = len(results) - sweep_count
    print(
        f"Completed {sweep_count} sweep configurations and "
        f"{confirmation_count} confirmation runs"
    )
    print(f"CSV: {csv_path}")
    print(f"Summary: {markdown_path}")


def command_gui(args):
    from gui.main import main

    main()


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m fisher_ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    workstation_parser = subparsers.add_parser(
        "workstation",
        help="run continuous self-play and learning",
    )
    workstation_parser.add_argument("--config", default="fisher_config.json")
    workstation_parser.set_defaults(handler=command_workstation)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark self-play throughput",
    )
    benchmark_parser.add_argument("--config", default="fisher_config.json")
    benchmark_parser.set_defaults(handler=command_benchmark)

    gui_parser = subparsers.add_parser(
        "gui",
        help="play against the latest checkpoint",
    )
    gui_parser.set_defaults(handler=command_gui)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.handler(args)
