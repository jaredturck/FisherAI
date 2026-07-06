"""Define the FisherAI command-line interface."""

import argparse

import torch

from fisher_ai.benchmark import run_benchmark
from fisher_ai.pipeline import TrainingPipeline


def command_train(args):
    """Run the continuous or bounded training pipeline."""
    TrainingPipeline(args.config).run(iterations=args.iterations)


def command_benchmark(args):
    """Run the configured generation and training benchmark."""
    results, csv_path, markdown_path = run_benchmark(
        config_path=args.config,
        positions=args.positions,
    )
    print(
        f"Generation: "
        f"{results['generation']['positions_per_second']:.2f} positions/s"
    )
    print(
        f"Training: "
        f"{results['training']['positions_per_second']:.2f} positions/s"
    )
    print(f"CSV: {csv_path}")
    print(f"Summary: {markdown_path}")


def command_gui(_args):
    """Launch the interactive chess interface."""
    from gui.main import main as run_gui

    run_gui()


def build_parser():
    """Build the command-line parser and subcommands."""
    parser = argparse.ArgumentParser(prog="python -m fisher_ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="generate a window, train for three epochs, and repeat",
    )
    train_parser.add_argument("--config", default="fisher_config.json")
    train_parser.add_argument("--iterations", type=int)
    train_parser.set_defaults(handler=command_train)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark generation and training",
    )
    benchmark_parser.add_argument("--config", default="fisher_config.json")
    benchmark_parser.add_argument("--positions", type=int)
    benchmark_parser.set_defaults(handler=command_benchmark)

    gui_parser = subparsers.add_parser(
        "gui",
        help="play against the latest checkpoint",
    )
    gui_parser.set_defaults(handler=command_gui)
    return parser


def main():
    """Parse command-line arguments and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.handler(args)
