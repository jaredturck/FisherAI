import argparse

import torch

from fisher_ai.benchmark import run_benchmark
from fisher_ai.pipeline import PhasedTrainingPipeline


def command_workstation(args):
    pipeline = PhasedTrainingPipeline(args.config)
    pipeline.run(iterations=args.iterations)


def command_benchmark(args):
    results, csv_path, markdown_path = run_benchmark(
        config_path=args.config,
        positions=args.positions,
    )
    print(
        f"Generation: {results['generation']['positions_per_second']:.2f} positions/s"
    )
    print(f"Training: {results['training']['positions_per_second']:.2f} positions/s")
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
        help="run phased self-play generation and full-window training",
    )
    workstation_parser.add_argument("--config", default="fisher_config.json")
    workstation_parser.add_argument("--iterations", type=int)
    workstation_parser.set_defaults(handler=command_workstation)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark generation and training as separate phases",
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
    parser = build_parser()
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.handler(args)
