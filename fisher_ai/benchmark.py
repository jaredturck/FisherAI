"""Benchmark FisherAI self-play generation and model training."""

import csv
import gc
from datetime import datetime
from pathlib import Path

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.generation import WindowGenerator
from fisher_ai.network import FisherNetwork
from fisher_ai.trainer import AlphaZeroTrainer

DEFAULT_BENCHMARK_POSITIONS = 5000
BENCHMARK_DIR = Path("benchmarks")


def write_reports(output_dir, window_positions, generation, training):
    """Write the current benchmark CSV and Markdown reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    markdown_path = output_dir / "benchmark_summary.md"

    rows = [
        {
            "phase": "generation",
            "positions": window_positions,
            "seconds": generation["elapsed_seconds"],
            "positions_per_second": generation["positions_per_second"],
            "evaluations_per_second": generation["evaluations_per_second"],
            "optimizer_steps": "",
        },
        {
            "phase": "training",
            "positions": training["positions"],
            "seconds": training["elapsed_seconds"],
            "positions_per_second": training["positions_per_second"],
            "evaluations_per_second": "",
            "optimizer_steps": training["optimizer_steps"],
        },
    ]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    total_seconds = generation["elapsed_seconds"] + training["elapsed_seconds"]
    generation_percent = generation["elapsed_seconds"] / total_seconds * 100
    training_percent = training["elapsed_seconds"] / total_seconds * 100
    lines = [
        "# Fisher AI Benchmark",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Window positions: {window_positions:,}",
        f"Training epochs: {training['epochs']}",
        "",
        "| Phase | Seconds | Positions/s | Share |",
        "|---|---:|---:|---:|",
        (
            f"| Generation | {generation['elapsed_seconds']:.3f} | "
            f"{generation['positions_per_second']:.2f} | "
            f"{generation_percent:.1f}% |"
        ),
        (
            f"| Training | {training['elapsed_seconds']:.3f} | "
            f"{training['positions_per_second']:.2f} | "
            f"{training_percent:.1f}% |"
        ),
        f"| Total | {total_seconds:.3f} | | 100.0% |",
        "",
        f"Games completed: {generation['games']:,}",
        f"Neural evaluations/s: {generation['evaluations_per_second']:.2f}",
        (
            "Average inference batch: "
            f"{generation['average_inference_batch']:.2f}"
        ),
        f"Maximum inference batch: {generation['max_batch']}",
        f"Optimizer steps: {training['optimizer_steps']:,}",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    return csv_path, markdown_path


def run_benchmark(
    config_path="fisher_config.json",
    positions=None,
    output_dir=BENCHMARK_DIR,
):
    """Run one fixed generation and training benchmark."""
    config = load_config(config_path)
    positions = positions or DEFAULT_BENCHMARK_POSITIONS
    manager = CheckpointManager()
    checkpoint_path = manager.ensure(FisherNetwork())

    generator = WindowGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    window, generation = generator.generate(positions)
    del generator
    gc.collect()

    model = FisherNetwork()
    trainer = AlphaZeroTrainer(
        model,
        config.batch_size,
        device=available_device(config.device),
        checkpoint_manager=manager,
    )
    trainer.load_checkpoint(checkpoint_path)
    training = trainer.train_window(window)
    csv_path, markdown_path = write_reports(
        output_dir,
        window.position_count,
        generation,
        training,
    )

    del trainer
    del model
    del window
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        {
            "generation": generation,
            "training": training,
        },
        csv_path,
        markdown_path,
    )
