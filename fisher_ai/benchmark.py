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


def ensure_checkpoint(config):
    manager = CheckpointManager(config.runtime.checkpoint_dir)
    path = manager.latest_path()
    if path is None:
        path = manager.save(FisherNetwork(config.network), config, 0)
    return manager, path


def write_reports(output_dir, positions, generation, training):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    markdown_path = output_dir / "benchmark_summary.md"

    rows = [
        {
            "phase": "generation",
            "positions": positions,
            "elapsed_seconds": generation["elapsed_seconds"],
            "positions_per_second": generation["positions_per_second"],
            "evaluations_per_second": generation["evaluations_per_second"],
            "average_inference_batch": generation["average_inference_batch"],
            "maximum_inference_batch": generation["max_batch"],
            "optimizer_steps": "",
            "loss": "",
        },
        {
            "phase": "training",
            "positions": positions,
            "elapsed_seconds": training["elapsed_seconds"],
            "positions_per_second": training["positions_per_second"],
            "evaluations_per_second": "",
            "average_inference_batch": "",
            "maximum_inference_batch": "",
            "optimizer_steps": training["optimizer_steps"],
            "loss": training["loss"],
        },
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_seconds = generation["elapsed_seconds"] + training["elapsed_seconds"]
    lines = [
        "# Fisher AI Phased Benchmark",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Window positions: {positions:,}",
        "",
        "| Phase | Seconds | Positions/s | Other |",
        "|---|---:|---:|---:|",
        f"| Generation | {generation['elapsed_seconds']:.3f} | "
        f"{generation['positions_per_second']:.2f} | "
        f"{generation['evaluations_per_second']:.2f} evals/s |",
        f"| Training | {training['elapsed_seconds']:.3f} | "
        f"{training['positions_per_second']:.2f} | "
        f"{training['optimizer_steps']} optimizer steps |",
        f"| Total | {total_seconds:.3f} | {positions / max(total_seconds, 1e-6):.2f} | |",
        "",
        f"Peak compact window memory: {generation['memory_bytes'] / (1024 ** 2):.2f} MiB",
        f"Average inference batch: {generation['average_inference_batch']:.2f}",
        f"Maximum inference batch: {generation['max_batch']}",
        f"Final benchmark loss: {training['loss']:.6f}",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    return csv_path, markdown_path


def run_benchmark(config_path="fisher_config.json", positions=None, output_dir=None):
    config = load_config(config_path)
    positions = positions or config.runtime.benchmark_positions
    manager, checkpoint_path = ensure_checkpoint(config)

    generator = WindowGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    window, generation_metrics = generator.generate(positions)
    del generator
    gc.collect()

    model = FisherNetwork(config.network)
    trainer = AlphaZeroTrainer(
        model,
        config,
        device=available_device(config.runtime.device),
        checkpoint_manager=manager,
    )
    trainer.load_checkpoint(checkpoint_path)
    training_metrics = trainer.train_window(window, epochs=1)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.runtime.benchmark_dir) / timestamp

    csv_path, markdown_path = write_reports(
        output_dir,
        window.position_count,
        generation_metrics,
        training_metrics,
    )

    del trainer
    del model
    del window
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "generation": generation_metrics,
        "training": training_metrics,
    }, csv_path, markdown_path
