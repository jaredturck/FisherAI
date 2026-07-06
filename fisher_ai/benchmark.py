"""Benchmark FisherAI without instrumenting production training hot paths."""

import csv
import gc
import queue
import resource
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fisher_ai.benchmark_metrics import (
    CSV_FIELDS,
    distribution_row,
    metric_row,
)
from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.dataset import InMemoryWindow, materialize_mixed_batch
from fisher_ai.generation import STATUS_INTERVAL_SECONDS, WindowGenerator
from fisher_ai.network import FisherNetwork
from fisher_ai.trainer import EPOCHS_PER_WINDOW, AlphaZeroTrainer

DEFAULT_BENCHMARK_POSITIONS = 5000
BENCHMARK_DIR = Path("benchmarks")


def core_metric_rows(window_positions, generation, training):
    """Build stable high-level rows shared by every benchmark report."""
    generation_seconds = generation["elapsed_seconds"]
    training_seconds = training["elapsed_seconds"]
    total_seconds = generation_seconds + training_seconds
    rows = [
        metric_row(
            "phase",
            "pipeline",
            "generation",
            "elapsed_seconds",
            generation_seconds,
            unit="seconds",
            share_percent=generation_seconds / max(total_seconds, 1e-9) * 100,
        ),
        metric_row(
            "phase",
            "pipeline",
            "training",
            "elapsed_seconds",
            training_seconds,
            unit="seconds",
            share_percent=training_seconds / max(total_seconds, 1e-9) * 100,
        ),
        metric_row(
            "phase",
            "pipeline",
            "generate_and_train",
            "elapsed_seconds",
            total_seconds,
            unit="seconds",
            share_percent=100.0,
        ),
        metric_row(
            "throughput",
            "generation",
            "window",
            "positions_per_second",
            generation["positions_per_second"],
            unit="positions/second",
        ),
        metric_row(
            "throughput",
            "generation",
            "inference",
            "evaluations_per_second",
            generation["evaluations_per_second"],
            unit="evaluations/second",
        ),
        metric_row(
            "counter",
            "generation",
            "window",
            "requested_positions",
            window_positions,
            unit="positions",
        ),
        metric_row(
            "counter",
            "generation",
            "window",
            "completed_games",
            generation["games"],
            unit="games",
        ),
        metric_row(
            "counter",
            "generation",
            "inference",
            "evaluations",
            generation["evaluations"],
            unit="evaluations",
        ),
        metric_row(
            "counter",
            "generation",
            "inference",
            "batches",
            generation["batches"],
            unit="batches",
        ),
        metric_row(
            "distribution",
            "generation",
            "inference",
            "average_batch_size",
            generation["average_inference_batch"],
            unit="positions",
        ),
        metric_row(
            "distribution",
            "generation",
            "inference",
            "maximum_batch_size",
            generation["max_batch"],
            unit="positions",
        ),
        metric_row(
            "throughput",
            "training",
            "window",
            "positions_per_second",
            training["positions_per_second"],
            unit="positions/second",
        ),
        metric_row(
            "counter",
            "training",
            "window",
            "positions",
            training["positions"],
            unit="positions",
        ),
        metric_row(
            "counter",
            "training",
            "window",
            "epochs",
            training["epochs"],
            unit="epochs",
        ),
        metric_row(
            "counter",
            "training",
            "optimizer",
            "steps",
            training["optimizer_steps"],
            unit="steps",
        ),
        metric_row(
            "model",
            "training",
            "loss",
            "total",
            training["loss"],
        ),
        metric_row(
            "model",
            "training",
            "loss",
            "policy",
            training["policy_loss"],
        ),
        metric_row(
            "model",
            "training",
            "loss",
            "value",
            training["value_loss"],
        ),
        metric_row(
            "configuration",
            "training",
            "optimizer",
            "learning_rate",
            training["learning_rate"],
        ),
        metric_row(
            "counter",
            "training",
            "window",
            "fresh_positions_per_epoch",
            training["fresh_positions_per_epoch"],
            unit="positions",
        ),
        metric_row(
            "counter",
            "training",
            "window",
            "replay_positions_per_epoch",
            training["replay_positions_per_epoch"],
            unit="positions",
        ),
    ]
    return rows


def format_number(value, digits=3):
    """Format numeric report values while preserving strings."""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def measured_timing_rows(rows):
    """Return timing rows suitable for the Markdown detail table."""
    timings = []
    for row in rows:
        if row["unit"] != "seconds":
            continue
        if row["metric"] not in (
            "elapsed_seconds",
            "wait_seconds",
            "run_seconds",
            "batch_seconds",
            "round_seconds",
            "tensor_batch",
            "zero_grad",
            "forward_and_loss",
            "backward",
            "optimizer_step",
            "host_to_device",
            "channels_last_conversion",
            "device_to_host",
            "numpy_preallocated_copy",
            "stem",
            "residual_tower",
            "policy_head",
            "value_head",
        ):
            continue
        if not isinstance(row["value"], (int, float, np.number)):
            continue
        timings.append(row)
    return sorted(timings, key=lambda row: float(row["value"]), reverse=True)


def write_reports(
    output_dir,
    window_positions,
    generation,
    training,
    extra_rows=None,
):
    """Write exhaustive long-format CSV and concise Markdown summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    markdown_path = output_dir / "benchmark_summary.md"
    rows = core_metric_rows(window_positions, generation, training)
    rows.extend(extra_rows or [])

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total_seconds = generation["elapsed_seconds"] + training["elapsed_seconds"]
    generation_percent = generation["elapsed_seconds"] / total_seconds * 100
    training_percent = training["elapsed_seconds"] / total_seconds * 100
    lines = [
        "# Fisher AI Benchmark",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"CSV metric rows: {len(rows):,}",
        "",
        "## End-to-end pipeline",
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
        f"Window positions requested: {window_positions:,}",
        f"Games completed: {generation['games']:,}",
        f"Neural evaluations: {generation['evaluations']:,}",
        f"Neural evaluations/s: {generation['evaluations_per_second']:.2f}",
        (
            "Average inference batch: "
            f"{generation['average_inference_batch']:.2f}"
        ),
        f"Maximum inference batch: {generation['max_batch']}",
        f"Training epochs: {training['epochs']}",
        f"Optimizer steps: {training['optimizer_steps']:,}",
    ]

    blocking_rows = [
        row
        for row in rows
        if row["category"] == "blocking" and row["unit"] == "seconds"
    ]
    if blocking_rows:
        lines.extend(
            [
                "",
                "## Blocking and waiting",
                "",
                "| Scope | Operation | Total seconds | Calls | p95 | Max |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in blocking_rows:
            lines.append(
                f"| {row['scope']} | {row['component']}."
                f"{row['metric']} | {format_number(row['value'])} | "
                f"{format_number(row['count'])} | "
                f"{format_number(row['p95'], 6)} | "
                f"{format_number(row['max'], 6)} |"
            )

    timings = measured_timing_rows(rows)
    if timings:
        lines.extend(
            [
                "",
                "## Largest measured timings",
                "",
                "These timings can overlap and component diagnostics are "
                "separate from end-to-end throughput.",
                "",
                "| Scope | Component | Metric | Seconds |",
                "|---|---|---|---:|",
            ]
        )
        for row in timings[:20]:
            lines.append(
                f"| {row['scope']} | {row['component']} | "
                f"{row['metric']} | {format_number(row['value'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Production training modules are not instrumented. End-to-end "
            "rows use benchmark-only orchestration around existing public "
            "operations. Component rows are isolated diagnostics and are "
            "excluded from generation and training throughput.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def generation_benchmark_rows(timings, elapsed):
    """Convert benchmark-owned generation timings into detailed rows."""
    rows = [
        metric_row(
            "phase",
            "generation",
            "startup",
            "elapsed_seconds",
            timings["startup"],
            unit="seconds",
            share_percent=timings["startup"] / max(elapsed, 1e-9) * 100,
        ),
        metric_row(
            "phase",
            "generation",
            "shutdown",
            "elapsed_seconds",
            timings["shutdown"],
            unit="seconds",
            share_percent=timings["shutdown"] / max(elapsed, 1e-9) * 100,
        ),
        metric_row(
            "counter",
            "generation",
            "game_queue",
            "timeouts",
            timings["game_queue_timeouts"],
            unit="count",
        ),
    ]
    rows.extend(
        [
            distribution_row(
                "blocking",
                "generation",
                "game_queue",
                "wait_seconds",
                timings["game_queue_wait"],
                "seconds",
                share_base=elapsed,
                notes="Parent waiting for completed games",
            ),
            distribution_row(
                "blocking",
                "generation",
                "game_queue_timeout",
                "wait_seconds",
                timings["game_queue_timeout_wait"],
                "seconds",
                share_base=elapsed,
            ),
            distribution_row(
                "blocking",
                "generation",
                "game_queue_success",
                "wait_seconds",
                timings["game_queue_success_wait"],
                "seconds",
                share_base=elapsed,
            ),
            distribution_row(
                "compute",
                "generation",
                "shared_game_append",
                "elapsed_seconds",
                timings["game_append"],
                "seconds",
                share_base=elapsed,
            ),
            distribution_row(
                "blocking",
                "generation",
                "game_ack_queue",
                "wait_seconds",
                timings["game_ack"],
                "seconds",
                share_base=elapsed,
                notes="Usually non-blocking; measured around queue put",
            ),
            distribution_row(
                "compute",
                "generation",
                "child_failure_poll",
                "elapsed_seconds",
                timings["failure_poll"],
                "seconds",
                share_base=elapsed,
            ),
        ]
    )
    return rows


def run_generation_benchmark(generator, target_positions, timeout=None):
    """Run the real generator with benchmark-owned parent-loop timings."""
    window = InMemoryWindow(target_positions)
    started = time.monotonic()
    last_status = started
    previous_positions = 0
    previous_evaluations = 0
    timings = {
        "startup": 0.0,
        "shutdown": 0.0,
        "game_queue_wait": [],
        "game_queue_success_wait": [],
        "game_queue_timeout_wait": [],
        "game_queue_timeouts": 0,
        "game_append": [],
        "game_ack": [],
        "failure_poll": [],
    }

    try:
        startup_started = time.monotonic()
        generator.start()
        timings["startup"] = time.monotonic() - startup_started
        print(
            f"Generating {target_positions:,} positions with "
            f"{generator.actor_count} actors, "
            f"{generator.active_game_count} active games, "
            f"and {generator.device}",
            flush=True,
        )

        while not window.full:
            poll_started = time.perf_counter()
            failure = generator.process_failure()
            timings["failure_poll"].append(time.perf_counter() - poll_started)
            if failure:
                raise RuntimeError(failure)
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(
                    "Window generation did not finish before timeout"
                )

            wait_started = time.perf_counter()
            try:
                actor_id, slot_id, count = generator.game_queue.get(
                    timeout=0.5
                )
            except queue.Empty:
                wait_seconds = time.perf_counter() - wait_started
                timings["game_queue_wait"].append(wait_seconds)
                timings["game_queue_timeout_wait"].append(wait_seconds)
                timings["game_queue_timeouts"] += 1
                continue
            wait_seconds = time.perf_counter() - wait_started
            timings["game_queue_wait"].append(wait_seconds)
            timings["game_queue_success_wait"].append(wait_seconds)

            append_started = time.perf_counter()
            generator.game_shared.append_to_window(
                window,
                actor_id,
                slot_id,
                count,
            )
            timings["game_append"].append(time.perf_counter() - append_started)

            ack_started = time.perf_counter()
            generator.game_ack_queues[actor_id][slot_id].put(None)
            timings["game_ack"].append(time.perf_counter() - ack_started)

            now = time.monotonic()
            if now - last_status >= STATUS_INTERVAL_SECONDS:
                current = generator.metric_snapshot()
                interval = max(now - last_status, 1e-6)
                positions_per_second = (
                    window.position_count - previous_positions
                ) / interval
                evaluations_per_second = (
                    current["evaluations"] - previous_evaluations
                ) / interval
                print(
                    f"window={window.position_count:,}/"
                    f"{target_positions:,} "
                    f"positions/s={positions_per_second:.1f} "
                    f"evals/s={evaluations_per_second:.1f} "
                    f"games={window.game_count:,}",
                    flush=True,
                )
                previous_positions = window.position_count
                previous_evaluations = current["evaluations"]
                last_status = now
    finally:
        shutdown_started = time.monotonic()
        generator.stop(drain_window=window)
        timings["shutdown"] = time.monotonic() - shutdown_started

    elapsed = time.monotonic() - started
    metrics = generator.metric_snapshot()
    metrics.update(
        {
            "elapsed_seconds": elapsed,
            "games": window.game_count,
            "positions_per_second": (
                window.position_count / max(elapsed, 1e-6)
            ),
            "evaluations_per_second": (
                metrics["evaluations"] / max(elapsed, 1e-6)
            ),
            "average_inference_batch": (
                metrics["evaluations"] / max(metrics["batches"], 1)
            ),
        }
    )
    rows = generation_benchmark_rows(timings, elapsed)
    generator.release_ipc()
    return window, metrics, rows


def training_benchmark_rows(timings, elapsed):
    """Convert benchmark-owned training timings into detailed rows."""
    rows = [
        distribution_row(
            "compute",
            "training",
            "epoch_setup",
            "elapsed_seconds",
            timings["epoch_setup"],
            "seconds",
            share_base=elapsed,
        ),
        distribution_row(
            "compute",
            "training",
            "batch_materialization",
            "elapsed_seconds",
            timings["batch_materialization"],
            "seconds",
            share_base=elapsed,
        ),
        distribution_row(
            "compute",
            "training",
            "optimizer_batch",
            "elapsed_seconds",
            timings["train_batch"],
            "seconds",
            share_base=elapsed,
        ),
        distribution_row(
            "distribution",
            "training",
            "batch",
            "positions_per_batch",
            timings["batch_sizes"],
            "positions",
        ),
        distribution_row(
            "distribution",
            "training",
            "batch",
            "legal_action_width",
            timings["legal_widths"],
            "actions",
        ),
        distribution_row(
            "memory",
            "training",
            "batch",
            "materialized_bytes",
            timings["batch_bytes"],
            "bytes",
        ),
    ]
    return rows


def run_training_benchmark(trainer, window):
    """Run existing trainer operations with benchmark-owned timing."""
    started = time.perf_counter()
    loss_total = 0.0
    policy_loss_total = 0.0
    value_loss_total = 0.0
    optimizer_steps = 0
    learning_rate = trainer.learning_rate()
    fresh_positions = window.position_count
    replay_per_epoch = 0
    timings = {
        "epoch_setup": [],
        "batch_materialization": [],
        "train_batch": [],
        "batch_sizes": [],
        "legal_widths": [],
        "batch_bytes": [],
    }

    for _ in range(EPOCHS_PER_WINDOW):
        setup_started = time.perf_counter()
        fresh_indices = np.arange(fresh_positions, dtype=np.int64)
        replay_indices = np.empty(0, dtype=np.int64)
        source_flags = np.concatenate(
            (
                np.zeros(fresh_positions, dtype=np.bool_),
                np.ones(len(replay_indices), dtype=np.bool_),
            )
        )
        position_indices = np.concatenate((fresh_indices, replay_indices))
        order = trainer.rng.permutation(len(position_indices))
        source_flags = source_flags[order]
        position_indices = position_indices[order]
        timings["epoch_setup"].append(time.perf_counter() - setup_started)

        for start in range(0, len(order), trainer.batch_size):
            end = start + trainer.batch_size
            materialize_started = time.perf_counter()
            batch = materialize_mixed_batch(
                window,
                None,
                source_flags[start:end],
                position_indices[start:end],
            )
            timings["batch_materialization"].append(
                time.perf_counter() - materialize_started
            )
            timings["batch_sizes"].append(len(batch[0]))
            timings["legal_widths"].append(batch[1].shape[1])
            timings["batch_bytes"].append(sum(array.nbytes for array in batch))

            batch_started = time.perf_counter()
            loss, policy_loss, value_loss, learning_rate = trainer.train_batch(
                batch
            )
            timings["train_batch"].append(time.perf_counter() - batch_started)
            loss_total += loss
            policy_loss_total += policy_loss
            value_loss_total += value_loss
            optimizer_steps += 1

    elapsed = time.perf_counter() - started
    positions_per_epoch = fresh_positions + replay_per_epoch
    positions = positions_per_epoch * EPOCHS_PER_WINDOW
    metrics = {
        "elapsed_seconds": elapsed,
        "epochs": EPOCHS_PER_WINDOW,
        "positions": positions,
        "positions_per_second": positions / max(elapsed, 1e-6),
        "optimizer_steps": optimizer_steps,
        "loss": loss_total / optimizer_steps,
        "policy_loss": policy_loss_total / optimizer_steps,
        "value_loss": value_loss_total / optimizer_steps,
        "learning_rate": learning_rate,
        "fresh_positions_per_epoch": fresh_positions,
        "replay_positions_per_epoch": replay_per_epoch,
    }
    rows = training_benchmark_rows(timings, elapsed)
    return metrics, rows


def run_benchmark(
    config_path="fisher_config.json",
    positions=None,
    output_dir=BENCHMARK_DIR,
):
    """Run end-to-end throughput plus isolated component diagnostics."""
    from fisher_ai.benchmark_components import (
        environment_rows,
        nvidia_smi_rows,
        process_usage_rows,
        process_usage_snapshot,
        run_component_benchmarks,
        window_statistics_rows,
    )

    benchmark_started = time.perf_counter()
    rows = []
    config = load_config(config_path)
    positions = positions or DEFAULT_BENCHMARK_POSITIONS
    device = available_device(config.device)
    rows.extend(environment_rows(config, device))
    rows.extend(nvidia_smi_rows("before_benchmark"))

    manager_started = time.perf_counter()
    manager = CheckpointManager()
    model = FisherNetwork()
    checkpoint_path = manager.ensure(model)
    rows.append(
        metric_row(
            "phase",
            "setup",
            "checkpoint_ensure",
            "elapsed_seconds",
            time.perf_counter() - manager_started,
            unit="seconds",
        )
    )
    del model

    generator_construct_started = time.perf_counter()
    generator = WindowGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    rows.append(
        metric_row(
            "phase",
            "setup",
            "generator_construct",
            "elapsed_seconds",
            time.perf_counter() - generator_construct_started,
            unit="seconds",
        )
    )
    request_capacity = generator.request_capacity

    self_before = process_usage_snapshot(resource.RUSAGE_SELF)
    children_before = process_usage_snapshot(resource.RUSAGE_CHILDREN)
    window, generation, generation_rows = run_generation_benchmark(
        generator,
        positions,
    )
    self_after_generation = process_usage_snapshot(resource.RUSAGE_SELF)
    children_after_generation = process_usage_snapshot(
        resource.RUSAGE_CHILDREN
    )
    rows.extend(generation_rows)
    rows.extend(
        process_usage_rows(
            self_before,
            self_after_generation,
            "generation",
            "benchmark_parent",
        )
    )
    rows.extend(
        process_usage_rows(
            children_before,
            children_after_generation,
            "generation",
            "child_processes",
        )
    )
    rows.extend(window_statistics_rows(window, positions, generation, config))

    collection_started = time.perf_counter()
    gc.collect()
    rows.append(
        metric_row(
            "phase",
            "between_phases",
            "garbage_collection",
            "elapsed_seconds",
            time.perf_counter() - collection_started,
            unit="seconds",
        )
    )

    trainer_construct_started = time.perf_counter()
    model = FisherNetwork()
    trainer = AlphaZeroTrainer(
        model,
        config.batch_size,
        device=device,
        checkpoint_manager=manager,
    )
    rows.append(
        metric_row(
            "phase",
            "setup",
            "trainer_construct",
            "elapsed_seconds",
            time.perf_counter() - trainer_construct_started,
            unit="seconds",
        )
    )

    checkpoint_load_started = time.perf_counter()
    trainer.load_checkpoint(checkpoint_path)
    rows.append(
        metric_row(
            "phase",
            "setup",
            "training_checkpoint_load",
            "elapsed_seconds",
            time.perf_counter() - checkpoint_load_started,
            unit="seconds",
        )
    )

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_self_before = process_usage_snapshot(resource.RUSAGE_SELF)
    training, training_rows = run_training_benchmark(trainer, window)
    training_self_after = process_usage_snapshot(resource.RUSAGE_SELF)
    rows.extend(training_rows)
    rows.extend(
        process_usage_rows(
            training_self_before,
            training_self_after,
            "training",
            "benchmark_process",
        )
    )
    if torch.device(device).type == "cuda":
        rows.extend(
            [
                metric_row(
                    "resource",
                    "training",
                    "cuda_allocator",
                    "peak_allocated_bytes",
                    torch.cuda.max_memory_allocated(device),
                    unit="bytes",
                ),
                metric_row(
                    "resource",
                    "training",
                    "cuda_allocator",
                    "peak_reserved_bytes",
                    torch.cuda.max_memory_reserved(device),
                    unit="bytes",
                ),
            ]
        )

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    component_rows = run_component_benchmarks(
        window,
        config,
        checkpoint_path,
        manager,
        device,
        request_capacity,
    )
    rows.extend(component_rows)
    rows.extend(nvidia_smi_rows("after_benchmark"))

    report_started = time.perf_counter()
    csv_path, markdown_path = write_reports(
        output_dir,
        positions,
        generation,
        training,
        rows,
    )
    report_seconds = time.perf_counter() - report_started

    total_seconds = time.perf_counter() - benchmark_started
    final_rows = [
        metric_row(
            "phase",
            "reporting",
            "write_reports",
            "elapsed_seconds",
            report_seconds,
            unit="seconds",
        ),
        metric_row(
            "phase",
            "benchmark",
            "complete_run",
            "elapsed_seconds",
            total_seconds,
            unit="seconds",
            notes="Includes component diagnostics and report writing",
        ),
    ]
    rows.extend(final_rows)
    csv_path, markdown_path = write_reports(
        output_dir,
        positions,
        generation,
        training,
        rows,
    )

    del window
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        {
            "generation": generation,
            "training": training,
            "metric_rows": len(
                core_metric_rows(
                    positions,
                    generation,
                    training,
                )
            )
            + len(rows),
        },
        csv_path,
        markdown_path,
    )
