import copy
import csv
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import load_config, save_config
from fisher_ai.distributed import DistributedSelfPlayPool


@dataclass(frozen=True)
class BenchmarkProfile:
    profile_id: str
    games_per_actor: int
    pending_leaves: int
    target_batch: int
    maximum_batch: int
    batch_wait_ms: float


def benchmark_profiles():
    baseline = BenchmarkProfile("baseline", 10, 16, 512, 1024, 2.0)
    return [
        baseline,
        BenchmarkProfile("games_6", 6, 16, 512, 1024, 2.0),
        BenchmarkProfile("games_8", 8, 16, 512, 1024, 2.0),
        BenchmarkProfile("games_12", 12, 16, 512, 1024, 2.0),
        BenchmarkProfile("games_14", 14, 16, 512, 1024, 2.0),
        BenchmarkProfile("leaves_8", 10, 8, 512, 1024, 2.0),
        BenchmarkProfile("leaves_24", 10, 24, 512, 1024, 2.0),
        BenchmarkProfile("leaves_32", 10, 32, 512, 1024, 2.0),
        BenchmarkProfile("batch_256", 10, 16, 256, 512, 2.0),
        BenchmarkProfile("batch_768", 10, 16, 768, 1536, 2.0),
        BenchmarkProfile("batch_1024", 10, 16, 1024, 2048, 2.0),
        BenchmarkProfile("wait_1ms", 10, 16, 512, 1024, 1.0),
        BenchmarkProfile("wait_4ms", 10, 16, 512, 1024, 4.0),
        BenchmarkProfile("aggressive", 14, 24, 1024, 2048, 4.0),
    ]


def read_cpu_times():
    line = Path("/proc/stat").read_text().splitlines()[0]
    values = [int(value) for value in line.split()[1:]]
    idle = values[3] + values[4]
    return sum(values), idle


def cpu_utilization(start, end):
    total_delta = end[0] - start[0]
    idle_delta = end[1] - start[1]
    if total_delta <= 0:
        return 0.0
    return 100.0 * (1.0 - idle_delta / total_delta)


def read_gpu_metrics():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []

    command = [
        executable,
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []

    metrics = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        metrics.append((float(parts[0]), float(parts[1])))
    return metrics


def histogram_percentile(histogram, percentile):
    total = int(histogram.sum())
    if total == 0:
        return 0.0
    target = total * percentile
    cumulative = 0
    for batch_size, count in enumerate(histogram):
        cumulative += int(count)
        if cumulative >= target:
            return float(batch_size)
    return float(len(histogram) - 1)


def metric_delta(start, end, elapsed, queue_samples, gpu_samples, cpu_percent):
    histogram = end["histogram"] - start["histogram"]
    batches = end["batches"] - start["batches"]
    evaluations = end["evaluations"] - start["evaluations"]
    games = end["games"] - start["games"]
    positions = end["positions"] - start["positions"]
    plies = end["plies"] - start["plies"]

    gpu_count = max((len(sample) for sample in gpu_samples), default=0)
    gpu_utilization = []
    gpu_memory = []
    for gpu_index in range(gpu_count):
        values = [sample[gpu_index] for sample in gpu_samples if gpu_index < len(sample)]
        gpu_utilization.append(float(np.mean([value[0] for value in values])))
        gpu_memory.append(float(np.mean([value[1] for value in values])))

    queue_depths = [sample[0] for sample in queue_samples if sample[0] >= 0]
    replay_depths = [sample[1] for sample in queue_samples if sample[1] >= 0]

    return {
        "elapsed_seconds": elapsed,
        "moves_per_second": plies / elapsed,
        "evaluations_per_second": evaluations / elapsed,
        "positions_per_second": positions / elapsed,
        "games_per_hour": games * 3600.0 / elapsed,
        "completed_games": games,
        "completed_positions": positions,
        "evaluations": evaluations,
        "average_gpu_batch": evaluations / max(batches, 1),
        "median_gpu_batch": histogram_percentile(histogram, 0.5),
        "p95_gpu_batch": histogram_percentile(histogram, 0.95),
        "maximum_gpu_batch": int(np.flatnonzero(histogram)[-1]) if histogram.any() else 0,
        "average_queue_depth": float(np.mean(queue_depths)) if queue_depths else -1.0,
        "maximum_queue_depth": max(queue_depths, default=-1),
        "average_replay_queue_depth": float(np.mean(replay_depths)) if replay_depths else -1.0,
        "cpu_utilization": cpu_percent,
        "gpu_utilization": gpu_utilization,
        "gpu_memory_mib": gpu_memory,
        "outstanding_requests": end["outstanding_requests"],
        "blocked_slot_waits": end["blocked_slot_waits"] - start["blocked_slot_waits"],
    }


def flatten_result(profile, metrics):
    result = {
        "configuration_id": profile.profile_id,
        "games_per_actor": profile.games_per_actor,
        "pending_leaves": profile.pending_leaves,
        "target_batch": profile.target_batch,
        "maximum_batch": profile.maximum_batch,
        "batch_wait_ms": profile.batch_wait_ms,
        **{key: value for key, value in metrics.items() if not isinstance(value, list)},
    }
    for index, value in enumerate(metrics["gpu_utilization"]):
        result[f"gpu_{index}_utilization"] = value
    for index, value in enumerate(metrics["gpu_memory_mib"]):
        result[f"gpu_{index}_memory_mib"] = value
    return result


def write_benchmark_reports(results, output_dir, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    markdown_path = output_dir / "benchmark_summary.md"

    fieldnames = []
    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ranked = sorted(results, key=lambda row: row["moves_per_second"], reverse=True)
    baseline = next(
        (row for row in results if row["configuration_id"] == "baseline"),
        ranked[0] if ranked else None,
    )

    lines = [
        "# Fisher AI self-play benchmark",
        "",
        f"Generated: {metadata['generated_at']}",
        f"Actor processes: {metadata['actor_count']}",
        f"Warmup per configuration: {metadata['warmup_seconds']} seconds",
        f"Measurement per configuration: {metadata['measure_seconds']} seconds",
        f"Devices: {', '.join(metadata['devices'])}",
        "",
        "Results are ranked by completed real chess moves per second. GPU and CPU "
        "utilization are diagnostics rather than the primary score.",
        "",
        "## Ranked results",
        "",
        "| Rank | Configuration | Moves/s | Evals/s | Avg batch | P95 batch | CPU | Games/hour |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['configuration_id']} | "
            f"{row['moves_per_second']:.2f} | "
            f"{row['evaluations_per_second']:.1f} | "
            f"{row['average_gpu_batch']:.1f} | "
            f"{row['p95_gpu_batch']:.0f} | "
            f"{row['cpu_utilization']:.1f}% | "
            f"{row['games_per_hour']:.1f} |"
        )

    if ranked:
        best = ranked[0]
        improvement = 0.0
        if baseline and baseline["moves_per_second"] > 0:
            improvement = (
                best["moves_per_second"] / baseline["moves_per_second"] - 1.0
            ) * 100.0
        lines.extend(
            [
                "",
                "## Suggested manual settings",
                "",
                f"The highest-throughput result was `{best['configuration_id']}` at "
                f"{best['moves_per_second']:.2f} moves/s, "
                f"{improvement:+.1f}% relative to the baseline.",
                "",
                "```json",
                "{",
                f"  \"games_per_actor\": {best['games_per_actor']},",
                f"  \"parallel_searches\": {best['pending_leaves']},",
                f"  \"inference_batch_size\": {best['target_batch']},",
                f"  \"inference_max_batch_size\": {best['maximum_batch']},",
                f"  \"inference_batch_wait_ms\": {best['batch_wait_ms']}",
                "}",
                "```",
                "",
                "These values are recommendations only. The benchmark does not modify "
                "`fisher_config.json`.",
            ]
        )

    markdown_path.write_text("\n".join(lines) + "\n")
    return csv_path, markdown_path


def run_benchmark(
    config_path="fisher_config.json",
    warmup_seconds=5.0,
    measure_seconds=15.0,
    profile_limit=None,
    output_dir=None,
    actor_count=None,
    devices=None,
):
    base_config = load_config(config_path)
    actor_count = actor_count or base_config.runtime.actor_processes
    devices = devices or base_config.runtime.self_play_devices
    profiles = benchmark_profiles()
    if profile_limit is not None:
        profiles = profiles[:profile_limit]

    checkpoint_manager = CheckpointManager(base_config.runtime.checkpoint_dir)
    checkpoint_path = checkpoint_manager.latest_path()
    if checkpoint_path is None:
        raise RuntimeError("Run `python -m fisher_ai init` before benchmarking")
    checkpoint_path = str(Path(checkpoint_path).resolve())

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_config.runtime.benchmark_dir) / timestamp

    results = []
    for profile_index, profile in enumerate(profiles, start=1):
        print(
            f"[{profile_index}/{len(profiles)}] {profile.profile_id}: "
            f"games={profile.games_per_actor} leaves={profile.pending_leaves} "
            f"batch={profile.target_batch}/{profile.maximum_batch} "
            f"wait={profile.batch_wait_ms:g}ms",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="fisher_benchmark_") as temp_dir:
            temp_dir = Path(temp_dir)
            config = copy.deepcopy(base_config)
            config.search.parallel_searches = profile.pending_leaves
            config.runtime.games_per_actor = profile.games_per_actor
            config.runtime.inference_batch_size = profile.target_batch
            config.runtime.inference_max_batch_size = profile.maximum_batch
            config.runtime.inference_batch_wait_ms = profile.batch_wait_ms
            config.runtime.replay_path = str(temp_dir / "replay.lmdb")
            config.runtime.status_interval_seconds = 3600.0
            profile_config_path = temp_dir / "config.json"
            save_config(config, profile_config_path)

            pool = DistributedSelfPlayPool(
                config_path=profile_config_path,
                actor_count=actor_count,
                games_per_actor=profile.games_per_actor,
                devices=devices,
                checkpoint_path=checkpoint_path,
            )
            try:
                pool.start()
                time.sleep(warmup_seconds)
                failure = pool.process_failure()
                if failure:
                    raise RuntimeError(failure)

                start = pool.metric_snapshot()
                cpu_start = read_cpu_times()
                queue_samples = []
                gpu_samples = []
                start_time = time.monotonic()
                deadline = start_time + measure_seconds

                while time.monotonic() < deadline:
                    failure = pool.process_failure()
                    if failure:
                        raise RuntimeError(failure)
                    snapshot = pool.metric_snapshot()
                    queue_samples.append(
                        (
                            snapshot["request_queue_depth"],
                            snapshot["replay_queue_depth"],
                        )
                    )
                    gpu_samples.append(read_gpu_metrics())
                    time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))

                elapsed = time.monotonic() - start_time
                cpu_end = read_cpu_times()
                end = pool.metric_snapshot()
                metrics = metric_delta(
                    start,
                    end,
                    elapsed,
                    queue_samples,
                    gpu_samples,
                    cpu_utilization(cpu_start, cpu_end),
                )
                result = flatten_result(profile, metrics)
                results.append(result)
                print(
                    f"  moves/s={result['moves_per_second']:.2f} "
                    f"evals/s={result['evaluations_per_second']:.1f} "
                    f"avg_batch={result['average_gpu_batch']:.1f} "
                    f"cpu={result['cpu_utilization']:.1f}%",
                    flush=True,
                )
            finally:
                pool.stop()

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "actor_count": actor_count,
        "warmup_seconds": warmup_seconds,
        "measure_seconds": measure_seconds,
        "devices": devices,
    }
    csv_path, markdown_path = write_benchmark_reports(results, output_dir, metadata)
    return results, csv_path, markdown_path
