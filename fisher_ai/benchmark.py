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
from fisher_ai.distributed import (
    BATCHED_SCHEDULER,
    THREADED_SCHEDULER,
    DistributedSelfPlayPool,
)


@dataclass(frozen=True)
class BenchmarkProfile:
    profile_id: str
    scheduler: str
    actor_count: int
    games_per_actor: int
    pending_leaves: int
    max_inflight_requests_per_actor: int
    target_batch: int
    maximum_batch: int
    batch_wait_ms: float


def build_profile(
    profile_id,
    scheduler,
    actor_count,
    games_per_actor,
    pending_leaves,
    target_batch=512,
    maximum_batch=1024,
    batch_wait_ms=2.0,
    max_inflight_requests_per_actor=None,
):
    if max_inflight_requests_per_actor is None:
        if scheduler == BATCHED_SCHEDULER:
            max_inflight_requests_per_actor = 1
        else:
            max_inflight_requests_per_actor = max(8, games_per_actor)
    return BenchmarkProfile(
        profile_id=profile_id,
        scheduler=scheduler,
        actor_count=actor_count,
        games_per_actor=games_per_actor,
        pending_leaves=pending_leaves,
        max_inflight_requests_per_actor=max_inflight_requests_per_actor,
        target_batch=target_batch,
        maximum_batch=maximum_batch,
        batch_wait_ms=batch_wait_ms,
    )


def benchmark_profiles(actor_count=24):
    return [
        build_profile("baseline", BATCHED_SCHEDULER, actor_count, 6, 8),
        build_profile(
            "threaded_current_4x24",
            THREADED_SCHEDULER,
            actor_count,
            4,
            24,
        ),
        build_profile(
            "threaded_old_control_6x8",
            THREADED_SCHEDULER,
            actor_count,
            6,
            8,
        ),
        build_profile("batched_4x8", BATCHED_SCHEDULER, actor_count, 4, 8),
        build_profile("batched_4x16", BATCHED_SCHEDULER, actor_count, 4, 16),
        build_profile("batched_4x24", BATCHED_SCHEDULER, actor_count, 4, 24),
        build_profile("batched_4x32", BATCHED_SCHEDULER, actor_count, 4, 32),
        build_profile("batched_6x16", BATCHED_SCHEDULER, actor_count, 6, 16),
        build_profile("batched_6x24", BATCHED_SCHEDULER, actor_count, 6, 24),
        build_profile("batched_6x32", BATCHED_SCHEDULER, actor_count, 6, 32),
        build_profile("batched_8x16", BATCHED_SCHEDULER, actor_count, 8, 16),
        build_profile("batched_8x24", BATCHED_SCHEDULER, actor_count, 8, 24),
        build_profile(
            "batched_6x24_wait1",
            BATCHED_SCHEDULER,
            actor_count,
            6,
            24,
            batch_wait_ms=1.0,
        ),
        build_profile(
            "batched_6x24_batch768",
            BATCHED_SCHEDULER,
            actor_count,
            6,
            24,
            target_batch=768,
            maximum_batch=1024,
        ),
        build_profile(
            f"batched_{max(1, actor_count - 2)}actors_6x24",
            BATCHED_SCHEDULER,
            max(1, actor_count - 2),
            6,
            24,
        ),
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


def metric_delta(
    start,
    end,
    elapsed,
    queue_samples,
    gpu_samples,
    cpu_percent,
    actor_count,
):
    histogram = end["histogram"] - start["histogram"]
    batches = end["batches"] - start["batches"]
    evaluations = end["evaluations"] - start["evaluations"]
    actor_evaluations = end["actor_evaluations"] - start["actor_evaluations"]
    requests = end["inference_requests"] - start["inference_requests"]
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
    actor_capacity_ns = max(actor_count * elapsed * 1_000_000_000, 1)

    return {
        "request_capacity": end["request_capacity"],
        "elapsed_seconds": elapsed,
        "moves_per_second": plies / elapsed,
        "evaluations_per_second": evaluations / elapsed,
        "positions_per_second": positions / elapsed,
        "games_per_hour": games * 3600.0 / elapsed,
        "completed_games": games,
        "completed_positions": positions,
        "evaluations": evaluations,
        "inference_requests": requests,
        "average_actor_request": actor_evaluations / max(requests, 1),
        "average_gpu_batch": evaluations / max(batches, 1),
        "median_gpu_batch": histogram_percentile(histogram, 0.5),
        "p95_gpu_batch": histogram_percentile(histogram, 0.95),
        "maximum_gpu_batch": int(np.flatnonzero(histogram)[-1]) if histogram.any() else 0,
        "average_queue_depth": float(np.mean(queue_depths)) if queue_depths else -1.0,
        "maximum_queue_depth": max(queue_depths, default=-1),
        "average_replay_queue_depth": float(np.mean(replay_depths)) if replay_depths else -1.0,
        "cpu_utilization": cpu_percent,
        "actor_work_percent": 100.0
        * (end["actor_compute_ns"] - start["actor_compute_ns"])
        / actor_capacity_ns,
        "inference_wait_percent": 100.0
        * (end["response_wait_ns"] - start["response_wait_ns"])
        / actor_capacity_ns,
        "queue_wait_percent": 100.0
        * (end["queue_wait_ns"] - start["queue_wait_ns"])
        / actor_capacity_ns,
        "replay_wait_percent": 100.0
        * (end["replay_wait_ns"] - start["replay_wait_ns"])
        / actor_capacity_ns,
        "gpu_utilization": gpu_utilization,
        "gpu_memory_mib": gpu_memory,
        "outstanding_requests": end["outstanding_requests"],
        "blocked_slot_waits": end["blocked_slot_waits"] - start["blocked_slot_waits"],
    }


def flatten_result(profile, metrics, run_stage="sweep", confirmation_rank=""):
    result = {
        "configuration_id": profile.profile_id,
        "run_stage": run_stage,
        "confirmation_rank": confirmation_rank,
        "scheduler": profile.scheduler,
        "actor_count": profile.actor_count,
        "games_per_actor": profile.games_per_actor,
        "pending_leaves": profile.pending_leaves,
        "max_inflight_requests_per_actor": profile.max_inflight_requests_per_actor,
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


def append_ranked_table(lines, title, ranked):
    lines.extend(
        [
            f"## {title}",
            "",
            "| Rank | Configuration | Scheduler | Actors | Games | Leaves | Request cap | "
            "Batch | Wait | Moves/s | Evals/s | Request avg | GPU batch | CPU | "
            "Actor work | Inference wait |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['configuration_id']} | {row['scheduler']} | "
            f"{row['actor_count']} | {row['games_per_actor']} | "
            f"{row['pending_leaves']} | {row['request_capacity']} | "
            f"{row['target_batch']}/{row['maximum_batch']} | "
            f"{row['batch_wait_ms']:g} ms | "
            f"{row['moves_per_second']:.2f} | "
            f"{row['evaluations_per_second']:.1f} | "
            f"{row['average_actor_request']:.1f} | "
            f"{row['average_gpu_batch']:.1f} | "
            f"{row['cpu_utilization']:.1f}% | "
            f"{row['actor_work_percent']:.1f}% | "
            f"{row['inference_wait_percent']:.1f}% |"
        )
    lines.append("")


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

    sweep_results = [
        row for row in results if row.get("run_stage", "sweep") == "sweep"
    ]
    confirmation_results = [
        row for row in results if row.get("run_stage") == "confirmation"
    ]
    ranked_sweep = sorted(
        sweep_results,
        key=lambda row: row["moves_per_second"],
        reverse=True,
    )
    ranked_confirmation = sorted(
        confirmation_results,
        key=lambda row: row["moves_per_second"],
        reverse=True,
    )
    baseline = next(
        (row for row in sweep_results if row["configuration_id"] == "baseline"),
        ranked_sweep[0] if ranked_sweep else None,
    )

    lines = [
        "# Fisher AI self-play benchmark",
        "",
        f"Generated: {metadata['generated_at']}",
        f"Base actor processes: {metadata['actor_count']}",
        f"Sweep profiles: {metadata['sweep_profile_count']}",
        f"Warmup per run: {metadata['warmup_seconds']} seconds",
        f"Sweep measurement: {metadata['measure_seconds']} seconds",
        f"Confirmation measurement: {metadata['confirmation_seconds']} seconds",
        f"Confirmation runs: {metadata['confirmation_run_count']}",
        f"Devices: {', '.join(metadata['devices'])}",
        "",
        "The sweep compares the restored process-level batched scheduler with the "
        "threaded regression under the same inference servers. Actor work and wait "
        "percentages are timing diagnostics; system CPU utilization and completed "
        "moves per second remain the primary throughput evidence.",
        "",
    ]

    append_ranked_table(lines, "Initial sweep results", ranked_sweep)
    if ranked_confirmation:
        append_ranked_table(lines, "Confirmed top configurations", ranked_confirmation)

    recommendation_pool = ranked_confirmation or ranked_sweep
    if recommendation_pool:
        best = recommendation_pool[0]
        improvement = 0.0
        if baseline and baseline["moves_per_second"] > 0:
            improvement = (
                best["moves_per_second"] / baseline["moves_per_second"] - 1.0
            ) * 100.0
        result_kind = "confirmed" if ranked_confirmation else "sweep"
        lines.extend(
            [
                "## Suggested manual settings",
                "",
                f"The highest-throughput {result_kind} result was "
                f"`{best['configuration_id']}` at "
                f"{best['moves_per_second']:.2f} moves/s, "
                f"{improvement:+.1f}% relative to the batched old-patch control.",
                "",
                "```json",
                "{",
                f"  \"self_play_scheduler\": \"{best['scheduler']}\",",
                f"  \"actor_processes\": {best['actor_count']},",
                f"  \"games_per_actor\": {best['games_per_actor']},",
                f"  \"parallel_searches\": {best['pending_leaves']},",
                "  \"inference_request_batch_size\": 0,",
                "  \"max_inflight_requests_per_actor\": "
                f"{best['max_inflight_requests_per_actor']},",
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


def run_profile(
    base_config,
    profile,
    devices,
    checkpoint_path,
    warmup_seconds,
    measure_seconds,
    run_stage="sweep",
    confirmation_rank="",
):
    with tempfile.TemporaryDirectory(prefix="fisher_benchmark_") as temp_dir:
        temp_dir = Path(temp_dir)
        config = copy.deepcopy(base_config)
        config.search.parallel_searches = profile.pending_leaves
        config.runtime.self_play_scheduler = profile.scheduler
        config.runtime.actor_processes = profile.actor_count
        config.runtime.games_per_actor = profile.games_per_actor
        config.runtime.inference_request_batch_size = 0
        config.runtime.max_inflight_requests_per_actor = (
            profile.max_inflight_requests_per_actor
        )
        config.runtime.inference_batch_size = profile.target_batch
        config.runtime.inference_max_batch_size = profile.maximum_batch
        config.runtime.inference_batch_wait_ms = profile.batch_wait_ms
        config.runtime.replay_path = str(temp_dir / "replay.lmdb")
        config.runtime.status_interval_seconds = 3600.0
        profile_config_path = temp_dir / "config.json"
        save_config(config, profile_config_path)

        pool = DistributedSelfPlayPool(
            config_path=profile_config_path,
            actor_count=profile.actor_count,
            games_per_actor=profile.games_per_actor,
            devices=devices,
            checkpoint_path=checkpoint_path,
            scheduler=profile.scheduler,
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
            failure = pool.process_failure()
            if failure:
                raise RuntimeError(failure)
            cpu_end = read_cpu_times()
            end = pool.metric_snapshot()
            metrics = metric_delta(
                start,
                end,
                elapsed,
                queue_samples,
                gpu_samples,
                cpu_utilization(cpu_start, cpu_end),
                profile.actor_count,
            )
            return flatten_result(
                profile,
                metrics,
                run_stage=run_stage,
                confirmation_rank=confirmation_rank,
            )
        finally:
            pool.stop()


def print_profile(profile, prefix):
    print(
        f"{prefix} {profile.profile_id}: scheduler={profile.scheduler} "
        f"actors={profile.actor_count} games={profile.games_per_actor} "
        f"leaves={profile.pending_leaves} "
        f"batch={profile.target_batch}/{profile.maximum_batch} "
        f"wait={profile.batch_wait_ms:g}ms",
        flush=True,
    )


def print_result(result):
    print(
        f"  moves/s={result['moves_per_second']:.2f} "
        f"evals/s={result['evaluations_per_second']:.1f} "
        f"request_avg={result['average_actor_request']:.1f} "
        f"avg_batch={result['average_gpu_batch']:.1f} "
        f"cpu={result['cpu_utilization']:.1f}% "
        f"actor_work={result['actor_work_percent']:.1f}% "
        f"inference_wait={result['inference_wait_percent']:.1f}%",
        flush=True,
    )


def run_benchmark(
    config_path="fisher_config.json",
    warmup_seconds=5.0,
    measure_seconds=15.0,
    confirmation_seconds=30.0,
    confirm_top=3,
    profile_limit=None,
    output_dir=None,
    actor_count=None,
    devices=None,
):
    base_config = load_config(config_path)
    base_actor_count = actor_count or base_config.runtime.actor_processes
    devices = devices or base_config.runtime.self_play_devices
    profiles = benchmark_profiles(base_actor_count)
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
    sweep_results = []
    for profile_index, profile in enumerate(profiles, start=1):
        print_profile(profile, f"[{profile_index}/{len(profiles)}]")
        result = run_profile(
            base_config,
            profile,
            devices,
            checkpoint_path,
            warmup_seconds,
            measure_seconds,
        )
        sweep_results.append(result)
        results.append(result)
        print_result(result)

    confirmation_profiles = []
    if confirmation_seconds > 0 and confirm_top > 0:
        profile_by_id = {profile.profile_id: profile for profile in profiles}
        ranked_sweep = sorted(
            sweep_results,
            key=lambda row: row["moves_per_second"],
            reverse=True,
        )
        confirmation_count = min(confirm_top, len(ranked_sweep))
        for rank, result in enumerate(ranked_sweep[:confirmation_count], start=1):
            profile = profile_by_id[result["configuration_id"]]
            confirmation_profiles.append((rank, profile))

    for confirmation_index, (rank, profile) in enumerate(
        confirmation_profiles,
        start=1,
    ):
        print_profile(
            profile,
            f"[confirm {confirmation_index}/{len(confirmation_profiles)}]",
        )
        result = run_profile(
            base_config,
            profile,
            devices,
            checkpoint_path,
            warmup_seconds,
            confirmation_seconds,
            run_stage="confirmation",
            confirmation_rank=rank,
        )
        results.append(result)
        print_result(result)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "actor_count": base_actor_count,
        "sweep_profile_count": len(profiles),
        "warmup_seconds": warmup_seconds,
        "measure_seconds": measure_seconds,
        "confirmation_seconds": confirmation_seconds,
        "confirmation_run_count": len(confirmation_profiles),
        "devices": devices,
    }
    csv_path, markdown_path = write_benchmark_reports(results, output_dir, metadata)
    return results, csv_path, markdown_path
