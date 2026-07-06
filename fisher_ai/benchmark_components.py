"""Run benchmark-only component diagnostics outside production training."""

import os
import platform
import resource
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from fisher_ai.benchmark_metrics import distribution_row, metric_row
from fisher_ai.encoding import StateEncodingWorkspace, encode_states
from fisher_ai.game import GameState
from fisher_ai.generation import load_inference_model
from fisher_ai.mcts import MAX_LEGAL_ACTIONS, MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.trainer import AlphaZeroTrainer

MICRO_WARMUP_ROUNDS = 2
MICRO_TIMED_ROUNDS = 5
MCTS_TIMED_ROUNDS = 3
SAMPLE_STATE_COUNT = 64


class ZeroEvaluator:
    """Return neutral policy logits and values for actor-side MCTS timing."""

    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        policies = np.zeros(legal_actions.shape, dtype=np.float32)
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def synchronize_device(device):
    """Synchronize isolated CUDA diagnostics when required for timing."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def device_warmup_rounds(device):
    """Return conservative warmup counts for the active device."""
    return MICRO_WARMUP_ROUNDS if torch.device(device).type == "cuda" else 0


def device_timed_rounds(device):
    """Keep CPU fallback diagnostics useful without becoming excessive."""
    return MICRO_TIMED_ROUNDS if torch.device(device).type == "cuda" else 1


def process_usage_snapshot(kind):
    """Capture process resource counters for the current or child processes."""
    usage = resource.getrusage(kind)
    return {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def process_usage_rows(before, after, scope, component):
    """Convert process resource deltas into benchmark rows."""
    rows = []
    for metric, end_value in after.items():
        start_value = before[metric]
        if metric == "max_rss_kib":
            value = end_value
            unit = "KiB"
            notes = "Maximum reported resident set size"
        else:
            value = end_value - start_value
            unit = "seconds" if metric.endswith("seconds") else "count"
            notes = "Delta during measured scope"
        rows.append(
            metric_row(
                "resource",
                scope,
                component,
                metric,
                value,
                unit=unit,
                notes=notes,
            )
        )
    return rows


def read_memory_info():
    """Read selected Linux memory counters without adding dependencies."""
    values = {}
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as file:
        for line in file:
            name, raw_value = line.split(":", 1)
            if name in ("MemAvailable", "MemFree", "SwapFree", "SwapTotal"):
                values[name] = int(raw_value.strip().split()[0])
    return values


def cpu_model_name():
    """Return the first Linux CPU model name when available."""
    path = "/proc/cpuinfo"
    if not os.path.exists(path):
        return platform.processor()
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def environment_rows(config, device):
    """Describe software, hardware, and benchmark configuration."""
    rows = [
        metric_row(
            "metadata",
            "run",
            "environment",
            "python_version",
            platform.python_version(),
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "platform",
            platform.platform(),
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "cpu_model",
            cpu_model_name(),
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "logical_cpu_count",
            os.cpu_count() or 0,
            unit="count",
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "numpy_version",
            np.__version__,
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "torch_version",
            torch.__version__,
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "torch_cuda_version",
            torch.version.cuda or "unavailable",
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "cudnn_version",
            torch.backends.cudnn.version() or "unavailable",
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "configured_device",
            config.device,
        ),
        metric_row(
            "metadata",
            "run",
            "environment",
            "resolved_device",
            str(device),
        ),
    ]
    for name in (
        "actor_processes",
        "games_per_actor",
        "inference_batch_size",
        "inference_max_batch_size",
        "inference_batch_wait_ms",
        "simulations",
        "parallel_searches",
        "batch_size",
    ):
        rows.append(
            metric_row(
                "configuration",
                "run",
                "fisher_config",
                name,
                getattr(config, name),
                unit="count" if not name.endswith("_ms") else "milliseconds",
            )
        )

    for name, value in read_memory_info().items():
        rows.append(
            metric_row(
                "resource",
                "run",
                "system_memory",
                name.lower(),
                value,
                unit="KiB",
            )
        )

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            rows.extend(
                [
                    metric_row(
                        "metadata",
                        "run",
                        f"gpu_{index}",
                        "name",
                        properties.name,
                    ),
                    metric_row(
                        "metadata",
                        "run",
                        f"gpu_{index}",
                        "total_memory",
                        properties.total_memory,
                        unit="bytes",
                    ),
                    metric_row(
                        "metadata",
                        "run",
                        f"gpu_{index}",
                        "multiprocessor_count",
                        properties.multi_processor_count,
                        unit="count",
                    ),
                ]
            )
    return rows


def nvidia_smi_rows(scope):
    """Capture a low-overhead point-in-time NVIDIA device snapshot."""
    command = shutil.which("nvidia-smi")
    if command is None:
        return []
    query = (
        "index,name,utilization.gpu,utilization.memory,memory.used,"
        "memory.total,temperature.gpu,power.draw"
    )
    result = subprocess.run(
        [
            command,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return []

    rows = []
    fields = (
        "index",
        "name",
        "gpu_utilization",
        "memory_utilization",
        "memory_used",
        "memory_total",
        "temperature",
        "power_draw",
    )
    units = (
        "count",
        "",
        "percent",
        "percent",
        "MiB",
        "MiB",
        "celsius",
        "watts",
    )
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        gpu_scope = f"gpu_{values[0]}"
        for field, value, unit in zip(fields, values, units, strict=True):
            rows.append(
                metric_row(
                    "resource",
                    scope,
                    gpu_scope,
                    field,
                    value,
                    unit=unit,
                    notes="Point-in-time nvidia-smi snapshot",
                )
            )
    return rows


def array_bytes(arrays):
    """Return the total storage occupied by NumPy arrays."""
    return sum(array.nbytes for array in arrays)


def window_statistics_rows(window, target_positions, generation, config):
    """Describe generated workload shape and retained memory."""
    count = window.position_count
    legal_lengths = window.legal_lengths[:count]
    game_lengths = np.diff(np.asarray([0, *window.game_ends]))
    arrays = [getattr(window, name) for name in window.ARRAY_NAMES]
    rows = [
        metric_row(
            "workload",
            "generation",
            "window",
            "target_positions",
            target_positions,
            unit="positions",
        ),
        metric_row(
            "workload",
            "generation",
            "window",
            "retained_positions",
            count,
            unit="positions",
        ),
        metric_row(
            "workload",
            "generation",
            "window",
            "position_overshoot",
            count - target_positions,
            unit="positions",
        ),
        metric_row(
            "workload",
            "generation",
            "window",
            "completed_games",
            window.game_count,
            unit="games",
        ),
        metric_row(
            "workload",
            "generation",
            "window",
            "allocated_window_bytes",
            array_bytes(arrays),
            unit="bytes",
        ),
        metric_row(
            "workload",
            "generation",
            "search",
            "evaluations_per_retained_position",
            generation["evaluations"] / max(count, 1),
            unit="evaluations/position",
        ),
        metric_row(
            "workload",
            "generation",
            "search",
            "requested_simulations_per_position",
            config.simulations,
            unit="simulations/position",
        ),
        metric_row(
            "workload",
            "generation",
            "inference",
            "target_batch_fill_percent",
            generation["average_inference_batch"]
            / max(config.inference_batch_size, 1)
            * 100.0,
            unit="percent",
        ),
        metric_row(
            "workload",
            "generation",
            "inference",
            "maximum_batch_fill_percent",
            generation["max_batch"]
            / max(config.inference_max_batch_size, 1)
            * 100.0,
            unit="percent",
        ),
    ]
    rows.append(
        distribution_row(
            "distribution",
            "generation",
            "window",
            "legal_moves_per_position",
            legal_lengths,
            "moves",
        )
    )
    rows.append(
        distribution_row(
            "distribution",
            "generation",
            "window",
            "positions_per_game",
            game_lengths,
            "positions",
        )
    )
    return rows


def build_sample_states(count=SAMPLE_STATE_COUNT):
    """Build deterministic legal positions for isolated CPU benchmarks."""
    states = []
    state = GameState()
    move_buffer = np.empty(MAX_LEGAL_ACTIONS, dtype=np.uint32)
    for index in range(count):
        states.append(state.copy())
        move_count, _ = state.board.fill_legal_moves(move_buffer)
        if move_count == 0 or state.is_rule_draw():
            state = GameState()
            continue
        move = int(move_buffer[index % move_count])
        state.push(move)
    return states


def legal_move_benchmark_rows(states):
    """Benchmark legal move generation across varied positions."""
    round_seconds = []
    legal_counts = []
    move_buffer = np.empty(MAX_LEGAL_ACTIONS, dtype=np.uint32)

    for _ in range(MICRO_WARMUP_ROUNDS):
        for state in states:
            state.board.fill_legal_moves(move_buffer)

    for _ in range(MICRO_TIMED_ROUNDS):
        started = time.perf_counter()
        round_counts = []
        for state in states:
            count, _ = state.board.fill_legal_moves(move_buffer)
            round_counts.append(count)
        round_seconds.append(time.perf_counter() - started)
        legal_counts.extend(round_counts)

    total_positions = len(states) * MICRO_TIMED_ROUNDS
    total_seconds = sum(round_seconds)
    return [
        distribution_row(
            "compute",
            "component",
            "legal_move_generation",
            "round_seconds",
            round_seconds,
            "seconds",
            notes="Isolated; no neural inference",
        ),
        metric_row(
            "throughput",
            "component",
            "legal_move_generation",
            "positions_per_second",
            total_positions / max(total_seconds, 1e-9),
            unit="positions/second",
        ),
        distribution_row(
            "distribution",
            "component",
            "legal_move_generation",
            "legal_moves_per_position",
            legal_counts,
            "moves",
        ),
    ]


def encoding_batch_sizes(request_capacity):
    """Return representative actor encoding batch sizes."""
    return sorted(
        {
            1,
            min(8, request_capacity),
            min(32, request_capacity),
            request_capacity,
        }
    )


def encoding_benchmark_rows(states, request_capacity):
    """Benchmark live-state encoding at actor-sized batches."""
    rows = []
    for batch_size in encoding_batch_sizes(request_capacity):
        batch_states = [
            states[index % len(states)] for index in range(batch_size)
        ]
        workspace = StateEncodingWorkspace(batch_size)
        output = np.empty((batch_size, 119, 8, 8), dtype=np.float16)
        for _ in range(MICRO_WARMUP_ROUNDS):
            encode_states(batch_states, output=output, workspace=workspace)

        samples = []
        for _ in range(MICRO_TIMED_ROUNDS):
            started = time.perf_counter()
            encode_states(batch_states, output=output, workspace=workspace)
            samples.append(time.perf_counter() - started)

        total_seconds = sum(samples)
        positions = batch_size * len(samples)
        component = f"state_encoding_batch_{batch_size}"
        rows.extend(
            [
                distribution_row(
                    "compute",
                    "component",
                    component,
                    "batch_seconds",
                    samples,
                    "seconds",
                ),
                metric_row(
                    "throughput",
                    "component",
                    component,
                    "positions_per_second",
                    positions / max(total_seconds, 1e-9),
                    unit="positions/second",
                ),
                metric_row(
                    "memory",
                    "component",
                    component,
                    "encoded_bytes_per_batch",
                    output.nbytes,
                    unit="bytes",
                ),
            ]
        )
    return rows


def mcts_benchmark_rows(config):
    """Benchmark actor-side MCTS with neural evaluation replaced by zeros."""
    samples = []
    nodes = []
    cached_states = []
    tree_count = min(4, max(1, config.games_per_actor))

    for round_index in range(MCTS_TIMED_ROUNDS + 1):
        states = build_sample_states(tree_count)
        search = MCTS(
            ZeroEvaluator(),
            simulations=config.simulations,
            parallel_searches=config.parallel_searches,
            seed=100 + round_index,
        )
        started = time.perf_counter()
        roots = search.run(states)
        elapsed = time.perf_counter() - started
        if round_index == 0:
            continue
        samples.append(elapsed)
        nodes.append(sum(root.next_free for root in roots))
        cached_states.append(sum(root.state_pool.count for root in roots))

    total_simulations = tree_count * config.simulations * MCTS_TIMED_ROUNDS
    total_seconds = sum(samples)
    return [
        distribution_row(
            "compute",
            "component",
            "actor_side_mcts",
            "run_seconds",
            samples,
            "seconds",
            notes="Neural evaluator replaced by zero logits and values",
        ),
        metric_row(
            "throughput",
            "component",
            "actor_side_mcts",
            "simulations_per_second",
            total_simulations / max(total_seconds, 1e-9),
            unit="simulations/second",
        ),
        distribution_row(
            "distribution",
            "component",
            "actor_side_mcts",
            "nodes_allocated_per_run",
            nodes,
            "nodes",
        ),
        distribution_row(
            "distribution",
            "component",
            "actor_side_mcts",
            "cached_states_per_run",
            cached_states,
            "states",
        ),
    ]


def materialization_batch_sizes(window, configured_batch_size):
    """Return representative training materialization sizes."""
    maximum = max(1, min(window.position_count, configured_batch_size))
    return sorted({1, min(32, maximum), min(256, maximum), maximum})


def materialization_benchmark_rows(window, configured_batch_size):
    """Benchmark window encoding and indexed batch assembly."""
    rows = []
    for batch_size in materialization_batch_sizes(
        window,
        configured_batch_size,
    ):
        indices = np.arange(batch_size, dtype=np.int64) % window.position_count
        for _ in range(MICRO_WARMUP_ROUNDS):
            window.materialize_batch(indices)

        samples = []
        batch = None
        for _ in range(MICRO_TIMED_ROUNDS):
            started = time.perf_counter()
            batch = window.materialize_batch(indices)
            samples.append(time.perf_counter() - started)

        total_seconds = sum(samples)
        component = f"batch_materialization_{batch_size}"
        rows.extend(
            [
                distribution_row(
                    "compute",
                    "component",
                    component,
                    "batch_seconds",
                    samples,
                    "seconds",
                ),
                metric_row(
                    "throughput",
                    "component",
                    component,
                    "positions_per_second",
                    batch_size * len(samples) / max(total_seconds, 1e-9),
                    unit="positions/second",
                ),
                metric_row(
                    "memory",
                    "component",
                    component,
                    "materialized_batch_bytes",
                    array_bytes(batch),
                    unit="bytes",
                ),
                metric_row(
                    "workload",
                    "component",
                    component,
                    "legal_action_width",
                    batch[1].shape[1],
                    unit="actions",
                ),
            ]
        )
    return rows


def repeat_first_row(array, batch_size):
    """Repeat one NumPy row into a contiguous benchmark batch."""
    return np.repeat(array[:1], batch_size, axis=0)


def inference_batch_sizes(config, request_capacity, device):
    """Return representative inference sizes for the active device."""
    maximum = config.inference_max_batch_size
    target = config.inference_batch_size
    if torch.device(device).type == "cpu":
        maximum = min(maximum, 32)
        target = min(target, maximum)
    request_capacity = min(request_capacity, maximum)
    return sorted({1, request_capacity, target, maximum})


def network_benchmark_rows(
    window,
    config,
    checkpoint_path,
    device,
    request_capacity,
):
    """Benchmark full neural inference and its major network sections."""
    rows = []
    model = load_inference_model(device, checkpoint_path)
    first_batch = window.materialize_batch(np.asarray([0]))
    base_states = first_batch[0]
    base_actions = first_batch[1]

    for batch_size in inference_batch_sizes(
        config,
        request_capacity,
        device,
    ):
        states_numpy = repeat_first_row(base_states, batch_size)
        actions_numpy = repeat_first_row(base_actions, batch_size)
        states = torch.from_numpy(states_numpy).to(device)
        actions = torch.from_numpy(actions_numpy).to(device)
        if torch.device(device).type == "cpu":
            states = states.float()
        else:
            states = states.contiguous(memory_format=torch.channels_last)

        warmup_rounds = device_warmup_rounds(device)
        timed_rounds = device_timed_rounds(device)
        with torch.inference_mode():
            for _ in range(warmup_rounds):
                with torch.autocast(
                    device_type=torch.device(device).type,
                    dtype=torch.float16,
                    enabled=torch.device(device).type == "cuda",
                ):
                    policy, _ = model(states)
                    policy.gather(1, actions)
            synchronize_device(device)

            samples = []
            for _ in range(timed_rounds):
                synchronize_device(device)
                started = time.perf_counter()
                with torch.autocast(
                    device_type=torch.device(device).type,
                    dtype=torch.float16,
                    enabled=torch.device(device).type == "cuda",
                ):
                    policy, _ = model(states)
                    policy.gather(1, actions)
                synchronize_device(device)
                samples.append(time.perf_counter() - started)

        total_seconds = sum(samples)
        component = f"network_inference_batch_{batch_size}"
        rows.extend(
            [
                distribution_row(
                    "compute",
                    "component",
                    component,
                    "batch_seconds",
                    samples,
                    "seconds",
                    notes="Includes model forward and legal-policy gather",
                ),
                metric_row(
                    "throughput",
                    "component",
                    component,
                    "positions_per_second",
                    batch_size * len(samples) / max(total_seconds, 1e-9),
                    unit="positions/second",
                ),
                metric_row(
                    "latency",
                    "component",
                    component,
                    "microseconds_per_position",
                    total_seconds
                    / max(batch_size * len(samples), 1)
                    * 1_000_000,
                    unit="microseconds/position",
                ),
            ]
        )

    stage_batch_size = min(
        config.inference_batch_size,
        inference_batch_sizes(config, request_capacity, device)[-1],
    )
    if torch.device(device).type == "cpu":
        stage_batch_size = 1
    rows.extend(
        network_stage_rows(
            model,
            base_states,
            stage_batch_size,
            device,
        )
    )
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return rows


def network_stage_rows(model, base_states, batch_size, device):
    """Benchmark the stem, residual tower, and both output heads."""
    states_numpy = repeat_first_row(base_states, batch_size)
    states = torch.from_numpy(states_numpy).to(device)
    if torch.device(device).type == "cpu":
        states = states.float()
    else:
        states = states.contiguous(memory_format=torch.channels_last)

    stage_samples = {
        "stem": [],
        "residual_tower": [],
        "policy_head": [],
        "value_head": [],
    }
    warmup_rounds = device_warmup_rounds(device)
    timed_rounds = device_timed_rounds(device)
    with torch.inference_mode():
        for round_index in range(warmup_rounds + timed_rounds):
            synchronize_device(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.float16,
                enabled=torch.device(device).type == "cuda",
            ):
                features = F.relu(model.stem_bn(model.stem_conv(states)))
            synchronize_device(device)
            stem_seconds = time.perf_counter() - started

            started = time.perf_counter()
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.float16,
                enabled=torch.device(device).type == "cuda",
            ):
                features = model.residual_tower(features)
            synchronize_device(device)
            tower_seconds = time.perf_counter() - started

            started = time.perf_counter()
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.float16,
                enabled=torch.device(device).type == "cuda",
            ):
                policy = F.relu(model.policy_bn(model.policy_conv1(features)))
                model.policy_conv2(policy).flatten(1)
            synchronize_device(device)
            policy_seconds = time.perf_counter() - started

            started = time.perf_counter()
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.float16,
                enabled=torch.device(device).type == "cuda",
            ):
                value = F.relu(model.value_bn(model.value_conv(features)))
                value = value.flatten(1)
                value = F.relu(model.value_linear1(value))
                torch.tanh(model.value_linear2(value)).squeeze(1)
            synchronize_device(device)
            value_seconds = time.perf_counter() - started

            if round_index < warmup_rounds:
                continue
            stage_samples["stem"].append(stem_seconds)
            stage_samples["residual_tower"].append(tower_seconds)
            stage_samples["policy_head"].append(policy_seconds)
            stage_samples["value_head"].append(value_seconds)

    rows = []
    for stage, samples in stage_samples.items():
        rows.append(
            distribution_row(
                "compute",
                "component",
                f"network_stage_batch_{batch_size}",
                stage,
                samples,
                "seconds",
                notes="Isolated stage timing synchronizes CUDA",
            )
        )
    return rows


def transfer_benchmark_rows(window, config, device):
    """Benchmark host copies and isolated CUDA transfer/layout costs."""
    batch_size = min(window.position_count, config.inference_batch_size)
    batch_size = max(1, batch_size)
    indices = np.arange(batch_size, dtype=np.int64) % window.position_count
    states = window.materialize_batch(indices)[0]
    destination = np.empty_like(states)
    copy_samples = []

    for _ in range(MICRO_WARMUP_ROUNDS):
        destination[:] = states
    for _ in range(MICRO_TIMED_ROUNDS):
        started = time.perf_counter()
        destination[:] = states
        copy_samples.append(time.perf_counter() - started)

    rows = [
        distribution_row(
            "memory",
            "component",
            f"state_copy_batch_{batch_size}",
            "numpy_preallocated_copy",
            copy_samples,
            "seconds",
        ),
        metric_row(
            "throughput",
            "component",
            f"state_copy_batch_{batch_size}",
            "effective_gib_per_second",
            states.nbytes
            * len(copy_samples)
            / max(sum(copy_samples), 1e-9)
            / (1024**3),
            unit="GiB/second",
        ),
    ]

    if torch.device(device).type != "cuda":
        return rows

    source = torch.from_numpy(states).pin_memory()
    h2d_samples = []
    layout_samples = []
    d2h_samples = []
    gpu_states = None

    for _ in range(MICRO_WARMUP_ROUNDS):
        gpu_states = source.to(device, non_blocking=True)
        gpu_states = gpu_states.contiguous(memory_format=torch.channels_last)
        gpu_states.cpu()
    synchronize_device(device)

    for _ in range(MICRO_TIMED_ROUNDS):
        synchronize_device(device)
        started = time.perf_counter()
        gpu_states = source.to(device, non_blocking=True)
        synchronize_device(device)
        h2d_samples.append(time.perf_counter() - started)

        started = time.perf_counter()
        gpu_states = gpu_states.contiguous(memory_format=torch.channels_last)
        synchronize_device(device)
        layout_samples.append(time.perf_counter() - started)

        started = time.perf_counter()
        gpu_states.cpu()
        synchronize_device(device)
        d2h_samples.append(time.perf_counter() - started)

    component = f"state_transfer_batch_{batch_size}"
    rows.extend(
        [
            distribution_row(
                "memory",
                "component",
                component,
                "host_to_device",
                h2d_samples,
                "seconds",
            ),
            distribution_row(
                "memory",
                "component",
                component,
                "channels_last_conversion",
                layout_samples,
                "seconds",
            ),
            distribution_row(
                "memory",
                "component",
                component,
                "device_to_host",
                d2h_samples,
                "seconds",
            ),
        ]
    )
    return rows


def training_step_benchmark_rows(
    window,
    config,
    checkpoint_path,
    manager,
    device,
):
    """Benchmark training stages on a separate diagnostic model."""
    model = FisherNetwork()
    trainer = AlphaZeroTrainer(
        model,
        config.batch_size,
        device=device,
        checkpoint_manager=manager,
    )
    trainer.load_checkpoint(checkpoint_path)
    batch_size = min(window.position_count, config.batch_size)
    indices = np.arange(batch_size, dtype=np.int64)
    batch = window.materialize_batch(indices)
    samples = {
        "tensor_batch": [],
        "learning_rate_update": [],
        "zero_grad": [],
        "forward_and_loss": [],
        "backward": [],
        "optimizer_step": [],
    }

    timed_rounds = device_timed_rounds(device)
    for _ in range(timed_rounds):
        synchronize_device(device)
        started = time.perf_counter()
        tensors = trainer.tensor_batch(batch)
        synchronize_device(device)
        samples["tensor_batch"].append(time.perf_counter() - started)

        started = time.perf_counter()
        trainer.set_learning_rate()
        samples["learning_rate_update"].append(time.perf_counter() - started)

        started = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        synchronize_device(device)
        samples["zero_grad"].append(time.perf_counter() - started)

        synchronize_device(device)
        started = time.perf_counter()
        with torch.autocast(
            device_type=trainer.device.type,
            dtype=torch.float16,
            enabled=trainer.device.type == "cuda",
        ):
            loss, _, _ = trainer.compute_loss(*tensors)
        synchronize_device(device)
        samples["forward_and_loss"].append(time.perf_counter() - started)

        started = time.perf_counter()
        trainer.scaler.scale(loss).backward()
        synchronize_device(device)
        samples["backward"].append(time.perf_counter() - started)

        started = time.perf_counter()
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.step += 1
        synchronize_device(device)
        samples["optimizer_step"].append(time.perf_counter() - started)

    rows = []
    for stage, values in samples.items():
        rows.append(
            distribution_row(
                "compute" if stage != "tensor_batch" else "memory",
                "component",
                f"training_step_batch_{batch_size}",
                stage,
                values,
                "seconds",
                notes=(
                    "Separate diagnostic model; not the measured training run"
                ),
            )
        )
    total_seconds = sum(sum(values) for values in samples.values())
    rows.append(
        metric_row(
            "throughput",
            "component",
            f"training_step_batch_{batch_size}",
            "positions_per_second",
            batch_size * timed_rounds / max(total_seconds, 1e-9),
            unit="positions/second",
        )
    )
    del trainer
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_component_benchmarks(
    window,
    config,
    checkpoint_path,
    manager,
    device,
    request_capacity,
):
    """Run all isolated diagnostics without changing production modules."""
    started = time.perf_counter()
    states = build_sample_states()
    rows = []
    rows.extend(legal_move_benchmark_rows(states))
    rows.extend(encoding_benchmark_rows(states, request_capacity))
    rows.extend(mcts_benchmark_rows(config))
    rows.extend(materialization_benchmark_rows(window, config.batch_size))
    rows.extend(transfer_benchmark_rows(window, config, device))
    rows.extend(
        network_benchmark_rows(
            window,
            config,
            checkpoint_path,
            device,
            request_capacity,
        )
    )
    rows.extend(
        training_step_benchmark_rows(
            window,
            config,
            checkpoint_path,
            manager,
            device,
        )
    )
    rows.append(
        metric_row(
            "phase",
            "component",
            "diagnostics",
            "elapsed_seconds",
            time.perf_counter() - started,
            unit="seconds",
            notes="Excluded from generation and training throughput",
        )
    )
    return rows
