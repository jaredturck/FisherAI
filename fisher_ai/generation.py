import gc
import os
import queue
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.dataset import InMemoryWindow
from fisher_ai.encoding import INPUT_PLANES, encode_state
from fisher_ai.mcts import MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.self_play import SelfPlayRunner


class SharedInferenceMemory:
    def __init__(
        self,
        descriptor=None,
        actor_count=0,
        max_request_batch=0,
        max_legal_actions=256,
    ):
        self.owner = descriptor is None
        self.segments = {}
        self.arrays = {}

        if self.owner:
            specs = {
                "states": (
                    (actor_count, max_request_batch, INPUT_PLANES, 8, 8),
                    np.float16,
                ),
                "legal_actions": (
                    (actor_count, max_request_batch, max_legal_actions),
                    np.uint16,
                ),
                "legal_lengths": (
                    (actor_count, max_request_batch),
                    np.uint16,
                ),
                "policy_logits": (
                    (actor_count, max_request_batch, max_legal_actions),
                    np.float32,
                ),
                "values": (
                    (actor_count, max_request_batch),
                    np.float32,
                ),
            }
            self.descriptor = {}

            for name, (shape, dtype) in specs.items():
                dtype = np.dtype(dtype)
                size = int(np.prod(shape)) * dtype.itemsize
                segment = shared_memory.SharedMemory(create=True, size=size)
                array = np.ndarray(shape, dtype=dtype, buffer=segment.buf)
                array.fill(0)
                self.segments[name] = segment
                self.arrays[name] = array
                self.descriptor[name] = {
                    "name": segment.name,
                    "shape": shape,
                    "dtype": dtype.str,
                }
        else:
            self.descriptor = descriptor
            for name, spec in descriptor.items():
                segment = shared_memory.SharedMemory(name=spec["name"])
                array = np.ndarray(
                    tuple(spec["shape"]),
                    dtype=np.dtype(spec["dtype"]),
                    buffer=segment.buf,
                )
                self.segments[name] = segment
                self.arrays[name] = array

        self.states = self.arrays["states"]
        self.legal_actions = self.arrays["legal_actions"]
        self.legal_lengths = self.arrays["legal_lengths"]
        self.policy_logits = self.arrays["policy_logits"]
        self.values = self.arrays["values"]

    def close(self):
        for segment in self.segments.values():
            segment.close()

    def unlink(self):
        if not self.owner:
            return

        for segment in self.segments.values():
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


class GenerationCancelled(Exception):
    pass


class RemoteEvaluator:
    def __init__(
        self,
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
        evaluated_positions,
        inference_requests,
        queue_wait_ns,
        response_wait_ns,
    ):
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.shared = SharedInferenceMemory(descriptor=shared_descriptor)
        self.stop_event = stop_event
        self.evaluated_positions = evaluated_positions
        self.inference_requests = inference_requests
        self.queue_wait_ns = queue_wait_ns
        self.response_wait_ns = response_wait_ns
        self.max_request_batch = self.shared.states.shape[1]
        self.max_legal_actions = self.shared.legal_actions.shape[2]
        self.request_id = 0
        self.current_queue_wait_ns = 0
        self.current_response_wait_ns = 0

    def reset_timing(self):
        self.current_queue_wait_ns = 0
        self.current_response_wait_ns = 0

    def wait_ns(self):
        return self.current_queue_wait_ns + self.current_response_wait_ns

    def evaluate(self, states, legal_actions=None):
        encoded_states = [encode_state(state) for state in states]
        return self.evaluate_encoded(encoded_states, legal_actions=legal_actions)

    def evaluate_encoded(self, encoded_states, legal_actions=None):
        if legal_actions is None:
            raise ValueError("Remote inference requires legal action lists")

        policies = []
        values = []
        for start in range(0, len(encoded_states), self.max_request_batch):
            end = min(start + self.max_request_batch, len(encoded_states))
            chunk_policies, chunk_values = self.evaluate_chunk(
                encoded_states[start:end],
                legal_actions[start:end],
            )
            policies.extend(chunk_policies)
            values.append(chunk_values)

        if not values:
            return [], np.asarray([], dtype=np.float32)
        return policies, np.concatenate(values)

    def evaluate_chunk(self, encoded_states, legal_actions):
        batch_size = len(encoded_states)
        if batch_size == 0:
            return [], np.asarray([], dtype=np.float32)

        for index, (encoded_state, actions) in enumerate(
            zip(encoded_states, legal_actions, strict=True)
        ):
            if len(actions) > self.max_legal_actions:
                raise RuntimeError(
                    f"Position has {len(actions)} legal moves, exceeding the configured "
                    f"maximum of {self.max_legal_actions}"
                )
            self.shared.states[self.actor_id, index] = encoded_state
            length = len(actions)
            self.shared.legal_lengths[self.actor_id, index] = length
            self.shared.legal_actions[self.actor_id, index, :length] = actions

        self.request_id += 1
        request_id = self.request_id
        request = (self.actor_id, batch_size, request_id)
        self.inference_requests[self.actor_id] += 1

        queue_started = time.perf_counter_ns()
        while not self.stop_event.is_set():
            try:
                self.request_queue.put(request, timeout=0.5)
                break
            except queue.Full:
                continue
        else:
            raise GenerationCancelled("Generation stopped before inference submission")

        elapsed = time.perf_counter_ns() - queue_started
        self.queue_wait_ns[self.actor_id] += elapsed
        self.current_queue_wait_ns += elapsed

        response_started = time.perf_counter_ns()
        while not self.stop_event.is_set():
            try:
                response_id, error = self.response_queue.get(timeout=0.5)
                break
            except queue.Empty:
                continue
        else:
            raise GenerationCancelled("Generation stopped before inference response")

        elapsed = time.perf_counter_ns() - response_started
        self.response_wait_ns[self.actor_id] += elapsed
        self.current_response_wait_ns += elapsed

        if response_id == -1 and error:
            raise RuntimeError(error)
        if response_id != request_id:
            raise RuntimeError(
                f"Inference response mismatch for actor {self.actor_id}: "
                f"expected {request_id}, received {response_id}"
            )
        if error:
            raise RuntimeError(error)

        policies = []
        for index, actions in enumerate(legal_actions):
            length = len(actions)
            policies.append(
                self.shared.policy_logits[self.actor_id, index, :length].copy()
            )
        values = self.shared.values[self.actor_id, :batch_size].copy()
        self.evaluated_positions[self.actor_id] += batch_size
        return policies, values

    def close(self):
        self.shared.close()


def configure_worker_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def pin_actor_to_cpu(actor_id):
    if not hasattr(os, "sched_setaffinity"):
        return

    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        cpu_id = available_cpus[actor_id % len(available_cpus)]
        os.sched_setaffinity(0, {cpu_id})
    except (OSError, IndexError):
        pass


def load_inference_model(config, device, checkpoint_path):
    model = FisherNetwork(config.network)
    manager = CheckpointManager(config.runtime.checkpoint_dir)
    step, _ = manager.load(model, path=checkpoint_path, device=device)
    model.to(device)
    if config.runtime.channels_last and torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    model.eval()
    return model, step


def inference_server_main(
    device_name,
    config_path,
    checkpoint_path,
    shared_descriptor,
    request_queue,
    response_queues,
    stop_event,
    inference_batches,
    inference_positions,
    inference_max_batch,
    checkpoint_step,
    server_ready,
):
    configure_worker_threads()
    for response_queue in response_queues:
        response_queue.cancel_join_thread()
    config = load_config(config_path)
    device_name = available_device(device_name)
    device = torch.device(device_name)
    shared = SharedInferenceMemory(descriptor=shared_descriptor)
    model = None

    try:
        model, step = load_inference_model(config, device, checkpoint_path)
        checkpoint_step.value = step
        server_ready.value = 1
        target_batch = config.runtime.inference_batch_size
        maximum_batch = config.runtime.inference_max_batch_size
        max_legal_actions = shared.legal_actions.shape[2]
        pinned_states = torch.empty(
            (maximum_batch, INPUT_PLANES, 8, 8),
            dtype=torch.float16,
            pin_memory=device.type == "cuda",
        )
        pinned_actions = torch.empty(
            (maximum_batch, max_legal_actions),
            dtype=torch.int64,
            pin_memory=device.type == "cuda",
        )
        batch_wait_seconds = config.runtime.inference_batch_wait_ms / 1000.0
        deferred_request = None
        should_exit = False

        while not should_exit:
            if deferred_request is None:
                request = request_queue.get()
            else:
                request = deferred_request
                deferred_request = None
            if request is None:
                break

            requests = [request]
            state_count = request[1]
            if state_count > maximum_batch:
                raise RuntimeError(
                    f"Inference request of {state_count} positions exceeds maximum "
                    f"batch size {maximum_batch}"
                )
            deadline = time.monotonic() + batch_wait_seconds

            while state_count < target_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    next_request = request_queue.get(timeout=timeout)
                except queue.Empty:
                    break

                if next_request is None:
                    should_exit = True
                    break
                if state_count + next_request[1] > maximum_batch:
                    deferred_request = next_request
                    break

                requests.append(next_request)
                state_count += next_request[1]

            offset = 0
            request_offsets = []
            lengths = np.zeros(state_count, dtype=np.int64)
            max_legal_moves = 0
            pinned_actions[:state_count].zero_()

            for actor_id, batch_size, request_id in requests:
                end = offset + batch_size
                pinned_states[offset:end].copy_(
                    torch.from_numpy(shared.states[actor_id, :batch_size])
                )
                actor_lengths = shared.legal_lengths[actor_id, :batch_size].astype(
                    np.int64,
                    copy=True,
                )
                lengths[offset:end] = actor_lengths
                actor_max = int(actor_lengths.max(initial=0))
                max_legal_moves = max(max_legal_moves, actor_max)
                if actor_max:
                    pinned_actions[offset:end, :actor_max].copy_(
                        torch.from_numpy(
                            shared.legal_actions[
                                actor_id,
                                :batch_size,
                                :actor_max,
                            ].astype(np.int64, copy=False)
                        )
                    )
                request_offsets.append((actor_id, request_id, offset, end))
                offset = end

            state_tensor = pinned_states[:state_count].to(device, non_blocking=True)
            if device.type != "cuda":
                state_tensor = state_tensor.float()
            elif config.runtime.channels_last:
                state_tensor = state_tensor.contiguous(memory_format=torch.channels_last)

            action_tensor = pinned_actions[
                :state_count,
                :max_legal_moves,
            ].to(device, non_blocking=True)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    policy, values = model(state_tensor)
                    gathered = policy.gather(1, action_tensor)

            gathered = gathered.float().cpu().numpy()
            values = values.float().cpu().numpy()

            for actor_id, request_id, start, end in request_offsets:
                batch_lengths = lengths[start:end]
                for local_index, length in enumerate(batch_lengths):
                    shared.policy_logits[
                        actor_id,
                        local_index,
                        :length,
                    ] = gathered[start + local_index, :length]
                shared.values[actor_id, : end - start] = values[start:end]
                response_queues[actor_id].put((request_id, None))

            inference_batches.value += 1
            inference_positions.value += state_count
            inference_max_batch.value = max(inference_max_batch.value, state_count)

    except Exception as error:
        server_ready.value = -1
        stop_event.set()
        message = f"Inference server failed: {error}"
        for response_queue in response_queues:
            try:
                response_queue.put_nowait((-1, message))
            except queue.Full:
                pass
        raise
    finally:
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        shared.close()


def put_completed_game(target_queue, value):
    while True:
        try:
            target_queue.put(value, timeout=0.5)
            return
        except queue.Full:
            continue


def actor_main(
    actor_id,
    config_path,
    shared_descriptor,
    request_queue,
    response_queue,
    game_queue,
    stop_event,
    games_completed,
    positions_completed,
    plies_completed,
    evaluated_positions,
    inference_requests,
    queue_wait_ns,
    response_wait_ns,
    actor_compute_ns,
    checkpoint_step,
    generated_game_offset,
    games_per_actor,
):
    configure_worker_threads()
    request_queue.cancel_join_thread()
    game_queue.cancel_join_thread()
    config = load_config(config_path)
    if config.runtime.pin_actor_cpus:
        pin_actor_to_cpu(actor_id)

    evaluator = RemoteEvaluator(
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
        evaluated_positions,
        inference_requests,
        queue_wait_ns,
        response_wait_ns,
    )
    seed = config.runtime.seed + actor_id * 1000
    runner = SelfPlayRunner(
        MCTS(evaluator, config.search, seed=seed),
        config.search,
        training_config=config.training,
        seed=seed,
    )
    sessions = [
        runner.create_session(checkpoint_step=checkpoint_step, allow_resignation=False)
        for _ in range(games_per_actor)
    ]

    try:
        while not stop_event.is_set():
            allow_resignation = (
                generated_game_offset + int(sum(games_completed))
                >= config.training.resignation_enabled_after_games
            )
            before_plies = sum(len(session.moves) for session in sessions)
            evaluator.reset_timing()
            started = time.perf_counter_ns()
            finished_indices = runner.advance_sessions(
                sessions,
                allow_resignation=allow_resignation,
            )
            elapsed = time.perf_counter_ns() - started
            actor_compute_ns[actor_id] += max(elapsed - evaluator.wait_ns(), 0)
            after_plies = sum(len(session.moves) for session in sessions)
            plies_completed[actor_id] += after_plies - before_plies

            for index in finished_indices:
                record = sessions[index].build_record()
                put_completed_game(game_queue, record)
                games_completed[actor_id] += 1
                positions_completed[actor_id] += len(record.samples)
                sessions[index] = runner.create_session(
                    checkpoint_step=checkpoint_step,
                    allow_resignation=allow_resignation,
                )
    except GenerationCancelled:
        return
    except Exception:
        stop_event.set()
        raise
    finally:
        evaluator.close()


def round_request_capacity(value):
    if value <= 32:
        return max(1, value)
    return ((value + 31) // 32) * 32


class WindowGenerator:
    def __init__(
        self,
        config_path="fisher_config.json",
        checkpoint_path=None,
        actor_count=None,
        games_per_actor=None,
        device=None,
        generated_game_offset=0,
    ):
        import multiprocessing as mp

        self.context = mp.get_context("spawn")
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.actor_count = actor_count or self.config.runtime.actor_processes
        self.games_per_actor = games_per_actor or self.config.runtime.games_per_actor
        self.config.runtime.games_per_actor = self.games_per_actor
        self.device = available_device(device or self.config.runtime.device)
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.generated_game_offset = int(generated_game_offset)
        self.request_capacity = round_request_capacity(
            self.games_per_actor * self.config.search.parallel_searches
        )
        if self.request_capacity > self.config.runtime.inference_max_batch_size:
            raise ValueError(
                f"Actor request capacity {self.request_capacity} exceeds "
                f"inference_max_batch_size={self.config.runtime.inference_max_batch_size}"
            )

        self.stop_event = self.context.Event()
        self.request_queue = self.context.Queue(
            maxsize=self.config.runtime.inference_queue_size
        )
        self.response_queues = [
            self.context.Queue(maxsize=2) for _ in range(self.actor_count)
        ]
        self.game_queue = self.context.Queue(
            maxsize=self.config.runtime.game_queue_size
        )
        self.shared = SharedInferenceMemory(
            actor_count=self.actor_count,
            max_request_batch=self.request_capacity,
            max_legal_actions=self.config.runtime.max_legal_actions,
        )
        self.games_completed = self.context.Array("q", self.actor_count, lock=False)
        self.positions_completed = self.context.Array("q", self.actor_count, lock=False)
        self.plies_completed = self.context.Array("q", self.actor_count, lock=False)
        self.evaluated_positions = self.context.Array("q", self.actor_count, lock=False)
        self.inference_requests = self.context.Array("q", self.actor_count, lock=False)
        self.queue_wait_ns = self.context.Array("q", self.actor_count, lock=False)
        self.response_wait_ns = self.context.Array("q", self.actor_count, lock=False)
        self.actor_compute_ns = self.context.Array("q", self.actor_count, lock=False)
        self.inference_batches = self.context.Value("q", 0, lock=False)
        self.inference_positions = self.context.Value("q", 0, lock=False)
        self.inference_max_batch = self.context.Value("q", 0, lock=False)
        self.checkpoint_step = self.context.Value("q", 0, lock=False)
        self.server_ready = self.context.Value("b", 0, lock=False)
        self.actor_processes = []
        self.server_process = None
        self.started = False

    def start(self):
        if self.started:
            return

        self.server_process = self.context.Process(
            target=inference_server_main,
            name="fisher-inference",
            args=(
                self.device,
                self.config_path,
                self.checkpoint_path,
                self.shared.descriptor,
                self.request_queue,
                self.response_queues,
                self.stop_event,
                self.inference_batches,
                self.inference_positions,
                self.inference_max_batch,
                self.checkpoint_step,
                self.server_ready,
            ),
        )
        self.server_process.start()
        self.started = True

        deadline = time.monotonic() + 60.0
        while self.server_ready.value == 0:
            if self.server_process.exitcode is not None:
                raise RuntimeError(
                    f"Inference server exited with code {self.server_process.exitcode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference server did not become ready within 60 seconds")
            time.sleep(0.05)

        if self.server_ready.value < 0:
            raise RuntimeError("Inference server failed during startup")

        for actor_id in range(self.actor_count):
            process = self.context.Process(
                target=actor_main,
                name=f"fisher-actor-{actor_id:02d}",
                args=(
                    actor_id,
                    self.config_path,
                    self.shared.descriptor,
                    self.request_queue,
                    self.response_queues[actor_id],
                    self.game_queue,
                    self.stop_event,
                    self.games_completed,
                    self.positions_completed,
                    self.plies_completed,
                    self.evaluated_positions,
                    self.inference_requests,
                    self.queue_wait_ns,
                    self.response_wait_ns,
                    self.actor_compute_ns,
                    self.checkpoint_step.value,
                    self.generated_game_offset,
                    self.games_per_actor,
                ),
            )
            process.start()
            self.actor_processes.append(process)

    def process_failure(self):
        processes = [self.server_process, *self.actor_processes]
        for process in processes:
            if process is not None and process.exitcode is not None:
                return f"{process.name} exited with code {process.exitcode}"
        return None

    def metric_snapshot(self):
        try:
            queue_depth = self.request_queue.qsize()
        except (NotImplementedError, OSError):
            queue_depth = -1

        return {
            "games": int(sum(self.games_completed)),
            "positions": int(sum(self.positions_completed)),
            "plies": int(sum(self.plies_completed)),
            "evaluations": int(self.inference_positions.value),
            "actor_evaluations": int(sum(self.evaluated_positions)),
            "inference_requests": int(sum(self.inference_requests)),
            "batches": int(self.inference_batches.value),
            "max_batch": int(self.inference_max_batch.value),
            "request_queue_depth": int(queue_depth),
            "queue_wait_ns": int(sum(self.queue_wait_ns)),
            "response_wait_ns": int(sum(self.response_wait_ns)),
            "actor_compute_ns": int(sum(self.actor_compute_ns)),
        }

    def generate(self, target_positions, timeout=None, progress=True):
        window = InMemoryWindow(target_positions)
        started = time.monotonic()
        last_status = started
        previous = self.metric_snapshot()

        try:
            self.start()
            if progress:
                print(
                    f"Generating {target_positions:,} positions with "
                    f"{self.actor_count} actors, "
                    f"{self.actor_count * self.games_per_actor} active games, "
                    f"and {self.device}",
                    flush=True,
                )

            while not window.full:
                failure = self.process_failure()
                if failure:
                    raise RuntimeError(failure)
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise TimeoutError("Window generation did not finish before timeout")

                try:
                    game = self.game_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                window.add_game(game)
                now = time.monotonic()
                if progress and now - last_status >= self.config.runtime.status_interval_seconds:
                    current = self.metric_snapshot()
                    elapsed = max(now - last_status, 1e-6)
                    positions_per_second = (
                        current["positions"] - previous["positions"]
                    ) / elapsed
                    evaluations_per_second = (
                        current["evaluations"] - previous["evaluations"]
                    ) / elapsed
                    print(
                        f"window={window.position_count:,}/{target_positions:,} "
                        f"positions/s={positions_per_second:.1f} "
                        f"evals/s={evaluations_per_second:.1f} "
                        f"games={current['games']:,} "
                        f"queue={current['request_queue_depth']}",
                        flush=True,
                    )
                    previous = current
                    last_status = now
        finally:
            self.stop(drain_window=window)

        elapsed = time.monotonic() - started
        metrics = self.metric_snapshot()
        metrics.update(
            {
                "elapsed_seconds": elapsed,
                "window_positions": window.position_count,
                "positions_per_second": window.position_count / max(elapsed, 1e-6),
                "evaluations_per_second": metrics["evaluations"] / max(elapsed, 1e-6),
                "average_inference_batch": metrics["evaluations"]
                / max(metrics["batches"], 1),
                "memory_bytes": window.memory_bytes,
            }
        )
        self.release_ipc()
        return window, metrics

    def release_ipc(self):
        self.request_queue = None
        self.response_queues = []
        self.game_queue = None
        self.shared = None
        self.stop_event = None
        self.games_completed = None
        self.positions_completed = None
        self.plies_completed = None
        self.evaluated_positions = None
        self.inference_requests = None
        self.queue_wait_ns = None
        self.response_wait_ns = None
        self.actor_compute_ns = None
        self.inference_batches = None
        self.inference_positions = None
        self.inference_max_batch = None
        self.checkpoint_step = None
        self.server_ready = None
        self.actor_processes = []
        self.server_process = None
        gc.collect()

    def stop(self, drain_window=None):
        if not self.started:
            if self.shared is not None:
                self.shared.close()
                self.shared.unlink()
            return

        self.stop_event.set()
        deadline = time.monotonic() + 10.0

        while any(process.is_alive() for process in self.actor_processes):
            if drain_window is not None:
                try:
                    drain_window.add_game(self.game_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
            else:
                time.sleep(0.1)

            for process in self.actor_processes:
                process.join(timeout=0)
            if time.monotonic() >= deadline:
                break

        for process in self.actor_processes:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()

        if drain_window is not None:
            while True:
                try:
                    drain_window.add_game(self.game_queue.get_nowait())
                except queue.Empty:
                    break

        try:
            self.request_queue.put(None, timeout=1)
        except queue.Full:
            pass

        if self.server_process is not None:
            self.server_process.join(timeout=30)
            if self.server_process.is_alive():
                self.server_process.terminate()
                self.server_process.join()
            self.server_process.close()

        self.request_queue.cancel_join_thread()
        self.request_queue.close()
        for response_queue in self.response_queues:
            response_queue.cancel_join_thread()
            response_queue.close()
        self.game_queue.cancel_join_thread()
        self.game_queue.close()
        self.shared.close()
        self.shared.unlink()
        self.started = False
