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
from fisher_ai.encoding import INPUT_PLANES
from fisher_ai.mcts import MAX_LEGAL_ACTIONS, MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.self_play import SelfPlayRunner

STATUS_INTERVAL_SECONDS = 10.0


class SharedInferenceMemory:
    def __init__(
        self,
        descriptor=None,
        actor_count=0,
        max_request_batch=0,
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
                    (
                        actor_count,
                        max_request_batch,
                        MAX_LEGAL_ACTIONS,
                    ),
                    np.uint16,
                ),
                "legal_lengths": (
                    (actor_count, max_request_batch),
                    np.uint16,
                ),
                "policy_logits": (
                    (
                        actor_count,
                        max_request_batch,
                        MAX_LEGAL_ACTIONS,
                    ),
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
    ):
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.shared = SharedInferenceMemory(descriptor=shared_descriptor)
        self.stop_event = stop_event
        self.max_request_batch = self.shared.states.shape[1]
        self.request_id = 0

    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        batch_size = len(encoded_states)
        policies = np.empty(
            (batch_size, legal_actions.shape[1]),
            dtype=np.float32,
        )
        values = np.empty(batch_size, dtype=np.float32)

        for start in range(0, batch_size, self.max_request_batch):
            end = min(start + self.max_request_batch, batch_size)
            chunk_policies, chunk_values = self.evaluate_chunk(
                encoded_states[start:end],
                legal_actions[start:end],
                legal_lengths[start:end],
            )
            policies[start:end, : chunk_policies.shape[1]] = chunk_policies
            values[start:end] = chunk_values

        return policies, values

    def evaluate_chunk(self, encoded_states, legal_actions, legal_lengths):
        batch_size = len(encoded_states)
        max_actions = int(legal_lengths.max(initial=0))
        self.shared.states[self.actor_id, :batch_size] = encoded_states
        self.shared.legal_lengths[
            self.actor_id,
            :batch_size,
        ] = legal_lengths
        self.shared.legal_actions[
            self.actor_id,
            :batch_size,
            :max_actions,
        ] = legal_actions[:, :max_actions]

        self.request_id += 1
        request_id = self.request_id
        request = (self.actor_id, batch_size, request_id)

        while not self.stop_event.is_set():
            try:
                self.request_queue.put(request, timeout=0.5)
                break
            except queue.Full:
                continue
        else:
            raise GenerationCancelled(
                "Generation stopped before inference submission"
            )

        while not self.stop_event.is_set():
            try:
                response_id, error = self.response_queue.get(timeout=0.5)
                break
            except queue.Empty:
                continue
        else:
            raise GenerationCancelled(
                "Generation stopped before inference response"
            )

        if response_id == -1 and error:
            raise RuntimeError(error)
        if response_id != request_id:
            raise RuntimeError(
                f"Inference response mismatch for actor {self.actor_id}: "
                f"expected {request_id}, received {response_id}"
            )
        if error:
            raise RuntimeError(error)

        policies = self.shared.policy_logits[
            self.actor_id,
            :batch_size,
            :max_actions,
        ].copy()
        values = self.shared.values[self.actor_id, :batch_size].copy()
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


def load_inference_model(device, checkpoint_path):
    model = FisherNetwork()
    CheckpointManager().load(model, path=checkpoint_path, device=device)
    model.to(device)
    if torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    model.eval()
    return model


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
    server_ready,
):
    configure_worker_threads()
    for response_queue in response_queues:
        response_queue.cancel_join_thread()

    config = load_config(config_path)
    device = torch.device(available_device(device_name))
    shared = SharedInferenceMemory(descriptor=shared_descriptor)
    model = None

    try:
        model = load_inference_model(device, checkpoint_path)
        server_ready.value = 1
        target_batch = config.inference_batch_size
        maximum_batch = config.inference_max_batch_size
        pinned_states = torch.empty(
            (maximum_batch, INPUT_PLANES, 8, 8),
            dtype=torch.float16,
            pin_memory=device.type == "cuda",
        )
        pinned_actions = torch.empty(
            (maximum_batch, MAX_LEGAL_ACTIONS),
            dtype=torch.int64,
            pin_memory=device.type == "cuda",
        )
        batch_wait_seconds = config.inference_batch_wait_ms / 1000.0
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
            assert state_count <= maximum_batch
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
                actor_lengths = shared.legal_lengths[
                    actor_id,
                    :batch_size,
                ].astype(np.int64, copy=True)
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

            states = pinned_states[:state_count].to(
                device,
                non_blocking=True,
            )
            if device.type == "cpu":
                states = states.float()
            else:
                states = states.contiguous(memory_format=torch.channels_last)

            actions = pinned_actions[
                :state_count,
                :max_legal_moves,
            ].to(device, non_blocking=True)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    policy, values = model(states)
                    gathered = policy.gather(1, actions)

            gathered = gathered.float().cpu().numpy()
            values = values.float().cpu().numpy()

            for actor_id, request_id, start, end in request_offsets:
                for local_index, length in enumerate(lengths[start:end]):
                    shared.policy_logits[
                        actor_id,
                        local_index,
                        :length,
                    ] = gathered[start + local_index, :length]
                shared.values[actor_id, : end - start] = values[start:end]
                response_queues[actor_id].put((request_id, None))

            inference_batches.value += 1
            inference_positions.value += state_count
            inference_max_batch.value = max(
                inference_max_batch.value,
                state_count,
            )

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


def put_completed_game(target_queue, game, stop_event):
    while not stop_event.is_set():
        try:
            target_queue.put(game, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def actor_main(
    actor_id,
    config_path,
    shared_descriptor,
    request_queue,
    response_queue,
    game_queue,
    stop_event,
    games_per_actor,
):
    configure_worker_threads()
    request_queue.cancel_join_thread()
    game_queue.cancel_join_thread()
    pin_actor_to_cpu(actor_id)
    config = load_config(config_path)

    evaluator = RemoteEvaluator(
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
    )
    search = MCTS(
        evaluator,
        simulations=config.simulations,
        parallel_searches=config.parallel_searches,
        seed=7 + actor_id,
    )
    runner = SelfPlayRunner(search, seed=7 + actor_id)
    sessions = [runner.create_session() for _ in range(games_per_actor)]

    try:
        while not stop_event.is_set():
            finished_indices = runner.advance_sessions(sessions)
            for index in finished_indices:
                record = sessions[index].build_record()
                if not put_completed_game(game_queue, record, stop_event):
                    return
                sessions[index] = runner.create_session()
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
    ):
        import multiprocessing as mp

        self.context = mp.get_context("spawn")
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.actor_count = self.config.actor_processes
        self.games_per_actor = self.config.games_per_actor
        self.device = available_device(self.config.device)
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.request_capacity = round_request_capacity(
            self.games_per_actor * self.config.parallel_searches
        )
        if self.request_capacity > self.config.inference_max_batch_size:
            raise ValueError(
                f"Actor request capacity {self.request_capacity} exceeds "
                f"inference_max_batch_size="
                f"{self.config.inference_max_batch_size}"
            )

        self.stop_event = self.context.Event()
        self.request_queue = self.context.Queue(
            maxsize=max(self.actor_count * 4, 8)
        )
        self.response_queues = [
            self.context.Queue(maxsize=2) for _ in range(self.actor_count)
        ]
        self.game_queue = self.context.Queue(
            maxsize=max(self.actor_count * 2, 8)
        )
        self.shared = SharedInferenceMemory(
            actor_count=self.actor_count,
            max_request_batch=self.request_capacity,
        )
        self.inference_batches = self.context.Value("q", 0, lock=False)
        self.inference_positions = self.context.Value("q", 0, lock=False)
        self.inference_max_batch = self.context.Value("q", 0, lock=False)
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
                self.server_ready,
            ),
        )
        self.server_process.start()
        self.started = True

        deadline = time.monotonic() + 60.0
        while self.server_ready.value == 0:
            if self.server_process.exitcode is not None:
                raise RuntimeError(
                    "Inference server exited with code "
                    f"{self.server_process.exitcode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Inference server did not become ready within 60 seconds"
                )
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
                    self.games_per_actor,
                ),
            )
            process.start()
            self.actor_processes.append(process)

    def process_failure(self):
        for process in [self.server_process, *self.actor_processes]:
            if process is not None and process.exitcode is not None:
                return f"{process.name} exited with code {process.exitcode}"
        return None

    def metric_snapshot(self):
        return {
            "evaluations": int(self.inference_positions.value),
            "batches": int(self.inference_batches.value),
            "max_batch": int(self.inference_max_batch.value),
        }

    def generate(self, target_positions, timeout=None, progress=True):
        window = InMemoryWindow(target_positions)
        started = time.monotonic()
        last_status = started
        previous_positions = 0
        previous_evaluations = 0

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
                if (
                    timeout is not None
                    and time.monotonic() - started >= timeout
                ):
                    raise TimeoutError(
                        "Window generation did not finish before timeout"
                    )

                try:
                    game = self.game_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                window.add_game(game)
                now = time.monotonic()
                if progress and now - last_status >= STATUS_INTERVAL_SECONDS:
                    current = self.metric_snapshot()
                    elapsed = max(now - last_status, 1e-6)
                    positions_per_second = (
                        window.position_count - previous_positions
                    ) / elapsed
                    evaluations_per_second = (
                        current["evaluations"] - previous_evaluations
                    ) / elapsed
                    print(
                        f"window={window.position_count:,}/"
                        f"{target_positions:,} "
                        f"positions/s={positions_per_second:.1f} "
                        f"evals/s={evaluations_per_second:.1f} "
                        f"games={len(window.games):,}",
                        flush=True,
                    )
                    previous_positions = window.position_count
                    previous_evaluations = current["evaluations"]
                    last_status = now
        finally:
            self.stop(drain_window=window)

        elapsed = time.monotonic() - started
        metrics = self.metric_snapshot()
        metrics.update(
            {
                "elapsed_seconds": elapsed,
                "games": len(window.games),
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
        self.release_ipc()
        return window, metrics

    def stop(self, drain_window=None):
        if not self.started:
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

    def release_ipc(self):
        self.request_queue = None
        self.response_queues = []
        self.game_queue = None
        self.shared = None
        self.stop_event = None
        self.inference_batches = None
        self.inference_positions = None
        self.inference_max_batch = None
        self.server_ready = None
        self.actor_processes = []
        self.server_process = None
        gc.collect()
