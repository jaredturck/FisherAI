"""Generate self-play windows with multiprocessing and shared inference."""

import gc
import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.dataset import InMemoryWindow
from fisher_ai.encoding import INPUT_PLANES
from fisher_ai.game import MAX_GAME_PLIES
from fisher_ai.mcts import MAX_LEGAL_ACTIONS, MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.self_play import SelfPlayRunner

STATUS_INTERVAL_SECONDS = 10.0
ACTOR_SESSION_GROUPS = 2


class SharedArrayMemory:
    """Own or attach named NumPy arrays backed by shared memory."""

    def __init__(self, specs, descriptor=None):
        self.owner = descriptor is None
        self.segments = {}
        self.arrays = {}

        if self.owner:
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

    def close(self):
        """Close every attached shared-memory segment."""
        for segment in self.segments.values():
            segment.close()

    def unlink(self):
        """Unlink every owned shared-memory segment."""
        if not self.owner:
            return
        for segment in self.segments.values():
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


class SharedInferenceMemory(SharedArrayMemory):
    """Expose shared request and response arrays for neural inference."""

    def __init__(
        self,
        descriptor=None,
        actor_count=0,
        slot_count=ACTOR_SESSION_GROUPS,
        max_request_batch=0,
    ):
        specs = {
            "states": (
                (
                    actor_count,
                    slot_count,
                    max_request_batch,
                    INPUT_PLANES,
                    8,
                    8,
                ),
                np.float16,
            ),
            "legal_actions": (
                (
                    actor_count,
                    slot_count,
                    max_request_batch,
                    MAX_LEGAL_ACTIONS,
                ),
                np.int64,
            ),
            "legal_lengths": (
                (actor_count, slot_count, max_request_batch),
                np.uint16,
            ),
            "policy_logits": (
                (
                    actor_count,
                    slot_count,
                    max_request_batch,
                    MAX_LEGAL_ACTIONS,
                ),
                np.float32,
            ),
            "values": (
                (actor_count, slot_count, max_request_batch),
                np.float32,
            ),
        }
        super().__init__(specs, descriptor=descriptor)
        self.states = self.arrays["states"]
        self.legal_actions = self.arrays["legal_actions"]
        self.legal_lengths = self.arrays["legal_lengths"]
        self.policy_logits = self.arrays["policy_logits"]
        self.values = self.arrays["values"]


class SharedGameMemory(SharedArrayMemory):
    """Expose shared arrays used to transfer completed games."""

    def __init__(
        self,
        descriptor=None,
        actor_count=0,
        slot_count=ACTOR_SESSION_GROUPS,
    ):
        leading = (actor_count, slot_count, MAX_GAME_PLIES)
        specs = {
            "snapshot_bitboards": ((*leading, 12), np.uint64),
            "snapshot_repetitions": (leading, np.uint8),
            "current_colors": (leading, np.bool_),
            "plies": (leading, np.uint16),
            "castling_masks": (leading, np.uint8),
            "halfmove_clocks": (leading, np.uint8),
            "legal_lengths": (leading, np.uint16),
            "legal_actions": (
                (*leading, MAX_LEGAL_ACTIONS),
                np.uint16,
            ),
            "visit_counts": (
                (*leading, MAX_LEGAL_ACTIONS),
                np.uint16,
            ),
            "values": (leading, np.float32),
        }
        super().__init__(specs, descriptor=descriptor)
        self.snapshot_bitboards = self.arrays["snapshot_bitboards"]
        self.snapshot_repetitions = self.arrays["snapshot_repetitions"]
        self.current_colors = self.arrays["current_colors"]
        self.plies = self.arrays["plies"]
        self.castling_masks = self.arrays["castling_masks"]
        self.halfmove_clocks = self.arrays["halfmove_clocks"]
        self.legal_lengths = self.arrays["legal_lengths"]
        self.legal_actions = self.arrays["legal_actions"]
        self.visit_counts = self.arrays["visit_counts"]
        self.values = self.arrays["values"]

    def write(self, actor_id, slot_id, game):
        """Write one completed game into its actor transfer slot."""
        count = game.position_count
        key = (actor_id, slot_id, slice(0, count))
        self.snapshot_bitboards[key] = game.snapshot_bitboards
        self.snapshot_repetitions[key] = game.snapshot_repetitions
        self.current_colors[key] = game.current_colors
        self.plies[key] = game.plies
        self.castling_masks[key] = game.castling_masks
        self.halfmove_clocks[key] = game.halfmove_clocks
        self.legal_lengths[key] = game.legal_lengths
        self.legal_actions[key] = game.legal_actions
        self.visit_counts[key] = game.visit_counts
        self.values[key] = game.values
        return count

    def append_to_window(self, window, actor_id, slot_id, count):
        """Append one shared game slot into the training window."""
        key = (actor_id, slot_id, slice(0, count))
        window.add_arrays(
            self.snapshot_bitboards[key],
            self.snapshot_repetitions[key],
            self.current_colors[key],
            self.plies[key],
            self.castling_masks[key],
            self.halfmove_clocks[key],
            self.legal_lengths[key],
            self.legal_actions[key],
            self.visit_counts[key],
            self.values[key],
        )


class GenerationCancelled(Exception):
    """Signal that self-play generation was intentionally cancelled."""

    pass


class RemoteEvaluator:
    """Submit encoded positions to the shared inference server."""

    def __init__(
        self,
        actor_id,
        slot_id,
        request_queue,
        response_queue,
        shared,
        stop_event,
    ):
        self.actor_id = actor_id
        self.slot_id = slot_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.shared = shared
        self.stop_event = stop_event
        self.max_request_batch = self.shared.states.shape[2]
        self.request_id = 0

    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        """Evaluate an encoded batch through chunked shared requests."""
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
        """Submit one inference chunk and collect its outputs."""
        batch_size = len(encoded_states)
        max_actions = int(legal_lengths.max(initial=0))
        actor_id = self.actor_id
        slot_id = self.slot_id
        self.shared.states[actor_id, slot_id, :batch_size] = encoded_states
        self.shared.legal_lengths[
            actor_id,
            slot_id,
            :batch_size,
        ] = legal_lengths
        self.shared.legal_actions[
            actor_id,
            slot_id,
            :batch_size,
        ] = legal_actions

        self.request_id += 1
        request_id = self.request_id
        request = (actor_id, slot_id, batch_size, request_id)

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
                f"Inference response mismatch for actor {actor_id}, "
                f"slot {slot_id}: expected {request_id}, "
                f"received {response_id}"
            )
        if error:
            raise RuntimeError(error)

        policies = self.shared.policy_logits[
            actor_id,
            slot_id,
            :batch_size,
            :max_actions,
        ].copy()
        values = self.shared.values[
            actor_id,
            slot_id,
            :batch_size,
        ].copy()
        return policies, values


def configure_worker_threads():
    """Limit numerical libraries to one thread per actor process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def pin_actor_to_cpu(actor_id):
    """Pin one actor process to an available CPU core."""
    if not hasattr(os, "sched_setaffinity"):
        return

    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        cpu_id = available_cpus[actor_id % len(available_cpus)]
        os.sched_setaffinity(0, {cpu_id})
    except (OSError, IndexError):
        pass


def load_inference_model(device, checkpoint_path):
    """Load the inference network and optional checkpoint weights."""
    model = FisherNetwork()
    CheckpointManager().load(model, path=checkpoint_path, device=device)
    model.to(device)
    if torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    model.eval()
    return model


def collect_requests(
    request_queue,
    first_request,
    target_batch,
    maximum_batch,
    wait_seconds,
):
    """Drain compatible inference requests into one server batch."""
    requests = [first_request]
    state_count = first_request[2]
    deferred_request = None
    should_exit = False
    deadline = time.monotonic() + wait_seconds

    while state_count < target_batch:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            break
        try:
            request = request_queue.get(timeout=timeout)
        except queue.Empty:
            break
        if request is None:
            should_exit = True
            break
        if state_count + request[2] > maximum_batch:
            deferred_request = request
            break
        requests.append(request)
        state_count += request[2]

    while state_count < maximum_batch and not should_exit:
        try:
            request = request_queue.get_nowait()
        except queue.Empty:
            break
        if request is None:
            should_exit = True
            break
        if state_count + request[2] > maximum_batch:
            deferred_request = request
            break
        requests.append(request)
        state_count += request[2]

    return requests, state_count, deferred_request, should_exit


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
    """Serve batched GPU inference requests until generation stops."""
    configure_worker_threads()
    for actor_queues in response_queues:
        for response_queue in actor_queues:
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
        lengths = np.empty(maximum_batch, dtype=np.int64)
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

            (
                requests,
                state_count,
                deferred_request,
                should_exit,
            ) = collect_requests(
                request_queue,
                request,
                target_batch,
                maximum_batch,
                batch_wait_seconds,
            )

            offset = 0
            request_offsets = []
            max_legal_moves = 0
            for actor_id, slot_id, batch_size, request_id in requests:
                end = offset + batch_size
                pinned_states[offset:end].copy_(
                    torch.from_numpy(
                        shared.states[actor_id, slot_id, :batch_size]
                    )
                )
                actor_lengths = shared.legal_lengths[
                    actor_id,
                    slot_id,
                    :batch_size,
                ]
                lengths[offset:end] = actor_lengths
                actor_max = int(actor_lengths.max(initial=0))
                max_legal_moves = max(max_legal_moves, actor_max)
                pinned_actions[offset:end].copy_(
                    torch.from_numpy(
                        shared.legal_actions[
                            actor_id,
                            slot_id,
                            :batch_size,
                        ]
                    )
                )
                request_offsets.append(
                    (actor_id, slot_id, request_id, offset, end, actor_max)
                )
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

            for (
                actor_id,
                slot_id,
                request_id,
                start,
                end,
                actor_max,
            ) in request_offsets:
                shared.policy_logits[
                    actor_id,
                    slot_id,
                    : end - start,
                    :actor_max,
                ] = gathered[start:end, :actor_max]
                shared.values[
                    actor_id,
                    slot_id,
                    : end - start,
                ] = values[start:end]
                response_queues[actor_id][slot_id].put((request_id, None))

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
        for actor_queues in response_queues:
            for response_queue in actor_queues:
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


def put_game_message(game_queue, message, stop_event):
    """Queue a completed-game message while respecting cancellation."""
    while not stop_event.is_set():
        try:
            game_queue.put(message, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def wait_for_game_ack(ack_queue, stop_event):
    """Wait until the parent acknowledges a transferred game slot."""
    while not stop_event.is_set():
        try:
            ack_queue.get(timeout=0.5)
            return True
        except queue.Empty:
            continue
    return False


def actor_session_worker(
    actor_id,
    slot_id,
    config,
    inference_shared,
    game_shared,
    request_queue,
    response_queue,
    game_queue,
    game_ack_queue,
    stop_event,
    games_per_group,
    error_queue,
):
    """Generate games for one actor session group."""
    evaluator = RemoteEvaluator(
        actor_id,
        slot_id,
        request_queue,
        response_queue,
        inference_shared,
        stop_event,
    )
    search = MCTS(
        evaluator,
        simulations=config.simulations,
        parallel_searches=config.parallel_searches,
        seed=7 + actor_id * ACTOR_SESSION_GROUPS + slot_id,
    )
    runner = SelfPlayRunner(
        search,
        seed=7 + actor_id * ACTOR_SESSION_GROUPS + slot_id,
    )
    sessions = [runner.create_session() for _ in range(games_per_group)]

    try:
        while not stop_event.is_set():
            finished_indices = runner.advance_sessions(sessions)
            for index in finished_indices:
                record = sessions[index].build_record()
                count = game_shared.write(actor_id, slot_id, record)
                message = (actor_id, slot_id, count)
                if not put_game_message(game_queue, message, stop_event):
                    return
                if not wait_for_game_ack(game_ack_queue, stop_event):
                    return
                sessions[index] = runner.create_session()
    except GenerationCancelled:
        return
    except Exception as error:
        error_queue.put(repr(error))
        stop_event.set()


def actor_main(
    actor_id,
    config_path,
    inference_descriptor,
    game_descriptor,
    request_queue,
    response_queues,
    game_queue,
    game_ack_queues,
    stop_event,
    games_per_actor,
):
    """Run both self-play session groups for one actor process."""
    configure_worker_threads()
    request_queue.cancel_join_thread()
    game_queue.cancel_join_thread()
    pin_actor_to_cpu(actor_id)
    config = load_config(config_path)
    inference_shared = SharedInferenceMemory(descriptor=inference_descriptor)
    game_shared = SharedGameMemory(descriptor=game_descriptor)
    error_queue = queue.Queue()
    workers = []

    try:
        for slot_id in range(ACTOR_SESSION_GROUPS):
            games_per_group = games_per_actor // ACTOR_SESSION_GROUPS
            if slot_id < games_per_actor % ACTOR_SESSION_GROUPS:
                games_per_group += 1
            if not games_per_group:
                continue

            worker = threading.Thread(
                target=actor_session_worker,
                args=(
                    actor_id,
                    slot_id,
                    config,
                    inference_shared,
                    game_shared,
                    request_queue,
                    response_queues[slot_id],
                    game_queue,
                    game_ack_queues[slot_id],
                    stop_event,
                    games_per_group,
                    error_queue,
                ),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        while any(worker.is_alive() for worker in workers):
            try:
                error = error_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            raise RuntimeError(error)
    finally:
        if not error_queue.empty():
            stop_event.set()
        for worker in workers:
            worker.join(timeout=1)
        inference_shared.close()
        game_shared.close()


def round_request_capacity(value):
    """Round an actor request capacity to an efficient boundary."""
    if value <= 32:
        return max(1, value)
    return ((value + 31) // 32) * 32


class WindowGenerator:
    """Coordinate actors and inference to produce one self-play window."""

    def __init__(
        self,
        config_path="fisher_config.json",
        checkpoint_path=None,
    ):
        self.context = mp.get_context("spawn")
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.actor_count = self.config.actor_processes
        self.games_per_actor = self.config.games_per_actor
        self.device = available_device(self.config.device)
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        largest_group = (
            self.games_per_actor + ACTOR_SESSION_GROUPS - 1
        ) // ACTOR_SESSION_GROUPS
        self.request_capacity = round_request_capacity(
            largest_group * self.config.parallel_searches
        )
        if self.request_capacity > self.config.inference_max_batch_size:
            raise ValueError(
                f"Actor request capacity {self.request_capacity} exceeds "
                "inference_max_batch_size="
                f"{self.config.inference_max_batch_size}"
            )

        self.stop_event = self.context.Event()
        self.request_queue = self.context.Queue(
            maxsize=max(self.actor_count * ACTOR_SESSION_GROUPS * 4, 8)
        )
        self.response_queues = [
            [
                self.context.Queue(maxsize=2)
                for _ in range(ACTOR_SESSION_GROUPS)
            ]
            for _ in range(self.actor_count)
        ]
        self.game_queue = self.context.Queue(
            maxsize=max(self.actor_count * ACTOR_SESSION_GROUPS * 2, 8)
        )
        self.game_ack_queues = [
            [
                self.context.Queue(maxsize=1)
                for _ in range(ACTOR_SESSION_GROUPS)
            ]
            for _ in range(self.actor_count)
        ]
        self.shared = SharedInferenceMemory(
            actor_count=self.actor_count,
            max_request_batch=self.request_capacity,
        )
        self.game_shared = SharedGameMemory(actor_count=self.actor_count)
        self.inference_batches = self.context.Value("q", 0, lock=False)
        self.inference_positions = self.context.Value("q", 0, lock=False)
        self.inference_max_batch = self.context.Value("q", 0, lock=False)
        self.server_ready = self.context.Value("b", 0, lock=False)
        self.actor_processes = []
        self.server_process = None
        self.started = False

    @property
    def active_game_count(self):
        """Return the total number of concurrently active games."""
        return self.actor_count * self.games_per_actor

    def start(self):
        """Start shared inference and self-play actor processes."""
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
                    self.game_shared.descriptor,
                    self.request_queue,
                    self.response_queues[actor_id],
                    self.game_queue,
                    self.game_ack_queues[actor_id],
                    self.stop_event,
                    self.games_per_actor,
                ),
            )
            process.start()
            self.actor_processes.append(process)

    def process_failure(self):
        """Raise the first failure reported by a child process."""
        for process in [self.server_process, *self.actor_processes]:
            if process is not None and process.exitcode is not None:
                return f"{process.name} exited with code {process.exitcode}"
        return None

    def metric_snapshot(self):
        """Return current neural inference throughput metrics."""
        return {
            "evaluations": int(self.inference_positions.value),
            "batches": int(self.inference_batches.value),
            "max_batch": int(self.inference_max_batch.value),
        }

    def consume_game(self, window, timeout=None):
        """Consume one completed game into the active window."""
        if timeout is None:
            actor_id, slot_id, count = self.game_queue.get_nowait()
        else:
            actor_id, slot_id, count = self.game_queue.get(timeout=timeout)
        self.game_shared.append_to_window(window, actor_id, slot_id, count)
        self.game_ack_queues[actor_id][slot_id].put(None)

    def generate(self, target_positions, timeout=None, progress=True):
        """Generate a fresh in-memory window of self-play positions."""
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
                    f"{self.active_game_count} active games, "
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
                    self.consume_game(window, timeout=0.5)
                except queue.Empty:
                    continue

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
                        f"games={window.game_count:,}",
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
        self.release_ipc()
        return window, metrics

    def stop(self, drain_window=None):
        """Stop child processes and optionally drain completed games."""
        if not self.started:
            self.shared.close()
            self.shared.unlink()
            self.game_shared.close()
            self.game_shared.unlink()
            return

        self.stop_event.set()
        deadline = time.monotonic() + 10.0

        while any(process.is_alive() for process in self.actor_processes):
            if drain_window is not None:
                try:
                    self.consume_game(drain_window, timeout=0.1)
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
                    self.consume_game(drain_window)
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
        for actor_queues in self.response_queues:
            for response_queue in actor_queues:
                response_queue.cancel_join_thread()
                response_queue.close()
        self.game_queue.cancel_join_thread()
        self.game_queue.close()
        for actor_queues in self.game_ack_queues:
            for ack_queue in actor_queues:
                ack_queue.cancel_join_thread()
                ack_queue.close()
        self.shared.close()
        self.shared.unlink()
        self.game_shared.close()
        self.game_shared.unlink()
        self.started = False

    def release_ipc(self):
        """Close and unlink all interprocess communication resources."""
        self.request_queue = None
        self.response_queues = []
        self.game_queue = None
        self.game_ack_queues = []
        self.shared = None
        self.game_shared = None
        self.stop_event = None
        self.inference_batches = None
        self.inference_positions = None
        self.inference_max_batch = None
        self.server_ready = None
        self.actor_processes = []
        self.server_process = None
        gc.collect()
