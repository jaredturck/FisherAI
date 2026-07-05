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
from fisher_ai.encoding import INPUT_PLANES, encode_state
from fisher_ai.mcts import MCTS
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import ReplayBuffer
from fisher_ai.self_play import SelfPlayRunner

BATCHED_SCHEDULER = "batched"
THREADED_SCHEDULER = "threaded"
VALID_SCHEDULERS = {BATCHED_SCHEDULER, THREADED_SCHEDULER}


class SharedInferenceMemory:
    def __init__(
        self,
        descriptor=None,
        actor_count=0,
        slots_per_actor=0,
        max_request_batch=0,
        max_legal_actions=256,
    ):
        self.owner = descriptor is None
        self.segments = {}
        self.arrays = {}

        if self.owner:
            specs = {
                "states": (
                    (
                        actor_count,
                        slots_per_actor,
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
                        slots_per_actor,
                        max_request_batch,
                        max_legal_actions,
                    ),
                    np.uint16,
                ),
                "legal_lengths": (
                    (actor_count, slots_per_actor, max_request_batch),
                    np.uint16,
                ),
                "policy_logits": (
                    (
                        actor_count,
                        slots_per_actor,
                        max_request_batch,
                        max_legal_actions,
                    ),
                    np.float32,
                ),
                "values": (
                    (actor_count, slots_per_actor, max_request_batch),
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


class SelfPlayCancelled(Exception):
    pass


def add_counter(counter, actor_id, value, lock=None):
    if lock is None:
        counter[actor_id] += value
        return
    with lock:
        counter[actor_id] += value


class RemoteEvaluator:
    """Blocking evaluator used by the process-level batched actor scheduler."""

    def __init__(
        self,
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
        evaluated_positions,
        outstanding_requests,
        blocked_slot_waits,
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
        self.outstanding_requests = outstanding_requests
        self.blocked_slot_waits = blocked_slot_waits
        self.inference_requests = inference_requests
        self.queue_wait_ns = queue_wait_ns
        self.response_wait_ns = response_wait_ns
        self.max_request_batch = self.shared.states.shape[2]
        self.max_legal_actions = self.shared.legal_actions.shape[3]
        self.request_id = 0
        self.thread_timing = threading.local()

    def reset_thread_timing(self):
        self.thread_timing.queue_wait_ns = 0
        self.thread_timing.response_wait_ns = 0

    def thread_wait_ns(self):
        return (
            getattr(self.thread_timing, "queue_wait_ns", 0)
            + getattr(self.thread_timing, "response_wait_ns", 0)
        )

    def record_queue_wait(self, elapsed):
        self.queue_wait_ns[self.actor_id] += elapsed
        self.thread_timing.queue_wait_ns = (
            getattr(self.thread_timing, "queue_wait_ns", 0) + elapsed
        )

    def record_response_wait(self, elapsed):
        self.response_wait_ns[self.actor_id] += elapsed
        self.thread_timing.response_wait_ns = (
            getattr(self.thread_timing, "response_wait_ns", 0) + elapsed
        )

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

        slot_id = 0
        for index, (encoded_state, actions) in enumerate(
            zip(encoded_states, legal_actions, strict=True)
        ):
            if len(actions) > self.max_legal_actions:
                raise RuntimeError(
                    f"Position has {len(actions)} legal moves, exceeding the configured "
                    f"maximum of {self.max_legal_actions}"
                )
            self.shared.states[self.actor_id, slot_id, index] = encoded_state
            length = len(actions)
            self.shared.legal_lengths[self.actor_id, slot_id, index] = length
            self.shared.legal_actions[
                self.actor_id,
                slot_id,
                index,
                :length,
            ] = actions

        self.request_id += 1
        request_id = self.request_id
        request = (self.actor_id, slot_id, batch_size, request_id)
        self.outstanding_requests[self.actor_id] += 1
        self.inference_requests[self.actor_id] += 1

        try:
            queue_started = time.perf_counter_ns()
            while not self.stop_event.is_set():
                try:
                    self.request_queue.put(request, timeout=0.5)
                    break
                except queue.Full:
                    continue
            else:
                raise SelfPlayCancelled(
                    "Self-play stopped before submitting an inference request"
                )
            self.record_queue_wait(time.perf_counter_ns() - queue_started)

            response_started = time.perf_counter_ns()
            while not self.stop_event.is_set():
                try:
                    response_id, response_slot, error = self.response_queue.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            else:
                raise SelfPlayCancelled(
                    "Self-play inference stopped before returning a response"
                )
            self.record_response_wait(time.perf_counter_ns() - response_started)

            if response_id == -1 and error:
                raise RuntimeError(error)
            if response_id != request_id or response_slot != slot_id:
                raise RuntimeError(
                    f"Inference response mismatch for actor {self.actor_id}: "
                    f"expected request {request_id} slot {slot_id}, received "
                    f"request {response_id} slot {response_slot}"
                )
            if error:
                raise RuntimeError(error)

            policies = []
            for index, actions in enumerate(legal_actions):
                length = len(actions)
                policies.append(
                    self.shared.policy_logits[
                        self.actor_id,
                        slot_id,
                        index,
                        :length,
                    ].copy()
                )
            values = self.shared.values[
                self.actor_id,
                slot_id,
                :batch_size,
            ].copy()
            self.evaluated_positions[self.actor_id] += batch_size
            return policies, values
        finally:
            if self.outstanding_requests[self.actor_id] > 0:
                self.outstanding_requests[self.actor_id] -= 1

    def close(self):
        self.shared.close()


class PendingInference:
    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.event = threading.Event()
        self.error = None


class ConcurrentRemoteEvaluator:
    """Multi-slot evaluator retained only for threaded scheduler comparisons."""

    def __init__(
        self,
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
        evaluated_positions,
        outstanding_requests,
        blocked_slot_waits,
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
        self.outstanding_requests = outstanding_requests
        self.blocked_slot_waits = blocked_slot_waits
        self.inference_requests = inference_requests
        self.queue_wait_ns = queue_wait_ns
        self.response_wait_ns = response_wait_ns
        self.slot_count = self.shared.states.shape[1]
        self.max_request_batch = self.shared.states.shape[2]
        self.max_legal_actions = self.shared.legal_actions.shape[3]
        self.available_slots = queue.Queue(maxsize=self.slot_count)
        for slot_id in range(self.slot_count):
            self.available_slots.put(slot_id)
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.thread_timing = threading.local()
        self.request_id = 0
        self.dispatcher = threading.Thread(
            target=self.dispatch_responses,
            name=f"fisher-response-{actor_id:02d}",
            daemon=True,
        )
        self.dispatcher.start()

    def reset_thread_timing(self):
        self.thread_timing.queue_wait_ns = 0
        self.thread_timing.response_wait_ns = 0

    def thread_wait_ns(self):
        return (
            getattr(self.thread_timing, "queue_wait_ns", 0)
            + getattr(self.thread_timing, "response_wait_ns", 0)
        )

    def record_queue_wait(self, elapsed):
        add_counter(self.queue_wait_ns, self.actor_id, elapsed, self.counter_lock)
        self.thread_timing.queue_wait_ns = (
            getattr(self.thread_timing, "queue_wait_ns", 0) + elapsed
        )

    def record_response_wait(self, elapsed):
        add_counter(self.response_wait_ns, self.actor_id, elapsed, self.counter_lock)
        self.thread_timing.response_wait_ns = (
            getattr(self.thread_timing, "response_wait_ns", 0) + elapsed
        )

    def dispatch_responses(self):
        while not self.stop_event.is_set():
            try:
                response = self.response_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if response is None:
                return

            request_id, slot_id, error = response
            if request_id == -1 and error:
                with self.pending_lock:
                    pending_requests = list(self.pending.values())
                for pending in pending_requests:
                    pending.error = error
                    pending.event.set()
                continue

            with self.pending_lock:
                pending = self.pending.get(request_id)
            if pending is None or pending.slot_id != slot_id:
                continue
            pending.error = error
            pending.event.set()

    def next_request_id(self):
        with self.pending_lock:
            self.request_id += 1
            return self.request_id

    def acquire_slot(self):
        try:
            return self.available_slots.get_nowait()
        except queue.Empty:
            add_counter(self.blocked_slot_waits, self.actor_id, 1, self.counter_lock)
            while not self.stop_event.is_set():
                try:
                    return self.available_slots.get(timeout=0.5)
                except queue.Empty:
                    continue
        raise SelfPlayCancelled(
            "Self-play stopped while waiting for an inference slot"
        )

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

        slot_id = self.acquire_slot()
        request_id = self.next_request_id()
        pending = PendingInference(slot_id)

        try:
            for index, (encoded_state, actions) in enumerate(
                zip(encoded_states, legal_actions, strict=True)
            ):
                if len(actions) > self.max_legal_actions:
                    raise RuntimeError(
                        f"Position has {len(actions)} legal moves, exceeding the configured "
                        f"maximum of {self.max_legal_actions}"
                    )
                self.shared.states[self.actor_id, slot_id, index] = encoded_state
                length = len(actions)
                self.shared.legal_lengths[self.actor_id, slot_id, index] = length
                self.shared.legal_actions[
                    self.actor_id,
                    slot_id,
                    index,
                    :length,
                ] = actions

            with self.pending_lock:
                self.pending[request_id] = pending
            add_counter(
                self.outstanding_requests,
                self.actor_id,
                1,
                self.counter_lock,
            )
            add_counter(
                self.inference_requests,
                self.actor_id,
                1,
                self.counter_lock,
            )

            request = (self.actor_id, slot_id, batch_size, request_id)
            queue_started = time.perf_counter_ns()
            while not self.stop_event.is_set():
                try:
                    self.request_queue.put(request, timeout=0.5)
                    break
                except queue.Full:
                    continue
            else:
                raise SelfPlayCancelled(
                    "Self-play stopped before submitting an inference request"
                )
            self.record_queue_wait(time.perf_counter_ns() - queue_started)

            response_started = time.perf_counter_ns()
            while not pending.event.wait(timeout=0.5):
                if self.stop_event.is_set():
                    raise SelfPlayCancelled(
                        "Self-play inference stopped before returning a response"
                    )
            self.record_response_wait(time.perf_counter_ns() - response_started)

            if self.stop_event.is_set() and pending.error is None:
                raise SelfPlayCancelled(
                    "Self-play inference stopped before returning a response"
                )
            if pending.error:
                raise RuntimeError(pending.error)

            policies = []
            for index, actions in enumerate(legal_actions):
                length = len(actions)
                policies.append(
                    self.shared.policy_logits[
                        self.actor_id,
                        slot_id,
                        index,
                        :length,
                    ].copy()
                )
            values = self.shared.values[
                self.actor_id,
                slot_id,
                :batch_size,
            ].copy()
            add_counter(
                self.evaluated_positions,
                self.actor_id,
                batch_size,
                self.counter_lock,
            )
            return policies, values
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            with self.counter_lock:
                if self.outstanding_requests[self.actor_id] > 0:
                    self.outstanding_requests[self.actor_id] -= 1
            self.available_slots.put(slot_id)

    def cancel_pending(self):
        with self.pending_lock:
            pending_requests = list(self.pending.values())
        for pending in pending_requests:
            pending.event.set()

    def close(self):
        self.cancel_pending()
        try:
            self.response_queue.put_nowait(None)
        except queue.Full:
            pass
        self.dispatcher.join(timeout=2)
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


def build_checkpoint_manager(config):
    return CheckpointManager(config.runtime.checkpoint_dir)


def load_inference_model(config, device, checkpoint_path=None):
    model = FisherNetwork(config.network)
    manager = build_checkpoint_manager(config)
    path = Path(checkpoint_path) if checkpoint_path else manager.latest_path()
    if path is None:
        path = manager.save(model, config, 0)
    step, _ = manager.load(model, path=path, device=device)
    model.to(device)
    if config.runtime.channels_last and torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    model.eval()
    return model, manager, path, step


def inference_server_main(
    server_id,
    device_name,
    config_path,
    checkpoint_path,
    shared_descriptor,
    request_queue,
    response_queues,
    stop_event,
    inference_batches,
    inference_positions,
    inference_max_batches,
    inference_histogram,
    checkpoint_steps,
    server_ready,
):
    configure_worker_threads()
    config = load_config(config_path)
    device_name = available_device(device_name)
    device = torch.device(device_name)
    shared = SharedInferenceMemory(descriptor=shared_descriptor)
    model = None

    try:
        model, manager, loaded_path, step = load_inference_model(
            config,
            device,
            checkpoint_path=checkpoint_path,
        )
        checkpoint_steps[server_id] = step
        server_ready[server_id] = 1
        target_batch = config.runtime.inference_batch_size
        maximum_batch = config.runtime.inference_max_batch_size
        max_legal_actions = shared.legal_actions.shape[3]
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
        reload_deadline = time.monotonic() + config.runtime.checkpoint_reload_seconds
        deferred_request = None
        should_exit = False
        histogram_width = maximum_batch + 1

        while not should_exit:
            if deferred_request is None:
                request = request_queue.get()
            else:
                request = deferred_request
                deferred_request = None
            if request is None:
                break

            requests = [request]
            state_count = request[2]
            if state_count > maximum_batch:
                raise RuntimeError(
                    f"Inference request of {state_count} positions exceeds maximum GPU "
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
                if state_count + next_request[2] > maximum_batch:
                    deferred_request = next_request
                    break

                requests.append(next_request)
                state_count += next_request[2]

            offset = 0
            request_offsets = []
            lengths = np.zeros(state_count, dtype=np.int64)
            max_legal_moves = 0
            pinned_actions[:state_count].zero_()

            for actor_id, slot_id, batch_size, request_id in requests:
                end = offset + batch_size
                pinned_states[offset:end].copy_(
                    torch.from_numpy(shared.states[actor_id, slot_id, :batch_size])
                )
                actor_lengths = shared.legal_lengths[
                    actor_id,
                    slot_id,
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
                                slot_id,
                                :batch_size,
                                :actor_max,
                            ].astype(np.int64, copy=False)
                        )
                    )
                request_offsets.append((actor_id, slot_id, request_id, offset, end))
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

            for actor_id, slot_id, request_id, start, end in request_offsets:
                batch_lengths = lengths[start:end]
                for local_index, length in enumerate(batch_lengths):
                    shared.policy_logits[
                        actor_id,
                        slot_id,
                        local_index,
                        :length,
                    ] = gathered[start + local_index, :length]
                shared.values[actor_id, slot_id, : end - start] = values[start:end]
                response_queues[actor_id].put((request_id, slot_id, None))

            inference_batches[server_id] += 1
            inference_positions[server_id] += state_count
            inference_max_batches[server_id] = max(
                inference_max_batches[server_id],
                state_count,
            )
            inference_histogram[server_id * histogram_width + state_count] += 1

            if checkpoint_path is None and time.monotonic() >= reload_deadline:
                latest_path = manager.latest_path()
                if latest_path is not None and latest_path != loaded_path:
                    step, _ = manager.load(model, path=latest_path, device=device)
                    model.eval()
                    loaded_path = latest_path
                    checkpoint_steps[server_id] = step
                reload_deadline = time.monotonic() + config.runtime.checkpoint_reload_seconds

    except Exception as error:
        server_ready[server_id] = -1
        stop_event.set()
        message = f"Inference server {server_id} failed: {error}"
        for response_queue in response_queues:
            try:
                response_queue.put_nowait((-1, -1, message))
            except queue.Full:
                pass
        raise
    finally:
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        shared.close()


def put_with_stop(target_queue, value, stop_event):
    while not stop_event.is_set():
        try:
            target_queue.put(value, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def checkpoint_step(checkpoint_steps):
    return int(max(checkpoint_steps)) if len(checkpoint_steps) else 0


def create_runner(actor_id, config, evaluator, seed_offset=0):
    seed = config.runtime.seed + actor_id * 1000 + seed_offset
    search = MCTS(evaluator, config.search, seed=seed)
    return SelfPlayRunner(
        search,
        config.search,
        training_config=config.training,
        seed=seed,
    )


def run_batched_actor(
    actor_id,
    config,
    evaluator,
    game_queue,
    stop_event,
    games_completed,
    positions_completed,
    plies_completed,
    replay_game_count,
    checkpoint_steps,
    games_per_actor,
    actor_loop_ns,
    actor_compute_ns,
    replay_wait_ns,
):
    runner = create_runner(actor_id, config, evaluator)
    sessions = [
        runner.create_session(
            checkpoint_step=checkpoint_step(checkpoint_steps),
            allow_resignation=False,
        )
        for _ in range(games_per_actor)
    ]

    while not stop_event.is_set():
        allow_resignation = (
            replay_game_count.value
            >= config.training.resignation_enabled_after_games
        )
        before_plies = sum(len(session.moves) for session in sessions)
        evaluator.reset_thread_timing()
        loop_started = time.perf_counter_ns()
        finished_indices = runner.advance_sessions(
            sessions,
            allow_resignation=allow_resignation,
        )
        elapsed = time.perf_counter_ns() - loop_started
        inference_wait = evaluator.thread_wait_ns()
        actor_loop_ns[actor_id] += elapsed
        actor_compute_ns[actor_id] += max(elapsed - inference_wait, 0)
        after_plies = sum(len(session.moves) for session in sessions)
        plies_completed[actor_id] += after_plies - before_plies

        for index in finished_indices:
            session = sessions[index]
            record = session.build_record()
            replay_started = time.perf_counter_ns()
            if not put_with_stop(game_queue, record, stop_event):
                return
            replay_wait_ns[actor_id] += time.perf_counter_ns() - replay_started
            games_completed[actor_id] += 1
            positions_completed[actor_id] += len(record.samples)
            sessions[index] = runner.create_session(
                checkpoint_step=checkpoint_step(checkpoint_steps),
                allow_resignation=allow_resignation,
            )


def play_game_thread(
    actor_id,
    game_id,
    config,
    evaluator,
    game_queue,
    stop_event,
    games_completed,
    positions_completed,
    plies_completed,
    replay_game_count,
    checkpoint_steps,
    counter_lock,
    error_queue,
    actor_loop_ns,
    actor_compute_ns,
    replay_wait_ns,
):
    try:
        runner = create_runner(actor_id, config, evaluator, seed_offset=game_id)
        while not stop_event.is_set():
            allow_resignation = (
                replay_game_count.value
                >= config.training.resignation_enabled_after_games
            )
            session = runner.create_session(
                checkpoint_step=checkpoint_step(checkpoint_steps),
                allow_resignation=allow_resignation,
            )

            while not stop_event.is_set() and not session.finished:
                before = len(session.moves)
                evaluator.reset_thread_timing()
                loop_started = time.perf_counter_ns()
                runner.advance_sessions(
                    [session],
                    allow_resignation=allow_resignation,
                )
                elapsed = time.perf_counter_ns() - loop_started
                inference_wait = evaluator.thread_wait_ns()
                with counter_lock:
                    actor_loop_ns[actor_id] += elapsed
                    actor_compute_ns[actor_id] += max(elapsed - inference_wait, 0)
                    plies_completed[actor_id] += len(session.moves) - before

            if stop_event.is_set():
                return

            record = session.build_record()
            replay_started = time.perf_counter_ns()
            if not put_with_stop(game_queue, record, stop_event):
                return
            replay_elapsed = time.perf_counter_ns() - replay_started
            with counter_lock:
                replay_wait_ns[actor_id] += replay_elapsed
                games_completed[actor_id] += 1
                positions_completed[actor_id] += len(record.samples)
    except SelfPlayCancelled:
        return
    except Exception as error:
        error_queue.put(str(error))
        stop_event.set()


def run_threaded_actor(
    actor_id,
    config,
    evaluator,
    game_queue,
    stop_event,
    games_completed,
    positions_completed,
    plies_completed,
    replay_game_count,
    checkpoint_steps,
    games_per_actor,
    actor_loop_ns,
    actor_compute_ns,
    replay_wait_ns,
):
    counter_lock = threading.Lock()
    error_queue = queue.Queue()
    threads = []

    for game_id in range(games_per_actor):
        thread = threading.Thread(
            target=play_game_thread,
            name=f"fisher-game-{actor_id:02d}-{game_id:02d}",
            args=(
                actor_id,
                game_id,
                config,
                evaluator,
                game_queue,
                stop_event,
                games_completed,
                positions_completed,
                plies_completed,
                replay_game_count,
                checkpoint_steps,
                counter_lock,
                error_queue,
                actor_loop_ns,
                actor_compute_ns,
                replay_wait_ns,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    try:
        while True:
            try:
                error = error_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                if not any(thread.is_alive() for thread in threads):
                    raise RuntimeError(
                        "All self-play game threads stopped unexpectedly"
                    ) from None
                continue
            raise RuntimeError(error)
    finally:
        evaluator.cancel_pending()
        for thread in threads:
            thread.join(timeout=2)


def actor_main(
    actor_id,
    scheduler,
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
    outstanding_requests,
    blocked_slot_waits,
    inference_requests,
    queue_wait_ns,
    response_wait_ns,
    actor_loop_ns,
    actor_compute_ns,
    replay_wait_ns,
    replay_game_count,
    checkpoint_steps,
    games_per_actor,
):
    configure_worker_threads()
    config = load_config(config_path)
    if config.runtime.pin_actor_cpus:
        pin_actor_to_cpu(actor_id)

    evaluator_class = (
        RemoteEvaluator if scheduler == BATCHED_SCHEDULER else ConcurrentRemoteEvaluator
    )
    evaluator = evaluator_class(
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
        evaluated_positions,
        outstanding_requests,
        blocked_slot_waits,
        inference_requests,
        queue_wait_ns,
        response_wait_ns,
    )

    try:
        if scheduler == BATCHED_SCHEDULER:
            run_batched_actor(
                actor_id,
                config,
                evaluator,
                game_queue,
                stop_event,
                games_completed,
                positions_completed,
                plies_completed,
                replay_game_count,
                checkpoint_steps,
                games_per_actor,
                actor_loop_ns,
                actor_compute_ns,
                replay_wait_ns,
            )
        else:
            run_threaded_actor(
                actor_id,
                config,
                evaluator,
                game_queue,
                stop_event,
                games_completed,
                positions_completed,
                plies_completed,
                replay_game_count,
                checkpoint_steps,
                games_per_actor,
                actor_loop_ns,
                actor_compute_ns,
                replay_wait_ns,
            )
    except SelfPlayCancelled:
        return
    except Exception:
        stop_event.set()
        raise
    finally:
        evaluator.close()


def replay_writer_main(
    config_path,
    game_queue,
    replay_game_count,
    replay_position_count,
):
    configure_worker_threads()
    config = load_config(config_path)
    replay = ReplayBuffer(
        config.runtime.replay_path,
        max_positions=config.training.replay_max_positions,
    )
    replay_game_count.value = replay.game_count
    replay_position_count.value = replay.position_count

    try:
        while True:
            game = game_queue.get()
            if game is None:
                break

            games = [game]
            while len(games) < config.runtime.replay_write_batch:
                try:
                    next_game = game_queue.get_nowait()
                except queue.Empty:
                    break
                if next_game is None:
                    game_queue.put(None)
                    break
                games.append(next_game)

            replay.add_games(games)
            replay_game_count.value = replay.game_count
            replay_position_count.value = replay.position_count
    finally:
        replay.close()


def round_request_capacity(value):
    if value <= 32:
        return max(1, value)
    return ((value + 31) // 32) * 32


def resolve_request_capacity(config, scheduler, games_per_actor):
    configured = max(0, int(config.runtime.inference_request_batch_size))
    if scheduler == BATCHED_SCHEDULER:
        required = games_per_actor * config.search.parallel_searches
    else:
        required = config.search.parallel_searches
    return round_request_capacity(max(configured, required))


class DistributedSelfPlayPool:
    def __init__(
        self,
        config_path="fisher_config.json",
        actor_count=None,
        games_per_actor=None,
        devices=None,
        checkpoint_path=None,
        scheduler=None,
    ):
        import multiprocessing as mp

        self.context = mp.get_context("spawn")
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.actor_count = actor_count or self.config.runtime.actor_processes
        self.games_per_actor = games_per_actor or self.config.runtime.games_per_actor
        self.scheduler = scheduler or self.config.runtime.self_play_scheduler
        if self.scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                f"Unknown self-play scheduler {self.scheduler!r}; expected one of "
                f"{sorted(VALID_SCHEDULERS)}"
            )
        self.config.runtime.games_per_actor = self.games_per_actor
        self.devices = devices or self.config.runtime.self_play_devices
        self.devices = [available_device(device) for device in self.devices]
        self.devices = list(dict.fromkeys(self.devices)) or ["cpu"]
        self.checkpoint_path = (
            str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        )
        self.request_capacity = resolve_request_capacity(
            self.config,
            self.scheduler,
            self.games_per_actor,
        )
        if self.request_capacity > self.config.runtime.inference_max_batch_size:
            raise ValueError(
                f"Resolved actor inference request capacity {self.request_capacity} exceeds "
                f"inference_max_batch_size={self.config.runtime.inference_max_batch_size}. "
                "Increase the maximum GPU batch size or reduce games_per_actor or "
                "parallel_searches."
            )
        self.slots_per_actor = (
            1
            if self.scheduler == BATCHED_SCHEDULER
            else self.config.runtime.max_inflight_requests_per_actor
        )
        if self.slots_per_actor < 1:
            raise ValueError("At least one inference slot per actor is required")

        self.stop_event = self.context.Event()
        self.request_queue = self.context.Queue(
            maxsize=self.config.runtime.inference_queue_size
        )
        response_size = max(2, self.slots_per_actor * 2)
        self.response_queues = [
            self.context.Queue(maxsize=response_size)
            for _ in range(self.actor_count)
        ]
        self.game_queue = self.context.Queue(
            maxsize=self.config.runtime.replay_queue_size
        )
        self.shared = SharedInferenceMemory(
            actor_count=self.actor_count,
            slots_per_actor=self.slots_per_actor,
            max_request_batch=self.request_capacity,
            max_legal_actions=self.config.runtime.max_legal_actions,
        )
        self.games_completed = self.context.Array("q", self.actor_count, lock=False)
        self.positions_completed = self.context.Array("q", self.actor_count, lock=False)
        self.plies_completed = self.context.Array("q", self.actor_count, lock=False)
        self.evaluated_positions = self.context.Array("q", self.actor_count, lock=False)
        self.outstanding_requests = self.context.Array("q", self.actor_count, lock=False)
        self.blocked_slot_waits = self.context.Array("q", self.actor_count, lock=False)
        self.inference_requests = self.context.Array("q", self.actor_count, lock=False)
        self.queue_wait_ns = self.context.Array("q", self.actor_count, lock=False)
        self.response_wait_ns = self.context.Array("q", self.actor_count, lock=False)
        self.actor_loop_ns = self.context.Array("q", self.actor_count, lock=False)
        self.actor_compute_ns = self.context.Array("q", self.actor_count, lock=False)
        self.replay_wait_ns = self.context.Array("q", self.actor_count, lock=False)
        self.inference_batches = self.context.Array("q", len(self.devices), lock=False)
        self.inference_positions = self.context.Array("q", len(self.devices), lock=False)
        self.inference_max_batches = self.context.Array("q", len(self.devices), lock=False)
        histogram_width = self.config.runtime.inference_max_batch_size + 1
        self.inference_histogram = self.context.Array(
            "q",
            len(self.devices) * histogram_width,
            lock=False,
        )
        self.checkpoint_steps = self.context.Array("q", len(self.devices), lock=False)
        self.server_ready = self.context.Array("b", len(self.devices), lock=False)
        self.replay_game_count = self.context.Value("q", 0)
        self.replay_position_count = self.context.Value("q", 0)
        self.actor_processes = []
        self.server_processes = []
        self.writer_process = None
        replay = ReplayBuffer(
            self.config.runtime.replay_path,
            max_positions=self.config.training.replay_max_positions,
        )
        self.replay_game_count.value = replay.game_count
        self.replay_position_count.value = replay.position_count
        replay.close()
        self.started = False

    def start(self):
        if self.started:
            return

        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"

        self.started = True
        self.writer_process = self.context.Process(
            target=replay_writer_main,
            name="fisher-replay-writer",
            args=(
                self.config_path,
                self.game_queue,
                self.replay_game_count,
                self.replay_position_count,
            ),
        )
        self.writer_process.start()

        for server_id, device in enumerate(self.devices):
            process = self.context.Process(
                target=inference_server_main,
                name=f"fisher-inference-{server_id}",
                args=(
                    server_id,
                    device,
                    self.config_path,
                    self.checkpoint_path,
                    self.shared.descriptor,
                    self.request_queue,
                    self.response_queues,
                    self.stop_event,
                    self.inference_batches,
                    self.inference_positions,
                    self.inference_max_batches,
                    self.inference_histogram,
                    self.checkpoint_steps,
                    self.server_ready,
                ),
            )
            process.start()
            self.server_processes.append(process)

        ready_deadline = time.monotonic() + 60.0
        while not all(value == 1 for value in self.server_ready):
            failure = self.process_failure()
            if failure:
                raise RuntimeError(failure)
            if any(value < 0 for value in self.server_ready):
                raise RuntimeError("An inference server failed during startup")
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("Inference servers did not become ready within 60 seconds")
            time.sleep(0.05)

        for actor_id in range(self.actor_count):
            process = self.context.Process(
                target=actor_main,
                name=f"fisher-actor-{actor_id:02d}",
                args=(
                    actor_id,
                    self.scheduler,
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
                    self.outstanding_requests,
                    self.blocked_slot_waits,
                    self.inference_requests,
                    self.queue_wait_ns,
                    self.response_wait_ns,
                    self.actor_loop_ns,
                    self.actor_compute_ns,
                    self.replay_wait_ns,
                    self.replay_game_count,
                    self.checkpoint_steps,
                    self.games_per_actor,
                ),
            )
            process.start()
            self.actor_processes.append(process)

    def process_failure(self):
        processes = self.server_processes + self.actor_processes
        if self.writer_process is not None:
            processes.append(self.writer_process)

        for process in processes:
            if process.exitcode is not None:
                return f"{process.name} exited with code {process.exitcode}"
        return None

    def batch_histogram(self):
        width = self.config.runtime.inference_max_batch_size + 1
        histogram = np.zeros(width, dtype=np.int64)
        for server_id in range(len(self.devices)):
            start = server_id * width
            histogram += np.asarray(
                self.inference_histogram[start : start + width],
                dtype=np.int64,
            )
        return histogram

    def metric_snapshot(self):
        try:
            request_depth = self.request_queue.qsize()
        except (NotImplementedError, OSError):
            request_depth = -1
        try:
            replay_depth = self.game_queue.qsize()
        except (NotImplementedError, OSError):
            replay_depth = -1

        return {
            "scheduler": self.scheduler,
            "request_capacity": self.request_capacity,
            "games": int(sum(self.games_completed)),
            "positions": int(sum(self.positions_completed)),
            "plies": int(sum(self.plies_completed)),
            "evaluations": int(sum(self.inference_positions)),
            "actor_evaluations": int(sum(self.evaluated_positions)),
            "inference_requests": int(sum(self.inference_requests)),
            "batches": int(sum(self.inference_batches)),
            "max_batch": int(max(self.inference_max_batches, default=0)),
            "histogram": self.batch_histogram(),
            "outstanding_requests": int(sum(self.outstanding_requests)),
            "blocked_slot_waits": int(sum(self.blocked_slot_waits)),
            "request_queue_depth": int(request_depth),
            "replay_queue_depth": int(replay_depth),
            "replay_games": int(self.replay_game_count.value),
            "replay_positions": int(self.replay_position_count.value),
            "server_evaluations": [int(value) for value in self.inference_positions],
            "queue_wait_ns": int(sum(self.queue_wait_ns)),
            "response_wait_ns": int(sum(self.response_wait_ns)),
            "actor_loop_ns": int(sum(self.actor_loop_ns)),
            "actor_compute_ns": int(sum(self.actor_compute_ns)),
            "replay_wait_ns": int(sum(self.replay_wait_ns)),
        }

    @staticmethod
    def percentile_from_histogram(histogram, percentile):
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

    def monitor(self, target_games=None, timeout=None, external_processes=None):
        start_time = time.monotonic()
        last_time = start_time
        next_status = start_time
        previous = self.metric_snapshot()
        initial_replay_games = self.replay_game_count.value

        print(
            f"Self-play pool started: scheduler={self.scheduler}, "
            f"{self.actor_count} actors, "
            f"{self.actor_count * self.games_per_actor} active games, "
            f"request_capacity={self.request_capacity}, "
            f"{len(self.devices)} inference GPUs ({', '.join(self.devices)})",
            flush=True,
        )

        while True:
            time.sleep(0.5)
            now = time.monotonic()

            failure = self.process_failure()
            if failure:
                raise RuntimeError(failure)

            for process in external_processes or []:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"External process exited with code {process.returncode}"
                    )

            if target_games is not None:
                written_games = self.replay_game_count.value - initial_replay_games
                if written_games >= target_games:
                    return

            if timeout is not None and now - start_time >= timeout:
                raise TimeoutError("Distributed self-play did not finish before the timeout")

            if now < next_status:
                continue

            current = self.metric_snapshot()
            elapsed = max(now - last_time, 1e-6)
            game_rate = (current["games"] - previous["games"]) * 3600 / elapsed
            move_rate = (current["plies"] - previous["plies"]) / elapsed
            position_rate = (current["positions"] - previous["positions"]) / elapsed
            evaluation_rate = (
                current["evaluations"] - previous["evaluations"]
            ) / elapsed
            delta_batches = current["batches"] - previous["batches"]
            delta_evaluations = current["evaluations"] - previous["evaluations"]
            delta_requests = current["inference_requests"] - previous["inference_requests"]
            average_batch = delta_evaluations / max(delta_batches, 1)
            average_request = delta_evaluations / max(delta_requests, 1)
            histogram = current["histogram"] - previous["histogram"]
            median_batch = self.percentile_from_histogram(histogram, 0.5)
            p95_batch = self.percentile_from_histogram(histogram, 0.95)
            capacity_ns = self.actor_count * elapsed * 1_000_000_000
            compute_percent = 100.0 * (
                current["actor_compute_ns"] - previous["actor_compute_ns"]
            ) / max(capacity_ns, 1)
            inference_wait_percent = 100.0 * (
                current["response_wait_ns"] - previous["response_wait_ns"]
            ) / max(capacity_ns, 1)

            print(
                f"self-play games={current['games']:,} "
                f"replay_games={current['replay_games']:,} "
                f"replay_positions={current['replay_positions']:,} "
                f"active={self.actor_count * self.games_per_actor:,} "
                f"moves/s={move_rate:.1f} games/hour={game_rate:.1f} "
                f"positions/s={position_rate:.1f} evals/s={evaluation_rate:.1f} "
                f"request_avg={average_request:.1f} batch_avg={average_batch:.1f} "
                f"batch_p50={median_batch:.0f} batch_p95={p95_batch:.0f} "
                f"actor_compute={compute_percent:.1f}% "
                f"inference_wait={inference_wait_percent:.1f}% "
                f"queue={current['request_queue_depth']} "
                f"inflight={current['outstanding_requests']} "
                f"replay_queue={current['replay_queue_depth']}",
                flush=True,
            )

            last_time = now
            next_status = now + self.config.runtime.status_interval_seconds
            previous = current

    def stop(self):
        if not self.started:
            self.shared.close()
            self.shared.unlink()
            return

        self.stop_event.set()

        for process in self.actor_processes:
            process.join(timeout=10)
        for process in self.actor_processes:
            if process.is_alive():
                process.terminate()
                process.join()

        for _ in self.server_processes:
            try:
                self.request_queue.put(None, timeout=1)
            except queue.Full:
                break
        for process in self.server_processes:
            process.join(timeout=30)
        for process in self.server_processes:
            if process.is_alive():
                process.terminate()
                process.join()

        try:
            self.game_queue.put(None, timeout=1)
        except queue.Full:
            pass
        if self.writer_process is not None:
            self.writer_process.join(timeout=30)
            if self.writer_process.is_alive():
                self.writer_process.terminate()
                self.writer_process.join()

        self.request_queue.close()
        for response_queue in self.response_queues:
            response_queue.close()
        self.game_queue.close()
        self.shared.close()
        self.shared.unlink()
        self.started = False

    def run(self, target_games=None, timeout=None):
        try:
            self.start()
            self.monitor(target_games=target_games, timeout=timeout)
        finally:
            self.stop()
