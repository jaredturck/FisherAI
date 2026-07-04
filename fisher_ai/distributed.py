import os
import queue
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


class SharedInferenceMemory:
    def __init__(self, descriptor=None, actor_count=0, max_request_batch=0, max_legal_actions=256):
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
                "legal_lengths": ((actor_count, max_request_batch), np.uint16),
                "policy_logits": (
                    (actor_count, max_request_batch, max_legal_actions),
                    np.float32,
                ),
                "values": ((actor_count, max_request_batch), np.float32),
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


class RemoteEvaluator:
    def __init__(
        self,
        actor_id,
        request_queue,
        response_queue,
        shared_descriptor,
        stop_event,
        evaluated_positions,
    ):
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.shared = SharedInferenceMemory(descriptor=shared_descriptor)
        self.stop_event = stop_event
        self.evaluated_positions = evaluated_positions
        self.max_request_batch = self.shared.states.shape[1]
        self.max_legal_actions = self.shared.legal_actions.shape[2]
        self.request_id = 0

    def evaluate(self, states, legal_actions=None):
        if legal_actions is None:
            raise ValueError("Remote inference requires legal action lists")

        policies = []
        values = []

        for start in range(0, len(states), self.max_request_batch):
            end = min(start + self.max_request_batch, len(states))
            chunk_policies, chunk_values = self.evaluate_chunk(
                states[start:end],
                legal_actions[start:end],
            )
            policies.extend(chunk_policies)
            values.append(chunk_values)

        return policies, np.concatenate(values)

    def evaluate_chunk(self, states, legal_actions):
        batch_size = len(states)
        if batch_size == 0:
            return [], np.asarray([], dtype=np.float32)

        for index, (state, actions) in enumerate(zip(states, legal_actions, strict=True)):
            if len(actions) > self.max_legal_actions:
                raise RuntimeError(
                    f"Position has {len(actions)} legal moves, exceeding the configured "
                    f"maximum of {self.max_legal_actions}"
                )

            self.shared.states[self.actor_id, index] = encode_state(state)
            length = len(actions)
            self.shared.legal_lengths[self.actor_id, index] = length
            self.shared.legal_actions[self.actor_id, index, :length] = actions

        self.request_id += 1
        request_id = self.request_id
        self.request_queue.put((self.actor_id, batch_size, request_id))

        while True:
            try:
                response_id, error = self.response_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if self.stop_event.is_set():
                    raise RuntimeError(
                        "Self-play inference stopped before returning a response"
                    ) from None

        if error:
            raise RuntimeError(error)
        if response_id != request_id:
            raise RuntimeError(
                f"Inference response mismatch for actor {self.actor_id}: "
                f"expected {request_id}, received {response_id}"
            )

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


def build_checkpoint_manager(config):
    return CheckpointManager(
        config.runtime.checkpoint_dir,
        keep_recent=config.training.checkpoint_keep_recent,
        milestone_interval=config.training.checkpoint_milestone_interval,
    )


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
        max_server_batch = shared.states.shape[0] * shared.states.shape[1]
        pinned_states = torch.empty(
            (max_server_batch, INPUT_PLANES, 8, 8),
            dtype=torch.float16,
            pin_memory=device.type == "cuda",
        )
        pinned_actions = torch.empty(
            (max_server_batch, shared.legal_actions.shape[2]),
            dtype=torch.int64,
            pin_memory=device.type == "cuda",
        )
        batch_wait_seconds = config.runtime.inference_batch_wait_ms / 1000.0
        reload_deadline = time.monotonic() + config.runtime.checkpoint_reload_seconds
        should_exit = False

        while not should_exit:
            request = request_queue.get()
            if request is None:
                break

            requests = [request]
            state_count = request[1]
            deadline = time.monotonic() + batch_wait_seconds

            while state_count < config.runtime.inference_batch_size:
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

                requests.append(next_request)
                state_count += next_request[1]

            offset = 0
            request_offsets = []
            lengths = []
            max_legal_moves = 0

            for actor_id, batch_size, request_id in requests:
                end = offset + batch_size
                pinned_states[offset:end].copy_(
                    torch.from_numpy(shared.states[actor_id, :batch_size])
                )
                actor_lengths = shared.legal_lengths[actor_id, :batch_size].astype(
                    np.int64,
                    copy=True,
                )
                lengths.extend(int(length) for length in actor_lengths)
                actor_max = int(actor_lengths.max(initial=0))
                max_legal_moves = max(max_legal_moves, actor_max)
                if actor_max:
                    pinned_actions[offset:end, :actor_max].copy_(
                        torch.from_numpy(
                            shared.legal_actions[actor_id, :batch_size, :actor_max].astype(
                                np.int64,
                                copy=False,
                            )
                        )
                    )
                request_offsets.append((actor_id, request_id, offset, end))
                offset = end

            state_tensor = pinned_states[:state_count].to(device, non_blocking=True)
            if device.type != "cuda":
                state_tensor = state_tensor.float()
            elif config.runtime.channels_last:
                state_tensor = state_tensor.contiguous(memory_format=torch.channels_last)

            action_tensor = pinned_actions[:state_count, :max_legal_moves].to(
                device,
                non_blocking=True,
            )

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
                    shared.policy_logits[actor_id, local_index, :length] = gathered[
                        start + local_index,
                        :length,
                    ]
                shared.values[actor_id, : end - start] = values[start:end]
                response_queues[actor_id].put((request_id, None))

            inference_batches[server_id] += 1
            inference_positions[server_id] += state_count

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


def actor_main(
    actor_id,
    server_id,
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
    replay_game_count,
    checkpoint_steps,
    games_per_actor,
):
    configure_worker_threads()
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
    )
    search = MCTS(evaluator, config.search, seed=config.runtime.seed + actor_id)
    runner = SelfPlayRunner(
        search,
        config.search,
        training_config=config.training,
        seed=config.runtime.seed + actor_id,
    )
    sessions = [
        runner.create_session(
            checkpoint_step=int(checkpoint_steps[server_id]),
            allow_resignation=False,
        )
        for _ in range(games_per_actor)
    ]

    try:
        while not stop_event.is_set():
            allow_resignation = (
                replay_game_count.value >= config.training.resignation_enabled_after_games
            )
            finished_indices = runner.advance_sessions(
                sessions,
                allow_resignation=allow_resignation,
            )

            for index in finished_indices:
                session = sessions[index]
                record = session.build_record()
                game_queue.put(record)
                games_completed[actor_id] += 1
                positions_completed[actor_id] += len(record.samples)
                plies_completed[actor_id] += len(record.moves)
                sessions[index] = runner.create_session(
                    checkpoint_step=int(checkpoint_steps[server_id]),
                    allow_resignation=allow_resignation,
                )
    except RuntimeError:
        if not stop_event.is_set():
            stop_event.set()
            raise
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


class DistributedSelfPlayPool:
    def __init__(
        self,
        config_path="fisher_config.json",
        actor_count=None,
        games_per_actor=None,
        devices=None,
        checkpoint_path=None,
    ):
        import multiprocessing as mp

        self.context = mp.get_context("spawn")
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        self.actor_count = actor_count or self.config.runtime.actor_processes
        self.games_per_actor = games_per_actor or self.config.runtime.games_per_actor
        self.config.runtime.games_per_actor = self.games_per_actor
        self.devices = devices or self.config.runtime.self_play_devices
        self.devices = [available_device(device) for device in self.devices]
        self.devices = list(dict.fromkeys(self.devices)) or ["cpu"]
        self.checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        self.stop_event = self.context.Event()
        self.request_queues = [
            self.context.Queue(maxsize=self.config.runtime.inference_queue_size)
            for _ in self.devices
        ]
        self.response_queues = [self.context.Queue(maxsize=1) for _ in range(self.actor_count)]
        self.game_queue = self.context.Queue(maxsize=self.config.runtime.replay_queue_size)
        self.max_request_batch = self.games_per_actor * self.config.search.parallel_searches
        self.shared = SharedInferenceMemory(
            actor_count=self.actor_count,
            max_request_batch=self.max_request_batch,
            max_legal_actions=self.config.runtime.max_legal_actions,
        )
        self.games_completed = self.context.Array("q", self.actor_count, lock=False)
        self.positions_completed = self.context.Array("q", self.actor_count, lock=False)
        self.plies_completed = self.context.Array("q", self.actor_count, lock=False)
        self.evaluated_positions = self.context.Array("q", self.actor_count, lock=False)
        self.inference_batches = self.context.Array("q", len(self.devices), lock=False)
        self.inference_positions = self.context.Array("q", len(self.devices), lock=False)
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
                    self.request_queues[server_id],
                    self.response_queues,
                    self.stop_event,
                    self.inference_batches,
                    self.inference_positions,
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
            server_id = actor_id % len(self.devices)
            process = self.context.Process(
                target=actor_main,
                name=f"fisher-actor-{actor_id:02d}",
                args=(
                    actor_id,
                    server_id,
                    self.config_path,
                    self.shared.descriptor,
                    self.request_queues[server_id],
                    self.response_queues[actor_id],
                    self.game_queue,
                    self.stop_event,
                    self.games_completed,
                    self.positions_completed,
                    self.plies_completed,
                    self.evaluated_positions,
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

    def monitor(self, target_games=None, timeout=None, external_processes=None):
        start_time = time.monotonic()
        last_time = start_time
        next_status = start_time
        last_games = sum(self.games_completed)
        last_positions = sum(self.positions_completed)
        last_evaluations = sum(self.inference_positions)
        initial_replay_games = self.replay_game_count.value

        print(
            f"Self-play pool started: {self.actor_count} actors, "
            f"{self.actor_count * self.games_per_actor} active games, "
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

            elapsed = max(now - last_time, 1e-6)
            games = sum(self.games_completed)
            positions = sum(self.positions_completed)
            plies = sum(self.plies_completed)
            evaluations = sum(self.inference_positions)
            batches = sum(self.inference_batches)
            game_rate = (games - last_games) * 3600 / elapsed
            position_rate = (positions - last_positions) / elapsed
            evaluation_rate = (evaluations - last_evaluations) / elapsed
            average_batch = evaluations / max(batches, 1)

            print(
                f"self-play games={games:,} replay_games={self.replay_game_count.value:,} "
                f"replay_positions={self.replay_position_count.value:,} "
                f"active={self.actor_count * self.games_per_actor:,} "
                f"plies={plies:,} games/hour={game_rate:.1f} "
                f"positions/s={position_rate:.1f} evals/s={evaluation_rate:.1f} "
                f"avg_gpu_batch={average_batch:.1f}",
                flush=True,
            )

            last_time = now
            next_status = now + self.config.runtime.status_interval_seconds
            last_games = games
            last_positions = positions
            last_evaluations = evaluations

    def stop(self):
        if not self.started:
            self.shared.close()
            self.shared.unlink()
            return

        self.stop_event.set()

        for process in self.actor_processes:
            process.join(timeout=30)
        for process in self.actor_processes:
            if process.is_alive():
                process.terminate()
                process.join()

        for request_queue in self.request_queues:
            request_queue.put(None)
        for process in self.server_processes:
            process.join(timeout=30)
        for process in self.server_processes:
            if process.is_alive():
                process.terminate()
                process.join()

        self.game_queue.put(None)
        if self.writer_process is not None:
            self.writer_process.join(timeout=30)
            if self.writer_process.is_alive():
                self.writer_process.terminate()
                self.writer_process.join()

        for request_queue in self.request_queues:
            request_queue.close()
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
