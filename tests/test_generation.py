import queue
import threading

import numpy as np

from fisher_ai.generation import RemoteEvaluator, SharedInferenceMemory
from fisher_ai.mcts import MCTS
from fisher_ai.self_play import SelfPlayRunner


class RecordingEvaluator:
    def __init__(self):
        self.batch_sizes = []

    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        self.batch_sizes.append(len(encoded_states))
        policies = np.zeros(legal_actions.shape, dtype=np.float32)
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def test_self_play_batches_sessions_into_shared_inference_calls():
    evaluator = RecordingEvaluator()
    search = MCTS(
        evaluator,
        simulations=1,
        parallel_searches=1,
        seed=7,
    )
    runner = SelfPlayRunner(search, seed=7)
    sessions = [runner.create_session(), runner.create_session()]

    runner.advance_sessions(sessions)

    assert evaluator.batch_sizes == [2, 2]


def test_remote_evaluator_uses_dedicated_shared_memory_slot():
    shared = SharedInferenceMemory(
        actor_count=1,
        slot_count=2,
        max_request_batch=64,
    )
    stop_event = threading.Event()
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    evaluator = RemoteEvaluator(
        0,
        1,
        request_queue,
        response_queue,
        shared,
        stop_event,
    )
    result = []

    def evaluate():
        states = np.zeros((40, 119, 8, 8), dtype=np.float16)
        actions = np.zeros((40, 256), dtype=np.uint16)
        lengths = np.ones(40, dtype=np.uint16)
        result.append(evaluator.evaluate_encoded(states, actions, lengths))

    thread = threading.Thread(target=evaluate)
    thread.start()

    try:
        actor_id, slot_id, batch_size, request_id = request_queue.get(
            timeout=2
        )
        assert (actor_id, slot_id, batch_size) == (0, 1, 40)
        shared.policy_logits[0, 1, :40, 0] = 1.0
        shared.values[0, 1, :40] = 0.5
        response_queue.put((request_id, None))
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert len(result[0][0]) == 40
        assert np.allclose(result[0][1], 0.5)
    finally:
        stop_event.set()
        shared.close()
        shared.unlink()
