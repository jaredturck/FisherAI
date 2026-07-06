import numpy as np

from fisher_ai import chess
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS
from fisher_ai.self_play import SelfPlayRunner


class UniformEvaluator:
    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        policies = np.zeros(legal_actions.shape, dtype=np.float32)
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def test_self_play_builds_array_policy_and_value_targets():
    search = MCTS(
        UniformEvaluator(),
        simulations=4,
        parallel_searches=2,
        seed=3,
    )
    runner = SelfPlayRunner(search, seed=3)
    session = runner.create_session()
    session.state = GameState(chess.Board("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1"))

    while not session.finished:
        runner.advance_sessions([session])

    game = session.build_record()

    assert game.position_count > 0
    assert game.snapshot_bitboards.shape == (game.position_count, 12)
    assert game.legal_actions.shape[1] == 256
    assert np.all(game.legal_lengths > 0)
    assert np.all(game.visit_counts.sum(axis=1) > 0)
    assert set(game.values.tolist()).issubset({-1.0, 0.0, 1.0})
