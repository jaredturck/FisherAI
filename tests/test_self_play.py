import numpy as np

from fisher_ai import chess
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS
from fisher_ai.self_play import SelfPlayRunner


class UniformEvaluator:
    def evaluate_encoded(self, encoded_states, legal_actions):
        policies = [
            np.zeros(len(actions), dtype=np.float32)
            for actions in legal_actions
        ]
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def test_self_play_builds_compact_policy_and_value_targets():
    search = MCTS(
        UniformEvaluator(),
        simulations=4,
        parallel_searches=2,
        seed=3,
    )
    runner = SelfPlayRunner(search, seed=3)
    session = runner.create_session()
    session.state = GameState(chess.Board("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    session.snapshots = [session.state.history[-1]]

    while not session.finished:
        runner.advance_sessions([session])

    game = session.build_record()

    assert game.samples
    assert len(game.snapshots) == len(game.samples) + 1
    assert game.materialize_state(0).shape == (119, 8, 8)
    assert all(sample.legal_actions.size for sample in game.samples)
    assert all(sample.visit_counts.sum() > 0 for sample in game.samples)
    assert set(sample.value for sample in game.samples).issubset(
        {-1.0, 0.0, 1.0}
    )
