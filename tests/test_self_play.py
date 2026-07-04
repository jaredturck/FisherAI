import numpy as np

from fisher_ai.config import SearchConfig
from fisher_ai.mcts import MCTS
from fisher_ai.self_play import SelfPlayRunner


class UniformEvaluator:
    def evaluate(self, states, legal_actions=None):
        value = np.zeros(len(states), dtype=np.float32)
        if legal_actions is None:
            return np.zeros((len(states), 4672), dtype=np.float32), value
        return [np.zeros(len(actions), dtype=np.float32) for actions in legal_actions], value


def test_self_play_generates_full_search_training_records():
    config = SearchConfig(
        simulations=2,
        fast_simulations=1,
        full_search_fraction=1.0,
        parallel_searches=2,
        max_game_plies=4,
    )
    search = MCTS(UniformEvaluator(), config, seed=3)
    runner = SelfPlayRunner(search, config, seed=3)

    games = runner.play_games(2, checkpoint_step=12)

    assert len(games) == 2
    assert all(game.checkpoint_step == 12 for game in games)
    assert all(game.result == 0 for game in games)
    assert all(len(game.samples) == 4 for game in games)
    assert all(sample.state.shape == (119, 8, 8) for game in games for sample in game.samples)
    assert all(sample.policy_weight == 1 for game in games for sample in game.samples)


def test_fast_search_positions_mask_policy_training():
    config = SearchConfig(
        simulations=2,
        fast_simulations=1,
        full_search_fraction=0.0,
        parallel_searches=2,
        max_game_plies=2,
    )
    search = MCTS(UniformEvaluator(), config, seed=3)
    runner = SelfPlayRunner(search, config, seed=3)

    game = runner.play_games(1)[0]

    assert all(sample.policy_weight == 0 for sample in game.samples)
