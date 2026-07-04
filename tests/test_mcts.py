import numpy as np

from fisher_ai.config import SearchConfig
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS, MCTSNode


class UniformEvaluator:
    def evaluate(self, states, legal_actions=None):
        value = np.zeros(len(states), dtype=np.float32)
        if legal_actions is None:
            return np.zeros((len(states), 4672), dtype=np.float32), value
        return [np.zeros(len(actions), dtype=np.float32) for actions in legal_actions], value


class BiasedEvaluator:
    def __init__(self, action):
        self.action = action

    def evaluate(self, states, legal_actions=None):
        value = np.zeros(len(states), dtype=np.float32)
        if legal_actions is None:
            policy = np.zeros((len(states), 4672), dtype=np.float32)
            policy[:, self.action] = 8
            return policy, value

        policies = []
        for actions in legal_actions:
            policy = np.zeros(len(actions), dtype=np.float32)
            matches = np.flatnonzero(actions == self.action)
            if len(matches):
                policy[matches[0]] = 8
            policies.append(policy)
        return policies, value


def test_mcts_completes_requested_simulations_with_parallel_leaves():
    config = SearchConfig(simulations=16, parallel_searches=4, max_game_plies=40)
    state = GameState(max_game_plies=40)
    search = MCTS(UniformEvaluator(), config)
    root = search.run([state], roots=[MCTSNode(state=state)])[0]

    assert root.visit_count == 16
    assert root.virtual_visit_count == 0
    assert sum(child.visit_count for child in root.children.values()) == 16


def test_mcts_accepts_different_simulation_counts_per_root():
    config = SearchConfig(simulations=16, parallel_searches=4, max_game_plies=40)
    states = [GameState(max_game_plies=40), GameState(max_game_plies=40)]
    roots = [MCTSNode(state=state) for state in states]
    search = MCTS(UniformEvaluator(), config)
    roots = search.run(states, roots=roots, simulations=[4, 9])

    assert [root.visit_count for root in roots] == [4, 9]


def test_mcts_finds_mate_in_one():
    config = SearchConfig(simulations=64, parallel_searches=8, max_game_plies=40)
    state = GameState.from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1", max_game_plies=40)
    search = MCTS(UniformEvaluator(), config)
    root = search.run([state], roots=[MCTSNode(state=state)])[0]
    action = search.choose_action(root, greedy=True)
    move = root.children[action].move

    child = state.child(move)
    assert child.board.is_checkmate()


def test_policy_prior_guides_early_search():
    config = SearchConfig(simulations=8, parallel_searches=2, c_puct=2.0, max_game_plies=40)
    state = GameState(max_game_plies=40)
    root = MCTSNode(state=state)
    uniform = MCTS(UniformEvaluator(), config)
    uniform.evaluate_and_expand([root])
    target_action = root.child_actions[0]

    search = MCTS(BiasedEvaluator(target_action), config)
    root = MCTSNode(state=state)
    root = search.run([state], roots=[root])[0]

    assert root.children[target_action].visit_count == max(
        child.visit_count for child in root.children.values()
    )
