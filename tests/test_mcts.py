import numpy as np

from fisher_ai.config import SearchConfig
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS, MCTSTree


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
    root = search.run([state], roots=[MCTSTree(config.tree_capacity)])[0]

    assert root.visit_count == 16
    assert root.virtual_visit_count == 0
    assert root.child_visits.sum() == 16


def test_mcts_accepts_different_simulation_counts_per_root():
    config = SearchConfig(simulations=16, parallel_searches=4, max_game_plies=40)
    states = [GameState(max_game_plies=40), GameState(max_game_plies=40)]
    roots = [MCTSTree(config.tree_capacity) for _ in states]
    search = MCTS(UniformEvaluator(), config)
    roots = search.run(states, roots=roots, simulations=[4, 9])

    assert [root.visit_count for root in roots] == [4, 9]


def test_mcts_finds_mate_in_one():
    config = SearchConfig(simulations=64, parallel_searches=8, max_game_plies=40)
    state = GameState.from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1", max_game_plies=40)
    search = MCTS(UniformEvaluator(), config)
    root = search.run([state], roots=[MCTSTree(config.tree_capacity)])[0]
    action = search.choose_action(root, greedy=True)
    move = root.move_for_action(action)

    child = state.child(move)
    assert child.board.is_checkmate()


def test_policy_prior_guides_early_search():
    config = SearchConfig(simulations=8, parallel_searches=2, c_puct=2.0, max_game_plies=40)
    state = GameState(max_game_plies=40)
    uniform = MCTS(UniformEvaluator(), config)
    root = uniform.run(
        [state],
        roots=[MCTSTree(config.tree_capacity)],
        simulations=0,
    )[0]
    target_action = int(root.child_actions[0])

    search = MCTS(BiasedEvaluator(target_action), config)
    root = search.run([state], roots=[MCTSTree(config.tree_capacity)])[0]
    actions, visits = search.visit_counts(root)
    target_index = int(np.flatnonzero(actions == target_action)[0])

    assert visits[target_index] == visits.max()


def test_tree_uses_preallocated_array_records_and_reuses_selected_subtree():
    config = SearchConfig(simulations=8, parallel_searches=2, tree_capacity=1024)
    state = GameState(max_game_plies=40)
    search = MCTS(UniformEvaluator(), config)
    tree = search.run([state], roots=[MCTSTree(config.tree_capacity)])[0]

    assert tree.record_count > 1
    assert tree.actions.shape == (tree.capacity,)
    assert tree.moves.shape == (tree.capacity,)
    assert tree.first_children.shape == (tree.capacity,)
    assert not hasattr(tree, "children")

    action = search.choose_action(tree, greedy=True)
    child_visits = int(tree.child_visits[np.flatnonzero(tree.child_actions == action)[0]])
    move = tree.advance(action)
    state.push(move)

    assert tree.visit_count == child_visits
    search.run([state], roots=[tree], simulations=2)
    assert tree.visit_count == child_visits + 2


def test_tree_grows_before_search_when_configured_capacity_is_too_small():
    config = SearchConfig(simulations=1, parallel_searches=1, tree_capacity=16)
    state = GameState(max_game_plies=40)
    search = MCTS(UniformEvaluator(), config)
    tree = MCTSTree(config.tree_capacity)

    search.run([state], roots=[tree])

    assert tree.capacity >= 513
    assert tree.visit_count == 1
