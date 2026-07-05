import numpy as np

from fisher_ai import chess
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS


class UniformEvaluator:
    def evaluate_encoded(self, encoded_states, legal_actions):
        policies = [
            np.zeros(len(actions), dtype=np.float32)
            for actions in legal_actions
        ]
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


class BiasedEvaluator:
    def __init__(self, action):
        self.action = action

    def evaluate_encoded(self, encoded_states, legal_actions):
        policies = []
        for actions in legal_actions:
            policy = np.zeros(len(actions), dtype=np.float32)
            matches = np.flatnonzero(actions == self.action)
            if len(matches):
                policy[matches[0]] = 8
            policies.append(policy)
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


def test_mcts_completes_requested_simulations_with_parallel_leaves():
    state = GameState()
    search = MCTS(
        UniformEvaluator(),
        simulations=16,
        parallel_searches=4,
    )
    root = search.run([state])[0]
    _, visits = search.visit_counts(root)

    assert root.visit_count == 16
    assert root.virtual_visit_count == 0
    assert visits.sum() == 16


def test_mcts_finds_mate_in_one():
    state = GameState(chess.Board("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    search = MCTS(
        UniformEvaluator(),
        simulations=64,
        parallel_searches=8,
    )
    root = search.run([state])[0]
    action = search.choose_action(root, greedy=True)
    child = state.copy()
    child.push(root.move_for_action(action))

    assert child.board.is_checkmate()


def test_policy_prior_guides_early_search():
    state = GameState()
    target_action = int(MCTS.legal_action_data(state)[0][0])
    search = MCTS(
        BiasedEvaluator(target_action),
        simulations=8,
        parallel_searches=2,
    )
    root = search.run([state])[0]
    actions, visits = search.visit_counts(root)
    target_index = int(np.flatnonzero(actions == target_action)[0])

    assert visits[target_index] == visits.max()


def test_tree_uses_array_records_and_reuses_selected_subtree():
    state = GameState()
    search = MCTS(
        UniformEvaluator(),
        simulations=8,
        parallel_searches=2,
    )
    tree = search.run([state])[0]

    assert tree.actions.shape == (tree.capacity,)
    assert tree.moves.shape == (tree.capacity,)
    assert tree.first_children.shape == (tree.capacity,)
    assert not hasattr(tree, "children")

    actions, visits = search.visit_counts(tree)
    action = int(actions[np.argmax(visits)])
    child_visits = int(visits[np.argmax(visits)])
    state.push(tree.advance(action))

    assert tree.visit_count == child_visits
    search.run([state], roots=[tree])
    assert tree.visit_count == child_visits + 8
