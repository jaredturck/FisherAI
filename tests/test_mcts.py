import numpy as np

from fisher_ai import chess
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS, MCTSStatePool


class UniformEvaluator:
    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        policies = np.zeros(legal_actions.shape, dtype=np.float32)
        values = np.zeros(len(encoded_states), dtype=np.float32)
        return policies, values


class BiasedEvaluator:
    def __init__(self, action):
        self.action = action

    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        policies = np.zeros(legal_actions.shape, dtype=np.float32)
        for index, length in enumerate(legal_lengths):
            actions = legal_actions[index, :length]
            matches = np.flatnonzero(actions == self.action)
            if len(matches):
                policies[index, matches[0]] = 8
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


def test_checkmate_takes_precedence_over_rule_draw():
    state = GameState(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 100 51"))

    assert state.terminal_status() == chess.CHECKMATE
    assert state.terminal_value() == -1.0


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

    assert child.terminal_status() == chess.CHECKMATE


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


def test_expanded_nodes_keep_cached_game_states():
    state = GameState()
    search = MCTS(
        UniformEvaluator(),
        simulations=16,
        parallel_searches=4,
    )
    tree = search.run([state])[0]
    expanded_nodes = np.flatnonzero(
        tree.first_children[: tree.next_free] != -1
    )

    assert len(expanded_nodes) > 1
    assert np.all(tree.state_slots[expanded_nodes] >= 0)
    assert tree.state_pool.count >= len(expanded_nodes)


def test_terminal_leaf_values_are_cached():
    state = GameState(chess.Board("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    search = MCTS(
        UniformEvaluator(),
        simulations=32,
        parallel_searches=4,
    )
    tree = search.run([state])[0]

    assert np.isfinite(tree.terminal_values[: tree.next_free]).any()


def test_dense_state_pool_counts_root_and_cached_path_repetitions():
    state = GameState()
    for move in (
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    ):
        state.push(chess.move_from_uci(move))

    pool = MCTSStatePool(8)
    root_slot = pool.store_root(state)
    position_hash = int(state.position_hashes[state.position_hash_length - 1])

    assert pool.repetition_count_for(root_slot, position_hash) == 3

    child_slot = pool.store_child(state, root_slot, position_hash)
    assert pool.repetition_count_for(child_slot, position_hash) == 4
