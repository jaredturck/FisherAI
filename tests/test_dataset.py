import numpy as np

from fisher_ai import chess
from fisher_ai.dataset import GameRecord, InMemoryWindow
from fisher_ai.encoding import castling_rights_mask, encode_state
from fisher_ai.game import GameState
from fisher_ai.mcts import MAX_LEGAL_ACTIONS


def make_record(position_count=4, action_count=2):
    state = GameState()
    snapshots = np.empty((position_count, 12), dtype=np.uint64)
    repetitions = np.empty(position_count, dtype=np.uint8)
    colors = np.empty(position_count, dtype=np.bool_)
    plies = np.empty(position_count, dtype=np.uint16)
    castling = np.empty(position_count, dtype=np.uint8)
    halfmoves = np.empty(position_count, dtype=np.uint8)
    lengths = np.full(position_count, action_count, dtype=np.uint16)
    actions = np.zeros(
        (position_count, MAX_LEGAL_ACTIONS),
        dtype=np.uint16,
    )
    counts = np.zeros_like(actions)
    values = np.empty(position_count, dtype=np.float32)
    moves = (
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1c4",
        "g8f6",
        "d2d3",
    )

    for index in range(position_count):
        state.current_bitboards(snapshots[index])
        repetitions[index] = state.repetition_count
        colors[index] = state.board.turn
        plies[index] = state.board.ply()
        castling[index] = castling_rights_mask(state.board)
        halfmoves[index] = state.board.halfmove_clock
        actions[index, :action_count] = np.arange(1, action_count + 1)
        counts[index, :action_count] = np.arange(
            action_count,
            0,
            -1,
        )
        values[index] = 1 if index % 2 == 0 else -1
        if index + 1 < position_count:
            state.push(chess.move_from_uci(moves[index]))

    return GameRecord(
        snapshots,
        repetitions,
        colors,
        plies,
        castling,
        halfmoves,
        lengths,
        actions,
        counts,
        values,
    )


def test_array_record_materializes_the_original_states_through_window():
    expected_state = GameState()
    record = make_record(4)
    window = InMemoryWindow(4)
    window.add_game(record)
    states = window.materialize_batch(np.arange(4))[0]

    for index, move in enumerate(("e2e4", "e7e5", "g1f3", "b8c6")):
        expected = encode_state(expected_state).astype(np.float16)
        np.testing.assert_array_equal(states[index], expected)
        expected_state.push(chess.move_from_uci(move))


def test_window_shuffles_every_position_without_replacement():
    window = InMemoryWindow(6)
    window.add_game(make_record(4))
    window.add_game(make_record(4))
    indices = window.shuffled_indices(np.random.default_rng(7))

    assert window.position_count == 8
    assert window.game_count == 2
    assert sorted(indices.tolist()) == list(range(8))
    assert len(np.unique(indices)) == 8

    batch = window.materialize_batch(indices[:3])
    assert batch[0].shape == (3, 119, 8, 8)
    assert batch[1].shape == (3, 2)
    assert len(batch) == 5


def test_window_uses_contiguous_structure_of_arrays():
    window = InMemoryWindow(4)
    window.add_game(make_record(4))

    assert window.snapshot_bitboards.flags.c_contiguous
    assert window.legal_actions.flags.c_contiguous
    assert window.visit_counts.flags.c_contiguous
    assert window.game_starts[:4].tolist() == [0, 0, 0, 0]
