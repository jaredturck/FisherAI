import numpy as np

from fisher_ai import chess
from fisher_ai.dataset import GameRecord, InMemoryWindow, PositionTarget
from fisher_ai.encoding import castling_rights_mask, encode_state
from fisher_ai.game import GameState


def make_record(position_count=4):
    state = GameState()
    snapshots = [state.history[-1]]
    samples = []
    moves = ("e2e4", "e7e5", "g1f3", "b8c6")

    for index in range(position_count):
        samples.append(
            PositionTarget(
                state.board.turn,
                state.board.ply(),
                castling_rights_mask(state.board),
                state.board.halfmove_clock,
                [1, 2],
                [3, 1],
                value=1 if index % 2 == 0 else -1,
            )
        )
        if index + 1 < position_count:
            state.push(chess.Move.from_uci(moves[index]))
            snapshots.append(state.history[-1])

    return GameRecord(snapshots, samples)


def test_compact_record_materializes_the_original_state():
    state = GameState()
    record = make_record(4)

    for index, move in enumerate(("e2e4", "e7e5", "g1f3", "b8c6")):
        expected = encode_state(state).astype(np.float16)
        np.testing.assert_array_equal(
            record.materialize_state(index),
            expected,
        )
        state.push(chess.Move.from_uci(move))


def test_window_shuffles_every_position_without_replacement():
    window = InMemoryWindow(6)
    window.add_game(make_record(4))
    window.add_game(make_record(4))
    indices = window.shuffled_indices(np.random.default_rng(7))

    assert window.position_count == 8
    assert sorted(indices.tolist()) == list(range(8))
    assert len(np.unique(indices)) == 8

    batch = window.materialize_batch(indices[:3])
    assert batch[0].shape == (3, 119, 8, 8)
    assert batch[1].shape == (3, 2)
    assert len(batch) == 5
