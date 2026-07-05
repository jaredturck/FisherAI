import numpy as np

from fisher_ai import chess
from fisher_ai.dataset import GameRecord, InMemoryWindow, PositionTarget
from fisher_ai.encoding import castling_rights_mask, encode_state
from fisher_ai.game import GameState


def make_record(position_count=4):
    state = GameState(max_game_plies=20)
    snapshots = [state.history[-1]]
    samples = []
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]

    for index in range(position_count):
        samples.append(
            PositionTarget(
                snapshot_index=len(snapshots) - 1,
                current_color=state.board.turn,
                ply=state.board.ply(),
                castling_mask=castling_rights_mask(state.board),
                halfmove_clock=state.board.halfmove_clock,
                legal_actions=[1, 2],
                visit_counts=[3, 1],
                value=1 if index % 2 == 0 else -1,
                policy_weight=index % 2,
            )
        )
        if index + 1 < position_count:
            state.push(chess.Move.from_uci(moves[index]))
            snapshots.append(state.history[-1])

    return GameRecord(
        snapshots=snapshots,
        samples=samples,
        max_game_plies=state.max_game_plies,
    )


def test_compact_record_materializes_the_original_state():
    state = GameState(max_game_plies=20)
    snapshots = [state.history[-1]]
    samples = []

    for uci in ("e2e4", "e7e5", "g1f3"):
        samples.append(
            PositionTarget(
                len(snapshots) - 1,
                state.board.turn,
                state.board.ply(),
                castling_rights_mask(state.board),
                state.board.halfmove_clock,
                [1],
                [1],
            )
        )
        expected = encode_state(state).astype(np.float16)
        record = GameRecord(
            snapshots=list(snapshots),
            samples=list(samples),
            max_game_plies=state.max_game_plies,
        )
        np.testing.assert_array_equal(
            record.materialize_state(len(samples) - 1),
            expected,
        )
        state.push(chess.Move.from_uci(uci))
        snapshots.append(state.history[-1])


def test_window_has_exact_size_and_shuffles_without_replacement():
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
