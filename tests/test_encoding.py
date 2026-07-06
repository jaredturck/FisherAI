import numpy as np

from fisher_ai import chess
from fisher_ai.encoding import (
    ACTION_SIZE,
    encode_state,
    encode_states,
    move_to_action,
)
from fisher_ai.game import GameState


def legal_moves(board):
    buffer = np.empty(256, dtype=np.uint32)
    count, _ = board.fill_legal_moves(buffer)
    return buffer[:count].copy()


def test_initial_state_has_expected_shape_and_piece_counts():
    state = GameState()
    encoded = encode_state(state)

    assert encoded.shape == (119, 8, 8)
    current_frame = encoded[98:112]
    assert current_frame[:6].sum() == 16
    assert current_frame[6:12].sum() == 16
    assert encoded[114, 0, 0] == 1
    assert encoded[115, 0, 0] == 1
    assert encoded[116, 0, 0] == 1
    assert encoded[117, 0, 0] == 1


def test_batch_encoding_matches_single_state_encoding():
    states = [GameState(), GameState()]
    states[1].push(chess.move_from_uci("e2e4"))

    batch = encode_states(states)

    np.testing.assert_array_equal(batch[0], encode_state(states[0]))
    np.testing.assert_array_equal(batch[1], encode_state(states[1]))


def test_black_to_move_is_rotated_to_current_player_perspective():
    state = GameState(chess.Board("8/8/8/8/8/8/4p3/4K2k b - - 0 1"))
    encoded = encode_state(state)
    current_frame = encoded[98:112]

    assert current_frame[0, 6, 3] == 1
    assert encoded[112, 0, 0] == 0


def test_legal_moves_have_unique_action_indices():
    state = GameState()

    for _ in range(50):
        moves = legal_moves(state.board)
        actions = [move_to_action(move, state.board.turn) for move in moves]
        assert len(actions) == len(set(actions))
        assert all(0 <= action < ACTION_SIZE for action in actions)

        if state.is_terminal():
            break
        state.push(int(moves[0]))


def test_castling_and_underpromotions_use_distinct_actions():
    castling_action = move_to_action(
        chess.move_from_uci("e1g1"),
        chess.WHITE,
    )
    promotions = [
        move_to_action(chess.move_from_uci("a7a8n"), chess.WHITE),
        move_to_action(chess.move_from_uci("a7a8b"), chess.WHITE),
        move_to_action(chess.move_from_uci("a7a8r"), chess.WHITE),
        move_to_action(chess.move_from_uci("a7a8q"), chess.WHITE),
    ]

    assert len(set(promotions)) == 4
    assert 0 <= castling_action < ACTION_SIZE


def test_repetition_planes_mark_second_and_third_occurrence():
    state = GameState()
    moves = ("g1f3", "g8f6", "f3g1", "f6g8")

    for move in moves:
        state.push(chess.move_from_uci(move))

    current_frame = encode_state(state)[98:112]
    assert np.all(current_frame[12] == 1)
    assert np.all(current_frame[13] == 0)

    for move in moves:
        state.push(chess.move_from_uci(move))

    current_frame = encode_state(state)[98:112]
    assert np.all(current_frame[12] == 1)
    assert np.all(current_frame[13] == 1)


def test_game_state_copy_owns_independent_dense_arrays():
    state = GameState()
    state.push(chess.move_from_uci("e2e4"))
    copied = state.copy()

    original_hash = state.board.position_hash()
    assert not np.shares_memory(
        state.history_bitboards,
        copied.history_bitboards,
    )
    copied.push(chess.move_from_uci("e7e5"))
    assert state.board.position_hash() == original_hash
    assert copied.board.position_hash() != original_hash


def test_cached_repetition_count_does_not_recompute_position_hash(monkeypatch):
    state = GameState()

    def fail_position_hash(board):
        raise AssertionError("position hash should not be recomputed")

    monkeypatch.setattr(chess.Board, "position_hash", fail_position_hash)

    assert state.current_repetition_count() == 1
