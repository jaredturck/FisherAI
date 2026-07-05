import numpy as np

from fisher_ai import chess
from fisher_ai.encoding import ACTION_SIZE, encode_state, move_to_action
from fisher_ai.game import GameState


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


def test_black_to_move_is_rotated_to_current_player_perspective():
    state = GameState(chess.Board("8/8/8/8/8/8/4p3/4K2k b - - 0 1"))
    encoded = encode_state(state)
    current_frame = encoded[98:112]

    assert current_frame[0, 6, 3] == 1
    assert encoded[112, 0, 0] == 0


def test_legal_moves_have_unique_action_indices():
    state = GameState()

    for _ in range(50):
        actions = [
            move_to_action(move, state.board.turn)
            for move in state.board.legal_moves
        ]
        assert len(actions) == len(set(actions))
        assert all(0 <= action < ACTION_SIZE for action in actions)

        if state.is_terminal():
            break
        state.push(next(iter(state.board.legal_moves)))


def test_castling_and_underpromotions_use_distinct_actions():
    castling_action = move_to_action(
        chess.Move.from_uci("e1g1"),
        chess.WHITE,
    )
    promotions = [
        move_to_action(chess.Move.from_uci("a7a8n"), chess.WHITE),
        move_to_action(chess.Move.from_uci("a7a8b"), chess.WHITE),
        move_to_action(chess.Move.from_uci("a7a8r"), chess.WHITE),
        move_to_action(chess.Move.from_uci("a7a8q"), chess.WHITE),
    ]

    assert len(set(promotions)) == 4
    assert 0 <= castling_action < ACTION_SIZE


def test_repetition_planes_mark_second_and_third_occurrence():
    state = GameState()
    moves = ("g1f3", "g8f6", "f3g1", "f6g8")

    for move in moves:
        state.push(chess.Move.from_uci(move))

    current_frame = encode_state(state)[98:112]
    assert np.all(current_frame[12] == 1)
    assert np.all(current_frame[13] == 0)

    for move in moves:
        state.push(chess.Move.from_uci(move))

    current_frame = encode_state(state)[98:112]
    assert np.all(current_frame[12] == 1)
    assert np.all(current_frame[13] == 1)


def test_game_state_copy_shares_immutable_history_snapshots():
    state = GameState()
    state.push(chess.Move.from_uci("e2e4"))
    copied = state.copy()

    original_key = state.board.position_key()
    assert list(state.history)[-1] is list(copied.history)[-1]
    copied.push(chess.Move.from_uci("e7e5"))
    assert state.board.position_key() == original_key
    assert copied.board.position_key() != original_key
