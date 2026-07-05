import numpy as np

from fisher_ai import chess
from fisher_ai.encoding import ACTION_SIZE, encode_state, legal_action_map, move_to_action
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
    state = GameState.from_fen("8/8/8/8/8/8/4p3/4K2k b - - 0 1")
    encoded = encode_state(state)
    current_frame = encoded[98:112]

    black_pawn_plane = current_frame[0]
    assert black_pawn_plane[6, 3] == 1
    assert encoded[112, 0, 0] == 0


def test_legal_moves_have_unique_action_indices():
    state = GameState()

    for _ in range(50):
        mapping = legal_action_map(state)
        assert len(mapping) == state.board.legal_moves.count()
        assert all(0 <= action < ACTION_SIZE for action in mapping)

        if state.is_terminal():
            break

        move = list(state.board.legal_moves)[0]
        state.push(move)


def test_castling_and_underpromotions_use_distinct_actions():
    castling = chess.Move.from_uci("e1g1")
    castling_action = move_to_action(castling, chess.WHITE)

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
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]

    for uci in moves:
        state.push(chess.Move.from_uci(uci))

    encoded = encode_state(state)
    current_frame = encoded[98:112]
    assert np.all(current_frame[12] == 1)
    assert np.all(current_frame[13] == 0)

    for uci in moves:
        state.push(chess.Move.from_uci(uci))

    encoded = encode_state(state)
    current_frame = encoded[98:112]
    assert np.all(current_frame[12] == 1)
    assert np.all(current_frame[13] == 1)


def test_snapshot_piece_planes_match_current_board():
    state = GameState()
    moves = [
        "e2e4",
        "c7c5",
        "g1f3",
        "d7d6",
        "d2d4",
        "c5d4",
        "f3d4",
        "g8f6",
        "b1c3",
    ]

    for uci in moves:
        state.push(chess.Move.from_uci(uci))
        expected = np.zeros((12, 8, 8), dtype=np.uint8)
        for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
            for piece_index, piece_type in enumerate(
                (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
            ):
                for square in state.board.pieces(piece_type, color):
                    row = chess.square_rank(square)
                    col = chess.square_file(square)
                    expected[color_index * 6 + piece_index, row, col] = 1

        np.testing.assert_array_equal(state.history[-1].piece_planes, expected)


def test_game_state_copy_shares_immutable_history_snapshots():
    state = GameState()
    state.push(chess.Move.from_uci("e2e4"))
    copied = state.copy()

    original_key = state.board.position_key()
    assert list(state.history)[-1] is list(copied.history)[-1]
    copied.push(chess.Move.from_uci("e7e5"))
    assert state.board.position_key() == original_key
    assert copied.board.position_key() != original_key
