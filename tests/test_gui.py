from fisher_ai import chess
from fisher_ai.game import GameState
from gui.main import (
    BOARD_X,
    BOARD_Y,
    PIECE_RENDER_OFFSETS,
    SQUARE_SIZE,
    ChessGUI,
)


def test_gui_square_mapping():
    gui = object.__new__(ChessGUI)

    assert gui.square_from_screen((BOARD_X + 1, BOARD_Y + 1)) == 56
    bottom_right = (
        BOARD_X + 7 * SQUARE_SIZE + 1,
        BOARD_Y + 7 * SQUARE_SIZE + 1,
    )
    assert gui.square_from_screen(bottom_right) == 7
    assert gui.square_from_screen((0, 0)) is None


def test_gui_defaults_promotion_to_queen():
    gui = object.__new__(ChessGUI)
    gui.state = GameState.from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1")

    move = gui.choose_human_move(48, 56)

    assert move == chess.Move.from_uci("a7a8q")


def test_gui_has_offsets_for_every_piece():
    expected_pieces = {
        (color, piece_type)
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in (
            chess.KING,
            chess.QUEEN,
            chess.BISHOP,
            chess.KNIGHT,
            chess.ROOK,
            chess.PAWN,
        )
    }

    assert set(PIECE_RENDER_OFFSETS) == expected_pieces
    assert PIECE_RENDER_OFFSETS[(chess.WHITE, chess.PAWN)] == (16, -4)
    assert PIECE_RENDER_OFFSETS[(chess.BLACK, chess.QUEEN)] == (-19, 1)
