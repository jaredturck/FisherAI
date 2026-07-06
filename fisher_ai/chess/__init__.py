"""FisherAI's minimal in-house standard chess module."""

from fisher_ai.chess.bitboards import square_file, square_rank
from fisher_ai.chess.board import BLACK, WHITE, Board
from fisher_ai.chess.move import (
    BISHOP,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    encode_move,
    move_from_square,
    move_from_uci,
    move_promotion,
    move_to_square,
    move_to_uci,
)

__all__ = [
    "BLACK",
    "BISHOP",
    "Board",
    "KING",
    "KNIGHT",
    "PAWN",
    "QUEEN",
    "ROOK",
    "WHITE",
    "encode_move",
    "move_from_square",
    "move_from_uci",
    "move_promotion",
    "move_to_square",
    "move_to_uci",
    "square_file",
    "square_rank",
]
