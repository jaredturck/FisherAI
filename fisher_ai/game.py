from collections import deque

import numpy as np

from fisher_ai import chess

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
MAX_GAME_PLIES = 320


class PositionSnapshot:
    def __init__(self, board, repetition_count):
        self.repetition_count = int(repetition_count)
        self.bitboards = np.zeros(12, dtype=np.uint64)

        for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
            for piece_index, piece_type in enumerate(PIECE_TYPES):
                self.bitboards[color_index * 6 + piece_index] = np.uint64(
                    int(board.pieces_mask(piece_type, color))
                )


class GameState:
    def __init__(self, board=None):
        self.board = board.copy() if board else chess.Board()
        self.repetition_counts = {}
        self.history = deque(maxlen=8)

        key = self.board.position_key()
        self.repetition_counts[key] = 1
        self.history.append(PositionSnapshot(self.board, 1))

    def copy(self):
        state = object.__new__(GameState)
        state.board = self.board.copy()
        state.repetition_counts = self.repetition_counts.copy()
        state.history = deque(self.history, maxlen=8)
        return state

    def push(self, move):
        self.board.push(move)
        key = self.board.position_key()
        count = self.repetition_counts.get(key, 0) + 1
        self.repetition_counts[key] = count
        self.history.append(PositionSnapshot(self.board, count))

    def current_repetition_count(self):
        return self.repetition_counts.get(self.board.position_key(), 1)

    def is_terminal(self):
        if self.board.is_checkmate():
            return True
        if self.board.ply() >= MAX_GAME_PLIES:
            return True
        if self.current_repetition_count() >= 3:
            return True
        if self.board.halfmove_clock >= 100:
            return True
        return (
            self.board.is_stalemate() or self.board.is_insufficient_material()
        )

    def terminal_value(self):
        assert self.is_terminal()
        return -1.0 if self.board.is_checkmate() else 0.0

    def final_result(self):
        assert self.is_terminal()
        if not self.board.is_checkmate():
            return 0
        return -1 if self.board.turn == chess.WHITE else 1
