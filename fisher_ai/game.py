from collections import deque

import numpy as np

from fisher_ai import chess

MAX_GAME_PLIES = 320


class PositionSnapshot:
    def __init__(self, board, repetition_count):
        self.repetition_count = int(repetition_count)
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        self.bitboards = np.asarray(
            (
                board.pawns & white,
                board.knights & white,
                board.bishops & white,
                board.rooks & white,
                board.queens & white,
                board.kings & white,
                board.pawns & black,
                board.knights & black,
                board.bishops & black,
                board.rooks & black,
                board.queens & black,
                board.kings & black,
            ),
            dtype=np.uint64,
        )


class GameState:
    def __init__(self, board=None):
        self.board = board.copy() if board else chess.Board()
        self.repetition_counts = {}
        self.history = deque(maxlen=8)

        key = self.board.position_key()
        self.repetition_counts[key] = 1
        self.repetition_count = 1
        self.history.append(PositionSnapshot(self.board, 1))

    def copy(self):
        state = object.__new__(GameState)
        state.board = self.board.copy()
        state.repetition_counts = self.repetition_counts.copy()
        state.repetition_count = self.repetition_count
        state.history = deque(self.history, maxlen=8)
        return state

    def push(self, move):
        self.board.push(move)
        key = self.board.position_key()
        count = self.repetition_counts.get(key, 0) + 1
        self.repetition_counts[key] = count
        self.repetition_count = count
        self.history.append(PositionSnapshot(self.board, count))

    def current_repetition_count(self):
        return self.repetition_count

    def is_rule_draw(self):
        if self.board.ply() >= MAX_GAME_PLIES:
            return True
        if self.repetition_count >= 3:
            return True
        if self.board.halfmove_clock >= 100:
            return True
        return self.board.is_insufficient_material()

    def is_terminal(self):
        if self.is_rule_draw():
            return True
        return not any(self.board.legal_moves)

    def terminal_value(self):
        return -1.0 if self.board.is_check() else 0.0

    def final_result(self):
        if not self.board.is_check():
            return 0
        return -1 if self.board.turn == chess.WHITE else 1
