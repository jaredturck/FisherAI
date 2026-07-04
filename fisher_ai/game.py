from collections import deque

import chess
import chess.polyglot
import numpy as np

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


class PositionSnapshot:
    def __init__(self, board, repetition_count):
        self.board = board.copy(stack=False)
        self.repetition_count = repetition_count
        self.piece_planes = self.build_piece_planes(self.board)

    @staticmethod
    def build_piece_planes(board):
        planes = np.zeros((12, 8, 8), dtype=np.uint8)
        for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
            for piece_index, piece_type in enumerate(PIECE_TYPES):
                for square in board.pieces(piece_type, color):
                    row = chess.square_rank(square)
                    col = chess.square_file(square)
                    planes[color_index * 6 + piece_index, row, col] = 1
        return planes

    def copy(self):
        return self


class GameState:
    def __init__(self, board=None, max_game_plies=512):
        self.board = board.copy(stack=False) if board else chess.Board()
        self.max_game_plies = max_game_plies
        self.repetition_counts = {}
        self.history = deque(maxlen=8)

        key = self.position_key(self.board)
        self.repetition_counts[key] = 1
        self.history.append(PositionSnapshot(self.board, 1))

    @staticmethod
    def position_key(board):
        return chess.polyglot.zobrist_hash(board)

    @classmethod
    def from_fen(cls, fen, max_game_plies=512):
        return cls(chess.Board(fen), max_game_plies=max_game_plies)

    def copy(self):
        state = object.__new__(GameState)
        state.board = self.board.copy(stack=False)
        state.max_game_plies = self.max_game_plies
        state.repetition_counts = self.repetition_counts.copy()
        state.history = deque(self.history, maxlen=8)
        return state

    def push(self, move):
        self.board.push(move)
        key = self.position_key(self.board)
        count = self.repetition_counts.get(key, 0) + 1
        self.repetition_counts[key] = count
        self.history.append(PositionSnapshot(self.board, count))

    def push_moves(self, moves):
        for move in moves:
            self.push(move)

    def child(self, move):
        state = self.copy()
        state.push(move)
        return state

    def legal_moves(self):
        return list(self.board.legal_moves)

    def current_repetition_count(self):
        return self.repetition_counts.get(self.position_key(self.board), 1)

    def is_terminal(self):
        if self.board.is_checkmate():
            return True
        if self.board.ply() >= self.max_game_plies:
            return True
        if self.current_repetition_count() >= 3:
            return True
        if self.board.halfmove_clock >= 100:
            return True
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return True
        return self.board.is_seventyfive_moves()

    def terminal_value(self):
        assert self.is_terminal()
        return -1.0 if self.board.is_checkmate() else 0.0

    def final_result(self):
        assert self.is_terminal()
        if not self.board.is_checkmate():
            return 0
        return -1 if self.board.turn == chess.WHITE else 1
