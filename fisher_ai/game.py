import numpy as np

from fisher_ai import chess

MAX_GAME_PLIES = 320
HISTORY_LENGTH = 8
PIECE_PLANES = 12


class GameState:
    __slots__ = (
        "board",
        "history_bitboards",
        "history_repetitions",
        "history_length",
        "position_hashes",
        "position_hash_length",
        "repetition_count",
        "legal_move_buffer",
        "terminal_status_cache",
    )

    def __init__(self, board=None):
        self.board = board.copy() if board else chess.Board()
        self.history_bitboards = np.zeros(
            (HISTORY_LENGTH, PIECE_PLANES),
            dtype=np.uint64,
        )
        self.history_repetitions = np.zeros(
            HISTORY_LENGTH,
            dtype=np.uint8,
        )
        self.history_length = 0
        self.position_hashes = np.zeros(
            MAX_GAME_PLIES + 1,
            dtype=np.uint64,
        )
        self.position_hash_length = 1
        self.position_hashes[0] = self.board.position_hash()
        self.repetition_count = 1
        self.legal_move_buffer = np.empty(256, dtype=np.uint32)
        self.terminal_status_cache = -1
        self._append_snapshot(1)

    def copy(self):
        state = object.__new__(GameState)
        state.board = self.board.copy()
        state.history_bitboards = self.history_bitboards.copy()
        state.history_repetitions = self.history_repetitions.copy()
        state.history_length = self.history_length
        state.position_hashes = self.position_hashes.copy()
        state.position_hash_length = self.position_hash_length
        state.repetition_count = self.repetition_count
        state.legal_move_buffer = np.empty(256, dtype=np.uint32)
        state.terminal_status_cache = self.terminal_status_cache
        return state

    def copy_from(self, other):
        self.board.copy_from(other.board)
        self.history_bitboards[:] = other.history_bitboards
        self.history_repetitions[:] = other.history_repetitions
        self.history_length = other.history_length
        self.position_hashes[:] = other.position_hashes
        self.position_hash_length = other.position_hash_length
        self.repetition_count = other.repetition_count
        self.terminal_status_cache = other.terminal_status_cache
        return self

    def _append_snapshot(self, repetition_count):
        if self.history_length < HISTORY_LENGTH:
            row = self.history_length
            self.history_length += 1
        else:
            self.history_bitboards[:-1] = self.history_bitboards[1:]
            self.history_repetitions[:-1] = self.history_repetitions[1:]
            row = HISTORY_LENGTH - 1

        self.board.fill_piece_bitboards(self.history_bitboards[row])
        self.history_repetitions[row] = repetition_count

    def push(self, move):
        self.board.push(move)
        self.terminal_status_cache = -1
        position_hash = self.board.position_hash()
        length = self.position_hash_length
        self.position_hashes[length] = position_hash
        self.position_hash_length = length + 1
        count = int(
            np.count_nonzero(
                self.position_hashes[: length + 1] == position_hash
            )
        )
        self.repetition_count = count
        self._append_snapshot(count)

    def current_bitboards(self, output=None):
        if output is None:
            output = np.empty(PIECE_PLANES, dtype=np.uint64)
        output[:] = self.history_bitboards[self.history_length - 1]
        return output

    def current_repetition_count(self):
        return self.repetition_count

    def is_rule_draw(self):
        if self.board.ply_count >= MAX_GAME_PLIES:
            return True
        if self.repetition_count >= 3:
            return True
        if self.board.halfmove_clock >= 100:
            return True
        return self.board.is_insufficient_material()

    def terminal_status(self):
        if self.terminal_status_cache >= 0:
            return self.terminal_status_cache
        _, status = self.board.fill_legal_moves(self.legal_move_buffer)
        if status == chess.CHECKMATE:
            self.terminal_status_cache = status
            return status
        if self.is_rule_draw():
            status = chess.STALEMATE
        self.terminal_status_cache = status
        return status

    def is_terminal(self):
        return self.terminal_status() != chess.ONGOING

    def terminal_value(self):
        return -1.0 if self.terminal_status() == chess.CHECKMATE else 0.0

    def final_result(self):
        if self.terminal_status() != chess.CHECKMATE:
            return 0
        return -1 if self.board.turn == chess.WHITE else 1
