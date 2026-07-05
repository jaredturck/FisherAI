"""Minimal standard-chess board adapted from python-chess 1.11.2.

This module intentionally supports only the behavior FisherAI uses: standard
FEN positions, legal move generation, move application, terminal detection,
castling state, piece bitboards, and repetition identity.
"""

from fisher_ai.chess.bitboards import (
    A1,
    A8,
    B1,
    B8,
    BB_A1,
    BB_A8,
    BB_ALL,
    BB_CORNERS,
    BB_DARK_SQUARES,
    BB_DIAG_ATTACKS,
    BB_DIAG_MASKS,
    BB_E1,
    BB_E8,
    BB_EMPTY,
    BB_FILE_ATTACKS,
    BB_FILE_MASKS,
    BB_H1,
    BB_H8,
    BB_KING_ATTACKS,
    BB_KNIGHT_ATTACKS,
    BB_LIGHT_SQUARES,
    BB_PAWN_ATTACKS,
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
    BB_RANK_ATTACKS,
    BB_RANK_MASKS,
    BB_RANKS,
    BB_SQUARES,
    C1,
    C8,
    D1,
    D8,
    E1,
    E8,
    F1,
    F8,
    G1,
    G8,
    H1,
    H8,
    between,
    msb,
    parse_square,
    popcount,
    ray,
    scan_forward,
    scan_reversed,
    square_rank,
)
from fisher_ai.chess.move import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, Move

WHITE = True
BLACK = False
COLORS = (WHITE, BLACK)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PIECE_FROM_SYMBOL = {
    "p": PAWN,
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
    "k": KING,
}


class LegalMoveView:
    __slots__ = ("board",)

    def __init__(self, board):
        self.board = board

    def __iter__(self):
        return self.board.generate_legal_moves()

    def __bool__(self):
        return any(self.board.generate_legal_moves())

    def count(self):
        return len(list(self.board.generate_legal_moves()))


class Board:
    __slots__ = (
        "pawns",
        "knights",
        "bishops",
        "rooks",
        "queens",
        "kings",
        "occupied_co",
        "occupied",
        "turn",
        "castling_rights",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
    )

    def __init__(self, fen=STARTING_FEN):
        self.occupied_co = [BB_EMPTY, BB_EMPTY]
        if fen == STARTING_FEN:
            self.reset()
        else:
            self.set_fen(fen)

    @property
    def legal_moves(self):
        return LegalMoveView(self)

    def reset(self):
        self.pawns = BB_RANK_2 | BB_RANK_7
        self.knights = BB_SQUARES[1] | BB_SQUARES[6] | BB_SQUARES[57] | BB_SQUARES[62]
        self.bishops = BB_SQUARES[2] | BB_SQUARES[5] | BB_SQUARES[58] | BB_SQUARES[61]
        self.rooks = BB_CORNERS
        self.queens = BB_SQUARES[3] | BB_SQUARES[59]
        self.kings = BB_E1 | BB_E8
        self.occupied_co[WHITE] = BB_RANK_1 | BB_RANK_2
        self.occupied_co[BLACK] = BB_RANK_7 | BB_RANK_8
        self.occupied = self.occupied_co[WHITE] | self.occupied_co[BLACK]
        self.turn = WHITE
        self.castling_rights = BB_CORNERS
        self.ep_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

    def clear(self):
        self.pawns = BB_EMPTY
        self.knights = BB_EMPTY
        self.bishops = BB_EMPTY
        self.rooks = BB_EMPTY
        self.queens = BB_EMPTY
        self.kings = BB_EMPTY
        self.occupied_co[WHITE] = BB_EMPTY
        self.occupied_co[BLACK] = BB_EMPTY
        self.occupied = BB_EMPTY
        self.turn = WHITE
        self.castling_rights = BB_EMPTY
        self.ep_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

    def copy(self):
        board = object.__new__(Board)
        board.pawns = self.pawns
        board.knights = self.knights
        board.bishops = self.bishops
        board.rooks = self.rooks
        board.queens = self.queens
        board.kings = self.kings
        board.occupied_co = self.occupied_co.copy()
        board.occupied = self.occupied
        board.turn = self.turn
        board.castling_rights = self.castling_rights
        board.ep_square = self.ep_square
        board.halfmove_clock = self.halfmove_clock
        board.fullmove_number = self.fullmove_number
        return board

    def ply(self):
        return 2 * (self.fullmove_number - 1) + (self.turn == BLACK)

    def pieces_mask(self, piece_type, color):
        if piece_type == PAWN:
            pieces = self.pawns
        elif piece_type == KNIGHT:
            pieces = self.knights
        elif piece_type == BISHOP:
            pieces = self.bishops
        elif piece_type == ROOK:
            pieces = self.rooks
        elif piece_type == QUEEN:
            pieces = self.queens
        elif piece_type == KING:
            pieces = self.kings
        else:
            return BB_EMPTY
        return pieces & self.occupied_co[color]

    def pieces(self, piece_type, color):
        return scan_forward(self.pieces_mask(piece_type, color))

    def piece_type_at(self, square):
        mask = BB_SQUARES[square]
        if not self.occupied & mask:
            return None
        if self.pawns & mask:
            return PAWN
        if self.knights & mask:
            return KNIGHT
        if self.bishops & mask:
            return BISHOP
        if self.rooks & mask:
            return ROOK
        if self.queens & mask:
            return QUEEN
        return KING

    def king(self, color):
        king_mask = self.kings & self.occupied_co[color]
        return msb(king_mask) if king_mask else None

    def _remove_piece_at(self, square):
        piece_type = self.piece_type_at(square)
        if piece_type is None:
            return None

        mask = BB_SQUARES[square]
        if piece_type == PAWN:
            self.pawns ^= mask
        elif piece_type == KNIGHT:
            self.knights ^= mask
        elif piece_type == BISHOP:
            self.bishops ^= mask
        elif piece_type == ROOK:
            self.rooks ^= mask
        elif piece_type == QUEEN:
            self.queens ^= mask
        else:
            self.kings ^= mask

        self.occupied ^= mask
        self.occupied_co[WHITE] &= ~mask
        self.occupied_co[BLACK] &= ~mask
        return piece_type

    def _set_piece_at(self, square, piece_type, color):
        self._remove_piece_at(square)
        mask = BB_SQUARES[square]

        if piece_type == PAWN:
            self.pawns |= mask
        elif piece_type == KNIGHT:
            self.knights |= mask
        elif piece_type == BISHOP:
            self.bishops |= mask
        elif piece_type == ROOK:
            self.rooks |= mask
        elif piece_type == QUEEN:
            self.queens |= mask
        else:
            self.kings |= mask

        self.occupied |= mask
        self.occupied_co[color] |= mask

    def attacks_mask(self, square):
        square_mask = BB_SQUARES[square]
        if square_mask & self.pawns:
            color = bool(square_mask & self.occupied_co[WHITE])
            return BB_PAWN_ATTACKS[color][square]
        if square_mask & self.knights:
            return BB_KNIGHT_ATTACKS[square]
        if square_mask & self.kings:
            return BB_KING_ATTACKS[square]

        attacks = BB_EMPTY
        if square_mask & (self.bishops | self.queens):
            attacks |= BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
        if square_mask & (self.rooks | self.queens):
            attacks |= BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
            attacks |= BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
        return attacks

    def attackers_mask(self, color, square, occupied=None):
        occupied = self.occupied if occupied is None else occupied
        rank_pieces = BB_RANK_MASKS[square] & occupied
        file_pieces = BB_FILE_MASKS[square] & occupied
        diag_pieces = BB_DIAG_MASKS[square] & occupied
        queens_and_rooks = self.queens | self.rooks
        queens_and_bishops = self.queens | self.bishops

        attackers = (
            (BB_KING_ATTACKS[square] & self.kings)
            | (BB_KNIGHT_ATTACKS[square] & self.knights)
            | (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks)
            | (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks)
            | (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops)
            | (BB_PAWN_ATTACKS[not color][square] & self.pawns)
        )
        return attackers & self.occupied_co[color]

    def is_attacked_by(self, color, square, occupied=None):
        return bool(self.attackers_mask(color, square, occupied))

    def pin_mask(self, color, square):
        king = self.king(color)
        if king is None:
            return BB_ALL

        square_mask = BB_SQUARES[square]
        attack_sets = (
            (BB_FILE_ATTACKS, self.rooks | self.queens),
            (BB_RANK_ATTACKS, self.rooks | self.queens),
            (BB_DIAG_ATTACKS, self.bishops | self.queens),
        )

        for attacks, sliders in attack_sets:
            rays = attacks[king][0]
            if not rays & square_mask:
                continue

            snipers = rays & sliders & self.occupied_co[not color]
            for sniper in scan_reversed(snipers):
                if between(sniper, king) & (self.occupied | square_mask) == square_mask:
                    return ray(king, sniper)
            break

        return BB_ALL

    def checkers_mask(self):
        king = self.king(self.turn)
        return BB_EMPTY if king is None else self.attackers_mask(not self.turn, king)

    def is_check(self):
        return bool(self.checkers_mask())

    def generate_pseudo_legal_moves(self, from_mask=BB_ALL, to_mask=BB_ALL):
        our_pieces = self.occupied_co[self.turn]

        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            moves = self.attacks_mask(from_square) & ~our_pieces & to_mask
            for to_square in scan_reversed(moves):
                yield Move(from_square, to_square)

        if from_mask & self.kings:
            yield from self.generate_castling_moves(from_mask, to_mask)

        pawns = self.pawns & our_pieces & from_mask
        if not pawns:
            return

        for from_square in scan_reversed(pawns):
            targets = (
                BB_PAWN_ATTACKS[self.turn][from_square]
                & self.occupied_co[not self.turn]
                & to_mask
            )
            for to_square in scan_reversed(targets):
                if square_rank(to_square) in (0, 7):
                    yield Move(from_square, to_square, QUEEN)
                    yield Move(from_square, to_square, ROOK)
                    yield Move(from_square, to_square, BISHOP)
                    yield Move(from_square, to_square, KNIGHT)
                else:
                    yield Move(from_square, to_square)

        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

        for to_square in scan_reversed(single_moves & to_mask):
            from_square = to_square + (8 if self.turn == BLACK else -8)
            if square_rank(to_square) in (0, 7):
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield Move(from_square, to_square)

        for to_square in scan_reversed(double_moves & to_mask):
            from_square = to_square + (16 if self.turn == BLACK else -16)
            yield Move(from_square, to_square)

        if self.ep_square is not None:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

    def generate_pseudo_legal_ep(self, from_mask=BB_ALL, to_mask=BB_ALL):
        if self.ep_square is None or not BB_SQUARES[self.ep_square] & to_mask:
            return
        if BB_SQUARES[self.ep_square] & self.occupied:
            return

        capturers = (
            self.pawns
            & self.occupied_co[self.turn]
            & from_mask
            & BB_PAWN_ATTACKS[not self.turn][self.ep_square]
            & BB_RANKS[4 if self.turn else 3]
        )
        for capturer in scan_reversed(capturers):
            yield Move(capturer, self.ep_square)

    def _slider_blockers(self, king):
        rooks_and_queens = self.rooks | self.queens
        bishops_and_queens = self.bishops | self.queens
        snipers = (
            (BB_RANK_ATTACKS[king][0] & rooks_and_queens)
            | (BB_FILE_ATTACKS[king][0] & rooks_and_queens)
            | (BB_DIAG_ATTACKS[king][0] & bishops_and_queens)
        )

        blockers = BB_EMPTY
        for sniper in scan_reversed(snipers & self.occupied_co[not self.turn]):
            between_pieces = between(king, sniper) & self.occupied
            if between_pieces and BB_SQUARES[msb(between_pieces)] == between_pieces:
                blockers |= between_pieces
        return blockers & self.occupied_co[self.turn]

    def _ep_skewered(self, king, capturer):
        last_double = self.ep_square + (-8 if self.turn == WHITE else 8)
        occupancy = (
            self.occupied
            & ~BB_SQUARES[last_double]
            & ~BB_SQUARES[capturer]
            | BB_SQUARES[self.ep_square]
        )

        horizontal_attackers = self.occupied_co[not self.turn] & (self.rooks | self.queens)
        if BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers:
            return True

        diagonal_attackers = self.occupied_co[not self.turn] & (self.bishops | self.queens)
        return bool(
            BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & occupancy] & diagonal_attackers
        )

    def is_en_passant(self, move):
        return bool(
            self.ep_square == move.to_square
            and self.pawns & BB_SQUARES[move.from_square]
            and abs(move.to_square - move.from_square) in (7, 9)
            and not self.occupied & BB_SQUARES[move.to_square]
        )

    @staticmethod
    def is_castling(move):
        return abs(move.to_square - move.from_square) == 2

    def _is_safe(self, king, blockers, move):
        if move.from_square == king:
            if self.is_castling(move):
                return True
            return not self.is_attacked_by(not self.turn, move.to_square)

        if self.is_en_passant(move):
            return bool(
                self.pin_mask(self.turn, move.from_square) & BB_SQUARES[move.to_square]
                and not self._ep_skewered(king, move.from_square)
            )

        return bool(
            not blockers & BB_SQUARES[move.from_square]
            or ray(move.from_square, move.to_square) & BB_SQUARES[king]
        )

    def _generate_evasions(self, king, checkers, from_mask=BB_ALL, to_mask=BB_ALL):
        sliders = checkers & (self.bishops | self.rooks | self.queens)
        attacked = BB_EMPTY
        for checker in scan_reversed(sliders):
            attacked |= ray(king, checker) & ~BB_SQUARES[checker]

        if BB_SQUARES[king] & from_mask:
            destinations = (
                BB_KING_ATTACKS[king]
                & ~self.occupied_co[self.turn]
                & ~attacked
                & to_mask
            )
            for to_square in scan_reversed(destinations):
                yield Move(king, to_square)

        checker = msb(checkers)
        if BB_SQUARES[checker] != checkers:
            return

        target = between(king, checker) | checkers
        yield from self.generate_pseudo_legal_moves(~self.kings & from_mask, target & to_mask)

        if self.ep_square is not None and not BB_SQUARES[self.ep_square] & target:
            last_double = self.ep_square + (-8 if self.turn == WHITE else 8)
            if last_double == checker:
                yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

    def generate_legal_moves(self, from_mask=BB_ALL, to_mask=BB_ALL):
        king = self.king(self.turn)
        if king is None:
            yield from self.generate_pseudo_legal_moves(from_mask, to_mask)
            return

        blockers = self._slider_blockers(king)
        checkers = self.attackers_mask(not self.turn, king)
        moves = (
            self._generate_evasions(king, checkers, from_mask, to_mask)
            if checkers
            else self.generate_pseudo_legal_moves(from_mask, to_mask)
        )
        for move in moves:
            if self._is_safe(king, blockers, move):
                yield move

    def _attacked_for_king(self, path, occupied):
        return any(
            self.attackers_mask(not self.turn, square, occupied)
            for square in scan_reversed(path)
        )

    def generate_castling_moves(self, from_mask=BB_ALL, to_mask=BB_ALL):
        rights = self.clean_castling_rights()

        if self.turn == WHITE:
            king_square = E1
            king_mask = BB_E1
            if not from_mask & king_mask:
                return

            if rights & BB_H1 and to_mask & BB_SQUARES[G1]:
                empty_path = BB_SQUARES[F1] | BB_SQUARES[G1]
                if not self.occupied & empty_path and not self._attacked_for_king(
                    BB_E1 | BB_SQUARES[F1] | BB_SQUARES[G1],
                    self.occupied ^ king_mask,
                ):
                    yield Move(king_square, G1)

            if rights & BB_A1 and to_mask & BB_SQUARES[C1]:
                empty_path = BB_SQUARES[B1] | BB_SQUARES[C1] | BB_SQUARES[D1]
                if not self.occupied & empty_path and not self._attacked_for_king(
                    BB_E1 | BB_SQUARES[D1] | BB_SQUARES[C1],
                    self.occupied ^ king_mask,
                ):
                    yield Move(king_square, C1)
        else:
            king_square = E8
            king_mask = BB_E8
            if not from_mask & king_mask:
                return

            if rights & BB_H8 and to_mask & BB_SQUARES[G8]:
                empty_path = BB_SQUARES[F8] | BB_SQUARES[G8]
                if not self.occupied & empty_path and not self._attacked_for_king(
                    BB_E8 | BB_SQUARES[F8] | BB_SQUARES[G8],
                    self.occupied ^ king_mask,
                ):
                    yield Move(king_square, G8)

            if rights & BB_A8 and to_mask & BB_SQUARES[C8]:
                empty_path = BB_SQUARES[B8] | BB_SQUARES[C8] | BB_SQUARES[D8]
                if not self.occupied & empty_path and not self._attacked_for_king(
                    BB_E8 | BB_SQUARES[D8] | BB_SQUARES[C8],
                    self.occupied ^ king_mask,
                ):
                    yield Move(king_square, C8)

    def clean_castling_rights(self):
        rights = self.castling_rights & self.rooks
        white_rights = rights & self.occupied_co[WHITE] & (BB_A1 | BB_H1)
        black_rights = rights & self.occupied_co[BLACK] & (BB_A8 | BB_H8)

        if not self.kings & self.occupied_co[WHITE] & BB_E1:
            white_rights = BB_EMPTY
        if not self.kings & self.occupied_co[BLACK] & BB_E8:
            black_rights = BB_EMPTY
        return white_rights | black_rights

    def has_kingside_castling_rights(self, color):
        return bool(self.clean_castling_rights() & (BB_H1 if color == WHITE else BB_H8))

    def has_queenside_castling_rights(self, color):
        return bool(self.clean_castling_rights() & (BB_A1 if color == WHITE else BB_A8))

    def push(self, move):
        from_mask = BB_SQUARES[move.from_square]
        to_mask = BB_SQUARES[move.to_square]
        piece_type = self.piece_type_at(move.from_square)
        if piece_type is None:
            raise ValueError(f"no piece at source square for move {move.uci()!r}")

        ep_square = self.ep_square
        captured_piece_type = self.piece_type_at(move.to_square)
        en_passant = self.is_en_passant(move)
        castling = piece_type == KING and self.is_castling(move)

        self.castling_rights = self.clean_castling_rights()
        self.castling_rights &= ~from_mask & ~to_mask
        if piece_type == KING:
            self.castling_rights &= ~(BB_RANK_1 if self.turn == WHITE else BB_RANK_8)

        self.ep_square = None
        self.halfmove_clock += 1
        if piece_type == PAWN or captured_piece_type is not None or en_passant:
            self.halfmove_clock = 0

        if self.turn == BLACK:
            self.fullmove_number += 1

        self._remove_piece_at(move.from_square)

        if en_passant:
            capture_square = ep_square + (-8 if self.turn == WHITE else 8)
            self._remove_piece_at(capture_square)

        if piece_type == PAWN:
            difference = move.to_square - move.from_square
            if difference == 16 and square_rank(move.from_square) == 1:
                self.ep_square = move.from_square + 8
            elif difference == -16 and square_rank(move.from_square) == 6:
                self.ep_square = move.from_square - 8

        if castling:
            if move.to_square == G1:
                rook_from, rook_to = H1, F1
            elif move.to_square == C1:
                rook_from, rook_to = A1, D1
            elif move.to_square == G8:
                rook_from, rook_to = H8, F8
            else:
                rook_from, rook_to = A8, D8

            self._remove_piece_at(rook_from)
            self._set_piece_at(move.to_square, KING, self.turn)
            self._set_piece_at(rook_to, ROOK, self.turn)
        else:
            placed_piece = move.promotion or piece_type
            self._set_piece_at(move.to_square, placed_piece, self.turn)

        self.turn = not self.turn

    def is_checkmate(self):
        return self.is_check() and not any(self.generate_legal_moves())

    def is_stalemate(self):
        return not self.is_check() and not any(self.generate_legal_moves())

    def has_insufficient_material(self, color):
        if self.occupied_co[color] & (self.pawns | self.rooks | self.queens):
            return False

        if self.occupied_co[color] & self.knights:
            return popcount(self.occupied_co[color]) <= 2 and not (
                self.occupied_co[not color] & ~self.kings & ~self.queens
            )

        if self.occupied_co[color] & self.bishops:
            same_color = not self.bishops & BB_DARK_SQUARES or not self.bishops & BB_LIGHT_SQUARES
            return same_color and not self.pawns and not self.knights

        return True

    def is_insufficient_material(self):
        return all(self.has_insufficient_material(color) for color in COLORS)

    def generate_legal_ep(self):
        for move in self.generate_legal_moves():
            if self.is_en_passant(move):
                yield move

    def has_legal_en_passant(self):
        return any(self.generate_legal_ep())

    def position_key(self):
        return (
            self.pawns,
            self.knights,
            self.bishops,
            self.rooks,
            self.queens,
            self.kings,
            self.occupied_co[WHITE],
            self.occupied_co[BLACK],
            self.turn,
            self.clean_castling_rights(),
            self.ep_square if self.has_legal_en_passant() else None,
        )

    def set_fen(self, fen):
        parts = fen.strip().split()
        if len(parts) != 6:
            raise ValueError(f"expected six FEN fields: {fen!r}")

        board_fen, turn, castling, ep_square, halfmove, fullmove = parts
        rows = board_fen.split("/")
        if len(rows) != 8:
            raise ValueError(f"expected eight FEN ranks: {fen!r}")

        self.clear()
        for fen_rank, row in enumerate(rows):
            file_index = 0
            for symbol in row:
                if symbol.isdigit():
                    file_index += int(symbol)
                    continue

                piece_type = PIECE_FROM_SYMBOL.get(symbol.lower())
                if piece_type is None or file_index >= 8:
                    raise ValueError(f"invalid board field in FEN: {fen!r}")

                square = (7 - fen_rank) * 8 + file_index
                self._set_piece_at(square, piece_type, symbol.isupper())
                file_index += 1

            if file_index != 8:
                raise ValueError(f"invalid rank width in FEN: {fen!r}")

        if turn == "w":
            self.turn = WHITE
        elif turn == "b":
            self.turn = BLACK
        else:
            raise ValueError(f"invalid side to move in FEN: {fen!r}")

        self.castling_rights = BB_EMPTY
        if castling != "-":
            for symbol in castling:
                if symbol == "K":
                    self.castling_rights |= BB_H1
                elif symbol == "Q":
                    self.castling_rights |= BB_A1
                elif symbol == "k":
                    self.castling_rights |= BB_H8
                elif symbol == "q":
                    self.castling_rights |= BB_A8
                else:
                    raise ValueError(f"invalid castling field in FEN: {fen!r}")

        self.ep_square = None if ep_square == "-" else parse_square(ep_square)
        self.halfmove_clock = int(halfmove)
        self.fullmove_number = int(fullmove)
        if self.halfmove_clock < 0 or self.fullmove_number < 1:
            raise ValueError(f"invalid move counters in FEN: {fen!r}")
