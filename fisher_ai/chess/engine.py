"""Purpose-built standard-chess kernel used internally by FisherAI."""

import numpy as np

from fisher_ai.chess._tables import (
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
    BB_RANK_4,
    BB_RANK_5,
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
    ZOBRIST_CASTLING,
    ZOBRIST_EP_FILE,
    ZOBRIST_PIECES,
    ZOBRIST_TURN,
    between,
    msb,
    parse_square,
    popcount,
    ray,
    square_file,
    square_name,
    square_rank,
)

WHITE = True
BLACK = False
COLORS = (WHITE, BLACK)

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

FROM_MASK = 0x3F
TO_MASK = 0x3F
TO_SHIFT = 6
PROMOTION_SHIFT = 12
MOVE_BASE_MASK = 0x7FFF
MOVING_PIECE_SHIFT = 15
CAPTURED_PIECE_SHIFT = 18
EN_PASSANT_FLAG = 1 << 21
CASTLING_FLAG = 1 << 22
DOUBLE_PAWN_FLAG = 1 << 23

WHITE_KINGSIDE = 1
WHITE_QUEENSIDE = 2
BLACK_KINGSIDE = 4
BLACK_QUEENSIDE = 8
ALL_CASTLING_RIGHTS = 15

ONGOING = 0
CHECKMATE = 1
STALEMATE = 2

ACTION_PLANES = 73
ACTION_SIZE = 64 * ACTION_PLANES

QUEEN_DIRECTIONS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
KNIGHT_DIRECTIONS = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)
UNDERPROMOTION_PIECES = (KNIGHT, BISHOP, ROOK)

PIECE_FROM_SYMBOL = {
    "p": PAWN,
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
    "k": KING,
}
PROMOTION_FROM_SYMBOL = {
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
}
PROMOTION_TO_SYMBOL = ("", "n", "b", "r", "q")


def encode_move(from_square, to_square, promotion=None):
    """Pack move coordinates and promotion into the engine move format."""
    promotion_code = 0 if not promotion else int(promotion) - 1
    return (
        int(from_square)
        | int(to_square) << TO_SHIFT
        | promotion_code << PROMOTION_SHIFT
    )


def move_from_square(move):
    """Return the source square encoded in a packed move."""
    return int(move) & FROM_MASK


def move_to_square(move):
    """Return the destination square encoded in a packed move."""
    return int(move) >> TO_SHIFT & TO_MASK


def move_promotion_code(move):
    """Return the promotion code encoded in a packed move."""
    return int(move) >> PROMOTION_SHIFT & 7


def move_promotion(move):
    """Return the promoted piece encoded in a packed move."""
    code = move_promotion_code(move)
    return None if code == 0 else code + 1


def move_piece(move):
    """Return the moving piece encoded in an annotated move."""
    return int(move) >> MOVING_PIECE_SHIFT & 7


def move_captured_piece(move):
    """Return the captured piece encoded in an annotated move."""
    return int(move) >> CAPTURED_PIECE_SHIFT & 7


def move_to_uci(move):
    """Convert a packed move into UCI notation."""
    value = square_name(move_from_square(move)) + square_name(
        move_to_square(move)
    )
    promotion_code = move_promotion_code(move)
    if promotion_code:
        value += PROMOTION_TO_SYMBOL[promotion_code]
    return value


def move_from_uci(uci):
    """Parse UCI notation into a packed move."""
    if len(uci) not in (4, 5):
        raise ValueError(f"expected UCI move of length 4 or 5: {uci!r}")

    from_square = parse_square(uci[:2])
    to_square = parse_square(uci[2:4])
    if from_square == to_square:
        raise ValueError(
            f"source and destination squares are identical: {uci!r}"
        )

    promotion = None
    if len(uci) == 5:
        try:
            promotion = PROMOTION_FROM_SYMBOL[uci[4]]
        except KeyError:
            raise ValueError(f"invalid promotion piece: {uci!r}") from None

    return encode_move(from_square, to_square, promotion)


def canonical_square(square, current_color):
    """Orient a square from the current player perspective."""
    return square if current_color == WHITE else 63 - square


def _calculate_action(from_square, to_square, promotion, current_color):
    canonical_from = canonical_square(from_square, current_color)
    canonical_to = canonical_square(to_square, current_color)
    from_row = square_rank(canonical_from)
    from_col = square_file(canonical_from)
    to_row = square_rank(canonical_to)
    to_col = square_file(canonical_to)
    row_delta = to_row - from_row
    col_delta = to_col - from_col

    if promotion in UNDERPROMOTION_PIECES:
        if row_delta != 1 or col_delta not in (-1, 0, 1):
            return -1
        direction_index = col_delta + 1
        piece_index = promotion - KNIGHT
        move_plane = 64 + direction_index * 3 + piece_index
        return move_plane * 64 + canonical_from

    delta = (row_delta, col_delta)
    if delta in KNIGHT_DIRECTIONS:
        move_plane = 56 + KNIGHT_DIRECTIONS.index(delta)
        return move_plane * 64 + canonical_from

    distance = max(abs(row_delta), abs(col_delta))
    if distance < 1 or distance > 7:
        return -1
    direction = (
        0 if row_delta == 0 else row_delta // abs(row_delta),
        0 if col_delta == 0 else col_delta // abs(col_delta),
    )
    if direction not in QUEEN_DIRECTIONS:
        return -1
    if not (
        row_delta == 0 or col_delta == 0 or abs(row_delta) == abs(col_delta)
    ):
        return -1

    move_plane = QUEEN_DIRECTIONS.index(direction) * 7 + distance - 1
    return move_plane * 64 + canonical_from


PACKED_MOVE_COUNT = 1 << 15


def _build_action_lookup():
    """Build the packed move-to-policy-action lookup table."""
    lookup = np.full((2, PACKED_MOVE_COUNT), -1, dtype=np.int16)
    for color in (BLACK, WHITE):
        for from_square in range(64):
            for to_square in range(64):
                base = from_square | to_square << TO_SHIFT
                lookup[int(color), base] = _calculate_action(
                    from_square,
                    to_square,
                    None,
                    color,
                )
                for promotion_code in range(1, 5):
                    promotion = promotion_code + 1
                    base = (
                        from_square
                        | to_square << TO_SHIFT
                        | promotion_code << PROMOTION_SHIFT
                    )
                    action_promotion = (
                        promotion
                        if promotion in UNDERPROMOTION_PIECES
                        else None
                    )
                    lookup[int(color), base] = _calculate_action(
                        from_square,
                        to_square,
                        action_promotion,
                        color,
                    )
    return lookup


PACKED_ACTION_LOOKUP = _build_action_lookup()


def move_to_action(move, current_color):
    """Map a packed move to its neural policy action."""
    base_move = int(move) & MOVE_BASE_MASK
    action = int(PACKED_ACTION_LOOKUP[int(current_color), base_move])
    if action < 0:
        raise AssertionError("move could not be mapped to a policy action")
    return action


class Board:
    """Represent and update one standard chess position with bitboards."""

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
        "ep_hash_file",
        "halfmove_clock",
        "ply_count",
        "zobrist_hash",
    )

    def __init__(self, fen=STARTING_FEN):
        self.occupied_co = [BB_EMPTY, BB_EMPTY]
        if fen == STARTING_FEN:
            self.reset()
        else:
            self.set_fen(fen)

    def reset(self):
        """Reset the board to the standard starting position."""
        self.pawns = BB_RANK_2 | BB_RANK_7
        self.knights = (
            BB_SQUARES[1] | BB_SQUARES[6] | BB_SQUARES[57] | BB_SQUARES[62]
        )
        self.bishops = (
            BB_SQUARES[2] | BB_SQUARES[5] | BB_SQUARES[58] | BB_SQUARES[61]
        )
        self.rooks = BB_CORNERS
        self.queens = BB_SQUARES[3] | BB_SQUARES[59]
        self.kings = BB_E1 | BB_E8
        self.occupied_co[WHITE] = BB_RANK_1 | BB_RANK_2
        self.occupied_co[BLACK] = BB_RANK_7 | BB_RANK_8
        self.occupied = self.occupied_co[WHITE] | self.occupied_co[BLACK]
        self.turn = WHITE
        self.castling_rights = ALL_CASTLING_RIGHTS
        self.ep_square = None
        self.ep_hash_file = -1
        self.halfmove_clock = 0
        self.ply_count = 0
        self.zobrist_hash = self._compute_zobrist()

    def clear(self):
        """Reset every board field to an empty position."""
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
        self.castling_rights = 0
        self.ep_square = None
        self.ep_hash_file = -1
        self.halfmove_clock = 0
        self.ply_count = 0
        self.zobrist_hash = 0

    def copy(self):
        """Return an independent copy of the board."""
        board = object.__new__(Board)
        board.occupied_co = [BB_EMPTY, BB_EMPTY]
        return board.copy_from(self)

    def copy_from(self, other):
        """Overwrite this board with another board state."""
        self.pawns = other.pawns
        self.knights = other.knights
        self.bishops = other.bishops
        self.rooks = other.rooks
        self.queens = other.queens
        self.kings = other.kings
        self.occupied_co[WHITE] = other.occupied_co[WHITE]
        self.occupied_co[BLACK] = other.occupied_co[BLACK]
        self.occupied = other.occupied
        self.turn = other.turn
        self.castling_rights = other.castling_rights
        self.ep_square = other.ep_square
        self.ep_hash_file = other.ep_hash_file
        self.halfmove_clock = other.halfmove_clock
        self.ply_count = other.ply_count
        self.zobrist_hash = other.zobrist_hash
        return self

    def fill_piece_bitboards(self, output):
        """Write the twelve color-piece bitboards into an output array."""
        white = self.occupied_co[WHITE]
        black = self.occupied_co[BLACK]
        output[0] = self.pawns & white
        output[1] = self.knights & white
        output[2] = self.bishops & white
        output[3] = self.rooks & white
        output[4] = self.queens & white
        output[5] = self.kings & white
        output[6] = self.pawns & black
        output[7] = self.knights & black
        output[8] = self.bishops & black
        output[9] = self.rooks & black
        output[10] = self.queens & black
        output[11] = self.kings & black
        return output

    def piece_type_at(self, square):
        """Return the piece type occupying a square."""
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

    def color_at(self, square):
        """Return the color occupying a square."""
        mask = BB_SQUARES[square]
        if self.occupied_co[WHITE] & mask:
            return WHITE
        if self.occupied_co[BLACK] & mask:
            return BLACK
        return None

    def king(self, color):
        """Return the king square for one color."""
        king_mask = self.kings & self.occupied_co[color]
        return msb(king_mask) if king_mask else None

    def _piece_bitboard(self, piece_type):
        if piece_type == PAWN:
            return self.pawns
        if piece_type == KNIGHT:
            return self.knights
        if piece_type == BISHOP:
            return self.bishops
        if piece_type == ROOK:
            return self.rooks
        if piece_type == QUEEN:
            return self.queens
        return self.kings

    def _set_piece_bitboard(self, piece_type, value):
        if piece_type == PAWN:
            self.pawns = value
        elif piece_type == KNIGHT:
            self.knights = value
        elif piece_type == BISHOP:
            self.bishops = value
        elif piece_type == ROOK:
            self.rooks = value
        elif piece_type == QUEEN:
            self.queens = value
        else:
            self.kings = value

    def _set_piece_raw(self, square, piece_type, color):
        mask = BB_SQUARES[square]
        self._set_piece_bitboard(
            piece_type,
            self._piece_bitboard(piece_type) | mask,
        )
        self.occupied |= mask
        self.occupied_co[color] |= mask

    def _remove_known_piece(self, square, piece_type, color):
        mask = BB_SQUARES[square]
        self._set_piece_bitboard(
            piece_type,
            self._piece_bitboard(piece_type) & ~mask,
        )
        self.occupied &= ~mask
        self.occupied_co[color] &= ~mask
        self.zobrist_hash ^= ZOBRIST_PIECES[int(color)][piece_type - 1][square]

    def _set_known_piece(self, square, piece_type, color):
        mask = BB_SQUARES[square]
        self._set_piece_bitboard(
            piece_type,
            self._piece_bitboard(piece_type) | mask,
        )
        self.occupied |= mask
        self.occupied_co[color] |= mask
        self.zobrist_hash ^= ZOBRIST_PIECES[int(color)][piece_type - 1][square]

    def attacks_mask(self, square):
        """Return every square attacked by the piece on a square."""
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
            attacks |= BB_DIAG_ATTACKS[square][
                BB_DIAG_MASKS[square] & self.occupied
            ]
        if square_mask & (self.rooks | self.queens):
            attacks |= BB_RANK_ATTACKS[square][
                BB_RANK_MASKS[square] & self.occupied
            ]
            attacks |= BB_FILE_ATTACKS[square][
                BB_FILE_MASKS[square] & self.occupied
            ]
        return attacks

    def attackers_mask(self, color, square, occupied=None):
        """Return pieces of one color attacking a square."""
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
        """Report whether one color attacks a square."""
        return bool(self.attackers_mask(color, square, occupied))

    def pin_mask(self, color, square):
        """Return the legal movement ray for a potentially pinned piece."""
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
            while snipers:
                sniper = snipers.bit_length() - 1
                snipers ^= BB_SQUARES[sniper]
                if (
                    between(sniper, king) & (self.occupied | square_mask)
                    == square_mask
                ):
                    return ray(king, sniper)
            break

        return BB_ALL

    def checkers_mask(self):
        """Return the enemy pieces checking the side to move."""
        king = self.king(self.turn)
        return (
            BB_EMPTY
            if king is None
            else self.attackers_mask(not self.turn, king)
        )

    def is_check(self):
        """Report whether the side to move is in check."""
        return bool(self.checkers_mask())

    @staticmethod
    def is_castling(move):
        """Report whether a packed move is a castling move."""
        move = int(move)
        if move & CASTLING_FLAG:
            return True
        return abs((move >> TO_SHIFT & TO_MASK) - (move & FROM_MASK)) == 2

    def is_en_passant(self, move):
        """Report whether a packed move is an en passant capture."""
        move = int(move)
        if move & EN_PASSANT_FLAG:
            return True
        from_square = move & FROM_MASK
        to_square = move >> TO_SHIFT & TO_MASK
        return bool(
            self.ep_square == to_square
            and self.pawns & BB_SQUARES[from_square]
            and abs(to_square - from_square) in (7, 9)
            and not self.occupied & BB_SQUARES[to_square]
        )

    def _slider_blockers(self, king):
        rooks_and_queens = self.rooks | self.queens
        bishops_and_queens = self.bishops | self.queens
        snipers = (
            (BB_RANK_ATTACKS[king][0] & rooks_and_queens)
            | (BB_FILE_ATTACKS[king][0] & rooks_and_queens)
            | (BB_DIAG_ATTACKS[king][0] & bishops_and_queens)
        )

        blockers = BB_EMPTY
        snipers &= self.occupied_co[not self.turn]
        while snipers:
            sniper = snipers.bit_length() - 1
            snipers ^= BB_SQUARES[sniper]
            between_pieces = between(king, sniper) & self.occupied
            if (
                between_pieces
                and BB_SQUARES[msb(between_pieces)] == between_pieces
            ):
                blockers |= between_pieces
        return blockers & self.occupied_co[self.turn]

    def _ep_skewered(self, king, capturer):
        last_double = self.ep_square + (-8 if self.turn == WHITE else 8)
        occupancy = (
            self.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer]
            | BB_SQUARES[self.ep_square]
        )

        horizontal_attackers = self.occupied_co[not self.turn] & (
            self.rooks | self.queens
        )
        if (
            BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy]
            & horizontal_attackers
        ):
            return True

        diagonal_attackers = self.occupied_co[not self.turn] & (
            self.bishops | self.queens
        )
        return bool(
            BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & occupancy]
            & diagonal_attackers
        )

    def _is_safe(self, king, blockers, move):
        move = int(move)
        from_square = move & FROM_MASK
        to_square = move >> TO_SHIFT & TO_MASK
        annotated = bool(move >> MOVING_PIECE_SHIFT & 7)
        if from_square == king:
            if move & CASTLING_FLAG or (
                not annotated and abs(to_square - from_square) == 2
            ):
                return True
            return not self.is_attacked_by(not self.turn, to_square)

        en_passant = bool(move & EN_PASSANT_FLAG)
        if not annotated and not en_passant:
            en_passant = bool(
                self.ep_square == to_square
                and self.pawns & BB_SQUARES[from_square]
                and abs(to_square - from_square) in (7, 9)
                and not self.occupied & BB_SQUARES[to_square]
            )
        if en_passant:
            return bool(
                self.pin_mask(self.turn, from_square) & BB_SQUARES[to_square]
                and not self._ep_skewered(king, from_square)
            )

        return bool(
            not blockers & BB_SQUARES[from_square]
            or ray(from_square, to_square) & BB_SQUARES[king]
        )

    def _captured_piece(self, to_square):
        if not self.occupied_co[not self.turn] & BB_SQUARES[to_square]:
            return 0
        return self.piece_type_at(to_square) or 0

    def _fill_piece_moves(
        self,
        output,
        count,
        piece_type,
        pieces,
        from_mask,
        to_mask,
    ):
        pieces &= self.occupied_co[self.turn] & from_mask
        our_pieces = self.occupied_co[self.turn]
        piece_bits = piece_type << MOVING_PIECE_SHIFT
        while pieces:
            from_square = pieces.bit_length() - 1
            pieces ^= BB_SQUARES[from_square]
            if piece_type == KNIGHT:
                destinations = BB_KNIGHT_ATTACKS[from_square]
            elif piece_type == BISHOP:
                occupied = BB_DIAG_MASKS[from_square] & self.occupied
                destinations = BB_DIAG_ATTACKS[from_square][occupied]
            elif piece_type == ROOK:
                rank_occupied = BB_RANK_MASKS[from_square] & self.occupied
                file_occupied = BB_FILE_MASKS[from_square] & self.occupied
                destinations = (
                    BB_RANK_ATTACKS[from_square][rank_occupied]
                    | BB_FILE_ATTACKS[from_square][file_occupied]
                )
            elif piece_type == QUEEN:
                rank_occupied = BB_RANK_MASKS[from_square] & self.occupied
                file_occupied = BB_FILE_MASKS[from_square] & self.occupied
                diag_occupied = BB_DIAG_MASKS[from_square] & self.occupied
                destinations = (
                    BB_RANK_ATTACKS[from_square][rank_occupied]
                    | BB_FILE_ATTACKS[from_square][file_occupied]
                    | BB_DIAG_ATTACKS[from_square][diag_occupied]
                )
            else:
                destinations = BB_KING_ATTACKS[from_square]
            destinations &= ~our_pieces & to_mask
            while destinations:
                to_square = destinations.bit_length() - 1
                destinations ^= BB_SQUARES[to_square]
                captured_piece = self._captured_piece(to_square)
                output[count] = (
                    from_square
                    | to_square << TO_SHIFT
                    | piece_bits
                    | captured_piece << CAPTURED_PIECE_SHIFT
                )
                count += 1
        return count

    def _fill_castling_moves(self, output, count, from_mask, to_mask):
        if self.turn == WHITE:
            king_square = E1
            king_mask = BB_E1
            if not from_mask & king_mask:
                return count

            if (
                self.castling_rights & WHITE_KINGSIDE
                and to_mask & BB_SQUARES[G1]
            ):
                empty_path = BB_SQUARES[F1] | BB_SQUARES[G1]
                if (
                    not self.occupied & empty_path
                    and not self._attacked_for_king(
                        BB_E1 | BB_SQUARES[F1] | BB_SQUARES[G1],
                        self.occupied ^ king_mask,
                    )
                ):
                    output[count] = (
                        king_square
                        | G1 << TO_SHIFT
                        | KING << MOVING_PIECE_SHIFT
                        | CASTLING_FLAG
                    )
                    count += 1

            if (
                self.castling_rights & WHITE_QUEENSIDE
                and to_mask & BB_SQUARES[C1]
            ):
                empty_path = BB_SQUARES[B1] | BB_SQUARES[C1] | BB_SQUARES[D1]
                if (
                    not self.occupied & empty_path
                    and not self._attacked_for_king(
                        BB_E1 | BB_SQUARES[D1] | BB_SQUARES[C1],
                        self.occupied ^ king_mask,
                    )
                ):
                    output[count] = (
                        king_square
                        | C1 << TO_SHIFT
                        | KING << MOVING_PIECE_SHIFT
                        | CASTLING_FLAG
                    )
                    count += 1
            return count

        king_square = E8
        king_mask = BB_E8
        if not from_mask & king_mask:
            return count

        if self.castling_rights & BLACK_KINGSIDE and to_mask & BB_SQUARES[G8]:
            empty_path = BB_SQUARES[F8] | BB_SQUARES[G8]
            if not self.occupied & empty_path and not self._attacked_for_king(
                BB_E8 | BB_SQUARES[F8] | BB_SQUARES[G8],
                self.occupied ^ king_mask,
            ):
                output[count] = (
                    king_square
                    | G8 << TO_SHIFT
                    | KING << MOVING_PIECE_SHIFT
                    | CASTLING_FLAG
                )
                count += 1

        if self.castling_rights & BLACK_QUEENSIDE and to_mask & BB_SQUARES[C8]:
            empty_path = BB_SQUARES[B8] | BB_SQUARES[C8] | BB_SQUARES[D8]
            if not self.occupied & empty_path and not self._attacked_for_king(
                BB_E8 | BB_SQUARES[D8] | BB_SQUARES[C8],
                self.occupied ^ king_mask,
            ):
                output[count] = (
                    king_square
                    | C8 << TO_SHIFT
                    | KING << MOVING_PIECE_SHIFT
                    | CASTLING_FLAG
                )
                count += 1
        return count

    def _fill_pseudo_legal_ep(
        self,
        output,
        count,
        from_mask=BB_ALL,
        to_mask=BB_ALL,
    ):
        if self.ep_square is None or not BB_SQUARES[self.ep_square] & to_mask:
            return count
        if BB_SQUARES[self.ep_square] & self.occupied:
            return count

        capturers = (
            self.pawns
            & self.occupied_co[self.turn]
            & from_mask
            & BB_PAWN_ATTACKS[not self.turn][self.ep_square]
            & BB_RANKS[4 if self.turn else 3]
        )
        while capturers:
            from_square = capturers.bit_length() - 1
            capturers ^= BB_SQUARES[from_square]
            output[count] = (
                from_square
                | self.ep_square << TO_SHIFT
                | PAWN << MOVING_PIECE_SHIFT
                | PAWN << CAPTURED_PIECE_SHIFT
                | EN_PASSANT_FLAG
            )
            count += 1
        return count

    def _fill_pseudo_legal_moves(
        self,
        output,
        from_mask=BB_ALL,
        to_mask=BB_ALL,
        count=0,
    ):
        count = self._fill_piece_moves(
            output,
            count,
            KNIGHT,
            self.knights,
            from_mask,
            to_mask,
        )
        count = self._fill_piece_moves(
            output,
            count,
            BISHOP,
            self.bishops,
            from_mask,
            to_mask,
        )
        count = self._fill_piece_moves(
            output,
            count,
            ROOK,
            self.rooks,
            from_mask,
            to_mask,
        )
        count = self._fill_piece_moves(
            output,
            count,
            QUEEN,
            self.queens,
            from_mask,
            to_mask,
        )
        count = self._fill_piece_moves(
            output,
            count,
            KING,
            self.kings,
            from_mask,
            to_mask,
        )

        if from_mask & self.kings:
            count = self._fill_castling_moves(
                output, count, from_mask, to_mask
            )

        pawns = self.pawns & self.occupied_co[self.turn] & from_mask
        if not pawns:
            return count

        pawn_copy = pawns
        while pawn_copy:
            from_square = pawn_copy.bit_length() - 1
            pawn_copy ^= BB_SQUARES[from_square]
            targets = (
                BB_PAWN_ATTACKS[self.turn][from_square]
                & self.occupied_co[not self.turn]
                & to_mask
            )
            while targets:
                to_square = targets.bit_length() - 1
                targets ^= BB_SQUARES[to_square]
                captured_piece = self._captured_piece(to_square)
                if square_rank(to_square) in (0, 7):
                    for promotion_code in (4, 3, 2, 1):
                        output[count] = (
                            from_square
                            | to_square << TO_SHIFT
                            | promotion_code << PROMOTION_SHIFT
                            | PAWN << MOVING_PIECE_SHIFT
                            | captured_piece << CAPTURED_PIECE_SHIFT
                        )
                        count += 1
                else:
                    output[count] = (
                        from_square
                        | to_square << TO_SHIFT
                        | PAWN << MOVING_PIECE_SHIFT
                        | captured_piece << CAPTURED_PIECE_SHIFT
                    )
                    count += 1

        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & BB_RANK_4
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & BB_RANK_5

        destinations = single_moves & to_mask
        while destinations:
            to_square = destinations.bit_length() - 1
            destinations ^= BB_SQUARES[to_square]
            from_square = to_square + (8 if self.turn == BLACK else -8)
            if square_rank(to_square) in (0, 7):
                for promotion_code in (4, 3, 2, 1):
                    output[count] = (
                        from_square
                        | to_square << TO_SHIFT
                        | promotion_code << PROMOTION_SHIFT
                        | PAWN << MOVING_PIECE_SHIFT
                    )
                    count += 1
            else:
                output[count] = (
                    from_square
                    | to_square << TO_SHIFT
                    | PAWN << MOVING_PIECE_SHIFT
                )
                count += 1

        destinations = double_moves & to_mask
        while destinations:
            to_square = destinations.bit_length() - 1
            destinations ^= BB_SQUARES[to_square]
            from_square = to_square + (16 if self.turn == BLACK else -16)
            output[count] = (
                from_square
                | to_square << TO_SHIFT
                | PAWN << MOVING_PIECE_SHIFT
                | DOUBLE_PAWN_FLAG
            )
            count += 1

        return self._fill_pseudo_legal_ep(
            output,
            count,
            from_mask,
            to_mask,
        )

    def _fill_evasions(
        self,
        output,
        king,
        checkers,
        from_mask=BB_ALL,
        to_mask=BB_ALL,
    ):
        count = 0
        sliders = checkers & (self.bishops | self.rooks | self.queens)
        attacked = BB_EMPTY
        while sliders:
            checker = sliders.bit_length() - 1
            sliders ^= BB_SQUARES[checker]
            attacked |= ray(king, checker) & ~BB_SQUARES[checker]

        if BB_SQUARES[king] & from_mask:
            destinations = (
                BB_KING_ATTACKS[king]
                & ~self.occupied_co[self.turn]
                & ~attacked
                & to_mask
            )
            while destinations:
                to_square = destinations.bit_length() - 1
                destinations ^= BB_SQUARES[to_square]
                captured_piece = self._captured_piece(to_square)
                output[count] = (
                    king
                    | to_square << TO_SHIFT
                    | KING << MOVING_PIECE_SHIFT
                    | captured_piece << CAPTURED_PIECE_SHIFT
                )
                count += 1

        checker = msb(checkers)
        if BB_SQUARES[checker] != checkers:
            return count

        target = between(king, checker) | checkers
        count = self._fill_pseudo_legal_moves(
            output,
            from_mask=~self.kings & from_mask,
            to_mask=target & to_mask,
            count=count,
        )

        if (
            self.ep_square is not None
            and not BB_SQUARES[self.ep_square] & target
        ):
            last_double = self.ep_square + (-8 if self.turn == WHITE else 8)
            if last_double == checker:
                count = self._fill_pseudo_legal_ep(
                    output,
                    count,
                    from_mask,
                    to_mask,
                )
        return count

    def fill_legal_moves(self, output, actions=None):
        """Write legal moves and policy actions into reusable buffers."""
        king = self.king(self.turn)
        if king is None:
            candidate_count = self._fill_pseudo_legal_moves(output)
            checkers = BB_EMPTY
        else:
            blockers = self._slider_blockers(king)
            checkers = self.attackers_mask(not self.turn, king)
            if checkers:
                candidate_count = self._fill_evasions(output, king, checkers)
            else:
                candidate_count = self._fill_pseudo_legal_moves(output)

            write_index = 0
            king_mask = BB_SQUARES[king]
            for read_index in range(candidate_count):
                move = int(output[read_index])
                from_square = move & FROM_MASK
                to_square = move >> TO_SHIFT & TO_MASK
                if from_square == king:
                    safe = bool(
                        move & CASTLING_FLAG
                    ) or not self.is_attacked_by(
                        not self.turn,
                        to_square,
                    )
                elif move & EN_PASSANT_FLAG:
                    safe = bool(
                        self.pin_mask(self.turn, from_square)
                        & BB_SQUARES[to_square]
                        and not self._ep_skewered(king, from_square)
                    )
                else:
                    safe = bool(
                        not blockers & BB_SQUARES[from_square]
                        or ray(from_square, to_square) & king_mask
                    )
                if safe:
                    output[write_index] = move
                    write_index += 1
            candidate_count = write_index

        if actions is not None and candidate_count:
            base_moves = output[:candidate_count] & MOVE_BASE_MASK
            mapped = PACKED_ACTION_LOOKUP[int(self.turn), base_moves]
            actions[:candidate_count] = mapped

        if candidate_count:
            return candidate_count, ONGOING
        return 0, CHECKMATE if checkers else STALEMATE

    def _attacked_for_king(self, path, occupied):
        while path:
            square = path.bit_length() - 1
            path ^= BB_SQUARES[square]
            if self.attackers_mask(not self.turn, square, occupied):
                return True
        return False

    def _legal_ep_file(self):
        if self.ep_square is None:
            return -1
        if BB_SQUARES[self.ep_square] & self.occupied:
            return -1

        capturers = (
            self.pawns
            & self.occupied_co[self.turn]
            & BB_PAWN_ATTACKS[not self.turn][self.ep_square]
            & BB_RANKS[4 if self.turn else 3]
        )
        if not capturers:
            return -1

        king = self.king(self.turn)
        if king is None:
            return square_file(self.ep_square)

        blockers = self._slider_blockers(king)
        checkers = self.attackers_mask(not self.turn, king)
        checker = msb(checkers) if checkers else None
        if checkers and BB_SQUARES[checker] != checkers:
            return -1

        while capturers:
            from_square = capturers.bit_length() - 1
            capturers ^= BB_SQUARES[from_square]
            if checkers:
                target = between(king, checker) | checkers
                last_double = self.ep_square + (
                    -8 if self.turn == WHITE else 8
                )
                if (
                    not BB_SQUARES[self.ep_square] & target
                    and last_double != checker
                ):
                    continue
            move = (
                from_square
                | self.ep_square << TO_SHIFT
                | PAWN << MOVING_PIECE_SHIFT
                | PAWN << CAPTURED_PIECE_SHIFT
                | EN_PASSANT_FLAG
            )
            if self._is_safe(king, blockers, move):
                return square_file(self.ep_square)
        return -1

    def _compute_zobrist(self):
        value = ZOBRIST_CASTLING[self.castling_rights]
        if self.turn == BLACK:
            value ^= ZOBRIST_TURN

        for color in COLORS:
            occupied = self.occupied_co[color]
            for piece_type in range(PAWN, KING + 1):
                pieces = self._piece_bitboard(piece_type) & occupied
                while pieces:
                    square = pieces.bit_length() - 1
                    pieces ^= BB_SQUARES[square]
                    value ^= ZOBRIST_PIECES[int(color)][piece_type - 1][square]

        if self.ep_hash_file >= 0:
            value ^= ZOBRIST_EP_FILE[self.ep_hash_file]
        return value

    def position_hash(self):
        """Return the incrementally maintained repetition hash."""
        return self.zobrist_hash

    def push(self, move):
        """Apply one packed move and update all board state."""
        move = int(move)
        from_square = move_from_square(move)
        to_square = move_to_square(move)
        promotion_code = move_promotion_code(move)
        moving_piece = move_piece(move) or self.piece_type_at(from_square)
        if moving_piece is None:
            raise ValueError(
                f"no piece at source square for move {move_to_uci(move)!r}"
            )

        color = self.turn
        opponent = not color
        en_passant = self.is_en_passant(move)
        castling = moving_piece == KING and self.is_castling(move)
        captured_piece = move_captured_piece(move)
        if en_passant:
            captured_piece = PAWN
        elif not captured_piece:
            captured_piece = self.piece_type_at(to_square) or 0

        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights]
        if self.ep_hash_file >= 0:
            self.zobrist_hash ^= ZOBRIST_EP_FILE[self.ep_hash_file]

        rights = self.castling_rights
        if moving_piece == KING:
            rights &= ~(
                WHITE_KINGSIDE | WHITE_QUEENSIDE
                if color == WHITE
                else BLACK_KINGSIDE | BLACK_QUEENSIDE
            )
        elif moving_piece == ROOK:
            if from_square == H1:
                rights &= ~WHITE_KINGSIDE
            elif from_square == A1:
                rights &= ~WHITE_QUEENSIDE
            elif from_square == H8:
                rights &= ~BLACK_KINGSIDE
            elif from_square == A8:
                rights &= ~BLACK_QUEENSIDE

        if captured_piece == ROOK and not en_passant:
            if to_square == H1:
                rights &= ~WHITE_KINGSIDE
            elif to_square == A1:
                rights &= ~WHITE_QUEENSIDE
            elif to_square == H8:
                rights &= ~BLACK_KINGSIDE
            elif to_square == A8:
                rights &= ~BLACK_QUEENSIDE
        self.castling_rights = rights

        self.halfmove_clock += 1
        if moving_piece == PAWN or captured_piece:
            self.halfmove_clock = 0

        self.ep_square = None
        self._remove_known_piece(from_square, moving_piece, color)

        if en_passant:
            capture_square = to_square + (-8 if color == WHITE else 8)
            self._remove_known_piece(capture_square, PAWN, opponent)
        elif captured_piece:
            self._remove_known_piece(to_square, captured_piece, opponent)

        if moving_piece == PAWN:
            difference = to_square - from_square
            if move & DOUBLE_PAWN_FLAG or abs(difference) == 16:
                self.ep_square = from_square + (8 if color == WHITE else -8)

        if castling:
            if to_square == G1:
                rook_from, rook_to = H1, F1
            elif to_square == C1:
                rook_from, rook_to = A1, D1
            elif to_square == G8:
                rook_from, rook_to = H8, F8
            else:
                rook_from, rook_to = A8, D8
            self._remove_known_piece(rook_from, ROOK, color)
            self._set_known_piece(rook_to, ROOK, color)

        placed_piece = promotion_code + 1 if promotion_code else moving_piece
        self._set_known_piece(to_square, placed_piece, color)

        self.turn = opponent
        self.ply_count += 1
        self.zobrist_hash ^= ZOBRIST_TURN
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights]
        self.ep_hash_file = self._legal_ep_file()
        if self.ep_hash_file >= 0:
            self.zobrist_hash ^= ZOBRIST_EP_FILE[self.ep_hash_file]

    def has_insufficient_material(self, color):
        """Report whether one side lacks sufficient mating material."""
        if self.occupied_co[color] & (self.pawns | self.rooks | self.queens):
            return False

        if self.occupied_co[color] & self.knights:
            return popcount(self.occupied_co[color]) <= 2 and not (
                self.occupied_co[not color] & ~self.kings & ~self.queens
            )

        if self.occupied_co[color] & self.bishops:
            same_color = (
                not self.bishops & BB_DARK_SQUARES
                or not self.bishops & BB_LIGHT_SQUARES
            )
            return same_color and not self.pawns and not self.knights

        return True

    def is_insufficient_material(self):
        """Report whether neither side can force checkmate by material."""
        return all(self.has_insufficient_material(color) for color in COLORS)

    def set_fen(self, fen):
        """Replace the board state from a six-field FEN string."""
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
                self._set_piece_raw(square, piece_type, symbol.isupper())
                file_index += 1

            if file_index != 8:
                raise ValueError(f"invalid rank width in FEN: {fen!r}")

        if turn == "w":
            self.turn = WHITE
        elif turn == "b":
            self.turn = BLACK
        else:
            raise ValueError(f"invalid side to move in FEN: {fen!r}")

        rights = 0
        if castling != "-":
            for symbol in castling:
                if symbol == "K":
                    rights |= WHITE_KINGSIDE
                elif symbol == "Q":
                    rights |= WHITE_QUEENSIDE
                elif symbol == "k":
                    rights |= BLACK_KINGSIDE
                elif symbol == "q":
                    rights |= BLACK_QUEENSIDE
                else:
                    raise ValueError(f"invalid castling field in FEN: {fen!r}")

        if not self.kings & self.occupied_co[WHITE] & BB_E1:
            rights &= ~(WHITE_KINGSIDE | WHITE_QUEENSIDE)
        if not self.rooks & self.occupied_co[WHITE] & BB_H1:
            rights &= ~WHITE_KINGSIDE
        if not self.rooks & self.occupied_co[WHITE] & BB_A1:
            rights &= ~WHITE_QUEENSIDE
        if not self.kings & self.occupied_co[BLACK] & BB_E8:
            rights &= ~(BLACK_KINGSIDE | BLACK_QUEENSIDE)
        if not self.rooks & self.occupied_co[BLACK] & BB_H8:
            rights &= ~BLACK_KINGSIDE
        if not self.rooks & self.occupied_co[BLACK] & BB_A8:
            rights &= ~BLACK_QUEENSIDE
        self.castling_rights = rights

        self.ep_square = None if ep_square == "-" else parse_square(ep_square)
        self.halfmove_clock = int(halfmove)
        fullmove_number = int(fullmove)
        if self.halfmove_clock < 0 or fullmove_number < 1:
            raise ValueError(f"invalid move counters in FEN: {fen!r}")
        self.ply_count = 2 * (fullmove_number - 1) + (self.turn == BLACK)
        self.ep_hash_file = self._legal_ep_file()
        self.zobrist_hash = self._compute_zobrist()
