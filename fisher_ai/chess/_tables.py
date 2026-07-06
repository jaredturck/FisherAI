"""Private bitboard and hashing tables for FisherAI's chess engine."""

FILE_NAMES = "abcdefgh"
RANK_NAMES = "12345678"
SQUARE_NAMES = tuple(
    file_name + rank_name
    for rank_name in RANK_NAMES
    for file_name in FILE_NAMES
)


def parse_square(name):
    """Convert an algebraic square name into its board index."""
    if (
        len(name) != 2
        or name[0] not in FILE_NAMES
        or name[1] not in RANK_NAMES
    ):
        raise ValueError(f"invalid square: {name!r}")
    return FILE_NAMES.index(name[0]) + RANK_NAMES.index(name[1]) * 8


def square_name(square):
    """Convert a board index into its algebraic square name."""
    return SQUARE_NAMES[square]


def square_file(square):
    """Return the file index of a board square."""
    return square & 7


def square_rank(square):
    """Return the rank index of a board square."""
    return square >> 3


def square_distance(a, b):
    """Return the Chebyshev distance between two squares."""
    return max(
        abs(square_file(a) - square_file(b)),
        abs(square_rank(a) - square_rank(b)),
    )


A1 = 0
B1 = 1
C1 = 2
D1 = 3
E1 = 4
F1 = 5
G1 = 6
H1 = 7
A8 = 56
B8 = 57
C8 = 58
D8 = 59
E8 = 60
F8 = 61
G8 = 62
H8 = 63

BB_EMPTY = 0
BB_ALL = 0xFFFF_FFFF_FFFF_FFFF
BB_SQUARES = tuple(1 << square for square in range(64))

BB_A1 = BB_SQUARES[A1]
BB_E1 = BB_SQUARES[E1]
BB_H1 = BB_SQUARES[H1]
BB_A8 = BB_SQUARES[A8]
BB_E8 = BB_SQUARES[E8]
BB_H8 = BB_SQUARES[H8]
BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8
BB_LIGHT_SQUARES = 0x55AA_55AA_55AA_55AA
BB_DARK_SQUARES = 0xAA55_AA55_AA55_AA55

BB_FILE_A = 0x0101_0101_0101_0101
BB_FILES = tuple(BB_FILE_A << file_index for file_index in range(8))

BB_RANK_1 = 0xFF
BB_RANK_2 = 0xFF << 8
BB_RANK_3 = 0xFF << 16
BB_RANK_4 = 0xFF << 24
BB_RANK_5 = 0xFF << 32
BB_RANK_6 = 0xFF << 40
BB_RANK_7 = 0xFF << 48
BB_RANK_8 = 0xFF << 56
BB_RANKS = (
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
)


def scan_reversed(bitboard):
    """Yield occupied squares from highest to lowest index."""
    while bitboard:
        square = bitboard.bit_length() - 1
        yield square
        bitboard ^= BB_SQUARES[square]


def msb(bitboard):
    """Return the highest occupied square in a bitboard."""
    return bitboard.bit_length() - 1


popcount = int.bit_count


def _sliding_attacks(square, occupied, deltas):
    attacks = BB_EMPTY

    for delta in deltas:
        target = square
        while True:
            target += delta
            if (
                not 0 <= target < 64
                or square_distance(target, target - delta) > 2
            ):
                break

            attacks |= BB_SQUARES[target]
            if occupied & BB_SQUARES[target]:
                break

    return attacks


def _step_attacks(square, deltas):
    return _sliding_attacks(square, BB_ALL, deltas)


BB_KNIGHT_ATTACKS = tuple(
    _step_attacks(square, (17, 15, 10, 6, -17, -15, -10, -6))
    for square in range(64)
)
BB_KING_ATTACKS = tuple(
    _step_attacks(square, (9, 8, 7, 1, -9, -8, -7, -1)) for square in range(64)
)
BB_PAWN_ATTACKS = tuple(
    tuple(_step_attacks(square, deltas) for square in range(64))
    for deltas in ((-7, -9), (7, 9))
)


def _edges(square):
    return ((BB_RANK_1 | BB_RANK_8) & ~BB_RANKS[square_rank(square)]) | (
        (BB_FILES[0] | BB_FILES[7]) & ~BB_FILES[square_file(square)]
    )


def _carry_rippler(mask):
    subset = BB_EMPTY
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break


def _attack_table(deltas):
    masks = []
    tables = []

    for square in range(64):
        mask = _sliding_attacks(square, 0, deltas) & ~_edges(square)
        attacks = {
            subset: _sliding_attacks(square, subset, deltas)
            for subset in _carry_rippler(mask)
        }
        masks.append(mask)
        tables.append(attacks)

    return tuple(masks), tuple(tables)


BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table((-9, -7, 7, 9))
BB_FILE_MASKS, BB_FILE_ATTACKS = _attack_table((-8, 8))
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table((-1, 1))


def _build_rays():
    rays = []
    for a, bb_a in enumerate(BB_SQUARES):
        row = []
        for b, bb_b in enumerate(BB_SQUARES):
            if BB_DIAG_ATTACKS[a][0] & bb_b:
                row.append(
                    (BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0])
                    | bb_a
                    | bb_b
                )
            elif BB_RANK_ATTACKS[a][0] & bb_b:
                row.append(BB_RANK_ATTACKS[a][0] | bb_a)
            elif BB_FILE_ATTACKS[a][0] & bb_b:
                row.append(BB_FILE_ATTACKS[a][0] | bb_a)
            else:
                row.append(BB_EMPTY)
        rays.append(tuple(row))
    return tuple(rays)


BB_RAYS = _build_rays()


def ray(a, b):
    """Return the complete aligned ray through two squares."""
    return BB_RAYS[a][b]


def between(a, b):
    """Return the squares strictly between two aligned squares."""
    bitboard = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b))
    return bitboard & (bitboard - 1)


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & BB_ALL
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & BB_ALL
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & BB_ALL
    return value ^ (value >> 31)


def _zobrist_values(count, seed=0xF15E_A11):
    values = []
    value = seed
    for _ in range(count):
        value = _splitmix64(value)
        values.append(value)
    return values


_ZOBRIST = iter(_zobrist_values(2 * 6 * 64 + 1 + 16 + 8))
ZOBRIST_PIECES = tuple(
    tuple(tuple(next(_ZOBRIST) for _ in range(64)) for _ in range(6))
    for _ in range(2)
)
ZOBRIST_TURN = next(_ZOBRIST)
ZOBRIST_CASTLING = tuple(next(_ZOBRIST) for _ in range(16))
ZOBRIST_EP_FILE = tuple(next(_ZOBRIST) for _ in range(8))
