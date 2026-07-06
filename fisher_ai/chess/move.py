"""Packed standard-chess move helpers."""

from fisher_ai.chess.bitboards import parse_square, square_name

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

FROM_MASK = 0x3F
TO_MASK = 0x3F
TO_SHIFT = 6
PROMOTION_SHIFT = 12

PROMOTION_FROM_SYMBOL = {
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
}
PROMOTION_TO_SYMBOL = {
    value: key for key, value in PROMOTION_FROM_SYMBOL.items()
}
PROMOTION_CODES = {
    None: 0,
    0: 0,
    KNIGHT: 1,
    BISHOP: 2,
    ROOK: 3,
    QUEEN: 4,
}
CODE_PROMOTIONS = (None, KNIGHT, BISHOP, ROOK, QUEEN)


def encode_move(from_square, to_square, promotion=None):
    return (
        int(from_square)
        | int(to_square) << TO_SHIFT
        | PROMOTION_CODES[promotion] << PROMOTION_SHIFT
    )


def move_from_square(move):
    return int(move) & FROM_MASK


def move_to_square(move):
    return int(move) >> TO_SHIFT & TO_MASK


def move_promotion_code(move):
    return int(move) >> PROMOTION_SHIFT


def move_promotion(move):
    return CODE_PROMOTIONS[move_promotion_code(move)]


def move_to_uci(move):
    value = square_name(move_from_square(move)) + square_name(
        move_to_square(move)
    )
    promotion = move_promotion(move)
    if promotion:
        value += PROMOTION_TO_SYMBOL[promotion]
    return value


def move_from_uci(uci):
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
