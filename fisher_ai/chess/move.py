"""Minimal standard UCI move type adapted from python-chess 1.11.2."""

from fisher_ai.chess.bitboards import parse_square, square_name

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

PROMOTION_FROM_SYMBOL = {
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
}
PROMOTION_TO_SYMBOL = {value: key for key, value in PROMOTION_FROM_SYMBOL.items()}


class Move:
    __slots__ = ("from_square", "to_square", "promotion")

    def __init__(self, from_square, to_square, promotion=None):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion

    def __eq__(self, other):
        return (
            isinstance(other, Move)
            and self.from_square == other.from_square
            and self.to_square == other.to_square
            and self.promotion == other.promotion
        )

    def __hash__(self):
        return hash((self.from_square, self.to_square, self.promotion))

    def uci(self):
        value = square_name(self.from_square) + square_name(self.to_square)
        if self.promotion:
            value += PROMOTION_TO_SYMBOL[self.promotion]
        return value

    def __str__(self):
        return self.uci()

    def __repr__(self):
        return f"Move.from_uci({self.uci()!r})"

    @classmethod
    def from_uci(cls, uci):
        if len(uci) not in (4, 5):
            raise ValueError(f"expected UCI move of length 4 or 5: {uci!r}")

        from_square = parse_square(uci[:2])
        to_square = parse_square(uci[2:4])
        if from_square == to_square:
            raise ValueError(f"source and destination squares are identical: {uci!r}")

        promotion = None
        if len(uci) == 5:
            try:
                promotion = PROMOTION_FROM_SYMBOL[uci[4]]
            except KeyError:
                raise ValueError(f"invalid promotion piece: {uci!r}") from None

        return cls(from_square, to_square, promotion)
