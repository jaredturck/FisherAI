import numpy as np

from fisher_ai import chess
from fisher_ai.game import MAX_GAME_PLIES

INPUT_PLANES = 119
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
UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def canonical_square(square, current_color):
    return square if current_color == chess.WHITE else 63 - square


def square_row_col(square):
    return chess.square_rank(square), chess.square_file(square)


def castling_rights_mask(board):
    mask = 0
    mask |= int(board.has_kingside_castling_rights(chess.WHITE))
    mask |= int(board.has_queenside_castling_rights(chess.WHITE)) << 1
    mask |= int(board.has_kingside_castling_rights(chess.BLACK)) << 2
    mask |= int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    return mask


def snapshot_planes(snapshot):
    return np.unpackbits(
        snapshot.bitboards.view(np.uint8),
        bitorder="little",
    ).reshape(12, 8, 8)


def encode_history(
    snapshots,
    current_color,
    ply,
    castling_mask,
    halfmove_clock,
    output=None,
):
    if output is None:
        output = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    else:
        output.fill(0)

    snapshots = list(snapshots)[-8:]
    start_plane = (8 - len(snapshots)) * 14

    for history_index, snapshot in enumerate(snapshots):
        plane_offset = start_plane + history_index * 14
        piece_planes = snapshot_planes(snapshot)

        if current_color == chess.WHITE:
            output[plane_offset : plane_offset + 12] = piece_planes
        else:
            output[plane_offset : plane_offset + 6] = piece_planes[
                6:12, ::-1, ::-1
            ]
            output[plane_offset + 6 : plane_offset + 12] = piece_planes[
                0:6, ::-1, ::-1
            ]

        if snapshot.repetition_count >= 2:
            output[plane_offset + 12].fill(1.0)
        if snapshot.repetition_count >= 3:
            output[plane_offset + 13].fill(1.0)

    output[112].fill(1.0 if current_color == chess.WHITE else 0.0)
    output[113].fill(min(ply, MAX_GAME_PLIES) / MAX_GAME_PLIES)

    if current_color == chess.WHITE:
        own_kingside = castling_mask & 1
        own_queenside = castling_mask >> 1 & 1
        opponent_kingside = castling_mask >> 2 & 1
        opponent_queenside = castling_mask >> 3 & 1
    else:
        own_kingside = castling_mask >> 2 & 1
        own_queenside = castling_mask >> 3 & 1
        opponent_kingside = castling_mask & 1
        opponent_queenside = castling_mask >> 1 & 1

    output[114].fill(float(own_kingside))
    output[115].fill(float(own_queenside))
    output[116].fill(float(opponent_kingside))
    output[117].fill(float(opponent_queenside))
    output[118].fill(min(halfmove_clock, 100) / 100.0)
    return output


def encode_state(state, output=None):
    return encode_history(
        state.history,
        state.board.turn,
        state.board.ply(),
        castling_rights_mask(state.board),
        state.board.halfmove_clock,
        output=output,
    )


def calculate_action(from_square, to_square, promotion, current_color):
    canonical_from = canonical_square(from_square, current_color)
    canonical_to = canonical_square(to_square, current_color)
    from_row, from_col = square_row_col(canonical_from)
    to_row, to_col = square_row_col(canonical_to)
    row_delta = to_row - from_row
    col_delta = to_col - from_col

    if promotion in UNDERPROMOTION_PIECES:
        if row_delta != 1 or col_delta not in (-1, 0, 1):
            return -1
        direction_index = col_delta + 1
        piece_index = UNDERPROMOTION_PIECES.index(promotion)
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


PROMOTION_LOOKUP_INDEX = {
    None: 0,
    chess.QUEEN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
}
ACTION_LOOKUP = np.full((2, 4, 64, 64), -1, dtype=np.int16)
for color in (chess.BLACK, chess.WHITE):
    for from_square in range(64):
        for to_square in range(64):
            ACTION_LOOKUP[int(color), 0, from_square, to_square] = (
                calculate_action(from_square, to_square, None, color)
            )
            for promotion_index, promotion in enumerate(
                UNDERPROMOTION_PIECES,
                start=1,
            ):
                ACTION_LOOKUP[
                    int(color),
                    promotion_index,
                    from_square,
                    to_square,
                ] = calculate_action(
                    from_square,
                    to_square,
                    promotion,
                    color,
                )


def move_to_action(move, current_color):
    promotion_index = PROMOTION_LOOKUP_INDEX[move.promotion]
    action = int(
        ACTION_LOOKUP[
            int(current_color),
            promotion_index,
            move.from_square,
            move.to_square,
        ]
    )
    assert action >= 0
    return action
