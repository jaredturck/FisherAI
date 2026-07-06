import numpy as np

from fisher_ai import chess
from fisher_ai.game import HISTORY_LENGTH, MAX_GAME_PLIES

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
PROMOTION_CODE_TO_ACTION_INDEX = np.asarray((0, 1, 2, 3, 0), dtype=np.uint8)


class StateEncodingWorkspace:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.history_bitboards = np.empty(
            (self.capacity, HISTORY_LENGTH, 12),
            dtype=np.uint64,
        )
        self.history_repetitions = np.empty(
            (self.capacity, HISTORY_LENGTH),
            dtype=np.uint8,
        )
        self.history_valid = np.empty(
            (self.capacity, HISTORY_LENGTH),
            dtype=np.bool_,
        )
        self.current_colors = np.empty(self.capacity, dtype=np.bool_)
        self.plies = np.empty(self.capacity, dtype=np.uint16)
        self.castling_masks = np.empty(self.capacity, dtype=np.uint8)
        self.halfmove_clocks = np.empty(self.capacity, dtype=np.uint8)


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


def bitboards_to_planes(bitboards):
    bitboards = np.ascontiguousarray(bitboards, dtype=np.uint64)
    byte_shape = (*bitboards.shape, 8)
    byte_view = bitboards.view(np.uint8).reshape(byte_shape)
    return np.unpackbits(
        byte_view,
        axis=-1,
        bitorder="little",
    ).reshape(*bitboards.shape, 8, 8)


def encode_history_batch(
    history_bitboards,
    history_repetitions,
    history_valid,
    current_colors,
    plies,
    castling_masks,
    halfmove_clocks,
    output=None,
):
    history_bitboards = np.asarray(history_bitboards, dtype=np.uint64)
    batch_size = len(history_bitboards)
    if output is None:
        output = np.zeros(
            (batch_size, INPUT_PLANES, 8, 8),
            dtype=np.float16,
        )
    else:
        output[:batch_size].fill(0)
        output = output[:batch_size]

    history_output = output[:, :112].reshape(
        batch_size,
        HISTORY_LENGTH,
        14,
        8,
        8,
    )
    piece_planes = bitboards_to_planes(history_bitboards)
    piece_planes *= np.asarray(history_valid)[..., None, None, None]

    current_colors = np.asarray(current_colors, dtype=np.bool_)
    white_rows = np.flatnonzero(current_colors)
    black_rows = np.flatnonzero(~current_colors)
    if len(white_rows):
        history_output[white_rows, :, :12] = piece_planes[white_rows]
    if len(black_rows):
        history_output[black_rows, :, :6] = piece_planes[
            black_rows,
            :,
            6:12,
            ::-1,
            ::-1,
        ]
        history_output[black_rows, :, 6:12] = piece_planes[
            black_rows,
            :,
            0:6,
            ::-1,
            ::-1,
        ]

    valid = np.asarray(history_valid, dtype=np.bool_)
    repetitions = np.asarray(history_repetitions)
    history_output[:, :, 12] = ((repetitions >= 2) & valid)[..., None, None]
    history_output[:, :, 13] = ((repetitions >= 3) & valid)[..., None, None]

    output[:, 112] = current_colors[:, None, None]
    output[:, 113] = (
        np.minimum(np.asarray(plies), MAX_GAME_PLIES) / MAX_GAME_PLIES
    )[:, None, None]

    castling_masks = np.asarray(castling_masks, dtype=np.uint8)
    own_kingside = np.where(
        current_colors,
        castling_masks & 1,
        castling_masks >> 2 & 1,
    )
    own_queenside = np.where(
        current_colors,
        castling_masks >> 1 & 1,
        castling_masks >> 3 & 1,
    )
    opponent_kingside = np.where(
        current_colors,
        castling_masks >> 2 & 1,
        castling_masks & 1,
    )
    opponent_queenside = np.where(
        current_colors,
        castling_masks >> 3 & 1,
        castling_masks >> 1 & 1,
    )
    output[:, 114] = own_kingside[:, None, None]
    output[:, 115] = own_queenside[:, None, None]
    output[:, 116] = opponent_kingside[:, None, None]
    output[:, 117] = opponent_queenside[:, None, None]
    output[:, 118] = (np.minimum(np.asarray(halfmove_clocks), 100) / 100.0)[
        :, None, None
    ]
    return output


def encode_states(states, output=None, workspace=None):
    batch_size = len(states)
    if workspace is None:
        workspace = StateEncodingWorkspace(batch_size)
    elif batch_size > workspace.capacity:
        raise ValueError(
            f"encoding batch {batch_size} exceeds workspace "
            f"capacity {workspace.capacity}"
        )

    history_bitboards = workspace.history_bitboards[:batch_size]
    history_repetitions = workspace.history_repetitions[:batch_size]
    history_valid = workspace.history_valid[:batch_size]
    current_colors = workspace.current_colors[:batch_size]
    plies = workspace.plies[:batch_size]
    castling_masks = workspace.castling_masks[:batch_size]
    halfmove_clocks = workspace.halfmove_clocks[:batch_size]
    history_bitboards.fill(0)
    history_repetitions.fill(0)
    history_valid.fill(False)

    for index, state in enumerate(states):
        length = state.history_length
        start = HISTORY_LENGTH - length
        history_bitboards[index, start:] = state.history_bitboards[:length]
        history_repetitions[index, start:] = state.history_repetitions[:length]
        history_valid[index, start:] = True
        current_colors[index] = state.board.turn
        plies[index] = state.board.ply()
        castling_masks[index] = castling_rights_mask(state.board)
        halfmove_clocks[index] = state.board.halfmove_clock

    return encode_history_batch(
        history_bitboards,
        history_repetitions,
        history_valid,
        current_colors,
        plies,
        castling_masks,
        halfmove_clocks,
        output=output,
    )


def encode_state(state, output=None):
    if output is None:
        batch_output = None
    else:
        batch_output = output[None]
    return encode_states([state], output=batch_output)[0]


def encode_window_batch(
    snapshot_bitboards,
    snapshot_repetitions,
    game_starts,
    indices,
    current_colors,
    plies,
    castling_masks,
    halfmove_clocks,
    output=None,
):
    indices = np.asarray(indices, dtype=np.int64)
    starts = np.asarray(game_starts, dtype=np.int64)[indices]
    offsets = np.arange(-7, 1, dtype=np.int64)
    history_indices = indices[:, None] + offsets
    history_valid = history_indices >= starts[:, None]
    history_indices = np.maximum(history_indices, starts[:, None])

    return encode_history_batch(
        np.asarray(snapshot_bitboards)[history_indices],
        np.asarray(snapshot_repetitions)[history_indices],
        history_valid,
        np.asarray(current_colors)[indices],
        np.asarray(plies)[indices],
        np.asarray(castling_masks)[indices],
        np.asarray(halfmove_clocks)[indices],
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

PACKED_MOVE_COUNT = 1 << 15
PACKED_ACTION_LOOKUP = np.full((2, PACKED_MOVE_COUNT), -1, dtype=np.int16)
_packed_moves = np.arange(PACKED_MOVE_COUNT, dtype=np.uint16)
_packed_from = _packed_moves & 63
_packed_to = _packed_moves >> 6 & 63
_packed_promotion = _packed_moves >> 12
_valid_promotion = _packed_promotion < len(PROMOTION_CODE_TO_ACTION_INDEX)
_promotion_indices = np.zeros(PACKED_MOVE_COUNT, dtype=np.uint8)
_promotion_indices[_valid_promotion] = PROMOTION_CODE_TO_ACTION_INDEX[
    _packed_promotion[_valid_promotion]
]
for color in (chess.BLACK, chess.WHITE):
    PACKED_ACTION_LOOKUP[int(color), _valid_promotion] = ACTION_LOOKUP[
        int(color),
        _promotion_indices[_valid_promotion],
        _packed_from[_valid_promotion],
        _packed_to[_valid_promotion],
    ]


def moves_to_actions(moves, current_color, output=None, count=None):
    moves = np.asarray(moves, dtype=np.uint16)
    if count is None:
        count = len(moves)
    actions = PACKED_ACTION_LOOKUP[int(current_color), moves[:count]]
    if np.any(actions < 0):
        raise AssertionError("move could not be mapped to a policy action")
    if output is None:
        return actions.astype(np.uint16, copy=True)
    output[:count] = actions
    return output


def move_to_action(move, current_color):
    moves = np.asarray((move,), dtype=np.uint16)
    return int(moves_to_actions(moves, current_color)[0])
