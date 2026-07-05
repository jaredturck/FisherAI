import numpy as np

from fisher_ai import chess

INPUT_PLANES = 119
ACTION_PLANES = 73
ACTION_SIZE = 64 * ACTION_PLANES

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

QUEEN_DIRECTIONS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]

KNIGHT_DIRECTIONS = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]

UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


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
    max_game_plies,
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
            output[plane_offset : plane_offset + 6] = piece_planes[6:12, ::-1, ::-1]
            output[plane_offset + 6 : plane_offset + 12] = piece_planes[0:6, ::-1, ::-1]

        if snapshot.repetition_count >= 2:
            output[plane_offset + 12].fill(1.0)
        if snapshot.repetition_count >= 3:
            output[plane_offset + 13].fill(1.0)

    output[112].fill(1.0 if current_color == chess.WHITE else 0.0)
    output[113].fill(min(ply, max_game_plies) / max_game_plies)

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


def encode_state(state):
    return encode_history(
        state.history,
        state.board.turn,
        state.board.ply(),
        state.max_game_plies,
        castling_rights_mask(state.board),
        state.board.halfmove_clock,
    )


def move_to_action(move, current_color):
    from_square = canonical_square(move.from_square, current_color)
    to_square = canonical_square(move.to_square, current_color)
    from_row, from_col = square_row_col(from_square)
    to_row, to_col = square_row_col(to_square)
    delta = (to_row - from_row, to_col - from_col)

    if move.promotion in UNDERPROMOTION_PIECES:
        direction_index = {-1: 0, 0: 1, 1: 2}[delta[1]]
        piece_index = UNDERPROMOTION_PIECES.index(move.promotion)
        move_plane = 64 + direction_index * 3 + piece_index
        return move_plane * 64 + from_square

    if delta in KNIGHT_DIRECTIONS:
        move_plane = 56 + KNIGHT_DIRECTIONS.index(delta)
        return move_plane * 64 + from_square

    distance = max(abs(delta[0]), abs(delta[1]))
    direction = (
        0 if delta[0] == 0 else delta[0] // abs(delta[0]),
        0 if delta[1] == 0 else delta[1] // abs(delta[1]),
    )
    assert direction in QUEEN_DIRECTIONS
    assert 1 <= distance <= 7
    move_plane = QUEEN_DIRECTIONS.index(direction) * 7 + distance - 1
    return move_plane * 64 + from_square


def legal_action_map(state):
    mapping = {}
    for move in state.board.legal_moves:
        action = move_to_action(move, state.board.turn)
        assert action not in mapping
        mapping[action] = move
    return mapping


def legal_actions(state):
    return np.asarray(sorted(legal_action_map(state)), dtype=np.int64)


def policy_from_visits(root):
    actions = root.child_actions.astype(np.int64, copy=True)
    visits = root.child_visits.astype(np.float32, copy=True)
    total = visits.sum()
    if total == 0:
        visits.fill(1.0 / len(visits))
    else:
        visits /= total
    return actions, visits
