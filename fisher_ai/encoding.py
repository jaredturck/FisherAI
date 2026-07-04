import chess
import numpy as np

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


def encode_state(state):
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    current_color = state.board.turn
    snapshots = list(state.history)
    start_plane = (8 - len(snapshots)) * 14

    for history_index, snapshot in enumerate(snapshots):
        plane_offset = start_plane + history_index * 14
        board = snapshot.board

        for square, piece in board.piece_map().items():
            square = canonical_square(square, current_color)
            row, col = square_row_col(square)
            piece_offset = PIECE_TYPES.index(piece.piece_type)
            player_offset = 0 if piece.color == current_color else 6
            planes[plane_offset + player_offset + piece_offset, row, col] = 1.0

        if snapshot.repetition_count >= 2:
            planes[plane_offset + 12].fill(1.0)
        if snapshot.repetition_count >= 3:
            planes[plane_offset + 13].fill(1.0)

    planes[112].fill(1.0 if current_color == chess.WHITE else 0.0)
    planes[113].fill(min(state.board.ply(), state.max_game_plies) / state.max_game_plies)

    own_color = current_color
    opponent_color = not current_color
    planes[114].fill(float(state.board.has_kingside_castling_rights(own_color)))
    planes[115].fill(float(state.board.has_queenside_castling_rights(own_color)))
    planes[116].fill(float(state.board.has_kingside_castling_rights(opponent_color)))
    planes[117].fill(float(state.board.has_queenside_castling_rights(opponent_color)))
    planes[118].fill(min(state.board.halfmove_clock, 100) / 100.0)

    return planes


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
    actions = np.asarray(sorted(root.children), dtype=np.int64)
    visits = np.asarray([root.children[action].visit_count for action in actions], dtype=np.float32)
    total = visits.sum()
    if total == 0:
        visits.fill(1.0 / len(visits))
    else:
        visits /= total
    return actions, visits
