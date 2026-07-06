import numpy as np

from fisher_ai import chess


def legal_moves(board):
    buffer = np.empty(256, dtype=np.uint32)
    count, status = board.fill_legal_moves(buffer)
    return buffer[:count].copy(), status


def perft(board, depth):
    if depth == 0:
        return 1

    moves, _ = legal_moves(board)
    nodes = 0
    for move in moves:
        child = board.copy()
        child.push(int(move))
        nodes += perft(child, depth - 1)
    return nodes


def test_starting_position_perft_depth_four():
    assert perft(chess.Board(), 4) == 197281


def test_complex_castling_position_perft_depth_two():
    board = chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    assert perft(board, 2) == 1779


def test_packed_move_round_trip():
    move = chess.move_from_uci("a7a8n")

    assert chess.move_to_uci(move) == "a7a8n"
    assert chess.move_from_square(move) == 48
    assert chess.move_to_square(move) == 56
    assert chess.move_promotion(move) == chess.KNIGHT


def test_generated_moves_include_piece_annotations():
    board = chess.Board()
    moves, _ = legal_moves(board)
    e2e4 = next(move for move in moves if chess.move_to_uci(move) == "e2e4")

    assert chess.move_piece(e2e4) == chess.PAWN
    assert chess.move_captured_piece(e2e4) == 0


def test_castling_moves_and_rook_relocation():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves, _ = legal_moves(board)
    move_names = {chess.move_to_uci(move) for move in moves}
    assert "e1g1" in move_names
    assert "e1c1" in move_names

    board.push(
        next(move for move in moves if chess.move_to_uci(move) == "e1g1")
    )
    assert board.piece_type_at(6) == chess.KING
    assert board.piece_type_at(5) == chess.ROOK
    assert board.castling_rights & 3 == 0


def test_en_passant_capture_removes_the_pawn():
    board = chess.Board("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    moves, _ = legal_moves(board)
    move = next(move for move in moves if chess.move_to_uci(move) == "e5d6")

    board.push(move)
    assert board.piece_type_at(43) == chess.PAWN
    assert board.piece_type_at(35) is None


def test_promotion_replaces_the_pawn():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    moves, _ = legal_moves(board)
    move = next(move for move in moves if chess.move_to_uci(move) == "a7a8n")
    board.push(move)
    assert board.piece_type_at(56) == chess.KNIGHT


def test_terminal_statuses():
    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    insufficient = chess.Board("8/8/8/8/8/8/4K3/7k w - - 0 1")

    _, checkmate_status = legal_moves(checkmate)
    _, stalemate_status = legal_moves(stalemate)
    assert checkmate_status == chess.CHECKMATE
    assert stalemate_status == chess.STALEMATE
    assert insufficient.is_insufficient_material()


def test_hash_ignores_non_capturable_en_passant_square():
    with_ep = chess.Board(
        "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2"
    )
    without_ep = chess.Board(
        "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2"
    )
    assert with_ep.position_hash() == without_ep.position_hash()


def test_incremental_hash_matches_full_recalculation():
    board = chess.Board()
    buffer = np.empty(256, dtype=np.uint32)

    for _ in range(80):
        count, status = board.fill_legal_moves(buffer)
        assert board.position_hash() == board._compute_zobrist()
        if status != chess.ONGOING:
            break
        board.push(int(buffer[count // 2]))
