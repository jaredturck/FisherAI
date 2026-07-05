from fisher_ai import chess


def perft(board, depth):
    if depth == 0:
        return 1

    nodes = 0
    for move in board.legal_moves:
        child = board.copy()
        child.push(move)
        nodes += perft(child, depth - 1)
    return nodes


def test_starting_position_perft_depth_three():
    assert perft(chess.Board(), 3) == 8902


def test_complex_castling_position_perft_depth_two():
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert perft(board, 2) == 1779


def test_castling_moves_and_rook_relocation():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves = {move.uci() for move in board.legal_moves}
    assert "e1g1" in moves
    assert "e1c1" in moves

    board.push(chess.Move.from_uci("e1g1"))
    assert board.piece_type_at(6) == chess.KING
    assert board.piece_type_at(5) == chess.ROOK
    assert not board.has_kingside_castling_rights(chess.WHITE)
    assert not board.has_queenside_castling_rights(chess.WHITE)


def test_en_passant_capture_removes_the_pawn():
    board = chess.Board("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    move = chess.Move.from_uci("e5d6")
    assert move.uci() in {legal_move.uci() for legal_move in board.legal_moves}

    board.push(move)
    assert board.piece_type_at(43) == chess.PAWN
    assert board.piece_type_at(35) is None


def test_promotion_replaces_the_pawn():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    board.push(chess.Move.from_uci("a7a8n"))
    assert board.piece_type_at(56) == chess.KNIGHT


def test_terminal_positions():
    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    insufficient = chess.Board("8/8/8/8/8/8/4K3/7k w - - 0 1")

    assert checkmate.is_checkmate()
    assert stalemate.is_stalemate()
    assert insufficient.is_insufficient_material()


def test_position_key_ignores_non_capturable_en_passant_square():
    with_ep = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2")
    without_ep = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2")
    assert with_ep.position_key() == without_ep.position_key()


def test_uci_position_commands_use_the_local_board():
    from fisher_ai.config import SearchConfig
    from fisher_ai.uci import UCIEngine

    engine = UCIEngine(object(), SearchConfig())
    engine.set_position("position startpos moves e2e4 e7e5 g1f3")
    expected = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        expected.push(chess.Move.from_uci(uci))
    assert engine.state.board.position_key() == expected.position_key()

    engine.set_position("position fen 8/8/8/8/8/8/4K3/7k w - - 0 1")
    assert engine.state.board.is_insufficient_material()
