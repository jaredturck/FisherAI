import chess

FEN1 = 'rnbqkb1r/ppppp1p1/5n1p/5p2/3P4/2P1P3/PP3PPP/RNBQKBNR w KQkq - 0 4'
FEN2 = 'rnbqkb1r/ppppp1p1/5n1p/5p2/3P4/2P1PP2/PP4PP/RNBQKBNR b KQkq - 0 4'

before = chess.Board(FEN1)
after = chess.Board(FEN2)

played = []
for move in before.legal_moves:
    cur = before.copy(stack=False)
    cur.push(move)

    if cur.fen() == after.fen():
        played.append(move)

print(played)
