import chess
import numpy as np
import torch

lookup = {
    (1,0) : 2,
    (2,0) : 3,
    (3,0) : 4,
    (4,0) : 5,
    (5,0) : 6,
    (6,0) : 7,
    (1,1) : 8,
    (2,1) : 9,
    (3,1) : 10,
    (4,1) : 11,
    (5,1) : 12,
    (6,1) : 13
}

def compare_boards():
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


def fen_to_tensor():
    FEN = 'rnbqkb1r/ppppp1p1/5n1p/5p2/3P4/2P1P3/PP3PPP/RNBQKBNR w KQkq - 0 4'
    board = chess.Board(FEN)
    
    arr = np.ones(64)
    for sq, sequence in board.piece_map().items():
        arr[sq] = lookup[(sequence.piece_type, int(sequence.color))]
    arr = arr[None, :]

    tensor = torch.as_tensor(arr.copy())

    return tensor

print(fen_to_tensor())
