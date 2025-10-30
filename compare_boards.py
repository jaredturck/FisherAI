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

piece_lookup = {0: ' ',1: '.',2: 'p',3: 'n',4: 'b',5: 'r',6: 'q',7: 'k',8: 'P',9: 'N',10: 'B',11: 'R',12: 'Q',13: 'K'}

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
    FEN = 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2'
    board = chess.Board(FEN)
    
    arr = np.ones(64)
    for sq, sequence in board.piece_map().items():
        arr[sq] = lookup[(sequence.piece_type, int(sequence.color))]
    arr = arr[None, :]

    tensor = torch.as_tensor(arr.copy())

    return tensor

def display_board(array):
        array = array.reshape(8,8)
        for i in range(8):
            print('|'.join(piece_lookup[int(array[i, j])] for j in range(8)))

print(fen_to_tensor())
display_board(fen_to_tensor())
