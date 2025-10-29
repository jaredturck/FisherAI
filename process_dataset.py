import io, time
import zstandard as zstd
import chess.pgn
import numpy as np
import torch

path = "lichess_db_standard_rated_2025-09.pgn.zst"

lookup = ['.', 'P', 'N', 'B', 'R', 'Q', 'K']

def display_board(array):
    for i in range(8):
        for j in range(8):
            p = lookup[int(array[i,j,0])]
            print(p.upper() if array[i,j,1] == 2 else p.lower(), end=' ')
        print('')

training_data = []
dctx = zstd.ZstdDecompressor()
start = time.time()
save_file = 1

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

with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
    text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
    while True:
        game = chess.pgn.read_game(text)
        if game is None:
            break

        board = game.board()
        moves = list(game.mainline_moves())
        game_array = np.ones((len(moves),64))

        for num,move in enumerate(moves):
            board.push(move)

            # Create array of the current board state
            for sq, piece in board.piece_map().items():
                game_array[num, sq] = lookup[(piece.piece_type, int(piece.color))]

        training_data.append(torch.from_numpy(game_array))

        if time.time() - start > 10:
            start = time.time()
            print(f'[+] Processed {len(training_data):,} games')

            if len(training_data) > 50_000:
                torch.save(training_data, f'training_data_{len(training_data):,}.pt')
                training_data = []
                save_file += 1
