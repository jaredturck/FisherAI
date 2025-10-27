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

with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
    text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
    while True:
        game = chess.pgn.read_game(text)
        if game is None:
            break

        board = game.board()
        moves = list(game.mainline_moves())
        game_array = np.zeros((len(moves),8,8,2))

        for num,move in enumerate(moves):
            board.push(move)

            # Create array of the current board state
            for sq, piece in board.piece_map().items():
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                game_array[num, r, f, 0] = piece.piece_type
                game_array[num, r, f, 1] = int(piece.color) + 1

        training_data.append(game_array)

        if time.time() - start > 10:
            start = time.time()
            print(f'[+] Processed {len(training_data):,} games')

            if len(training_data) > 50_000:
                torch.save(training_data, f'training_data_{len(training_data):,}.pt')
                training_data = []
                save_file += 1
