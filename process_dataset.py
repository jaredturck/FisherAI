import io
import zstandard as zstd
import chess.pgn
import numpy as np

path = "lichess_db_standard_rated_2025-09.pgn.zst"

lookup = ['.', 'P', 'N', 'B', 'R', 'Q', 'K']

def display_board(array):
    for i in range(8):
        for j in range(8):
            p = lookup[int(array[i,j,0])]
            print(p.upper() if array[i,j,1] == 2 else p.lower(), end=' ')
        print('')

dctx = zstd.ZstdDecompressor()
with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
    text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
    game_count = 0
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
            # array = np.zeros((8,8,2))
            for sq, piece in board.piece_map().items():
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                game_array[num, r, f, 0] = piece.piece_type
                game_array[num, r, f, 1] = int(piece.color) + 1

            # process board state after each move


        input('STOP')

