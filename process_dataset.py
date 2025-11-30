import io, time
import zstandard as zstd
import chess.pgn
import numpy as np
import torch

class PrepareDataset:
    def __init__(self):
        self.path = "lichess_db_standard_rated_2025-09.pgn.zst"
        self.piece_symbols = ['.', 'P', 'N', 'B', 'R', 'Q', 'K']
        self.training_data = []
        self.dctx = zstd.ZstdDecompressor()
        self.save_file = 1
        self.lookup = {(1,0) : 2,(2,0) : 3,(3,0) : 4,(4,0) : 5,(5,0) : 6,(6,0) : 7,(1,1) : 8,(2,1) : 9,(3,1) : 10,(4,1) : 11,(5,1) : 12,(6,1) : 13}

    def display_board(self, array):
        ''' Display a board from the array representation '''
        for i in range(8):
            for j in range(8):
                p = self.piece_symbols[int(array[i,j,0])]
                print(p.upper() if array[i,j,1] == 2 else p.lower(), end=' ')
            print('')

    def process_db(self):
        ''' Process the dataset and save training data '''

        start = time.time()
        with open(self.path, "rb") as fh, self.dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
            while True:
                game = chess.pgn.read_game(text)
                if game is None:
                    break

                result_str = game.headers['Result']
                if result_str == '1-0':
                    result = 1.0
                elif result_str == '0-1':
                    result = -1.0
                elif result_str == '1/2-1/2':
                    result = 0.0
                else:
                    continue

                board = game.board()
                moves = list(game.mainline_moves())
                if not moves:
                    continue

                game_array = np.ones((len(moves), 64), dtype=np.int64)

                for num,move in enumerate(moves):
                    board.push(move)
                    for sq, piece in board.piece_map().items():
                        game_array[num, sq] = self.lookup[(piece.piece_type, int(piece.color))]

                self.training_data.append((torch.from_numpy(game_array), result))

                if time.time() - start > 10:
                    start = time.time()
                    print(f'[+] Processed {len(self.training_data):,} games')
                    if len(self.training_data) > 50_000:
                        torch.save(self.training_data, f'training_data_{len(self.training_data):,}.pt')
                        self.training_data = []
                        self.save_file += 1
        
        if self.training_data:
            torch.save(self.training_data, f'training_data_{len(self.training_data):,}.pt')

if __name__ == "__main__":
    db = PrepareDataset()
    db.process_db()
