import io, time, zstandard, chess.pgn, torch, os
import numpy as np
import multiprocessing as mp

OUTPUT_DIR = 'datasets'

def process_chunk_worker(chunk_text, lookup, worker_id, shard_size, shard_prefix="training_data"):
    ''' Worker function to process a chunk of PGN text. '''
    text_io = io.StringIO(chunk_text)
    training_data = []
    save_file = 1
    processed_games = 0
    start = time.time()

    while True:
        game = chess.pgn.read_game(text_io)
        if game is None:
            break

        result_str = game.headers.get("Result", "*")
        if result_str == "1-0":
            result = 1.0
        elif result_str == "0-1":
            result = -1.0
        elif result_str == "1/2-1/2":
            result = 0.0
        else:
            continue

        board = game.board()
        moves = list(game.mainline_moves())
        if not moves:
            continue
        
        n = len(moves)
        game_array = np.ones((n, 64), dtype=np.int64)
        turns_array = np.empty(n, dtype=np.int8)
        moves_idx_array = np.empty(n, dtype=np.int64)

        for num, move in enumerate(moves):
            board.push(move)
            turns_array[num] = 1 if board.turn == chess.WHITE else 0
            moves_idx_array[num] = move.from_square * 64 + move.to_square
            for sq, piece in board.piece_map().items():
                game_array[num, sq] = lookup[(piece.piece_type, int(piece.color))]

        training_data.append((torch.from_numpy(game_array), torch.from_numpy(turns_array), torch.from_numpy(moves_idx_array), result))
        processed_games += 1

        if len(training_data) >= shard_size:
            fname = f"{shard_prefix}_w{worker_id}_{save_file:03d}_{len(training_data)}.pt"
            torch.save(training_data, os.path.join(OUTPUT_DIR, fname))
            training_data = []
            save_file += 1

        if time.time() - start > 10:
            start = time.time()
            print(f"[worker {worker_id}] Processed {processed_games:,} games in this chunk")

    if training_data:
        fname = f"{shard_prefix}_w{worker_id}_{save_file:03d}_{len(training_data)}.pt"
        torch.save(training_data, os.path.join(OUTPUT_DIR, fname))

class PrepareDataset:
    def __init__(self):
        self.path = "lichess_db_standard_rated_2025-09.pgn.zst"
        self.piece_symbols = ['.', 'P', 'N', 'B', 'R', 'Q', 'K']
        self.training_data = []
        self.dctx = zstandard.ZstdDecompressor()
        self.save_file = 1
        self.lookup = {
            (1, 0): 2, (2, 0): 3, (3, 0): 4, (4, 0): 5, (5, 0): 6, (6, 0): 7,
            (1, 1): 8, (2, 1): 9, (3, 1): 10, (4, 1): 11, (5, 1): 12, (6, 1): 13
        }

    def display_board(self, array):
        ''' Display a board from the encoded array format. '''
        for i in range(8):
            for j in range(8):
                p = self.piece_symbols[int(array[i, j, 0])]
                print(p.upper() if array[i, j, 1] == 2 else p.lower(), end=" ")
            print("")

    def process_db(self, chunk_size_bytes, num_workers, shard_size):
        ''' Process the compressed PGN database in chunks using multiprocessing. '''
        start_global = time.time()
        chunk_index = 0

        with open(self.path, "rb") as fh, self.dctx.stream_reader(fh) as reader:
            while True:
                decompressed = b""
                for i in range(10):
                    decompressed += reader.read(chunk_size_bytes)

                if not decompressed:
                    break

                chunk_index += 1
                print(f"[+] Read decompressed chunk {chunk_index} of size {len(decompressed) / (1024**2):.2f} MB")

                text_chunk = decompressed.decode("utf-8", errors="ignore")
                total_len = len(text_chunk)
                if total_len == 0:
                    continue

                part_len = total_len // num_workers
                processes = []

                for worker_id in range(num_workers):
                    start_idx = worker_id * part_len
                    end_idx = total_len if worker_id == num_workers - 1 else (worker_id + 1) * part_len

                    sub_text = text_chunk[start_idx:end_idx]
                    if not sub_text:
                        continue

                    p = mp.Process(
                        target=process_chunk_worker,
                        args=(sub_text, self.lookup, worker_id, shard_size, f"training_data_chunk{chunk_index}")
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                elapsed = time.time() - start_global
                print(f"[+] Finished chunk {chunk_index}, elapsed {elapsed/60:.1f} minutes")

        print(f"[+] All chunks processed in { (time.time() - start_global) / 60:.1f} minutes")

if __name__ == "__main__":
    db = PrepareDataset()
    db.process_db(chunk_size_bytes=10_000_000_000, num_workers=os.cpu_count(), shard_size=1_000_000_000)
