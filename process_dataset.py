import io, time, zstandard, chess.pgn, torch, os
import numpy as np
import multiprocessing as mp
import chess.engine

OUTPUT_DIR = 'datasets'

def process_chunk_worker(chunk_text, lookup, worker_id, no_pos_per_shared, shard_prefix="training_data"):
    ''' Worker function to process a chunk of PGN text. '''
    text_io = io.StringIO(chunk_text)
    training_data = []
    save_file = 1
    processed_games = 0
    start = time.time()

    stockfish_path = "/usr/bin/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({'Threads': 1, 'Hash': 16})
    eval_limit = chess.engine.Limit(time=0.01)

    GOOD_MARGIN_CP = 50
    no_positions_processed = 0
    no_pos_counter = 0

    while True:
        game = chess.pgn.read_game(text_io)
        if game is None:
            break

        result_str = game.headers.get("Result", "*")
        if result_str not in ("1-0", "0-1", "1/2-1/2"):
            continue

        board = game.board()
        moves = list(game.mainline_moves())
        if not moves:
            continue

        n = len(moves)
        game_array       = np.ones((n, 64), dtype=np.int64)
        turns_array      = np.empty(n, dtype=np.int8)
        values_array     = np.empty(n, dtype=np.float32)
        features_array   = np.empty((n, 6), dtype=np.float32)
        move_targets_arr = np.zeros((n, 64 * 64), dtype=np.float32)

        for num, human_move in enumerate(moves):
            game_array[num].fill(1)
            for sq, piece in board.piece_map().items():
                game_array[num, sq] = lookup[(piece.piece_type, int(piece.color))]

            turns_array[num] = 1 if board.turn == chess.WHITE else 0
            features_array[num, 0] = 1.0 if board.turn == chess.WHITE else 0.0
            features_array[num, 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
            features_array[num, 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
            features_array[num, 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
            features_array[num, 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
            features_array[num, 5] = 1.0 if board.ep_square is not None else 0.0

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                values_array[num] = 0.0
                continue

            legal_indices = []
            legal_cps     = []

            mover_color = board.turn

            for mv in legal_moves:
                board.push(mv)
                info  = engine.analyse(board, eval_limit)
                score = info["score"].pov(mover_color)
                board.pop()

                if score.is_mate():
                    mate_score = score.mate()
                    cp = 1000 * (1 if mate_score > 0 else -1)
                else:
                    cp = score.cp

                cp = max(-1000, min(1000, cp))
                idx = mv.from_square * 64 + mv.to_square

                legal_indices.append(idx)
                legal_cps.append(cp)

            best_cp = max(legal_cps)
            values_array[num] = best_cp / 1000.0

            threshold = best_cp - GOOD_MARGIN_CP
            targets = np.zeros(64 * 64, dtype=np.float32)

            for idx, cp in zip(legal_indices, legal_cps):
                if cp >= threshold:
                    targets[idx] = 1.0

            move_targets_arr[num] = targets
            board.push(human_move)
            no_positions_processed += 1
            no_pos_counter += 1

        training_data.append(
            (
                torch.from_numpy(game_array),
                torch.from_numpy(turns_array),
                torch.from_numpy(move_targets_arr),
                torch.from_numpy(values_array),
                torch.from_numpy(features_array),
            )
        )
        processed_games += 1

        if no_pos_counter >= no_pos_per_shared:
            fname = f"{shard_prefix}_w{worker_id}_{save_file:03d}_{len(training_data)}.pt"
            torch.save(training_data, os.path.join(OUTPUT_DIR, fname))
            training_data = []
            no_pos_counter = 0
            save_file += 1

        if time.time() - start > 10:
            start = time.time()
            print(f"[worker {worker_id}] Processed {no_positions_processed} positions in {processed_games:,} games in this chunk")

    if training_data:
        fname = f"{shard_prefix}_w{worker_id}_{save_file:03d}_{len(training_data)}.pt"
        torch.save(training_data, os.path.join(OUTPUT_DIR, fname))

    engine.quit()

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

    def process_db(self, chunk_size_bytes, num_workers, no_pos_per_shared):
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
                        args=(sub_text, self.lookup, worker_id, no_pos_per_shared, f"training_data_chunk{chunk_index}")
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
    db.process_db(chunk_size_bytes=1_000_000, num_workers=os.cpu_count(), no_pos_per_shared=1000)
