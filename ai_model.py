import sys
from torch.utils.data import Dataset
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch, os, time, datetime, math, chess, chess.engine, platform

lookup = {(1,0) : 2,(2,0) : 3,(3,0) : 4,(4,0) : 5,(5,0) : 6,(6,0) : 7,(1,1) : 8,(2,1) : 9,(3,1) : 10,(4,1) : 11,(5,1) : 12,(6,1) : 13}
piece_lookup = {0: ' ',1: '.',2: 'p',3: 'n',4: 'b',5: 'r',6: 'q',7: 'k',8: 'P',9: 'N',10: 'B',11: 'R',12: 'Q',13: 'K'}
rlookup = {v : k for k,v in lookup.items()}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_LOSS = 2

if platform.uname().node == 'Jared-PC':
    DATASET_PATH = 'datasets/'
    WEIGHTS_PATH = 'weights/'
    BATCH_SIZE = 1024
    MAX_FILES = 8
else:
    DATASET_PATH = 'datasets/'
    WEIGHTS_PATH = 'weights/'
    BATCH_SIZE = 16
    MAX_FILES = 64

class ChessDataset(Dataset):
    def __init__(self):
        self.training_data = []
        self.max_files = MAX_FILES

    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def read_data(self):
        self.training_data = []
        files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.pt')]
        files.sort()
        start_time = time.time()

        print('[+] Reading dataset')
        pos_count = 0
        for file_idx, fname in enumerate(files):
            if file_idx >= self.max_files:
                break

            path = os.path.join(DATASET_PATH, fname)
            data = torch.load(path, map_location="cpu", weights_only=False)

            for boards_tensor, turns_tensor, move_targets_tensor, values_tensor, features_tensor in data:
                boards_tensor       = boards_tensor.long()
                turns_tensor        = turns_tensor.long()
                move_targets_tensor = move_targets_tensor.float()
                values_tensor       = values_tensor.float()
                features_tensor     = features_tensor.float()
                num_positions       = boards_tensor.shape[0]

                for ply_idx in range(num_positions):
                    board64   = boards_tensor[ply_idx]
                    value     = float(values_tensor[ply_idx].item())
                    move_tgt  = move_targets_tensor[ply_idx]
                    feats     = features_tensor[ply_idx]

                    self.training_data.append((board64, value, move_tgt, feats))
                    pos_count += 1
                
                if time.time() - start_time > 10:
                    start_time = time.time()
                    print(f"[+] Loaded {pos_count:,} positions from {file_idx+1} of {min(len(files), self.max_files)} files")

        print(f"[+] Loaded {pos_count:,} positions")
    
    def collate_fn(self, batch):
        boards, values, move_targets, feats = zip(*batch)
        x = torch.stack(boards, dim=0)
        y = torch.tensor(values, dtype=torch.float32)
        t = torch.stack(move_targets, dim=0).float()
        f = torch.stack(feats, dim=0).float()
        return x, y, t, f

class FisherAI(Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.dataset = ChessDataset()
        self.dataloader_workers = max(2, os.cpu_count() // 2)
        self.optimizer = None
        self.max_epochs = 1000
        self.d_model = d_model

        self.piece_embedding = nn.Embedding(14, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(64, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.value_head  = nn.Linear(d_model * 64, 1)
        self.policy_head = nn.Linear(d_model, 64)

        self.feature_dim = 6
        self.feature_proj = nn.Linear(self.feature_dim, d_model)
        self.feature_buffer = np.empty(self.feature_dim, dtype=np.float32)

        self.stock_fish_path = '/usr/bin/stockfish'
        self.board_buffer = np.empty(64, dtype=np.int64)
        self.fen_buffer   = np.empty(64, dtype=np.int64)
        self.encode_buffer = np.empty(64, dtype=np.int64)

        self.register_buffer('pos_indices', torch.arange(64, dtype=torch.long))

    def encode_features(self, board):
        fb = self.feature_buffer
        fb[0] = 1.0 if board.turn == chess.WHITE else 0.0
        fb[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        fb[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        fb[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        fb[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        fb[5] = 1.0 if board.ep_square is not None else 0.0
        return fb

    def forward(self, board, features):
        board_emb = self.piece_embedding(board)

        B, S = board.shape
        pos = self.pos_indices[:S].unsqueeze(0).expand(B, S)
        pos_emb = self.pos_embedding(pos)

        feat_emb = torch.tanh(self.feature_proj(features))
        feat_emb = feat_emb.unsqueeze(1)

        x = board_emb + pos_emb + feat_emb
        x = self.dropout(x)
        x = self.encoder(x)

        x_flat = x.reshape(B, 64 * self.d_model)
        value  = self.value_head(x_flat).squeeze(-1)

        policy_per_square = self.policy_head(x)
        policy = policy_per_square.reshape(B, 64 * 64)

        return value, policy

    def train_model(self):
        self.dataset.read_data()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.dataloader_workers
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

        value_loss_func  = nn.MSELoss()
        policy_loss_func = nn.BCEWithLogitsLoss()

        self.load_weights()
        start = time.time()
        save_time = time.time()
        self.train()

        print(f'[+] Starting training on {DEVICE} for {len(self.dataset):,} positions, batch size {BATCH_SIZE}')
        for epoch in range(self.max_epochs):
            total_value_loss  = 0.0
            total_policy_loss = 0.0
            total_loss        = 0.0

            for batch_idx, (boards, values, move_targets, features) in enumerate(self.dataloader):
                boards       = boards.to(DEVICE, non_blocking=True)
                values       = values.to(DEVICE, non_blocking=True)
                move_targets = move_targets.to(DEVICE, non_blocking=True)
                features     = features.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                pred_value, policy_logits = self.forward(boards, features)

                loss_value  = value_loss_func(pred_value, values)
                loss_policy = policy_loss_func(policy_logits, move_targets)
                loss = loss_value + loss_policy

                loss.backward()
                self.optimizer.step()

                total_loss        += loss.item()
                total_value_loss  += loss_value.item()
                total_policy_loss += loss_policy.item()

                if time.time() - start > 10:
                    start = time.time()
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1} of {len(self.dataloader)}, Loss: {loss.item():.6f}')
                    if time.time() - save_time > 300:
                        save_time = time.time()
                        self.save_weights()
            
            avg_loss        = total_loss / len(self.dataloader)
            avg_value_loss  = total_value_loss / len(self.dataloader)
            avg_policy_loss = total_policy_loss / len(self.dataloader)
            print(f'[+] Epoch {epoch+1}, avg loss {avg_loss:.2f}, value loss {avg_value_loss:.2f}, policy loss {avg_policy_loss:.2f}')

            if avg_loss < TARGET_LOSS:
                print(f'[+] Target loss reached, stopping training, loss {avg_loss:.2f}')
                self.save_weights()
                return

    def save_weights(self):
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pt')]
        if len(files) > 10:
            oldest = min(files, key=os.path.getctime)
            os.remove(oldest)
            print(f'[+] Deleted oldest weight file {oldest}')

        fname = f'weights_{datetime.datetime.now().strftime("%d-%b-%Y_%H-%M")}.pt'
        torch.save({
            'weights': self.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None
        }, os.path.join(WEIGHTS_PATH, fname))
        print(f'[+] Saved weights {fname}')
    
    def load_weights(self):
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pt')]
        if files:
            max_file = max(files, key=os.path.getctime)
            weights_data = torch.load(max_file, map_location=DEVICE)
            if 'weights' in weights_data:
                self.load_state_dict(weights_data['weights'])
                print(f'[+] Loaded weights from {max_file}')

            if self.optimizer and 'optimizer' in weights_data and weights_data['optimizer'] is not None:
                self.optimizer.load_state_dict(weights_data['optimizer'])
                print(f'[+] Loaded optimizer state from {max_file}')

    def fen_to_tensor(self, fen):
        board = chess.Board(fen)
        self.encode_board_inplace(board, self.fen_buffer)
        return torch.from_numpy(self.fen_buffer).long().unsqueeze(0)
    
    def encode_board_inplace(self, board, dest):
        dest.fill(1)
        for sq, p in board.piece_map().items():
            dest[sq] = lookup[(p.piece_type, int(p.color))]
    
    def encode_board(self, board):
        self.encode_board_inplace(board, self.encode_buffer)
        return self.encode_buffer.copy()

    @torch.no_grad()
    def predict(self, array):
        self.eval()
        flat = array.view(-1).detach().cpu().numpy()

        board = chess.Board.empty()
        for sq, v in enumerate(flat):
            if int(v) > 1:
                ptype, color = rlookup[int(v)]
                board.set_piece_at(sq, chess.Piece(ptype, bool(color)))
        
        moves, _ = self.suggest_moves(board, k=1)
        if not moves:
            return array
        
        move = moves[0]
        board.push(move)

        self.encode_board_inplace(board, self.encode_buffer)
        return torch.from_numpy(self.encode_buffer.copy()).long().unsqueeze(0)

    def display_board(self, array):
        array = array.reshape(8,8)
        for i in range(8):
            print('|'.join(piece_lookup[int(array[i, j])] for j in range(8)))
    
    @torch.no_grad()
    def suggest_moves(self, board, k=5):
        self.eval()
        self.encode_board_inplace(board, self.board_buffer)
        board_tensor = torch.from_numpy(self.board_buffer).long().unsqueeze(0).to(DEVICE)

        feats_np = self.encode_features(board)
        feats_tensor = torch.from_numpy(feats_np).unsqueeze(0).to(DEVICE)

        value, policy_logits = self.forward(board_tensor, feats_tensor)
        policy_logits = policy_logits.squeeze(0)
        policy_probs = F.softmax(policy_logits, dim=-1)

        scored_moves = []
        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            score = policy_probs[idx].item()
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        top = [move for score, move in scored_moves[:k]]
        return top, float(value.item())
    
    @torch.no_grad()
    def best_move_from_fen(self, fen, k=5):
        board = chess.Board(fen)
        return self.best_move_from_board(board, k=k)
    
    @torch.no_grad()
    def best_move_from_board(self, board, k=5):
        moves, _ = self.suggest_moves(board, k)
        return moves[0] if moves else None

    def evaluate_position(self, board):
        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                return 1e4 if board.turn == chess.WHITE else -1e4
            elif result == '0-1':
                return 1e4 if board.turn == chess.BLACK else -1e4
            else:
                return 0.0
        
        self.encode_board_inplace(board, self.board_buffer)
        board_tensor = torch.from_numpy(self.board_buffer).long().unsqueeze(0).to(DEVICE)
        feats_np = self.encode_features(board)
        feats_tensor = torch.from_numpy(feats_np).unsqueeze(0).to(DEVICE)

        self.eval()
        value, _ = self.forward(board_tensor, feats_tensor)
        return float(value.item())
    
    def negamax_search(self, board, depth, alpha, beta, start_time, time_limit, k=5):
        if depth == 0 or board.is_game_over() or (time.time() - start_time) > time_limit:
            return self.evaluate_position(board)
        
        moves, _ = self.suggest_moves(board, k=k)

        if not moves:
            moves = list(board.legal_moves)

        best = float('-inf')
        for move in moves:
            if time.time() - start_time > time_limit:
                break

            board.push(move)
            score = -self.negamax_search(board, depth - 1, -beta, -alpha, start_time, time_limit, k)
            board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        
        return best
    
    def best_move_negamax(self, board, depth = 2, time_limit=5, k=5):
        start_time = time.time()
        best_move = None
        best_score = float('-inf')

        alpha = -math.inf
        beta = math.inf

        moves, _ = self.suggest_moves(board, k=k)
        if not moves:
            moves = list(board.legal_moves)
        if not moves:
            return None

        for move in moves:
            if time.time() - start_time > time_limit:
                break

            board.push(move)
            score = -self.negamax_search(board, depth - 1, -beta, -alpha, start_time, time_limit, k)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            
            if score > alpha:
                alpha = score
        
        if best_move is None and moves:
            best_move = moves[0]
        
        return best_move
    @torch.no_grad()
    def engine_vs_stockfish(self):
        '''Evaluate how good the engine is by playing it against Stockfish'''
        
        stock_fish_elo = 1320
        engine = chess.engine.SimpleEngine.popen_uci(self.stock_fish_path)
        engine.configure({'UCI_Elo' : stock_fish_elo, 'UCI_LimitStrength' : True})

        scores = {'stockfish' : 0, 'fisherai' : 0, 'draws' : 0}
        result_map = {'1-0' : 'stockfish', '0-1' : 'fisherai', '1/2-1/2' : 'draws'}
        no_games = 50

        print('[+] Starting evaulation')
        start = time.time()
        for game in range(no_games):
            board = chess.Board()
            while not board.is_game_over():

                # Stockfish move
                result = engine.play(board, chess.engine.Limit(time=0.05))
                board.push(result.move)

                if board.is_game_over():
                    break
                
                # FisherAI move
                move = self.best_move_negamax(board, depth=3, time_limit=5)
                if move is None:
                    break
                board.push(move)
            
            # Add scores
            scores[result_map[board.result()]] += 1
            print(f'Game {game+1} of {no_games}, {result_map.get(board.result())} won, result: {board.result()}')
        
        # Calculate FisherAI ELO
        print(f'\n{scores}')
        score = (scores['fisherai'] + 0.5 * scores['draws']) / no_games
        if score <= 0.0:
            elo = float('-inf')
        elif score >= 1.0:
            elo = float('inf')
        else:
            elo = stock_fish_elo + 400 * math.log10(score / (1 - score))
        
        print(f'[+] Evaluation completed {time.time() - start:.2f} seconds, FisherAI ELO {elo:.2f}')
        engine.quit()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = FisherAI().to(DEVICE)
            model.train_model()
        except KeyboardInterrupt:
            model.save_weights()
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        model = FisherAI().to(DEVICE)
        model.load_weights()
        model.engine_vs_stockfish()
    
    else:
        model = FisherAI().to(DEVICE)
        model.load_weights()
        while True:
            fen = input('Enter FEN: ')
            if not fen.strip():
                break
            move = model.best_move_from_fen(fen)
            print(f'Best move: {move}')
