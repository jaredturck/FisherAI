import sys
from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch, os, time, datetime, math, chess, chess.engine

lookup = {(1,0) : 2,(2,0) : 3,(3,0) : 4,(4,0) : 5,(5,0) : 6,(6,0) : 7,(1,1) : 8,(2,1) : 9,(3,1) : 10,(4,1) : 11,(5,1) : 12,(6,1) : 13}
piece_lookup = {0: ' ',1: '.',2: 'p',3: 'n',4: 'b',5: 'r',6: 'q',7: 'k',8: 'P',9: 'N',10: 'B',11: 'R',12: 'Q',13: 'K'}
rlookup = {v : k for k,v in lookup.items()}

DATASET_PATH = 'datasets/'
WEIGHTS_PATH = 'weights/'
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ChessDataset(Dataset):
    def __init__(self):
        self.training_data = []
        self.max_files = 4

    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def read_data(self):
        self.training_data = []
        files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.pt')]
        files.sort()

        pos_count = 0
        for file_idx, fname in enumerate(files):
            if file_idx >= self.max_files:
                break

            path = os.path.join(DATASET_PATH, fname)
            data = torch.load(path, map_location="cpu", weights_only=False)

            for boards_tensor, result_white in data:
                boards_tensor = boards_tensor.long()
                num_positions = boards_tensor.shape[0]

                for ply_idx in range(num_positions):
                    board64 = boards_tensor[ply_idx]
                    whites_turn = (ply_idx % 2 == 0)
                    value = float(result_white if whites_turn else -result_white)

                    self.training_data.append((board64, value))
                    pos_count += 1

        print(f"[+] Loaded {pos_count:,} positions")
    
    def collate_fn(self, batch):
        boards, values = zip(*batch)
        x = torch.stack(boards, dim=0)
        y = torch.tensor(values, dtype=torch.float32)
        return x, y

class FisherAI(Module):
    def __init__(self, emb_dim=16, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(14, emb_dim, padding_idx=0)
        in_dim = 64 * emb_dim + 6

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        self.stock_fish_path = '/usr/bin/stockfish'
        self.board_buffer = np.empty(64, dtype=np.int64)
        self.feature_buffer = np.empty(6, dtype=np.float32)

    def forward(self, board, feature_tensor):
        x = self.embedding(board)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, feature_tensor), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)
    
    def train_model(self):
        self.dataset.read_data()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.dataset.collate_fn)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.load_weights()
        start = time.time()
        save_time = time.time()

        print(f'[+] Training ({DEVICE}) started, d_model={self.d_model}, nheads={self.nheads}, dim_feedforward={self.dim_feedforward}, '
              f'layers={self.no_transformer_layers}, batch_size={BATCH_SIZE}')
        for epoch in range(100):
            total_loss = 0.0
            for n, (src, tgt) in enumerate(self.dataloader):
                B,T,S = src.shape
                src = src.to(DEVICE).view(-1, S)
                tgt = tgt.to(DEVICE).view(-1)
                self.optimizer.zero_grad()

                output = self.forward(src)
                loss = loss_func(output.view(-1, 14), tgt)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    start = time.time()
                    print(f'[+] Epoch {epoch+1}, batch {n+1} of {len(self.dataloader)}, loss: {loss.item():.4f}')

                    if time.time() - save_time > 600:
                        save_time = time.time()
                        self.save_weights()
                        print('[+] Weights saved')
    
            print(f'Epoch {epoch+1}, avg loss: {total_loss / len(self.dataloader):.4f}')
        
    def save_weights(self):
        fname = f'weights_{datetime.datetime.now().strftime('%d-%b-%Y_%H-%M')}.pt'
        torch.save({
            'weights': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(WEIGHTS_PATH, fname))
    
    def load_weights(self):
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pt')]
        if files:
            max_file = max(files, key=os.path.getctime)
            weights_data = torch.load(max_file, map_location=DEVICE)
            if 'weights' in weights_data:
                self.load_state_dict(weights_data['weights'])
                print(f'[+] Loaded weights from {max_file}')

            if self.optimizer and 'optimizer' in weights_data:
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
    
    def encode_features(self, board):
        fb = self.feature_buffer
        fb[0] = 1.0 if board.turn == chess.WHITE else 0.0
        fb[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        fb[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        fb[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        fb[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        fb[5] = 1.0 if board.ep_square is not None else 0.0
        return fb
    
    def predict(self, array):
        self.eval()
        x = array.to(DEVICE)
        with torch.no_grad():
            logp = F.log_softmax(self.forward(x).squeeze(0), dim=-1)
        
        arr = x.squeeze(0).cpu().tolist()
        board = chess.Board.empty()
        for sq, v in enumerate(array.view(-1)):
            if v > 1:
                ptype, color = rlookup[int(v)]
                board.set_piece_at(sq, chess.Piece(ptype, bool(color)))
            
            best_score, best_board = float('-inf'), np.array(arr, dtype=np.int64)
            for move in board.legal_moves:
                b2 = board.copy(stack=False)
                b2.push(move)
                target = self.encode_board(b2)

                idx = torch.as_tensor(target, device=logp.device, dtype=torch.long).view(-1, 1)
                score = logp.gather(1, idx).sum().item()

                if score > best_score:
                    best_score = score
                    best_board = target
        
        self.display_board(best_board)
        
        return torch.as_tensor([best_board], dtype=torch.long)
    
    def display_board(self, array):
        array = array.reshape(8,8)
        for i in range(8):
            print('|'.join(piece_lookup[int(array[i, j])] for j in range(8)))
    
    @torch.no_grad()
    def best_move_from_fen(self, fen):
        ''' Get the best move from a FEN string '''
        board = chess.Board(fen)
        return self.best_move_from_board(board)
    
    @torch.no_grad()
    def best_move_from_board(self, board):
        ''' Get the best move from a chess.Board object '''
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            board.push(move)
            score = self.evaluate_position(board)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    @torch.no_grad()
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
        feature_np = self.encode_features(board)
        feature_tensor = torch.from_numpy(feature_np).float().unsqueeze(0).to(DEVICE)

        self.eval()
        value = self.forward(board_tensor, feature_tensor)
        return float(value.item())
    
    def negamax_search(self, board, depth, alpha, beta, start_time, time_limit):
        if depth == 0 or board.is_game_over() or (time.time() - start_time) > time_limit:
            return self.evaluate_position(board)

        best = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax_search(board, depth - 1, -beta, -alpha, start_time, time_limit)
            board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

            if time.time() - start_time > time_limit:
                break
        
        return best
    
    def best_move_negamax(self, board, depth = 2, time_limit=5):
        start_time = time.time()
        best_move = None
        best_score = float('-inf')

        alpha = -math.inf
        beta = math.inf

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        for move in board.legal_moves:
            if time.time() - start_time > time_limit:
                break

            board.push(move)
            score = -self.negamax_search(board, depth - 1, -beta, -alpha, start_time, time_limit)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            
            if score > alpha:
                alpha = score
        
        if best_move is None:
            best_move = self.best_move_from_board(board)
        
        return best_move
    
    @torch.no_grad()
    def engine_vs_stockfish(self):
        ''' Evaulte how good the engine is by playing ti against stockfish '''
        
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
                move = self.best_move_negamax(board, depth=2, time_limit=5)
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
            move = model.best_move_from_fen(fen)
            print(f'Best move: {move}')
