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

    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def read_data(self):
        self.training_data = []
        for file in os.listdir(DATASET_PATH):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(DATASET_PATH, file), map_location='cpu', weights_only=False)
                for game in data:
                    g = torch.as_tensor(game, dtype=torch.long)
                    x = g[:-1]
                    y = g[1:]
                    self.training_data.append((x,y))
        
        print(f'[+] Loaded {len(self.training_data):,} games')
    
    def collate_fn(self, batch):
        x,y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)
        return x,y

class FisherAI(Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.nheads = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.no_transformer_layers = self.d_model // 128
        self.dropout = 0.05
        self.dataset = ChessDataset()
        self.optimizer = None
        self.stock_fish_path = '/usr/bin/stockfish'
        self.encode_buffer = np.empty(64, dtype=np.int64)
        self.fen_buffer = np.empty(64, dtype=np.int64)
        
        self.piece_embedding = nn.Embedding(14, self.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(64, self.d_model)
        self.em_dropout = nn.Dropout(self.dropout)

        self.encoder_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nheads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=self.no_transformer_layers
        )
        self.piece_head = nn.Linear(self.d_model, 14)

    def forward(self, x):

        B,S = x.shape
        pos = self.position_embedding(torch.arange(S, device=DEVICE)).unsqueeze(0).expand(B, S, -1)

        x = self.piece_head(
            self.encoder_layer(
                self.em_dropout(
                    self.piece_embedding(x) + pos
                )
            )
        )

        return x
    
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
        array = self.fen_buffer[None, :].copy()
        return torch.as_tensor(array, dtype=torch.long)
    
    def encode_board_inplace(self, board, dest):
        dest.fill(1)
        for sq, p in board.piece_map().items():
            dest[sq] = lookup[(p.piece_type, int(p.color))]
    
    def encode_board(self, board):
        self.encode_board_inplace(board, self.encode_buffer)
        return self.encode_buffer.copy()
    
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
        x = self.fen_to_tensor(fen).to(DEVICE)
        self.eval()
        logp = F.log_softmax(self.forward(x).squeeze(0), dim=-1)
        logp_np = logp.detach().cpu().numpy()

        best_score = float('-inf')
        best_move = None

        lagal_moves = list(board.legal_moves)
        for move in lagal_moves:
            board.push(move)
            self.encode_board_inplace(board, self.encode_buffer)
            score = logp_np[np.arange(64), self.encode_buffer].sum()
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
        
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
        for game in range(no_games):
            board = chess.Board()
            while not board.is_game_over():

                # Stockfish move
                result = engine.play(board, chess.engine.Limit(time=0.05))
                board.push(result.move)

                if board.is_game_over():
                    break
                
                # FisherAI move
                fen = board.fen()
                move = self.best_move_from_fen(fen)
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
        
        print(f'[+] Evaluation completed, FisherAI ELO {elo:.2f}')
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
