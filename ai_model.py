import sys
from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch, os, time, datetime, chess

lookup = {(1,0) : 2,(2,0) : 3,(3,0) : 4,(4,0) : 5,(5,0) : 6,(6,0) : 7,(1,1) : 8,(2,1) : 9,(3,1) : 10,(4,1) : 11,(5,1) : 12,(6,1) : 13}
piece_lookup = {0: ' ',1: '.',2: 'p',3: 'n',4: 'b',5: 'r',6: 'q',7: 'k',8: 'P',9: 'N',10: 'B',11: 'R',12: 'Q',13: 'K'}

DATASET_PATH = 'datasets/'
WEIGHTS_PATH = 'weights/'
BATCH_SIZE = 2
DEVICE = 'cuda'

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
        self.dropout = 0.1
        self.dataset = ChessDataset()
        self.optimizer = None
        
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

        print(f'[+] Training started, d_model={self.d_model}, nheads={self.nheads}, dim_feedforward={self.dim_feedforward}, '
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
        
        arr = np.ones(64, dtype=np.int64)
        for sq, sequence in board.piece_map().items():
            arr[sq] = lookup[(sequence.piece_type, int(sequence.color))]
        arr = arr[None, :]
        return torch.as_tensor(arr.copy())
    
    def encode_board(self, b):
        arr = np.ones(64, dtype=np.int64)
        for sq, p in b.piece_map().items():
            arr[sq] = lookup[(p.piece_type, int(p.color))]
        return arr
    
    def predict(self, fen):
        self.eval()
        board = chess.Board(fen)

        tensor = self.fen_to_tensor(fen).to(DEVICE)
        with torch.no_grad():
            logits = self.forward(tensor).squeeze(0)
            logp = F.log_softmax(logits, dim=-1)

        best_move, best_score, best_board = None, float('-inf'), None

        for move in board.legal_moves:
            b2 = board.copy(stack=False)
            b2.push(move)
            target = self.encode_board(b2)

            idx = torch.as_tensor(target, device=logp.device, dtype=torch.long).view(-1, 1)
            score = logp.gather(1, idx).sum().item()

            if score > best_score:
                best_score = score
                best_move = move
                best_board = target

        print(f'Predicted move: {best_move.uci()}')
        self.display_board(best_board)
    
    def display_board(self, array):
        array = array.reshape(8,8)
        for i in range(8):
            print('|'.join(piece_lookup[int(array[i, j])] for j in range(8)))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = FisherAI().to(DEVICE)
            model.train_model()
        except KeyboardInterrupt:
            model.save_weights()
    
    else:
        model = FisherAI().to(DEVICE)
        model.load_weights()
        while True:
            fen = input('Enter FEN: ')
            model.predict(fen)
