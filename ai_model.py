from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch, os, time

DATASET_PATH = 'datasets/'
BATCH_SIZE = 6
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
    
    def collate_fn(self, batch):
        x,y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)
        return x,y

class FisherAI(Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.nheads = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.no_transformer_layers = self.d_model // 128
        self.dropout = 0.1
        self.dataset = ChessDataset()
        
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
        start = time.time()

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
    
            print(f'Epoch {epoch+1}, avg loss: {total_loss / len(self.dataloader):.4f}')

if __name__ == '__main__':
    model = FisherAI().to(DEVICE)
    model.train_model()
