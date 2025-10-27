from torch.utils.data import Dataset
from torch.nn import Module
import torch.nn as nn
import torch, os

DATASET_PATH = 'datasets/'

class ChessDataset(Dataset):
    def __init__(self):
        self.training_data = []

    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        return self.training_data[idx], self.training_data[idx + 1]
    
    def read_data(self):
        samples = []
        for file in os.listdir(DATASET_PATH):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(DATASET_PATH, file))
                samples.append(data)
        
        self.training_data = torch.cat(samples, dim=0)

class FisherAI(Module):
    def __init__(self):
        super().__init__()

        self.d_model = 512
        self.nheads = 8
        self.dim_feedforward = 2048
        self.no_transformer_layers = 6
        self.dropout = 0.1
        
        self.piece_type_embedding = nn.Embedding(6, 2)
        self.color_embedding = nn.Embedding(2, 2)
        self.position_embedding = nn.Embedding(self.d_model, 2)
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

    def forward(self, x):
        piece_type = x[:, :, 0].long()
        color = x[:, :, 1].long()

        piece_type_emb = self.piece_type_embedding(piece_type)
        color_emb = self.color_embedding(color)

        x = piece_type_emb + color_emb + position_emb
        x = self.em_dropout(x)

        x = self.encoder_layer(x)

        return x
