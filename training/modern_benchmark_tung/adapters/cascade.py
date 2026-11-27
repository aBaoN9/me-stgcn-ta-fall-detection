"""Independent CascadeFormer small, CE-only, after Peng and Yilmaz."""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        width = 48
        self.stem = nn.Sequential(nn.Flatten(2), nn.Linear(264, width), nn.LayerNorm(width))
        self.position = nn.Parameter(torch.empty(1, 32, width))
        nn.init.normal_(self.position, std=0.02)
        self.first = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout,
                                                activation="gelu", batch_first=True, norm_first=True)
        self.second = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout,
                                                 activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, x):
        frames = self.stem(x.permute(0, 2, 3, 1))
        first = self.first(frames + self.position)
        refined = self.second(first)
        fused, _ = self.fusion(first, refined, refined, need_weights=False)
        return self.head(self.norm(first + fused).mean(dim=1))
