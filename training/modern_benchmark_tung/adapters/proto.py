"""ProtoGCN small, CE-only, after Liu et al."""
import torch
from torch import nn

from training.component_ablation.models import make_adjacency


class TopologyBlock(nn.Module):
    def __init__(self, stride, dropout, width=24):
        super().__init__()
        self.graph = nn.Parameter(make_adjacency())
        self.first = nn.Conv1d(width, 8, 1)
        self.second = nn.Conv1d(width, 8, 1)
        self.difference = nn.Conv2d(8, width, 1)
        self.affinity = nn.Conv2d(8, width, 1)
        self.values = nn.Conv2d(width, width, 1)
        self.scale = nn.Parameter(torch.zeros(2))
        self.temporal = nn.ModuleList(nn.Conv2d(width, width // 2, (3, 1),
            stride=(stride, 1), dilation=(d, 1), padding=(d, 0)) for d in (1, 2))
        self.bn = nn.BatchNorm2d(width)
        self.dropout = nn.Dropout(dropout)
        self.stride = stride

    def forward(self, x):
        mean = x.mean(2)
        q, k = self.first(mean), self.second(mean)
        difference = self.difference((q.unsqueeze(-1) - k.unsqueeze(-2)).tanh())
        affinity = self.affinity((q.unsqueeze(-1) * k.unsqueeze(-2)).tanh())
        graph = self.graph + self.scale[0] * difference + self.scale[1] * affinity
        spatial = torch.einsum("bcuv,bctv->bctu", graph, self.values(x))
        temporal = self.bn(torch.cat([branch(spatial) for branch in self.temporal], dim=1))
        return (x[:, :, ::self.stride] + self.dropout(temporal)).relu(), graph


class Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.stem = nn.Conv2d(8, 24, 1)
        self.first = TopologyBlock(1, dropout)
        self.second = TopologyBlock(2, dropout)
        self.query = nn.Linear(24, 16, bias=False)
        self.memory = nn.Linear(16, 24, bias=False)
        self.post = nn.Sequential(nn.Conv2d(24, 24, 1), nn.BatchNorm2d(24), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(24, 2)

    def forward(self, x):
        x, _ = self.first(self.stem(x))
        x, topology = self.second(x)
        weights = self.query(topology.permute(0, 2, 3, 1)).softmax(-1)
        reconstructed = self.dropout(self.memory(weights)).permute(0, 3, 1, 2)
        # CE drives the topology prototypes without a class-specific contrastive target.
        context = self.post(reconstructed).mean((2, 3))
        return self.head(self.dropout(context + x.mean((2, 3))))
