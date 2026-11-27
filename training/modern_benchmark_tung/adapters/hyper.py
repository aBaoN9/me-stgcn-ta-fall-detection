"""Adaptive multi-head hypergraph with anatomical virtual joints, after Zhou et al."""
import torch
from torch import nn

from training.component_ablation.models import EDGES

PARTS = (tuple(range(11)), (11, 12, 23, 24),
         (11, 13, 15, 17, 19, 21), (12, 14, 16, 18, 20, 22),
         (23, 25, 27, 29, 31), (24, 26, 28, 30, 32))


def augmented_graph():
    adjacency = torch.eye(33 + len(PARTS))
    for origin, destination in EDGES:
        adjacency[origin, destination] = adjacency[destination, origin] = 1
    for virtual, nodes in enumerate(PARTS, 33):
        adjacency[virtual, list(nodes)] = adjacency[list(nodes), virtual] = 1
    return adjacency


class HyperBlock(nn.Module):
    def __init__(self, stride, dropout, width=24, heads=4):
        super().__init__()
        self.heads = heads
        self.virtual = nn.Parameter(torch.randn(len(PARTS), width) * 0.02)
        self.register_buffer("base", augmented_graph())
        self.edges = nn.Parameter(torch.ones(heads, *self.base.shape))
        self.embedding = nn.Linear(width, 4 * heads)
        self.weight = nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(),
                                    nn.Linear(width, heads), nn.Tanh())
        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        self.values = nn.Conv2d(width, width, 1, groups=heads)
        self.temporal = nn.ModuleList(nn.Conv2d(width, width // 2, (3, 1),
            stride=(stride, 1), dilation=(d, 1), padding=(d, 0)) for d in (1, 2))
        self.norm = nn.BatchNorm2d(width)
        self.dropout = nn.Dropout(dropout)
        self.stride = stride

    def forward(self, x):
        batch, channels, frames, _ = x.shape
        virtual = self.virtual.T[None, :, None].expand(batch, channels, frames, -1)
        graph_x = torch.cat((x, virtual), dim=-1)
        node = graph_x.mean(dim=2).transpose(1, 2)
        relation = self.embedding(node).unflatten(-1, (self.heads, 4)).transpose(1, 2)
        distances = torch.cdist(relation.contiguous(), relation.contiguous())
        nearest, positions = distances.topk(9, largest=False)
        incidence = torch.zeros_like(distances).scatter(-1, positions, (-nearest).softmax(-1))
        weights = self.weight(node).transpose(1, 2).unsqueeze(-2)
        weighted = incidence * weights
        send = weighted / weighted.abs().sum(-1, keepdim=True).clamp_min(1e-6)
        receive = incidence / incidence.sum(-2, keepdim=True).clamp_min(1e-6)
        adaptive = send @ (weights * receive).transpose(-1, -2)
        base = self.base * self.edges
        adjacency = base / base.abs().sum(-1, keepdim=True).clamp_min(1e-6)
        adjacency = adjacency + self.scale.relu() * adaptive
        features = self.values(graph_x).unflatten(1, (self.heads, -1))
        gathered = torch.einsum("bhuv,bhctv->bhctu", adjacency, features)
        gathered = gathered.flatten(1, 2)[..., :33]
        temporal = torch.cat([layer(gathered) for layer in self.temporal], dim=1)
        return (x[:, :, ::self.stride] + self.dropout(self.norm(temporal))).relu()


class Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.stem = nn.Conv2d(8, 24, 1)
        self.blocks = nn.Sequential(HyperBlock(1, dropout), HyperBlock(2, dropout))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(24, 2))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))
