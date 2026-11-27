"""DHA-eGCN small, single fixed masked graph, after Nugroho et al."""
import torch
from torch import nn

from training.component_ablation.models import EDGES, make_adjacency


def part_ids():
    return torch.tensor([0] * 11 + [0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                                    0, 0, 3, 4, 3, 4, 3, 4, 3, 4])


def hops():
    dist = torch.full((33, 33), 8, dtype=torch.long)
    dist.fill_diagonal_(0)
    for left, right in EDGES:
        dist[left, right] = dist[right, left] = 1
    for joint in range(33):
        dist = torch.minimum(dist, dist[:, joint, None] + dist[None, joint, :])
    return dist


class DifferentialBlock(nn.Module):
    def __init__(self, width, stride, dropout):
        super().__init__()
        self.register_buffer("base", make_adjacency())
        self.register_buffer("hops", hops(), persistent=False)
        ids = part_ids()
        self.register_buffer("part", ids, persistent=False)
        membership = torch.nn.functional.one_hot(ids, num_classes=5).float()
        self.register_buffer("incidence", membership / membership.sum(0), persistent=False)
        self.query = nn.Linear(width, 2 * width)
        self.key = nn.Linear(width, 2 * width)
        self.context = nn.Linear(width, 2 * width, bias=False)
        self.value = nn.Linear(width, width)
        self.relative = nn.Parameter(torch.zeros(2, 9))
        self.lam = nn.Parameter(torch.tensor(-1.0))
        self.edge_weight = nn.Parameter(torch.ones(33, 33))
        self.graph = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.ModuleList(nn.Conv2d(width, width // 2, (3, 1),
            stride=(stride, 1), dilation=(d, 1), padding=(d, 0)) for d in (1, 2))
        self.bn = nn.BatchNorm2d(width)
        self.dropout = nn.Dropout(dropout)
        self.stride = stride

    def forward(self, x):
        x_norm = self.norm(x)
        joint_parts = torch.einsum("btvc,vp->btpc", x_norm, self.incidence)[:, :, self.part]
        q = self.query(x_norm).unflatten(-1, (2, -1)).transpose(2, 3)
        k = (self.key(x_norm) + self.context(joint_parts)).unflatten(-1, (2, -1)).transpose(2, 3)
        attention = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        attention = (attention + self.relative[:, self.hops]).softmax(-1)
        difference = attention[:, :, 0] - self.lam.sigmoid() * attention[:, :, 1]
        contextual = difference @ self.value(x_norm)
        grounded = (self.base * self.edge_weight) @ self.graph(x_norm)
        combined = (contextual + grounded).permute(0, 3, 1, 2)
        motion = self.bn(torch.cat([layer(combined) for layer in self.temporal], dim=1))
        return (x[:, ::self.stride] + self.dropout(motion.permute(0, 2, 3, 1))).relu()


class Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.stem = nn.Linear(8, 24)
        self.blocks = nn.Sequential(DifferentialBlock(24, 1, dropout),
                                    DifferentialBlock(24, 2, dropout))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(24, 2))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x.permute(0, 2, 3, 1))).mean((1, 2)))
