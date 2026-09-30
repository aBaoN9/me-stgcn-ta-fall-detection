import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from research07.features import augment_pose, make_features_v2


EDGES = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
         (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
         (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
         (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
         (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_adjacency():
    adjacency = np.eye(33, dtype=np.float32)
    for source, target in EDGES:
        adjacency[source, target] = adjacency[target, source] = 1
    inverse = np.maximum(adjacency.sum(axis=1), 1e-8) ** -0.5
    return torch.from_numpy(inverse[:, None] * adjacency * inverse[None])


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1)
        self.register_buffer("adjacency", adjacency.clone())
        self.edge_importance = nn.Parameter(torch.ones_like(adjacency))

    def forward(self, values):
        return torch.einsum("nctv,vw->nctw", self.projection(values), self.adjacency * self.edge_importance)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1, dropout=0.3):
        super().__init__()
        self.graph_conv = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.temporal_conv = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout))
        self.residual = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, values):
        return self.activation(self.temporal_conv(self.graph_conv(values)) + self.residual(values))


class SkeletonModel(nn.Module):
    def __init__(self, channels=8, attention=True, dropout=0.3):
        super().__init__()
        adjacency = make_adjacency()
        self.data_bn = nn.BatchNorm1d(channels * 33)
        self.blocks = nn.Sequential(STGCNBlock(channels, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 128, adjacency, stride=2, dropout=dropout),
            STGCNBlock(128, 128, adjacency, dropout=dropout))
        self.temporal_attention = nn.Sequential(nn.Conv1d(128, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1)) if attention else None
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, values):
        batch, channels, frames, nodes = values.shape
        values = values.permute(0, 3, 1, 2).contiguous().view(batch, nodes * channels, frames)
        values = self.data_bn(values).view(batch, nodes, channels, frames).permute(0, 2, 3, 1)
        temporal = self.blocks(values).mean(dim=3)
        weights = torch.softmax(self.temporal_attention(temporal), dim=2) if self.temporal_attention is not None else torch.ones_like(temporal[:, :1]) / temporal.shape[2]
        return self.classifier((temporal * weights).sum(dim=2)), weights.squeeze(1)


class PoseDataset(Dataset):
    def __init__(self, data, indices, channels, augment, seed):
        self.data, self.indices, self.channels, self.augment = data, indices, channels, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = self.indices[offset]
        pose, mask, duration = self.data["poses"][index], self.data["masks"][index], self.data["durations"][index]
        dropped = []
        if self.augment:
            pose, mask, duration, dropped = augment_pose(pose, mask, duration, self.rng)
        features = make_features_v2(pose, mask, duration)
        features[:5, :, dropped] = 0
        return torch.from_numpy(features[:self.channels]), torch.tensor(int(self.data["labels"][index]), dtype=torch.long)


def make_loader(data, indices, variant, config, augment, seed):
    return DataLoader(PoseDataset(data, indices, variant["channels"], augment, seed),
                      batch_size=config["batch_size"], shuffle=augment, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
