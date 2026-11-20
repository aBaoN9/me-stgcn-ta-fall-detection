import torch
from torch import nn

from training.component_ablation.models import EDGES
from ..models import flatten_joints

GRU = "GRU_v2_small"
CTR = "CTR_GCN_adapted"


class SmallGRU(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.normalize = nn.BatchNorm1d(264)
        self.recurrent = nn.GRU(264, 64, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, values):
        sequence = self.normalize(flatten_joints(values)).transpose(1, 2)
        _, hidden = self.recurrent(sequence)
        return self.classifier(hidden[-1])


def pose_graph():
    graph = torch.eye(33).repeat(3, 1, 1)
    graph[1:] = 0
    for source, target in EDGES:
        graph[1, target, source] = 1
        graph[2, source, target] = 1
    return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1)


class ChannelRefinement(nn.Module):
    def __init__(self, in_channels, width):
        super().__init__()
        self.prior = nn.Parameter(pose_graph())
        self.scale = nn.Parameter(torch.zeros(3))
        self.query = nn.Conv2d(in_channels, 12, 1)
        self.key = nn.Conv2d(in_channels, 12, 1)
        self.relation = nn.Conv2d(12, 3 * width, 1, groups=3)
        self.values = nn.Conv2d(in_channels, 3 * width, 1)
        self.width = width

    def forward(self, values):
        batch, _, frames, joints = values.shape
        query = self.query(values).mean(dim=2)
        key = self.key(values).mean(dim=2)
        relation = self.relation((query.unsqueeze(-1) - key.unsqueeze(-2)).tanh())
        topology = self.prior[None, :, None] + (
            self.scale[None, :, None, None, None]
            * relation.reshape(batch, 3, self.width, joints, joints)
        )
        features = self.values(values).reshape(batch, 3, self.width, frames, joints)
        return torch.einsum("ngcuv,ngctv->nctu", topology, features)


class GraphBlock(nn.Module):
    def __init__(self, in_channels, width, stride):
        super().__init__()
        self.spatial = ChannelRefinement(in_channels, width)
        self.normalize = nn.BatchNorm2d(width)
        self.temporal = nn.ModuleList(
            nn.Conv2d(width, width // 2, (3, 1), stride=(stride, 1),
                      dilation=(d, 1), padding=(d, 0), bias=False)
            for d in (1, 2)
        )
        self.out_norm = nn.BatchNorm2d(width)
        self.skip = (nn.Identity() if in_channels == width and stride == 1 else
                     nn.Sequential(nn.Conv2d(in_channels, width, 1, stride=(stride, 1), bias=False),
                                   nn.BatchNorm2d(width)))

    def forward(self, values):
        spatial = self.normalize(self.spatial(values)).relu()
        temporal = torch.cat([branch(spatial) for branch in self.temporal], dim=1)
        return (self.out_norm(temporal) + self.skip(values)).relu()


class AdaptedCTR(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.normalize = nn.BatchNorm1d(264)
        self.blocks = nn.Sequential(GraphBlock(8, 16, 1), GraphBlock(16, 24, 2),
                                    GraphBlock(24, 32, 2))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, values):
        flat = self.normalize(flatten_joints(values))
        skeleton = flat.reshape(-1, 33, 8, 32).permute(0, 2, 3, 1)
        return self.classifier(self.blocks(skeleton.contiguous()).mean(dim=(2, 3)))


def build_model(variant, dropout=0.3):
    builders = {GRU: SmallGRU, CTR: AdaptedCTR}
    try:
        builder = builders[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown variant: {variant}") from exc
    return builder(dropout)
