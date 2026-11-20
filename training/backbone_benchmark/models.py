import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from training.component_ablation.models import SkeletonModel

TCN = "TCN_v2_small"
A3 = "A3_motion_attention"


def flatten_joints(values):
    if values.ndim != 4 or values.shape[1:] != (8, 32, 33):
        raise ValueError("Expected (N,8,32,33)")
    return values.permute(0, 3, 1, 2).reshape(-1, 264, 32)


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.layers = nn.ModuleList()
        for width in (channels, 32):
            conv = nn.Conv1d(width, 32, 3, dilation=dilation)
            nn.init.normal_(conv.weight, 0, 0.01)
            self.layers.append(weight_norm(conv))
        self.skip = nn.Conv1d(channels, 32, 1) if channels != 32 else nn.Identity()
        if channels != 32:
            nn.init.normal_(self.skip.weight, 0, 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = self.skip(values)
        for conv in self.layers:
            values = self.dropout(F.relu(conv(F.pad(values, (self.padding, 0)))))
        return F.relu(values + residual)


class SmallTCN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(264)
        self.blocks = nn.Sequential(*(TemporalBlock(264 if d == 1 else 32, d, dropout)
                                      for d in (1, 2, 4, 8)))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, values):
        temporal = self.blocks(self.data_bn(flatten_joints(values)))
        return self.classifier(temporal[:, :, -1])


def build_model(variant, dropout=0.3):
    if variant == A3:
        return SkeletonModel(channels=8, attention=True, dropout=dropout)
    if variant == TCN:
        return SmallTCN(dropout)
    raise ValueError(f"Unknown variant: {variant}")


def logits_of(model, values):
    output = model(values)
    return output[0] if isinstance(output, tuple) else output
