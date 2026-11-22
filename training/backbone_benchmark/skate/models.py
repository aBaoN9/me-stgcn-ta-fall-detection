"""Small SkateFormer adaptation inspired by Do and Kim (ECCV 2024)."""
import torch
from torch import nn
from torch.nn import functional as F

from training.component_ablation.models import make_adjacency

SKATE = "SkateFormer_adapted_small"

# Six anatomical groups; padded slots 33--35 are excluded from attention and pooling.
JOINT_GRID = (
    (0, 1, 2, 3, 7, 9),
    (33, 4, 5, 6, 8, 10),
    (23, 11, 13, 15, 17, 19),
    (24, 12, 14, 16, 18, 20),
    (21, 25, 27, 29, 31, 34),
    (22, 26, 28, 30, 32, 35),
)


def partition_indices(frames, distant_time, distant_joints):
    time = torch.arange(frames)
    time = time.reshape(4, -1).T if distant_time else time.reshape(-1, 4)
    joints = torch.tensor(JOINT_GRID)
    if distant_joints:
        joints = joints.T
    return (time[:, None, :, None] * 36 + joints[None, :, None, :]).reshape(-1, 24)


class SkateBlock(nn.Module):
    def __init__(self, frames, width=32):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.map = nn.Linear(width, 2 * width)
        self.temporal = nn.Conv2d(width // 4, width // 4, (3, 1), padding=(1, 0),
                                  groups=width // 4)
        self.graph = nn.Parameter(make_adjacency())
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                 nn.GELU(), nn.Linear(2 * width, width))
        for i, (dt, dj) in enumerate(((False, False), (False, True), (True, False), (True, True))):
            index = partition_indices(frames, dt, dj).flatten()
            self.register_buffer(f"index_{i}", index, persistent=False)
            self.register_buffer(f"restore_{i}", index.argsort(), persistent=False)
            self.register_buffer(f"mask_{i}", index.reshape(-1, 24) % 36 < 33, persistent=False)

    def attention(self, values, branch):
        batch, frames, joints, channels = values.shape
        index = getattr(self, f"index_{branch}")
        restore = getattr(self, f"restore_{branch}")
        mask = getattr(self, f"mask_{branch}")[None, :, None, :]
        windows = values.reshape(batch, frames * joints, channels)[:, index]
        q, k, v = windows.reshape(batch, -1, 24, channels).chunk(3, dim=-1)
        output = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                                attn_mask=mask)
        return output.reshape(batch, frames * joints, -1)[:, restore].reshape(batch, frames, joints, -1)

    def forward(self, x):
        values = self.map(self.norm(x))
        graph, temporal, *attention = values.split((8, 8, 12, 12, 12, 12), dim=-1)
        graph = F.pad(torch.einsum("uv,btvc->btuc", self.graph, graph[:, :, :33]), (0, 0, 0, 3))
        temporal = self.temporal(temporal.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        branches = [graph, temporal, *(self.attention(part, i) for i, part in enumerate(attention))]
        x = x + self.proj(torch.cat(branches, dim=-1))
        return x + self.mlp(x)


class SkateFormerSmall(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.stem = nn.Linear(8, 32)
        self.joint_position = nn.Parameter(torch.zeros(1, 1, 33, 32))
        self.time_position = nn.Parameter(torch.zeros(1, 32, 1, 32))
        nn.init.normal_(self.joint_position, std=0.02)
        nn.init.normal_(self.time_position, std=0.02)
        self.first = SkateBlock(32)
        self.merge = nn.Conv2d(32, 32, (3, 1), stride=(2, 1), padding=(1, 0))
        self.second = SkateBlock(16)
        self.classifier = nn.Sequential(nn.LayerNorm(32), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, values):
        if values.ndim != 4 or values.shape[1:] != (8, 32, 33):
            raise ValueError("Expected (N,8,32,33) with MediaPipe landmark order")
        x = self.stem(values.permute(0, 2, 3, 1))
        x = self.first(F.pad(x + self.joint_position + self.time_position, (0, 0, 0, 3)))
        # Remove padded queries before the temporal projection and final pooling.
        x = self.merge(x[:, :, :33].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.second(F.pad(x, (0, 0, 0, 3)))
        return self.classifier(x[:, :, :33].mean(dim=(1, 2)))


def build_model(variant, dropout=0.3):
    if variant != SKATE:
        raise ValueError(f"Unknown variant: {variant}")
    return SkateFormerSmall(dropout)
