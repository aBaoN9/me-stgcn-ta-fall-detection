"""BlockGCN adapted to the GMDCSA-24 v2 contract (CE-adapted, author-size).

Ported from ZhouYuxuanYX/BlockGCN, model/BlockGCN.py at commit 7b45971a931c5b0ba60f3a7ba11a91e544de54fb
(Apache License 2.0; licence and NOTICE in THIRD_PARTY_NOTICES.md). BlockGC (per-head joint mixing fc1 +
grouped fc2), hop-distance relative encoding, multi-scale temporal conv, the 10-layer 128/256 layout and the
persistent-homology topology encoding (Topo + TopoTrans per layer) are kept. Changes:

* input (N,8,32,33): to_joint_embedding is Linear(8,128); one person (TopoTrans does not repeat, as the
  author's N-UCLA branch); 33-joint MediaPipe graph from graph.py, hop encoding with a separate bucket for
  joints in different graph components (the author code assumes a connected skeleton);
* topology input = relative_x, relative_y (channels 0-1) of the same augmented sample, i.e. the joint
  positions, as the author feeds raw joint coordinates;
* the author normalises the batch's distance matrices by the batch min/max; here each clip is normalised by
  its own min/max so a test score never depends on the other clips in its batch;
* torch_topological is not a dependency: dimension-0 Vietoris-Rips persistence of a point cloud equals
  (birth 0, death = minimum-spanning-tree edge lengths), computed exactly with Prim's algorithm. As in the
  author code the cloud is the rows of the normalised distance matrix (Euclidean cdist, no
  treat_as_distances). StructureElementLayer is re-implemented from torch_topological 8aca6db (BSD-3) and
  returns (B, 64) for any batch size;
* head dropout 0.3; two-class head; returns logits only; plain CE (the author's loss2 is zero).
"""
import math

import numpy as np
import torch
from torch import nn

from training.modern_benchmark_duc.graph import NUM_JOINTS, hop_matrix, spatial_graph


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class Zero(nn.Module):
    def forward(self, x):
        return 0


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 2, 3, 4)):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0
        branch_channels = out_channels // (len(dilations) + 2)
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride, dilation=dilation))
            for dilation in dilations])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)))
        self.apply(weights_init)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)  # author: residual=False


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, hops):
        super().__init__()
        self.num_heads = 8 if in_channels > 8 else 1
        V = A.shape[-1]
        self.fc1 = nn.Parameter(torch.stack([torch.stack([torch.eye(V) for _ in range(self.num_heads)], dim=0)
                                             for _ in range(3)], dim=0))
        self.fc2 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)])
        self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)) \
            if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        self.register_buffer("hops", torch.as_tensor(hops, dtype=torch.long), persistent=False)
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, int(hops.max()) + 1)))

    @staticmethod
    def L2_norm(weight):
        return torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        pos_emb = self.rpe[:, :, self.hops]
        for i in range(3):
            w1 = self.fc1[i] / self.L2_norm(self.fc1[i])
            w1 = w1 + pos_emb[i] / self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            z = torch.einsum("nhctv, hvw->nhctw", (x_in, w1)).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)
            y = z + y if y is not None else z
        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, hops, stride=1, kernel_size=5, dilations=(1, 2)):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, hops)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations)
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class TopoTrans(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(64, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):  # one person: no repeat (author's N-UCLA branch)
        return self.relu(self.bn(self.mlp(x))).unsqueeze(2).unsqueeze(3)


class StructureElementLayer(nn.Module):
    """torch_topological.nn.layers.StructureElementLayer (BSD-3, 8aca6db) for (birth, death, dim) triples."""

    def __init__(self, n_elements=64):
        super().__init__()
        self.n_elements, self.dim = n_elements, 2
        self.centres = nn.Parameter(torch.rand(n_elements, self.dim))
        self.sharpness = nn.Parameter(torch.ones(n_elements, self.dim) * 3)

    def forward(self, x):  # x: (B, P, 3) = (birth, death, homology dimension)
        diff = self.centres[None, :, None, :] - x[:, None, :, :2]                  # (B, E, P, 2)
        energy = (diff.pow(2) * self.sharpness.pow(2)[None, :, None, :]).sum(-1)   # (B, E, P)
        energy = energy + x[:, None, :, 2].pow(2)  # (2*dim - dim)^2 * 1, as in the source layer
        return torch.exp(-energy).sum(-1)                                          # (B, E)


def mst_deaths(points):
    """Dimension-0 Vietoris-Rips deaths of each cloud = its minimum-spanning-tree edge lengths (Prim)."""
    distances = torch.cdist(points, points, p=2)                    # (B, V, V)
    batch, count = distances.shape[:2]
    in_tree = torch.zeros(batch, count, dtype=torch.bool, device=points.device)
    in_tree[:, 0] = True
    best = distances[:, 0].clone()
    deaths = []
    for _ in range(count - 1):
        candidate = best.masked_fill(in_tree, float("inf"))
        value, index = candidate.min(dim=1)
        deaths.append(value)
        in_tree[torch.arange(batch, device=points.device), index] = True
        best = torch.minimum(best, distances[torch.arange(batch, device=points.device), index])
    return torch.stack(deaths, dim=1)                               # (B, V-1)


class Topo(nn.Module):
    def __init__(self):
        super().__init__()
        self.pl = StructureElementLayer(n_elements=64)

    def forward(self, joint):  # joint: (N, M, C, T, V)
        with torch.no_grad():
            x = joint.mean(1)
            x = x.unsqueeze(-1) - x.unsqueeze(-2)
            x = x.mean(-3)
            x = torch.norm(x, 2, dim=1)                              # (N, V, V)
            low = x.amin(dim=(1, 2), keepdim=True)
            span = (x.amax(dim=(1, 2), keepdim=True) - low).clamp_min(1e-12)
            x = (x - low) / span                                     # per-clip min/max (author: per batch)
            deaths = mst_deaths(x)
            diagram = torch.stack([torch.zeros_like(deaths), deaths, torch.zeros_like(deaths)], dim=-1)
        return self.pl(diagram)


class BlockGCNV2(nn.Module):
    def __init__(self, in_channels=8, num_classes=2, num_point=NUM_JOINTS, num_person=1, drop_out=0.3,
                 topology_channels=(0, 1)):
        super().__init__()
        A, hops = spatial_graph(), hop_matrix()
        self.num_point, self.topology_channels = num_point, list(topology_channels)
        self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, 128))
        widths = [(128, 128, 1), (128, 128, 1), (128, 128, 1), (128, 128, 1), (128, 256, 2),
                  (256, 256, 1), (256, 256, 1), (256, 256, 2), (256, 256, 1), (256, 256, 1)]
        self.layers = nn.ModuleList(TCN_GCN_unit(i, o, A, hops, stride=s) for i, o, s in widths)
        self.topo_trans = nn.ModuleList(TopoTrans(i) for i, _, _ in widths)
        self.topo = Topo()
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (8, 32, self.num_point):
            raise ValueError("Expected (N,8,32,33)")
        N, C, T, V = x.shape
        a = self.topo(x[:, self.topology_channels].unsqueeze(1))   # (N, 64)
        x = x.permute(0, 2, 3, 1).reshape(N * T, V, C)            # '(n m t) v c' with m = 1
        x = self.to_joint_embedding(x) + self.pos_embedding[:, :self.num_point]
        x = x.view(N, T, V, 128).permute(0, 2, 3, 1).reshape(N, V * 128, T)  # 'n (m v c) t'
        x = self.data_bn(x)
        x = x.view(N, V, 128, T).permute(0, 2, 3, 1).contiguous()  # (N, 128, T, V)
        for layer, trans in zip(self.layers, self.topo_trans):
            x = layer(x + trans(a))
        x = x.view(N, x.size(1), -1).mean(2)
        return self.fc(self.drop_out(x))
