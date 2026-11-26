"""DeGCN (released code) adapted to the GMDCSA-24 v2 contract (CE-adapted, author-size).

Ported from WoominM/DeGCN_pytorch, model/degcn.py + model/modules.py at commit
a0a261fbf4369095083a7b22fa371fb9448f69a8 (MIT License, Copyright (c) 2024 Woom; text in
THIRD_PARTY_NOTICES.md). Important: the deformable spatial module DeSGC is NOT in the authors' public release
("not included in the open-source release"); their released Basic_Block uses CTR_GC for space and the
deformable temporal DeTGC for time. This port reproduces that released model: 10 blocks 64/128/256,
eta = 4, two parallel streams on the same input. It is "DeGCN (released code)", not the paper's DeSGC.

Changes forced by the contract:
* input (N,8,32,33): the author's `in_channels == 3` test marks the input block (ST_GC, num_scale 1);
  here an explicit `first` flag does that for 8 channels; base_frame = 32 so DeTGC clamps to the real T;
* 33-joint MediaPipe spatial graph from graph.py instead of the NTU graph; one person;
* the author trains each stream head with its own CE and scores with the sum of stream logits; here the
  model returns that summed logit and the common recipe applies one class-weighted CE to it;
* head dropout 0.3; two-class heads.
"""
import math

import numpy as np
import torch
from torch import nn

from training.modern_benchmark_duc.graph import NUM_JOINTS, spatial_graph

LEAKY_ALPHA = 0.1


def init_param(modules):
    for m in modules:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, a=LEAKY_ALPHA, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ST_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A)
        self.Nh = A.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        v = self.conv(x).view(N, self.Nh, -1, T, V)
        return self.bn(torch.einsum("hvu,nhctu->nctv", self.A.to(v.dtype), v))


class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1):
        super().__init__()
        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale
        rel_channels = in_channels // 8
        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        N, C, T, V = x.size()
        q, k = self.conv1(x).mean(-2), self.conv2(x).mean(-2)
        v = self.conv3(x).view(N, self.num_scale, self.Nh, -1, T, V)
        weights = self.conv4(self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))).view(N, self.num_scale, self.Nh, -1, V, V)
        weights = weights * self.alpha.to(weights.dtype) + self.A.view(1, 1, self.Nh, 1, V, V).to(weights.dtype)
        x = torch.einsum("ngacvu, ngactu->ngctv", weights, v).contiguous().view(N, -1, T, V)
        return self.bn(x)


class DeTGC(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, dilation=1, num_frame=32):
        super().__init__()
        self.ks, self.stride, self.dilation, self.T, self.eta = kernel_size, stride, dilation, num_frame, eta
        ref = (self.ks + (self.ks - 1) * (self.dilation - 1) - 1) // 2
        self.tr = nn.Parameter(torch.linspace(-ref, ref, self.eta))
        self.conv_out = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
                                      nn.BatchNorm3d(out_channels))

    def forward(self, x):
        N, C, T, V = x.size()
        if T != self.T:
            raise ValueError(f"DeTGC expected T={self.T}, got {T}")
        Tout = T // self.stride
        dtype = x.dtype
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device)
        t0, tr = t0.view(1, 1, -1).expand(-1, self.eta, -1), self.tr.to(dtype).view(1, self.eta, 1)
        t = (t0 + tr).view(1, 1, -1, 1)
        tdn = t.detach().floor()
        tup = tdn + 1
        # Same as the author's x.gather(-2, index.expand(N, C, -1, V)): the index only varies along time.
        index1 = torch.clamp(tdn, 0, self.T - 1).long().view(-1)
        index2 = torch.clamp(tup, 0, self.T - 1).long().view(-1)
        alpha = tup - t
        x = x.index_select(2, index1) * alpha + x.index_select(2, index2) * (1 - alpha)
        return self.conv_out(x.view(N, C, self.eta, Tout, V)).squeeze(2)


class MultiScale_TemporalModeling(nn.Module):
    def __init__(self, in_channels, out_channels, eta, stride=1, num_scale=1, num_frame=32):
        super().__init__()
        scale_channels = out_channels // num_scale

        def branch(dilation):
            return nn.Sequential(PointWiseTCN(in_channels, scale_channels), nn.LeakyReLU(LEAKY_ALPHA),
                                 DeTGC(scale_channels, scale_channels, eta, kernel_size=5, stride=stride,
                                       dilation=dilation, num_frame=num_frame))

        self.tcn1, self.tcn2 = branch(1), branch(2)
        self.maxpool3x1 = nn.Sequential(PointWiseTCN(in_channels, scale_channels), nn.LeakyReLU(LEAKY_ALPHA),
                                        nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                                        nn.BatchNorm2d(scale_channels))
        self.conv1x1 = PointWiseTCN(in_channels, scale_channels, stride=stride)

    def forward(self, x):
        return torch.cat([self.tcn1(x), self.tcn2(x), self.maxpool3x1(x), self.conv1x1(x)], 1)


class Zero(nn.Module):
    def forward(self, x):
        return 0


class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, eta, stride=1, num_frame=32, residual=True, first=False):
        super().__init__()
        num_scale = 4
        self.num_scale = 1 if first else num_scale
        self.gcn = ST_GC(in_channels, out_channels, A) if first else CTR_GC(in_channels, out_channels, A, self.num_scale)
        self.tcn = MultiScale_TemporalModeling(out_channels, out_channels, eta, stride=stride, num_scale=num_scale,
                                               num_frame=num_frame)
        self.residual1 = PointWiseTCN(in_channels, out_channels, groups=self.num_scale) \
            if in_channels != out_channels else nn.Identity()
        if not residual:
            self.residual2 = Zero()
        elif in_channels == out_channels and stride == 1:
            self.residual2 = nn.Identity()
        else:
            self.residual2 = PointWiseTCN(in_channels, out_channels, stride=stride, groups=self.num_scale)
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        init_param(self.modules())

    def forward(self, x):
        res = x
        x = self.relu(self.gcn(x) + self.residual1(res))
        return self.relu(self.tcn(x) + self.residual2(res))


class DeGCNV2(nn.Module):
    def __init__(self, in_channels=8, num_classes=2, num_point=NUM_JOINTS, num_person=1, eta=4, num_stream=2,
                 drop_out=0.3, base_frame=32):
        super().__init__()
        A = spatial_graph()
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        c = 64
        blocks = [(in_channels, c, 1, False, base_frame), (c, c, 1, True, base_frame), (c, c, 1, True, base_frame),
                  (c, c, 1, True, base_frame), (c, 2 * c, 2, True, base_frame),
                  (2 * c, 2 * c, 1, True, base_frame // 2), (2 * c, 2 * c, 1, True, base_frame // 2),
                  (2 * c, 4 * c, 2, True, base_frame // 2), (4 * c, 4 * c, 1, True, base_frame // 4),
                  (4 * c, 4 * c, 1, True, base_frame // 4)]
        self.streams = nn.ModuleList(nn.Sequential(*[
            Basic_Block(i, o, A, eta, stride=s, num_frame=f, residual=r, first=index == 0)
            for index, (i, o, s, r, f) in enumerate(blocks)]) for _ in range(num_stream))
        self.fc = nn.ModuleList(nn.Linear(4 * c, num_classes) for _ in range(num_stream))
        for fc in self.fc:
            nn.init.normal_(fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (8, 32, self.num_point):
            raise ValueError("Expected (N,8,32,33)")
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)   # (N, M*V*C, T) with M = 1
        x = self.data_bn(x).view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        logits = []
        for stream, fc in zip(self.streams, self.fc):
            y = stream(x)
            y = y.view(N, y.size(1), -1).mean(2)
            logits.append(fc(self.drop_out(y)))
        return torch.stack(logits).sum(0)   # author scores with sum(output)
