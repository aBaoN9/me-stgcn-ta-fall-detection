"""JT-GraphFormer adapted to the GMDCSA-24 v2 contract (CE-adapted, author-size).

Ported from ZhengYaolin/JT-GraphFormer, model/gf2_new.py + model/pos_embed.py at commit
71fb81d177c151f029403cff9b7f2b0d93f8e01c (MIT License, Copyright (c) 2023 Yaolin Zheng; text in
THIRD_PARTY_NOTICES.md). The joint-trajectory graph (len_parts consecutive frames folded into the node axis),
the Spatio-Temporal Dijkstra Attention prior exp(-D)/11 + b, the 8 Graphformer blocks
[[64,64,16],[64,64,16],[64,128,32],[128,128,32],[128,256,64],[256,256,64]x3], 3 heads, kernel [3,5],
use_pes and the plain fc head (koopman = 'none', as the author's NTU config) are kept. Changes:

* input (N,8,32,33): num_channels 8, num_frames 32, len_parts 8 (author N-UCLA value; 32/8 = 4 groups),
  one person, JT graph of 33 x 8 = 264 nodes;
* Dijkstra matrix D = hop distance on the A3 MediaPipe edges (graph.py) instead of the NTU/N-UCLA tables,
  plus |i-j| for frame offsets inside a part, exactly as the author builds the block matrix. MediaPipe
  pose has three graph components; unreachable pairs get prior weight exp(-inf) = 0 (learnable b and att0s
  still connect them);
* the prior is a registered buffer (the author moved it to CUDA inside forward);
* every author dropout position (att_drop, dropout2d, dropout) p = 0.3; no Koopman head or extra loss.
"""
import math

import numpy as np
import torch
from torch import nn

from training.modern_benchmark_duc.graph import NUM_JOINTS, shortest_paths


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class Pos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()
        position = torch.from_numpy(np.array([st for _ in range(num_frames) for st in range(num_joints)])).unsqueeze(1).float()
        pe = torch.zeros(num_frames * num_joints, channels)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :, :x.size(2)]


class Graphformer_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim, num_frames, num_joints, num_heads, kernel_size,
                 use_pes=True, att_drop=0.):
        super().__init__()
        self.qkv_dim, self.in_channels, self.num_heads, self.use_pes = qkv_dim, in_channels, num_heads, use_pes
        pads, padt = int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2)
        if use_pes:
            self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints)
        self.out_nets = nn.Sequential(nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
                                      nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.tcn = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
                                 nn.BatchNorm2d(out_channels))
        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress, self.rest = nn.Identity(), nn.Identity()
        self.tan, self.relu, self.drop = nn.Tanh(), nn.LeakyReLU(0.1), nn.Dropout(att_drop)

    def forward(self, x, atten_weight):
        N, C, T, V = x.size()
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan(torch.einsum("nhctu,nhctv->nhuv", [q, k]) / (self.qkv_dim * T)) * self.alphas
        attention = attention * atten_weight.unsqueeze(0).unsqueeze(0) + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum("nctu,nhuv->nhctv", [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)
        return self.relu(self.tcn(xs) + self.rest(xs))


AUTHOR_CONFIG = ((64, 64, 16), (64, 64, 16), (64, 128, 32), (128, 128, 32),
                 (128, 256, 64), (256, 256, 64), (256, 256, 64), (256, 256, 64))


class JTGraphFormerV2(nn.Module):
    def __init__(self, num_channels=8, num_classes=2, num_joints=NUM_JOINTS, num_frames=32, len_parts=8,
                 num_heads=3, kernel_size=(3, 5), use_pes=True, config=AUTHOR_CONFIG, att_drop=0.3, dropout=0.3,
                 dropout2d=0.3):
        super().__init__()
        if num_frames % len_parts:
            raise ValueError("len_parts must divide num_frames")
        self.len_parts, self.out_channels = len_parts, config[-1][1]
        self.num_frames, self.num_joints = num_frames // len_parts, num_joints * len_parts
        self.base_joints = num_joints
        in_channels = config[0][0]
        self.input_map = nn.Sequential(nn.Conv2d(num_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
                                       nn.LeakyReLU(0.1))
        hops = torch.from_numpy(shortest_paths()).float()   # inf between graph components
        weight = torch.zeros(self.num_joints, self.num_joints)
        for i in range(len_parts):
            for j in range(len_parts):
                weight[i * num_joints:(i + 1) * num_joints, j * num_joints:(j + 1) * num_joints] = hops + abs(i - j)
        self.register_buffer("spatial_pos_weight", torch.exp(-weight))   # graphw_norm = False
        self.b = nn.Parameter(torch.ones(self.num_joints, self.num_joints) / num_joints)
        self.blocks = nn.ModuleList(Graphformer_Block(i, o, q, self.num_frames, self.num_joints, num_heads, kernel_size,
                                                      use_pes, att_drop) for i, o, q in config)
        self.fc = nn.Linear(self.out_channels, num_classes)
        self.drop_out, self.drop_out2d = nn.Dropout(dropout), nn.Dropout2d(dropout2d)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (8, self.num_frames * self.len_parts, self.base_joints):
            raise ValueError("Expected (N,8,32,33)")
        N, C, T, V = x.shape
        x = x.reshape(N, C, T // self.len_parts, V * self.len_parts)   # joint-trajectory nodes
        x = self.input_map(x)
        att_weight = self.spatial_pos_weight / 11 + self.b
        for block in self.blocks:
            x = block(x, att_weight)
        x = x.view(N, 1, self.out_channels, -1).permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)
        return self.fc(self.drop_out(x))
