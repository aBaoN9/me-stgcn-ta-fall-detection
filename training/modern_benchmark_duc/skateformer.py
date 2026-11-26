"""SkateFormer adapted to the GMDCSA-24 v2 contract (CE-adapted, author-size, not 'small').

Ported from KAIST-VICLab/SkateFormer, model/SkateFormer.py at commit
0249e19d96b33e9150760129bbede56464126ae7 (MIT License, Copyright (c) 2024 KAIST-VICLab; licence text in
THIRD_PARTY_NOTICES.md). Partition/reverse functions, the relative bias, the block (G-Conv, T-Conv,
four Skate-MSA branches, MLP) and the stage layout are kept. Changes, all forced by the input contract:

* input (N,8,32,33), one person, 8-channel stem (author formula C -> 2C -> 3C -> embed_dim);
* 33 MediaPipe joints. G-Conv, T-Conv, LayerNorm, MLP, downsampling BN and pooling run on the 33 real
  joints only. Skate-MSA needs equal parts, so the attention branch scatters joints into six parts of six
  (graph.SKATE_PARTS, 3 padded slots), masks padded keys with -inf and drops padded queries afterwards;
* temporal partition sizes N = M = 4 (author N-UCLA: 8) because T = 32 is halved by three stages to 4;
* index_t encoding uses t = 0..31 (the 32 landmarks); joint embedding has 33 columns;
* dropout positions (attention, projection/MLP, head) all p = 0.3; drop-path 0; timm replaced by torch
  equivalents (trunc_normal_ with absolute bounds, Mlp = fc-GELU-drop-fc-drop);
* two-class linear head; returns logits only.
"""
import math

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from training.modern_benchmark_duc.graph import NUM_JOINTS, skate_layout


def type_1_partition(input, partition_size):  # partition_size = [N, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    return partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)


def type_1_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    return partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)


def type_2_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    return partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)


def type_3_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    return partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)


def type_4_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)


PARTITIONS = (type_1_partition, type_2_partition, type_3_partition, type_4_partition)
REVERSES = (type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse)


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super().__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5  # author's scaling, kept
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.rel = rel
        if self.rel:
            if rel_type in ("type_1", "type_3"):
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
            else:
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads))
            self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        n0, n1 = self.partition_size
        if self.rel_type in ("type_1", "type_3"):
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n0, n0, -1)
            bias = bias.unsqueeze(1).unsqueeze(3).repeat(1, n1, 1, n1, 1, 1).view(self.attn_area, self.attn_area, -1)
        else:
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n0, n0, n1, n1, -1)
            bias = bias.permute(0, 2, 1, 3, 4).contiguous().view(self.attn_area, self.attn_area, -1)
        return bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, input, key_padding):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = attn.masked_fill(key_padding[:, None, None, :], float("-inf"))  # padded joint slots never attended
        attn = self.attn_drop(attn.softmax(dim=-1))
        return (attn @ v).transpose(1, 2).reshape(B_, N, -1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop):
        super().__init__()
        self.fc1, self.act, self.drop1 = nn.Linear(in_features, hidden_features), nn.GELU(), nn.Dropout(drop)
        self.fc2, self.drop2 = nn.Linear(hidden_features, in_features), nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, sizes, kernel_size=7, num_heads=32, attn_drop=0., drop=0., rel=True, mlp_ratio=4.):
        super().__init__()
        self.partition_size = sizes
        self.norm_1 = nn.LayerNorm(in_channels)
        self.mapping = nn.Linear(in_channels, 2 * in_channels, bias=True)
        self.gconv = nn.Parameter(torch.zeros(num_heads // 4, NUM_JOINTS, NUM_JOINTS))
        trunc_normal_(self.gconv, std=.02)
        self.tconv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0), groups=num_heads // 4)
        self.attention = nn.ModuleList(
            MultiHeadSelfAttention(in_channels // 8, f"type_{i + 1}", num_heads // 8, sizes[i], attn_drop, rel)
            for i in range(4))
        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.norm_2 = nn.LayerNorm(in_channels)
        self.mlp = Mlp(in_channels, int(mlp_ratio * in_channels), drop)
        layout = torch.from_numpy(skate_layout())
        self.register_buffer("gather_index", layout.clamp(min=0), persistent=False)
        self.register_buffer("slot_real", layout >= 0, persistent=False)
        slot_of_joint = torch.empty(NUM_JOINTS, dtype=torch.long)
        slot_of_joint[layout[layout >= 0]] = torch.nonzero(layout >= 0).flatten()
        self.register_buffer("slot_of_joint", slot_of_joint, persistent=False)

    def key_padding(self, index, frames, batch):
        flags = (~self.slot_real).float().view(1, 1, 1, -1).expand(1, 1, frames, -1).contiguous()
        size = self.partition_size[index]
        groups = PARTITIONS[index](flags, size).view(-1, size[0] * size[1]) > 0.5
        if groups.all(dim=1).any():
            raise ValueError("A Skate-MSA group contains only padded joints")
        return groups.repeat(batch, 1)

    def forward(self, input):
        B, C, T, V = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()
        skip = input
        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()
        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []
        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        y.append(torch.cat([torch.einsum("n c t u, v u -> n c t v", split_f_gconv[i], self.gconv[i])
                            for i in range(self.gconv.shape[0])], dim=1))
        y.append(self.tconv(split_f_conv[1]))
        padded = f_attn[..., self.gather_index] * self.slot_real.to(f_attn.dtype)  # (B, 3C/2, T, 36)
        width = padded.shape[-1]
        for i, branch in enumerate(torch.chunk(padded, 4, dim=1)):
            channels, size = branch.shape[1], self.partition_size[i]
            parts = PARTITIONS[i](branch, size).view(-1, size[0] * size[1], channels)
            out = REVERSES[i](self.attention[i](parts, self.key_padding(i, T, B)), (T, width), size)
            y.append(out[..., self.slot_of_joint])  # drop padded queries, back to 33 joints
        output = self.proj_drop(self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous()))
        output = skip + output
        output = output + self.mlp(self.norm_2(output))
        return output.permute(0, 3, 1, 2).contiguous()


class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0),
                                   stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.reduction(x))


class SkateFormerStage(nn.Module):
    def __init__(self, depth, in_channels, out_channels, first_depth, sizes, kernel_size, num_heads,
                 attn_drop, drop, rel, mlp_ratio):
        super().__init__()
        self.downsample = None if first_depth else PatchMergingTconv(in_channels, out_channels, kernel_size)
        self.blocks = nn.ModuleList(SkateFormerBlock(out_channels, sizes, kernel_size, num_heads, attn_drop, drop,
                                                     rel, mlp_ratio) for _ in range(depth))

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class SkateFormerV2(nn.Module):
    def __init__(self, in_channels=8, num_classes=2, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192),
                 embed_dim=96, num_frames=32, kernel_size=7, num_heads=32,
                 partition=((4, 6), (4, 6), (4, 6), (4, 6)), attn_drop=0.3, head_drop=0.3, drop=0.3,
                 rel=True, mlp_ratio=1.0):
        super().__init__()
        self.embed_dim, self.num_frames = embed_dim, num_frames
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, 1), nn.GELU(),
                                  nn.Conv2d(2 * in_channels, 3 * in_channels, 1), nn.GELU(),
                                  nn.Conv2d(3 * in_channels, embed_dim, 1))
        self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, NUM_JOINTS))
        trunc_normal_(self.joint_person_embedding, std=.02)
        self.stages = nn.ModuleList(
            SkateFormerStage(depth, embed_dim if index == 0 else channels[index - 1], channel, index == 0,
                             partition, kernel_size, num_heads, attn_drop, drop, rel, mlp_ratio)
            for index, (depth, channel) in enumerate(zip(depths, channels)))
        self.dropout = nn.Dropout(head_drop)
        self.head = nn.Linear(channels[-1], num_classes)

    def temporal_encoding(self, frames, device):
        index_t = torch.arange(frames, device=device, dtype=torch.float32)
        te = torch.zeros(frames, self.embed_dim, device=device)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device)
                             * -(math.log(10000.0) / self.embed_dim))
        te[:, 0::2] = torch.sin(index_t.unsqueeze(-1) * div_term)
        te[:, 1::2] = torch.cos(index_t.unsqueeze(-1) * div_term)
        return te

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (8, self.num_frames, NUM_JOINTS):
            raise ValueError("Expected (N,8,32,33)")
        output = self.stem(x)
        te = self.temporal_encoding(x.shape[2], x.device)
        output = output + torch.einsum("t c, c v -> c t v", te, self.joint_person_embedding).unsqueeze(0)
        for stage in self.stages:
            output = stage(output)
        return self.head(self.dropout(output.mean(dim=(2, 3))))
