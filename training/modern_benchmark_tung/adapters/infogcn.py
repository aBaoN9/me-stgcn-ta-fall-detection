"""Full-clip InfoGCN++ small with causal graph encoder and RK4 latent extrapolation."""
import torch
from torch import nn

from training.component_ablation.models import make_adjacency


class Model(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        width = 24
        self.register_buffer("graph", make_adjacency())
        self.register_buffer("causal", torch.ones(32, 32, dtype=torch.bool).triu(1),
                             persistent=False)
        self.stem = nn.Linear(8, width)
        self.joint = nn.Parameter(torch.randn(1, 1, 33, width) * 0.02)
        self.time = nn.Parameter(torch.randn(1, 32, 1, width) * 0.02)
        self.query = nn.Linear(width, 8)
        self.key = nn.Linear(width, 8)
        self.mix = nn.Linear(width, width)
        self.encoder = nn.TransformerEncoderLayer(width, 4, width * 2, dropout,
                                                  activation="gelu", batch_first=True, norm_first=True)
        self.ode_1 = nn.Linear(width, width)
        self.ode_2 = nn.Linear(width, width)
        self.ode_out = nn.Linear(width, width)
        self.latent_mean = nn.Linear(width, width)
        self.latent_scale = nn.Linear(width, width)
        self.decoder = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 2))

    def field(self, time, z):
        h = torch.nn.functional.leaky_relu(self.ode_1(self.graph @ (z + time * self.time)))
        h = torch.nn.functional.leaky_relu(self.ode_2(self.graph @ h))
        return self.ode_out(h)

    def extrapolate(self, z):
        k1 = self.field(0, z)
        k2 = self.field(0.5, z + 0.5 * k1)
        k3 = self.field(0.5, z + 0.5 * k2)
        k4 = self.field(1, z + k3)
        return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def forward(self, x):
        batch = x.shape[0]
        x = self.stem(x.permute(0, 2, 3, 1)) + self.joint + self.time
        dynamic = (self.query(x) @ self.key(x).transpose(-1, -2) / 8 ** 0.5).softmax(-1)
        x = x + self.mix((self.graph + dynamic) @ x)
        seq = x.transpose(1, 2).reshape(batch * 33, 32, -1)
        encoded = self.encoder(seq, src_mask=self.causal)
        encoded = encoded.reshape(batch, 33, 32, -1).transpose(1, 2)
        z = self.latent_mean(encoded)
        if self.training:
            scale = (0.5 * self.latent_scale(encoded).clamp(-8, 4)).exp()
            z = z + scale * torch.randn_like(z)
        future = self.extrapolate(z)
        enhanced = self.decoder(self.graph @ torch.cat((z, future), dim=-1))
        return self.head(enhanced.mean((1, 2)))
