"""Latent world model: multi-step rollout in JEPA space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(self, d_latent: int):
        super().__init__()
        self.cell = nn.GRUCell(d_latent, d_latent)
        self.head = nn.Linear(d_latent, d_latent)
        self.s0 = nn.Parameter(torch.zeros(d_latent))
        self.d_latent = d_latent

    def init_state(self, B, device):
        return self.s0.unsqueeze(0).expand(B, -1).contiguous().to(device)

    def step(self, s, z):
        s = self.cell(z, s)
        return s, self.head(s)

    def rollout(self, s, horizon):
        preds = []
        for _ in range(horizon):
            z_hat = self.head(s)
            preds.append(z_hat)
            s = self.cell(z_hat, s)
        return torch.stack(preds, 1), s

    def sequence_loss(self, z_seq):
        B, T, _ = z_seq.shape
        s = self.init_state(B, z_seq.device)
        loss = 0.0
        for t in range(T - 1):
            s, z_hat = self.step(s, z_seq[:, t])
            loss = loss + F.mse_loss(z_hat, z_seq[:, t + 1])
        return loss / (T - 1)
