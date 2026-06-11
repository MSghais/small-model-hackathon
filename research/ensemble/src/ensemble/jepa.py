"""JEPA latent predictor with EMA target encoder."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SegEncoder(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.enc = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, d)

    def forward(self, ids):
        h, _ = self.enc(self.tok(ids))
        return self.out(h.mean(dim=1))


class JEPA(nn.Module):
    def __init__(self, vocab_size: int, d_latent: int = 64, ema_m: float = 0.996):
        super().__init__()
        self.ctx_enc = _SegEncoder(vocab_size, d_latent)
        self.tgt_enc = copy.deepcopy(self.ctx_enc)
        for p in self.tgt_enc.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, 2 * d_latent),
            nn.GELU(),
            nn.Linear(2 * d_latent, d_latent),
        )
        self.m = ema_m
        self.d_latent = d_latent

    @property
    def enc(self):
        """Alias used by world-model track."""
        return self.ctx_enc

    @property
    def tgt(self):
        return self.tgt_enc

    @property
    def pred(self):
        return self.predictor

    @torch.no_grad()
    def ema_update(self):
        for p_t, p_c in zip(self.tgt_enc.parameters(), self.ctx_enc.parameters()):
            p_t.mul_(self.m).add_(p_c.detach(), alpha=1 - self.m)

    def ema(self):
        """Alias used by world-model track."""
        self.ema_update()

    def loss(self, seg_ctx, seg_tgt):
        z_hat = self.predictor(self.ctx_enc(seg_ctx))
        with torch.no_grad():
            z_tgt = self.tgt_enc(seg_tgt)
        pred = F.mse_loss(z_hat, z_tgt)
        var_reg = F.relu(1.0 - z_hat.std(dim=0)).mean()
        return pred + 0.5 * var_reg

    @torch.no_grad()
    def predict_next_latent(self, seg_ctx):
        return self.predictor(self.ctx_enc(seg_ctx))

    @torch.no_grad()
    def encode(self, seg):
        return self.tgt_enc(seg)
