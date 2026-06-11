"""Energy model: score candidate latents against world state."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyModel(nn.Module):
    def __init__(self, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_latent, 2 * d_latent),
            nn.GELU(),
            nn.Linear(2 * d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
        )
        self.d_latent = d_latent

    def energy(self, s, z):
        return self.net(torch.cat([s, z], -1)).squeeze(-1)

    def contrastive_loss(self, s, z_pos, z_negs=None, tau=0.5):
        B = s.size(0)
        s_rep = s.unsqueeze(1).expand(B, B, self.d_latent).reshape(
            B * B, self.d_latent
        )
        z_rep = z_pos.unsqueeze(0).expand(B, B, self.d_latent).reshape(
            B * B, self.d_latent
        )
        E = self.energy(s_rep, z_rep).view(B, B)
        if z_negs is not None:
            En = self.energy(
                s.repeat_interleave(z_negs.size(1), 0),
                z_negs.reshape(-1, self.d_latent),
            ).view(B, -1)
            E = torch.cat([E, En], dim=1)
        labels = torch.arange(B, device=s.device)
        return F.cross_entropy(-E / tau, labels)

    @torch.no_grad()
    def rank(self, s, candidates):
        return self.energy(s.expand(candidates.size(0), -1), candidates)
