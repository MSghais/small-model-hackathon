"""Bridge: align LLM hidden states with JEPA latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bridge(nn.Module):
    def __init__(self, d_llm_hidden: int, d_latent: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_llm_hidden, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(self, llm_hidden):
        return self.proj(llm_hidden.float().mean(dim=1))

    def info_nce(self, z1, z2, tau=0.07):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )
