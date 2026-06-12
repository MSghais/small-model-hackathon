"""Retrieval memory: embedder, vector store, and adapter router."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_emb: int = 64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_emb)
        self.enc = nn.GRU(d_emb, d_emb, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_emb, d_emb)
        self.d_emb = d_emb

    def forward(self, ids):
        h, _ = self.enc(self.tok(ids))
        return F.normalize(self.proj(h.mean(dim=1)), dim=-1)


class VectorStore:
    def __init__(self):
        self.keys, self.values = [], []

    def add(self, emb, payload):
        self.keys.append(emb.squeeze(0).detach().cpu())
        self.values.append(payload)

    def search(self, q, k=2):
        if not self.keys:
            return []
        K = torch.stack(self.keys)
        sims = (q.detach().cpu() @ K.t()).squeeze(0)
        top = sims.topk(min(k, len(self.keys))).indices
        return [self.values[i] for i in top]


class Router(nn.Module):
    def __init__(self, d_emb, n_adapters):
        super().__init__()
        self.fc = nn.Linear(d_emb, n_adapters)

    def forward(self, emb):
        return self.fc(emb).argmax(dim=-1)
