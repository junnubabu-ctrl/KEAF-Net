"""Core neural components for KEAF-Net.

This module provides an executable, dataset-independent baseline of the fusion
and iterative reasoning path. Dataset/model-specific encoders can be plugged
in through the same tensor contract.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveKnowledgeFusion(nn.Module):
    """Fuse visual, question, and retrieved-knowledge representations.

    Inputs are `[batch, dim]`. The gate learns per-feature weights and returns
    both the fused representation and gate values for analysis.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim * 3), nn.Sigmoid())
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual: torch.Tensor, question: torch.Tensor, knowledge: torch.Tensor):
        joined = torch.cat([visual, question, knowledge], dim=-1)
        gates = self.gate(joined)
        gated = joined * gates
        fused = self.norm(self.proj(gated) + (visual + question + knowledge) / 3.0)
        return fused, gates


class MultiHopReasoner(nn.Module):
    """Iterative GRU reasoning over a pooled graph/knowledge context."""

    def __init__(self, dim: int, hops: int = 3):
        super().__init__()
        self.hops = hops
        self.cell = nn.GRUCell(dim, dim)
        self.attn = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, state: torch.Tensor, nodes: torch.Tensor):
        # nodes: [B, N, D]
        for _ in range(self.hops):
            q = state.unsqueeze(1).expand_as(nodes)
            scores = self.attn(torch.cat([nodes, q], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            context = torch.sum(nodes * weights.unsqueeze(-1), dim=1)
            state = self.cell(context, state)
        return state


class VQAPredictionHead(nn.Module):
    def __init__(self, dim: int, num_answers: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_answers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
