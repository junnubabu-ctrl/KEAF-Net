"""
Multi-Hop Semantic Reasoning (MHSR).

Refines a query vector across T iterative GRU-based hops over the fused
heterogeneous graph representation. Each hop attends over all nodes, builds a
context vector, and updates the query with a GRU cell, enabling the model to
chain evidence across multiple reasoning steps.

Reference: KEAF-Net, Section 3.6.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSR(nn.Module):
    """Multi-hop reasoning over fused graph nodes.

    Args:
        dim: hidden dimension.
        hops: number of reasoning hops T (paper default 3).
    """

    def __init__(self, dim: int = 768, hops: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.hops = hops

        # Bilinear attention scoring: beta_i = softmax(q^T W h_i).
        self.attn_w = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, q0: torch.Tensor, nodes: torch.Tensor,
                node_mask: torch.Tensor | None = None):
        """
        Args:
            q0:        (B, d) initial query (from BERT [CLS]).
            nodes:     (B, N, d) fused node features from HGAF.
            node_mask: (B, N) optional 1/0 mask of valid nodes.
        Returns:
            q_final: (B, d) refined query after T hops.
            ctx:     (B, d) final context vector.
            attn_history: list of (B, N) attention maps, one per hop.
        """
        q = q0
        ctx = torch.zeros_like(q0)
        attn_history = []
        for _ in range(self.hops):
            # beta_i = softmax(q^T W h_i)
            scores = torch.einsum("bd,bnd->bn", self.attn_w(q), nodes)
            if node_mask is not None:
                scores = scores.masked_fill(node_mask == 0, float("-inf"))
            beta = torch.softmax(scores, dim=-1)  # (B, N)
            beta = torch.nan_to_num(beta, nan=0.0)
            attn_history.append(beta)

            ctx = torch.einsum("bn,bnd->bd", beta, nodes)  # (B, d)
            q = self.gru(ctx, q)  # update query

        return q, ctx, attn_history
