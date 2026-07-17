"""Heterogeneous Graph Adaptive Fusion (HGAF), Section 3.5.

A multi-head graph attention network with *type-specific* projections and
attention vectors. Because the attention coefficient in Eq. (14) is indexed by
the ordered node-type pair ``(type(i), type(j))``, a visual-knowledge link and a
visual-visual link are transformed differently even when the raw features look
alike (Proposition 2). A gated fusion (Eq. 17-18) then rescales each node by a
type-specific gate before the reasoning module reads the graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..structures import NODE_TYPES
from ..utils.scatter import scatter_softmax, scatter_sum

# masked attention logits use a large finite negative so empty neighbourhoods
# do not propagate NaNs through the softmax.
_NEG_INF = -1e9


class TypedGATLayer(nn.Module):
    """One type-aware multi-head GAT layer (Eq. 14-16)."""

    def __init__(self, dim: int, heads: int, dropout: float, slope: float):
        super().__init__()
        if dim % heads:
            raise ValueError(f"hidden_dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.d_head = dim // heads
        self.dim = dim
        # W_type(i): one projection per node type, producing all heads at once
        self.proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(NODE_TYPES))
        # a_{type(i),type(j)}: one attention vector per ordered type pair and head
        self.attn = nn.Parameter(torch.empty(NODE_TYPES * NODE_TYPES, heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.attn)
        self.leaky = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)

    def _project(self, h: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        out = h.new_zeros(h.shape)
        for t, lin in enumerate(self.proj):
            mask = node_type == t
            if mask.any():
                out[mask] = lin(h[mask])
        return out.view(h.size(0), self.heads, self.d_head)

    def forward(
        self, h: torch.Tensor, node_type: torch.Tensor,
        edge_index: torch.Tensor, edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = h.size(0)
        proj = self._project(h, node_type)              # [N, H, d_head]
        src, dst = edge_index[0], edge_index[1]         # message src -> dst (j -> i)
        h_src, h_dst = proj[src], proj[dst]

        pair = node_type[dst] * NODE_TYPES + node_type[src]   # (type(i), type(j))
        a = self.attn[pair]                             # [E, H, 2*d_head]
        logits = self.leaky((a * torch.cat([h_dst, h_src], dim=-1)).sum(-1))  # Eq. (14)
        logits = logits.masked_fill(~edge_mask.unsqueeze(-1), _NEG_INF)
        alpha = self.dropout(scatter_softmax(logits, dst, n))                 # Eq. (15)

        msg = alpha.unsqueeze(-1) * h_src
        agg = scatter_sum(msg, dst, n).reshape(n, self.dim)
        return torch.sigmoid(agg)                       # Eq. (16)


class HGAF(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float, slope: float):
        super().__init__()
        self.layers = nn.ModuleList(
            TypedGATLayer(dim, heads, dropout, slope) for _ in range(layers)
        )
        self.gate = nn.ModuleList(nn.Linear(dim, dim) for _ in range(NODE_TYPES))  # Eq. (17)

    def _gated_fusion(self, h: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        gate = h.new_zeros(h.shape)
        for t, lin in enumerate(self.gate):
            mask = node_type == t
            if mask.any():
                gate[mask] = torch.sigmoid(lin(h[mask]))
        return gate * h                                  # Eq. (18)

    def forward(
        self, node_feat: torch.Tensor, node_type: torch.Tensor,
        edge_index: torch.Tensor, node_keep: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused node states; filtered knowledge nodes are zeroed out.

        ``node_keep`` is the AKF gate (1 for visual/textual, 0/1 STE gate for
        knowledge). Edges touching a discarded triplet are masked so the
        filtered evidence never enters message passing.
        """
        n = node_feat.size(0)
        loop = torch.arange(n, device=node_feat.device).expand(2, n)  # self-loops
        edge_index = torch.cat([edge_index, loop], dim=1)
        valid = node_keep > 0
        edge_mask = (valid[edge_index[0]] & valid[edge_index[1]])

        h = node_feat
        for layer in self.layers:
            h = layer(h, node_type, edge_index, edge_mask)
        h = self._gated_fusion(h, node_type)
        return h * node_keep.unsqueeze(-1)
