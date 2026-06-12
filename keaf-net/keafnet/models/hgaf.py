"""
Heterogeneous Graph Adaptive Fusion (HGAF).

Folds visual regions, question tokens, and surviving knowledge triplets into a
single typed graph with five edge categories, then applies type-aware
multi-head graph attention followed by a gated fusion of the three modality
streams.

Edge types (R = 5):
    0: Visual-Visual    (spatial,    region IoU > 0.3)
    1: Textual-Textual  (sequential, sliding window 3)
    2: Visual-Knowledge (entity link)
    3: Textual-Knowledge(entity link)
    4: Visual-Textual   (cosine similarity > 0.4)

Reference: KEAF-Net, Section 3.5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EDGE_TYPES = 5


class TypeAwareGATLayer(nn.Module):
    """A single multi-head graph-attention layer with edge-type-specific
    projection matrices W_{tau}. Each edge type owns its own linear map, which
    lets the layer distinguish neighbour multisets that a type-agnostic GAT
    would collapse (Proposition 2 in the paper).
    """

    def __init__(self, dim: int, heads: int = 8, num_edge_types: int = NUM_EDGE_TYPES,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_edge_types = num_edge_types

        # One projection per edge type (applied to source & destination nodes).
        self.type_proj = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_edge_types)]
        )
        # Per-head attention vector a, shared across edge types.
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, adj_by_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:           (B, N, d) node features.
            adj_by_type: (B, R, N, N) binary adjacency, one slice per edge type.
        Returns:
            (B, N, d) updated node features.
        """
        b, n, d = h.shape
        # Aggregate messages per edge type, then sum across types.
        agg = torch.zeros_like(h)
        # Combine all edge types into a single attention bias mask.
        # For each edge type, project nodes and compute attention logits.
        combined_logits = h.new_full((b, self.heads, n, n), float("-inf"))
        proj_cache = []
        any_edge = (adj_by_type.sum(dim=1) > 0)  # (B, N, N)

        for r in range(self.num_edge_types):
            hr = self.type_proj[r](h)  # (B, N, d)
            hr = hr.view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
            proj_cache.append(hr)
            a_src, a_dst = self.attn[:, :self.head_dim], self.attn[:, self.head_dim:]
            # (B, H, N) source / destination contributions
            e_src = (hr * a_src.view(1, self.heads, 1, self.head_dim)).sum(-1)
            e_dst = (hr * a_dst.view(1, self.heads, 1, self.head_dim)).sum(-1)
            logits = self.leaky_relu(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2))
            mask_r = adj_by_type[:, r].unsqueeze(1).bool()  # (B,1,N,N)
            combined_logits = torch.where(mask_r, logits, combined_logits)

        # Softmax over neighbours (last dim). Rows with no edges -> uniform zero.
        attn = torch.softmax(combined_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Weighted sum of type-projected neighbour features (mean over types
        # that connect each pair is approximated by summing then normalising).
        msg = torch.zeros(b, self.heads, n, self.head_dim, device=h.device, dtype=h.dtype)
        for r in range(self.num_edge_types):
            mask_r = adj_by_type[:, r].unsqueeze(1).bool()
            attn_r = attn * mask_r
            msg = msg + torch.matmul(attn_r, proj_cache[r])

        out = msg.permute(0, 2, 1, 3).reshape(b, n, d)
        out = self.out_proj(out)
        return F.elu(out)


class GatedFusion(nn.Module):
    """Combines visual, textual, and knowledge node streams with learned gates:
        g_m = sigmoid(W_gm h_m + b_gm);  h_fused = sum_m g_m ⊙ h_m.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_v = nn.Linear(dim, dim)
        self.gate_t = nn.Linear(dim, dim)
        self.gate_k = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h_v: torch.Tensor, h_t: torch.Tensor, h_k: torch.Tensor) -> torch.Tensor:
        g_v = torch.sigmoid(self.gate_v(h_v))
        g_t = torch.sigmoid(self.gate_t(h_t))
        g_k = torch.sigmoid(self.gate_k(h_k))
        fused = g_v * h_v + g_t * h_t + g_k * h_k
        return self.out(fused)


class HGAF(nn.Module):
    """Heterogeneous Graph Adaptive Fusion module.

    Args:
        dim: hidden dimension.
        heads: attention heads (paper uses 8).
        layers: number of GAT layers (paper uses 2).
    """

    def __init__(self, dim: int = 768, heads: int = 8, layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TypeAwareGATLayer(dim, heads, NUM_EDGE_TYPES, dropout) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.fusion = GatedFusion(dim)

    def forward(self, h: torch.Tensor, adj_by_type: torch.Tensor,
                type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:           (B, N, d) initial node features (V, T, K concatenated).
            adj_by_type: (B, R, N, N) typed adjacency.
            type_ids:    (B, N) node-type id in {0:visual, 1:textual, 2:knowledge}.
        Returns:
            (B, N, d) fused node features.
        """
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h, adj_by_type))

        # Split node streams by type and fuse (masked means broadcast back).
        def stream(t):
            m = (type_ids == t).float().unsqueeze(-1)  # (B, N, 1)
            return h * m

        h_v, h_t, h_k = stream(0), stream(1), stream(2)
        fused = self.fusion(h_v, h_t, h_k)
        return h + fused
