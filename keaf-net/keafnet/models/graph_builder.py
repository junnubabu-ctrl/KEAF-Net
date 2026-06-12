"""
Heterogeneous graph construction.

Builds the typed adjacency tensor consumed by HGAF from visual, textual, and
knowledge node features. Five edge types are produced:

    0 V-V : visual-visual   (region IoU > iou_thresh, or kNN on features)
    1 T-T : textual-textual (sliding window of size `window`)
    2 V-K : visual-knowledge(cosine similarity > link_thresh)
    3 T-K : textual-knowledge(cosine similarity > link_thresh)
    4 V-T : visual-textual  (cosine similarity > sim_thresh)

In a full system the V-K / T-K edges come from entity linking between detector
labels / noun phrases and triplet entities. Here we approximate that linkage by
embedding cosine similarity, which keeps the module self-contained and
differentiable-friendly while preserving the five-edge-type structure.

Reference: KEAF-Net, Section 3.5.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .hgaf import NUM_EDGE_TYPES


def _cosine_block(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.bmm(a, b.transpose(1, 2))  # (B, Na, Nb)


def build_hetero_graph(v: torch.Tensor, t: torch.Tensor, k: torch.Tensor,
                       boxes: torch.Tensor | None = None,
                       window: int = 3, iou_thresh: float = 0.3,
                       sim_thresh: float = 0.4, link_thresh: float = 0.4):
    """Assemble nodes and a typed adjacency tensor.

    Args:
        v: (B, Nv, d) visual node features.
        t: (B, Nt, d) textual node features.
        k: (B, Nk, d) knowledge node features (filtered triplets).
        boxes: (B, Nv, 4) optional region boxes for IoU-based V-V edges.
        window: T-T sliding-window radius.
        *_thresh: similarity / IoU thresholds for the respective edge types.
    Returns:
        nodes:       (B, N, d) concatenated [V; T; K].
        adj_by_type: (B, R, N, N) binary adjacency per edge type.
        type_ids:    (B, N) node type in {0,1,2}.
    """
    b, nv, d = v.shape
    nt = t.shape[1]
    nk = k.shape[1]
    n = nv + nt + nk
    device = v.device

    nodes = torch.cat([v, t, k], dim=1)
    type_ids = torch.cat([
        torch.zeros(nv, device=device),
        torch.ones(nt, device=device),
        torch.full((nk,), 2.0, device=device),
    ]).long().unsqueeze(0).expand(b, n).contiguous()

    adj = torch.zeros(b, NUM_EDGE_TYPES, n, n, device=device)

    v0, t0, k0 = 0, nv, nv + nt  # offsets

    # --- edge type 0: V-V ---
    if boxes is not None:
        iou = _pairwise_iou(boxes)  # (B, Nv, Nv)
        vv = (iou > iou_thresh).float()
    else:
        sim_vv = _cosine_block(v, v)
        vv = (sim_vv > sim_thresh).float()
    adj[:, 0, v0:v0 + nv, v0:v0 + nv] = vv

    # --- edge type 1: T-T (sliding window) ---
    idx = torch.arange(nt, device=device)
    tt = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= window).float()
    adj[:, 1, t0:t0 + nt, t0:t0 + nt] = tt.unsqueeze(0).expand(b, nt, nt)

    # --- edge type 2: V-K ---
    if nk > 0:
        sim_vk = _cosine_block(v, k)
        vk = (sim_vk > link_thresh).float()
        adj[:, 2, v0:v0 + nv, k0:k0 + nk] = vk
        adj[:, 2, k0:k0 + nk, v0:v0 + nv] = vk.transpose(1, 2)

        # --- edge type 3: T-K ---
        sim_tk = _cosine_block(t, k)
        tk = (sim_tk > link_thresh).float()
        adj[:, 3, t0:t0 + nt, k0:k0 + nk] = tk
        adj[:, 3, k0:k0 + nk, t0:t0 + nt] = tk.transpose(1, 2)

    # --- edge type 4: V-T ---
    sim_vt = _cosine_block(v, t)
    vt = (sim_vt > sim_thresh).float()
    adj[:, 4, v0:v0 + nv, t0:t0 + nt] = vt
    adj[:, 4, t0:t0 + nt, v0:v0 + nv] = vt.transpose(1, 2)

    # Add self-loops on every node (helps message passing stability).
    eye = torch.eye(n, device=device).unsqueeze(0).expand(b, n, n)
    adj[:, 0] = adj[:, 0] + eye  # carried by the V-V slice; harmless duplication
    return nodes, adj, type_ids


def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    """boxes: (B, N, 4) in (x1, y1, x2, y2). Returns (B, N, N) IoU."""
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    area = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)

    lt_x = torch.maximum(x1.unsqueeze(-1), x1.unsqueeze(-2))
    lt_y = torch.maximum(y1.unsqueeze(-1), y1.unsqueeze(-2))
    rb_x = torch.minimum(x2.unsqueeze(-1), x2.unsqueeze(-2))
    rb_y = torch.minimum(y2.unsqueeze(-1), y2.unsqueeze(-2))
    inter = (rb_x - lt_x).clamp_min(0) * (rb_y - lt_y).clamp_min(0)
    union = area.unsqueeze(-1) + area.unsqueeze(-2) - inter
    return inter / union.clamp_min(1e-6)
