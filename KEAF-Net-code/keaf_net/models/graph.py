"""Heterogeneous graph construction (Algorithm 1, Section 3.5).

Each sample yields a typed graph over visual regions, question tokens and the
retained knowledge triplets. Edges follow the five rules in the paper:

    (a) visual-visual   : bounding-box IoU > 0.3
    (b) textual-textual : token distance <= 3
    (c) visual-knowledge: detector label matches a triplet entity
    (d) textual-knowledge: question noun phrase matches a triplet entity
    (e) visual-textual  : cosine similarity in the shared space > 0.4

The entity-linking matches (c, d) are resolved during preprocessing
(string then WordNet-lemma) and passed in as index pairs; the geometric and
similarity edges (a, b, e) are computed here from the features.
"""
from __future__ import annotations

import torch

from ..structures import (
    EDGE_TK,
    EDGE_TT,
    EDGE_VK,
    EDGE_VT,
    EDGE_VV,
    KNOWLEDGE,
    TEXTUAL,
    VISUAL,
)


def box_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for boxes in ``[x1, y1, x2, y2]`` format, shape [M, M]."""
    area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(dim=-1)
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp_min(1e-6)


def _undirected(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Stack a symmetric edge list so messages flow both ways."""
    s = torch.cat([src, dst])
    d = torch.cat([dst, src])
    return torch.stack([s, d], dim=0)


def build_sample_graph(
    v_feat: torch.Tensor,
    boxes: torch.Tensor,
    t_feat: torch.Tensor,
    k_feat: torch.Tensor,
    vk_pairs: torch.Tensor,
    tk_pairs: torch.Tensor,
    iou_threshold: float = 0.3,
    token_window: int = 3,
    visual_text_sim: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one sample's typed graph.

    ``vk_pairs`` / ``tk_pairs`` are ``[*, 2]`` long tensors of local
    (visual|text index, knowledge index) matches from the entity linker.
    Returns ``(node_feat, node_type, edge_index, edge_type)`` with knowledge
    nodes ordered last so the filter can address them as a contiguous block.
    """
    nv, nt, nk = v_feat.size(0), t_feat.size(0), k_feat.size(0)
    t_off, k_off = nv, nv + nt

    node_feat = torch.cat([v_feat, t_feat, k_feat], dim=0)
    node_type = torch.cat([
        node_feat.new_full((nv,), VISUAL, dtype=torch.long),
        node_feat.new_full((nt,), TEXTUAL, dtype=torch.long),
        node_feat.new_full((nk,), KNOWLEDGE, dtype=torch.long),
    ])

    edges: list[torch.Tensor] = []
    types: list[torch.Tensor] = []

    def add(ei: torch.Tensor, etype: int) -> None:
        if ei.numel() == 0:
            return
        edges.append(ei)
        types.append(ei.new_full((ei.size(1),), etype))

    # (a) visual-visual via IoU
    if nv > 1:
        iou = box_iou(boxes)
        iou.fill_diagonal_(0.0)
        vi, vj = (iou > iou_threshold).nonzero(as_tuple=True)
        add(torch.stack([vi, vj], dim=0), EDGE_VV)

    # (b) textual-textual via a sliding window
    if nt > 1:
        idx = torch.arange(nt, device=t_feat.device)
        ti, tj = torch.meshgrid(idx, idx, indexing="ij")
        win = (ti - tj).abs() <= token_window
        win &= ti != tj
        si, sj = win.nonzero(as_tuple=True)
        add(torch.stack([si + t_off, sj + t_off], dim=0), EDGE_TT)

    # (e) visual-textual via cosine similarity
    if nv and nt:
        vn = torch.nn.functional.normalize(v_feat, dim=-1)
        tn = torch.nn.functional.normalize(t_feat, dim=-1)
        sim = vn @ tn.t()
        vi, tj = (sim > visual_text_sim).nonzero(as_tuple=True)
        add(_undirected(vi, tj + t_off), EDGE_VT)

    # (c) visual-knowledge and (d) textual-knowledge from the entity linker
    if vk_pairs.numel():
        add(_undirected(vk_pairs[:, 0], vk_pairs[:, 1] + k_off), EDGE_VK)
    if tk_pairs.numel():
        add(_undirected(tk_pairs[:, 0] + t_off, tk_pairs[:, 1] + k_off), EDGE_TK)

    if edges:
        edge_index = torch.cat(edges, dim=1)
        edge_type = torch.cat(types)
    else:  # isolated nodes (e.g. no retained knowledge) still form a valid graph
        edge_index = node_feat.new_zeros((2, 0), dtype=torch.long)
        edge_type = node_feat.new_zeros((0,), dtype=torch.long)
    return node_feat, node_type, edge_index, edge_type
