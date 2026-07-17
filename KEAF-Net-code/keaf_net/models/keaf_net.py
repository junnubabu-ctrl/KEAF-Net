"""KEAF-Net: the full model (Eq. 1).

    A_hat = Classifier( MHSR( HGAF( V, T, K_hat ) ) )

The forward pass is split into three reusable stages so the trainer can run the
leave-one-out loop cheaply:

    assemble_graph -> filter -> reason

``assemble_graph`` projects the pre-extracted dual-stream features into the
shared 768-d space and builds the typed graph (Algorithm 1). ``filter`` is the
AKF gate. ``reason`` runs HGAF + MHSR for a given keep-mask, so the LOO loop
only re-runs the reasoning path rather than re-extracting features.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..structures import KEAFBatch
from .akf import AdaptiveKnowledgeFilter
from .graph import build_sample_graph
from .hgaf import HGAF
from .mhsr import MHSR


class KEAFNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # input adapters: bring each stream into the shared space (Section 3.2)
        self.region_proj = nn.Linear(cfg.region_feat_dim, d)
        self.knowledge_proj = nn.Linear(cfg.knowledge_feat_dim, d)
        self.visual_norm = nn.LayerNorm(d)
        self.text_norm = nn.LayerNorm(d)
        self.knowledge_norm = nn.LayerNorm(d)

        self.akf = AdaptiveKnowledgeFilter(d, cfg.akf_threshold_init, cfg.akf_temperature)
        self.hgaf = HGAF(d, cfg.gat_heads, cfg.gat_layers, cfg.dropout, cfg.leaky_relu_slope)
        self.mhsr = MHSR(d, cfg.num_hops, cfg.num_answers, cfg.dropout)

    # ------------------------------------------------------------------ graph
    def assemble_graph(self, samples: Sequence[dict[str, Any]]) -> KEAFBatch:
        """Project per-sample features and stitch them into one disjoint graph."""
        cfg = self.cfg
        feats, types, batches, qcls = [], [], [], []
        edge_index, edge_type = [], []
        targets, node_offset = [], 0
        for b, s in enumerate(samples):
            grid = s["grid"]
            region = self.region_proj(s["region"])
            v = self.visual_norm(torch.cat([grid, region], dim=0))
            boxes = torch.cat([s["grid_boxes"], s["region_boxes"]], dim=0)
            t = self.text_norm(s["tokens"])
            k = self.knowledge_norm(self.knowledge_proj(s["triplets"]))

            nf, nt, ei, et = build_sample_graph(
                v, boxes, t, k, s["vk_pairs"], s["tk_pairs"],
                cfg.iou_threshold, cfg.token_window, cfg.visual_text_sim,
            )
            feats.append(nf)
            types.append(nt)
            batches.append(nf.new_full((nf.size(0),), b, dtype=torch.long))
            edge_index.append(ei + node_offset)
            edge_type.append(et)
            qcls.append(s["qcls"])
            if s.get("target") is not None:
                targets.append(s["target"])
            node_offset += nf.size(0)

        return KEAFBatch(
            node_feat=torch.cat(feats, dim=0),
            node_type=torch.cat(types, dim=0),
            node_batch=torch.cat(batches, dim=0),
            edge_index=torch.cat(edge_index, dim=1),
            edge_type=torch.cat(edge_type, dim=0),
            qcls=torch.stack(qcls, dim=0),
            num_graphs=len(samples),
            answer_target=torch.stack(targets, dim=0) if targets else None,
            meta=[s.get("meta", {}) for s in samples],
        )

    # ------------------------------------------------------------------ stages
    def filter(self, batch: KEAFBatch):
        """Run AKF; returns (scores, gate, node_keep, knowledge node index)."""
        return self.akf(batch.node_feat, batch.node_type, batch.node_batch, batch.qcls)

    def reason(self, batch: KEAFBatch, node_keep: torch.Tensor, return_attention: bool = False):
        """HGAF + MHSR for a given keep-mask (Eq. 14-22)."""
        fused = self.hgaf(batch.node_feat, batch.node_type, batch.edge_index, node_keep)
        return self.mhsr(fused, batch.node_batch, batch.qcls, node_keep, return_attention)

    # ------------------------------------------------------------------ forward
    def forward(self, samples: Sequence[dict[str, Any]], return_attention: bool = False) -> dict:
        batch = self.assemble_graph(samples)
        scores, gate, node_keep, k_index = self.filter(batch)
        logits, aux = self.reason(batch, node_keep, return_attention)
        return {
            "logits": logits,
            "akf_scores": scores,
            "gate": gate,
            "node_keep": node_keep,
            "knowledge_index": k_index,
            "batch": batch,
            **aux,
        }
