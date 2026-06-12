"""
KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network.

End-to-end model wiring the dual-stream encoders, the Adaptive Knowledge Filter
(AKF), the Heterogeneous Graph Adaptive Fusion (HGAF) module, the Multi-Hop
Semantic Reasoning (MHSR) module, and the answer classifier.

    A_hat = Classifier(MHSR(HGAF(V, T, K_hat)))

Reference: KEAF-Net, Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .akf import AdaptiveKnowledgeFilter
from .encoders import TextEncoder, VisualEncoder
from .graph_builder import build_hetero_graph
from .hgaf import HGAF
from .mhsr import MHSR


@dataclass
class KEAFConfig:
    dim: int = 768
    num_answers: int = 3129
    num_regions: int = 36
    region_dim: int = 2048
    triplet_dim: int = 384          # Sentence-BERT all-MiniLM-L6-v2 dim
    max_triplets: int = 50          # P
    gat_heads: int = 8
    gat_layers: int = 2
    mhsr_hops: int = 3              # T
    akf_threshold: float = 0.5
    akf_temperature: float = 0.1    # tau
    loss_lambda: float = 0.3        # weight on L_AKF
    pretrained: bool = True


class Classifier(nn.Module):
    def __init__(self, dim: int, num_answers: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_answers),
        )

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, ctx], dim=-1))


class KEAFNet(nn.Module):
    def __init__(self, cfg: KEAFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim

        self.visual = VisualEncoder(d, cfg.num_regions, cfg.region_dim,
                                    pretrained=cfg.pretrained)
        self.text = TextEncoder(d, pretrained=cfg.pretrained)
        self.triplet_proj = nn.Linear(cfg.triplet_dim, d)

        self.akf = AdaptiveKnowledgeFilter(d, cfg.akf_threshold, cfg.akf_temperature)
        self.hgaf = HGAF(d, cfg.gat_heads, cfg.gat_layers)
        self.mhsr = MHSR(d, cfg.mhsr_hops)
        self.classifier = Classifier(d, cfg.num_answers)

    def encode(self, batch: dict):
        """Run the dual-stream encoders and project triplet embeddings."""
        v = self.visual(
            images=batch.get("images"),
            region_feats=batch.get("region_feats"),
            grid_feats=batch.get("grid_feats"),
        )
        t, q_cls = self.text(batch["input_ids"], batch.get("attention_mask"))
        k = self.triplet_proj(batch["triplet_feats"])  # (B, P, d)
        return v, t, q_cls, k

    def reason(self, v, t, q_cls, k, triplet_mask, boxes):
        """AKF -> graph -> HGAF -> MHSR -> classifier. Returns logits & aux."""
        filtered, alpha, keep = self.akf(k, q_cls, v, mask=triplet_mask)

        nodes, adj, type_ids = build_hetero_graph(v, t, filtered, boxes=boxes)
        fused = self.hgaf(nodes, adj, type_ids)

        q_final, ctx, attn = self.mhsr(q_cls, fused)
        logits = self.classifier(q_final, ctx)
        return logits, {"alpha": alpha, "keep": keep, "attn": attn}

    def forward(self, batch: dict):
        v, t, q_cls, k = self.encode(batch)
        logits, aux = self.reason(
            v, t, q_cls, k,
            triplet_mask=batch.get("triplet_mask"),
            boxes=batch.get("boxes"),
        )
        return logits, aux

    # ------------------------------------------------------------------ losses

    def vqa_loss(self, logits: torch.Tensor, soft_scores: torch.Tensor) -> torch.Tensor:
        """Soft-target binary cross-entropy over the answer vocabulary (the
        standard VQA loss). `soft_scores` is (B, num_answers) in [0, 1].
        """
        return F.binary_cross_entropy_with_logits(
            logits, soft_scores, reduction="mean"
        ) * logits.shape[1]

    @torch.no_grad()
    def loo_deltas(self, batch: dict, v, t, q_cls, k, sample_size: int = 10):
        """Estimate leave-one-out loss differences Delta_j for a random subset
        of triplets, used as AKF supervision. For each sampled triplet j,
        Delta_j = L(K_hat \\ {k_j}) - L(K_hat).

        This is a reference (un-vectorised) implementation favouring clarity;
        production code would batch the masked forward passes.
        """
        soft = batch["answer_scores"]
        triplet_mask = batch.get("triplet_mask")
        boxes = batch.get("boxes")
        b, p, _ = k.shape

        base_logits, _ = self.reason(v, t, q_cls, k, triplet_mask, boxes)
        base_loss = F.binary_cross_entropy_with_logits(
            base_logits, soft, reduction="none").mean(dim=1)  # (B,)

        delta = torch.zeros(b, p, device=k.device)
        idx = torch.randint(0, p, (sample_size,), device=k.device)
        for j in idx.tolist():
            k_drop = k.clone()
            k_drop[:, j] = 0.0
            logits_j, _ = self.reason(v, t, q_cls, k_drop, triplet_mask, boxes)
            loss_j = F.binary_cross_entropy_with_logits(
                logits_j, soft, reduction="none").mean(dim=1)
            delta[:, j] = loss_j - base_loss
        return delta

    def compute_loss(self, batch: dict, loo_sample_size: int = 10):
        """Full training loss L = L_VQA + lambda * L_AKF."""
        v, t, q_cls, k = self.encode(batch)
        logits, aux = self.reason(
            v, t, q_cls, k,
            triplet_mask=batch.get("triplet_mask"),
            boxes=batch.get("boxes"),
        )
        l_vqa = self.vqa_loss(logits, batch["answer_scores"])

        delta = self.loo_deltas(batch, v, t, q_cls, k, loo_sample_size)
        targets = self.akf.soft_targets_from_loo(delta)
        l_akf = self.akf.loss(aux["alpha"], targets, batch.get("triplet_mask"))

        loss = l_vqa + self.cfg.loss_lambda * l_akf
        return loss, {"loss": loss.item(), "l_vqa": l_vqa.item(), "l_akf": l_akf.item()}
