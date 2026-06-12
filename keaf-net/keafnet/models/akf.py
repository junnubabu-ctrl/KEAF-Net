"""
Adaptive Knowledge Filter (AKF).

The AKF gates retrieved knowledge triplets against a joint image-question
embedding, retaining only triplets whose learned relevance score exceeds a
learned threshold. Its supervision derives from leave-one-out (LOO)
prediction-loss differences rather than answer-overlap pseudo-labels, which
avoids label leakage during training.

Reference: KEAF-Net, Section 3.4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughThreshold(torch.autograd.Function):
    """Hard threshold on the forward pass, identity gradient on the backward
    pass (straight-through estimator). Lets the discrete keep/discard decision
    propagate gradients to the relevance scores and the learned threshold.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        # 1.0 where score > threshold else 0.0
        return (scores > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Identity gradient w.r.t. scores; pass a (negated) copy to threshold.
        return grad_output, -grad_output.sum(dim=-1, keepdim=True)


def ste_threshold(scores: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return StraightThroughThreshold.apply(scores, threshold)


class AdaptiveKnowledgeFilter(nn.Module):
    """Scores and filters candidate knowledge triplets.

    Args:
        dim: hidden dimension d of all embeddings.
        init_threshold: initial value of the learned gate threshold theta.
        temperature: tau used in temperature-scaled sigmoid for soft targets.
    """

    def __init__(self, dim: int = 768, init_threshold: float = 0.5,
                 temperature: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Compress the image-question context into a single vector h_IQ.
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

        # Three-way interaction scorer: [k_j ; h_IQ ; k_j ⊙ h_IQ] -> scalar.
        self.scorer = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        # Learned scalar threshold theta (kept in (0,1) via sigmoid at use time).
        self.threshold_logit = nn.Parameter(
            torch.logit(torch.tensor(float(init_threshold)))
        )

    def joint_context(self, q_cls: torch.Tensor, v_feats: torch.Tensor) -> torch.Tensor:
        """Build h_IQ = LayerNorm(W_q q_cls + W_v mean(V)).

        Args:
            q_cls:   (B, d) question [CLS] embedding.
            v_feats: (B, M, d) visual region/grid features.
        Returns:
            (B, d) joint image-question context.
        """
        v_mean = v_feats.mean(dim=1)
        return self.ln(self.q_proj(q_cls) + self.v_proj(v_mean))

    def score(self, k: torch.Tensor, h_iq: torch.Tensor) -> torch.Tensor:
        """Relevance score alpha_j for every triplet.

        Args:
            k:    (B, P, d) triplet embeddings.
            h_iq: (B, d) joint context.
        Returns:
            (B, P) scores in (0, 1).
        """
        b, p, d = k.shape
        h = h_iq.unsqueeze(1).expand(-1, p, -1)
        feat = torch.cat([k, h, k * h], dim=-1)  # (B, P, 3d)
        return torch.sigmoid(self.scorer(feat).squeeze(-1))  # (B, P)

    def forward(self, k: torch.Tensor, q_cls: torch.Tensor, v_feats: torch.Tensor,
                mask: torch.Tensor | None = None):
        """Filter triplets.

        Args:
            k:       (B, P, d) candidate triplet embeddings.
            q_cls:   (B, d) question [CLS] embedding.
            v_feats: (B, M, d) visual features.
            mask:    (B, P) optional 1/0 mask of valid (non-padding) triplets.
        Returns:
            filtered: (B, P, d) triplets with discarded entries zeroed out.
            alpha:    (B, P) relevance scores.
            keep:     (B, P) hard keep/discard decision (via STE).
        """
        h_iq = self.joint_context(q_cls, v_feats)
        alpha = self.score(k, h_iq)
        if mask is not None:
            alpha = alpha * mask

        theta = torch.sigmoid(self.threshold_logit)
        keep = ste_threshold(alpha, theta)  # (B, P), STE
        filtered = k * keep.unsqueeze(-1)
        return filtered, alpha, keep

    def loss(self, alpha: torch.Tensor, soft_targets: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
        """Retrieval-consistency loss L_AKF = mean (alpha_j - t_j)^2.

        Args:
            alpha:        (B, P) predicted relevance.
            soft_targets: (B, P) targets t_j = sg(sigmoid(Delta_j / tau)).
            mask:         (B, P) optional valid mask.
        """
        se = (alpha - soft_targets) ** 2
        if mask is not None:
            denom = mask.sum().clamp_min(1.0)
            return (se * mask).sum() / denom
        return se.mean()

    @torch.no_grad()
    def soft_targets_from_loo(self, delta: torch.Tensor) -> torch.Tensor:
        """Convert LOO loss differences Delta_j into soft targets.

        t_j = sigmoid(Delta_j / tau), detached (stop-gradient).

        A positive Delta_j means removing triplet j increased the loss, i.e.
        the triplet was helpful, so its target relevance is high.
        """
        return torch.sigmoid(delta / self.temperature)
