"""Adaptive Knowledge Filter (AKF), Section 3.4.

AKF scores each retrieved triplet against the joint image-question context and
keeps the ones above a learned threshold ``c_th``. The discrete keep/discard
decision uses a straight-through estimator so gradients still reach the score.
The filter's auxiliary target is a sampled leave-one-out (LOO) prediction-loss
difference (Eq. 11-13); that target is produced by the trainer because it needs
the full forward pass, so this module only owns the scoring and the MSE loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..structures import KNOWLEDGE, VISUAL
from ..utils.scatter import scatter_mean


class AdaptiveKnowledgeFilter(nn.Module):
    def __init__(self, hidden_dim: int, threshold_init: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.context = nn.Linear(2 * hidden_dim, hidden_dim)   # W_q | W_v, Eq. (8)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(3 * hidden_dim, 1)              # W_a, Eq. (9)
        # c_th is learned end-to-end; stored in logit space so it stays in (0, 1)
        self.threshold_logit = nn.Parameter(torch.logit(torch.tensor(threshold_init)))
        self.temperature = temperature

    @property
    def threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.threshold_logit)

    def context_vector(
        self, node_feat: torch.Tensor, node_type: torch.Tensor, node_batch: torch.Tensor,
        qcls: torch.Tensor,
    ) -> torch.Tensor:
        """h_IQ for every sample, Eq. (8)."""
        vis = node_type == VISUAL
        v_pool = scatter_mean(node_feat[vis], node_batch[vis], qcls.size(0))
        return self.context_norm(self.context(torch.cat([qcls, v_pool], dim=-1)))

    def forward(
        self, node_feat: torch.Tensor, node_type: torch.Tensor, node_batch: torch.Tensor,
        qcls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score and gate the knowledge nodes.

        Returns ``(scores, gate, node_keep, k_index)`` where ``scores`` and
        ``gate`` are over knowledge nodes only, ``node_keep`` is a per-node
        multiplier (1 for visual/textual, STE gate for knowledge) and
        ``k_index`` locates the knowledge nodes inside the flat batch.
        """
        h_iq = self.context_vector(node_feat, node_type, node_batch, qcls)

        k_index = (node_type == KNOWLEDGE).nonzero(as_tuple=False).squeeze(-1)
        k = node_feat[k_index]
        ctx = h_iq[node_batch[k_index]]
        scores = torch.sigmoid(self.score(torch.cat([k, ctx, k * ctx], dim=-1))).squeeze(-1)  # Eq. (9)

        # hard keep/discard with a straight-through estimator (Eq. 10)
        hard = (scores > self.threshold).float()
        gate = hard.detach() + scores - scores.detach()

        node_keep = node_feat.new_ones(node_feat.size(0))
        node_keep = node_keep.index_copy(0, k_index, gate)
        return scores, gate, node_keep, k_index

    def loss(self, scores: torch.Tensor, loo_delta: torch.Tensor) -> torch.Tensor:
        """Retrieval-consistency loss, Eq. (12)-(13).

        ``loo_delta`` holds L_VQA(K\\{j}) - L_VQA(K) for the sampled triplets;
        a positive value means the triplet was helping. The target is detached
        (stop-gradient) so only the score is trained.
        """
        target = torch.sigmoid(loo_delta / self.temperature).detach()
        return torch.mean((scores - target) ** 2)
