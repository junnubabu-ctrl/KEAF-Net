"""Multi-Hop Semantic Reasoning (MHSR), Section 3.6.

The query state starts from the question [CLS] vector and is refined over ``T``
hops. Each hop attends over the fused heterogeneous graph, pools an evidence
vector, and updates the query with a GRU cell (Eq. 19-21). The final query and
its attended evidence are concatenated for the answer classifier (Eq. 22). The
default ``T = 3`` is the depth selected in the hop study (Table 10).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.scatter import scatter_softmax, scatter_sum

_NEG_INF = -1e9


class MHSR(nn.Module):
    def __init__(self, dim: int, num_hops: int, num_answers: int, dropout: float):
        super().__init__()
        self.num_hops = num_hops
        self.hop_proj = nn.Linear(dim, dim, bias=False)   # W_hop, Eq. (20)
        self.gru = nn.GRUCell(dim, dim)                   # Eq. (21)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * dim, num_answers)  # Eq. (22)

    def forward(
        self, node_feat: torch.Tensor, node_batch: torch.Tensor,
        qcls: torch.Tensor, node_keep: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        num_graphs = qcls.size(0)
        proj = self.hop_proj(node_feat)                   # W_hop h_i, reused across hops
        valid = node_keep > 0

        q = qcls                                          # q^(0), Eq. (19)
        context = qcls.new_zeros(qcls.shape)
        hop_attn: list[torch.Tensor] = []
        for _ in range(self.num_hops):
            score = (proj * q[node_batch]).sum(-1)        # (q^(t))^T W_hop h_i
            score = score.masked_fill(~valid, _NEG_INF)
            beta = scatter_softmax(score, node_batch, num_graphs)            # Eq. (20)
            context = scatter_sum(beta.unsqueeze(-1) * node_feat, node_batch, num_graphs)
            q = self.gru(self.dropout(context), q)        # Eq. (21)
            if return_attention:
                hop_attn.append(beta.detach())

        logits = self.classifier(torch.cat([q, context], dim=-1))           # Eq. (22)
        aux = {"query": q, "context": context}
        if return_attention:
            aux["hop_attention"] = hop_attn
        return logits, aux
