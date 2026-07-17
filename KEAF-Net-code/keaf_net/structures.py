"""Shared tensor containers and node/edge type constants.

A mini-batch is stored as a single *disjoint* graph (PyG-style): node tensors
are concatenated along the node axis and ``node_batch`` records which sample
each node came from. Edges never cross samples, so the type-aware message
passing in HGAF runs once over the whole batch while MHSR still attends inside
each sample via segment softmax.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# node types (Section 3.5: visual / textual / knowledge)
VISUAL, TEXTUAL, KNOWLEDGE = 0, 1, 2
NODE_TYPES = 3

# edge types, ordered as in Algorithm 1 / Section 3.5
EDGE_VV, EDGE_TT, EDGE_VK, EDGE_TK, EDGE_VT = 0, 1, 2, 3, 4
EDGE_TYPES = 5


@dataclass
class KEAFBatch:
    node_feat: torch.Tensor      # [N, d]
    node_type: torch.Tensor      # [N]   long in {VISUAL, TEXTUAL, KNOWLEDGE}
    node_batch: torch.Tensor     # [N]   long, sample index in [0, B)
    edge_index: torch.Tensor     # [2, E] long, message passes edge_index[0] -> edge_index[1]
    edge_type: torch.Tensor      # [E]   long in {0..4}
    qcls: torch.Tensor           # [B, d] BERT [CLS] of the question
    num_graphs: int
    answer_target: torch.Tensor | None = None   # [B, num_answers] soft labels
    meta: list[dict[str, Any]] | None = None

    @property
    def num_nodes(self) -> int:
        return self.node_feat.size(0)

    @property
    def device(self) -> torch.device:
        return self.node_feat.device

    def knowledge_mask(self) -> torch.Tensor:
        """Boolean mask over nodes selecting knowledge triplets."""
        return self.node_type == KNOWLEDGE

    def to(self, device: torch.device | str) -> "KEAFBatch":
        move = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t
        return KEAFBatch(
            node_feat=move(self.node_feat),
            node_type=move(self.node_type),
            node_batch=move(self.node_batch),
            edge_index=move(self.edge_index),
            edge_type=move(self.edge_type),
            qcls=move(self.qcls),
            num_graphs=self.num_graphs,
            answer_target=move(self.answer_target),
            meta=self.meta,
        )
