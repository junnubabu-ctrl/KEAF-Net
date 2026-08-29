"""End-to-end KEAF-Net tensor model.

The class deliberately consumes encoded features. Heavy pretrained encoders and
external knowledge stores belong to adapters so licensing and experimental
choices remain explicit.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .modules import AdaptiveKnowledgeFusion, MultiHopReasoner, VQAPredictionHead


@dataclass
class KEAFNetConfig:
    visual_dim: int = 768
    question_dim: int = 768
    knowledge_dim: int = 768
    hidden_dim: int = 512
    num_answers: int = 3129
    reasoning_hops: int = 3
    dropout: float = 0.1


class KEAFNet(nn.Module):
    """Modular KEAF-Net implementation over encoded multimodal features."""

    def __init__(self, config: KEAFNetConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.visual_proj = nn.Linear(config.visual_dim, d)
        self.question_proj = nn.Linear(config.question_dim, d)
        self.knowledge_proj = nn.Linear(config.knowledge_dim, d)
        self.node_proj = nn.Linear(config.knowledge_dim, d)
        self.fusion = AdaptiveKnowledgeFusion(d)
        self.reasoner = MultiHopReasoner(d, config.reasoning_hops)
        self.head = VQAPredictionHead(d, config.num_answers, config.dropout)

    def forward(
        self,
        visual: torch.Tensor,
        question: torch.Tensor,
        knowledge: torch.Tensor,
        graph_nodes: torch.Tensor,
    ):
        v = self.visual_proj(visual)
        q = self.question_proj(question)
        k = self.knowledge_proj(knowledge)
        nodes = self.node_proj(graph_nodes)
        fused, gates = self.fusion(v, q, k)
        reasoned = self.reasoner(fused, nodes)
        logits = self.head(reasoned)
        return {"logits": logits, "fusion_gates": gates, "representation": reasoned}
