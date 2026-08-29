"""Paper 3 KEAF-Net core architecture.

Implements manuscript equations (8)-(21) over encoded visual, textual and
knowledge features. External encoders/retrievers are adapters so restricted
assets are never bundled into the repository.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Paper3Config:
    dim: int = 768
    num_answers: int = 3129
    gat_heads: int = 8
    gat_layers: int = 2
    hops: int = 3
    akf_threshold_init: float = 0.5
    akf_loss_weight: float = 0.3
    dropout: float = 0.1


class AdaptiveKnowledgeFilter(nn.Module):
    """Manuscript equations (8)-(11)."""
    def __init__(self, dim: int, threshold_init: float = 0.5):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim * 3, 1)
        # sigmoid(logit_theta) makes the learned threshold remain in (0,1).
        init = torch.tensor(threshold_init).clamp(1e-4, 1 - 1e-4)
        self.threshold_logit = nn.Parameter(torch.logit(init))

    @property
    def threshold(self):
        return torch.sigmoid(self.threshold_logit)

    def forward(self, visual_nodes, q_cls, knowledge_nodes, knowledge_mask=None):
        h_iq = self.norm(self.q_proj(q_cls) + self.v_proj(visual_nodes.mean(dim=1)))
        ctx = h_iq.unsqueeze(1).expand_as(knowledge_nodes)
        alpha = torch.sigmoid(self.score(torch.cat([knowledge_nodes, ctx, knowledge_nodes * ctx], -1)).squeeze(-1))
        if knowledge_mask is not None:
            alpha = alpha * knowledge_mask.float()
        # Straight-through soft gate: differentiable training while preserving threshold semantics.
        soft = torch.sigmoid((alpha - self.threshold) / 0.1)
        hard = (alpha > self.threshold).float()
        gate = hard.detach() - soft.detach() + soft
        return knowledge_nodes * gate.unsqueeze(-1), alpha, hard.bool(), h_iq

    @staticmethod
    def auxiliary_loss(alpha, labels, mask=None):
        loss = F.binary_cross_entropy(alpha, labels.float(), reduction="none")
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp_min(1.0)
        return loss.mean()


class TypedGraphAttentionLayer(nn.Module):
    """Type-aware multi-head message passing corresponding to eqs. (12)-(14)."""
    def __init__(self, dim: int, heads: int, num_node_types: int = 3, num_edge_types: int = 5, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.node_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_node_types)])
        self.edge_bias = nn.Embedding(num_edge_types, heads)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, node_types, adjacency, edge_types):
        b, n, d = nodes.shape
        typed = torch.zeros_like(nodes)
        for t, proj in enumerate(self.node_proj):
            mask = (node_types == t).unsqueeze(-1)
            typed = torch.where(mask, proj(nodes), typed)
        q = self.q(typed).view(b,n,self.heads,self.head_dim).transpose(1,2)
        k = self.k(typed).view(b,n,self.heads,self.head_dim).transpose(1,2)
        v = self.v(typed).view(b,n,self.heads,self.head_dim).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / self.head_dim**0.5
        bias = self.edge_bias(edge_types.clamp_min(0)).permute(0,3,1,2)
        scores = scores + bias
        eye = torch.eye(n, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        allowed = adjacency.bool() | eye
        scores = scores.masked_fill(~allowed.unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        msg = torch.matmul(self.dropout(attn), v).transpose(1,2).contiguous().view(b,n,d)
        return self.norm(nodes + self.dropout(self.out(msg))), attn


class HeterogeneousGraphAdaptiveFusion(nn.Module):
    """HGAF: two-layer typed GAT plus manuscript gated modality fusion."""
    def __init__(self, cfg: Paper3Config):
        super().__init__()
        self.layers = nn.ModuleList([TypedGraphAttentionLayer(cfg.dim, cfg.gat_heads, dropout=cfg.dropout) for _ in range(cfg.gat_layers)])
        self.gates = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim) for _ in range(3)])

    def forward(self, visual, text, knowledge, adjacency, edge_types):
        nodes = torch.cat([visual, text, knowledge], dim=1)
        b = nodes.size(0)
        types = torch.cat([
            torch.zeros(b, visual.size(1), dtype=torch.long, device=nodes.device),
            torch.ones(b, text.size(1), dtype=torch.long, device=nodes.device),
            torch.full((b, knowledge.size(1)), 2, dtype=torch.long, device=nodes.device),
        ], dim=1)
        attentions = []
        for layer in self.layers:
            nodes, attn = layer(nodes, types, adjacency, edge_types)
            attentions.append(attn)
        nv, nt = visual.size(1), text.size(1)
        hv = nodes[:,:nv].mean(1); ht = nodes[:,nv:nv+nt].mean(1); hk = nodes[:,nv+nt:].mean(1)
        gv, gt, gk = [torch.sigmoid(g(h)) for g,h in zip(self.gates, (hv,ht,hk))]
        fused = gv*hv + gt*ht + gk*hk
        return nodes, fused, (gv,gt,gk), attentions


class MultiHopSemanticReasoning(nn.Module):
    """Equations (17)-(20): three-hop attention and GRU refinement."""
    def __init__(self, cfg: Paper3Config):
        super().__init__()
        self.hops = cfg.hops
        self.hop_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.gru = nn.GRUCell(cfg.dim, cfg.dim)
        self.classifier = nn.Linear(cfg.dim*2, cfg.num_answers)

    def forward(self, q_cls, nodes):
        q = q_cls
        history = []
        c = nodes.mean(1)
        for _ in range(self.hops):
            scores = torch.einsum("bd,bnd->bn", q, self.hop_proj(nodes))
            beta = torch.softmax(scores, -1)
            c = torch.sum(nodes * beta.unsqueeze(-1), 1)
            q = self.gru(c, q)
            history.append(beta)
        logits = self.classifier(torch.cat([q,c], -1))
        return logits, q, history


class KEAFNetPaper3(nn.Module):
    """Feature-level implementation of the complete Paper 3 reasoning path."""
    def __init__(self, cfg: Paper3Config):
        super().__init__()
        self.cfg = cfg
        self.akf = AdaptiveKnowledgeFilter(cfg.dim, cfg.akf_threshold_init)
        self.hgaf = HeterogeneousGraphAdaptiveFusion(cfg)
        self.mhsr = MultiHopSemanticReasoning(cfg)

    def forward(self, visual, text, knowledge, adjacency, edge_types, knowledge_mask=None):
        q_cls = text[:,0]
        filtered, relevance, selected, h_iq = self.akf(visual, q_cls, knowledge, knowledge_mask)
        graph_nodes, fused, modality_gates, graph_attention = self.hgaf(visual, text, filtered, adjacency, edge_types)
        logits, query, hop_attention = self.mhsr(q_cls + fused, graph_nodes)
        return {"logits": logits, "knowledge_relevance": relevance, "knowledge_selected": selected,
                "joint_context": h_iq, "modality_gates": modality_gates, "graph_attention": graph_attention,
                "hop_attention": hop_attention, "reasoned_query": query}

    def loss(self, outputs, answer_targets, akf_labels=None, knowledge_mask=None):
        # answer_targets may be class indices or VQA soft target distributions.
        if answer_targets.ndim == 1:
            vqa = F.cross_entropy(outputs["logits"], answer_targets)
        else:
            vqa = F.binary_cross_entropy_with_logits(outputs["logits"], answer_targets.float())
        akf = outputs["logits"].new_zeros(())
        if akf_labels is not None:
            akf = self.akf.auxiliary_loss(outputs["knowledge_relevance"], akf_labels, knowledge_mask)
        return {"total": vqa + self.cfg.akf_loss_weight*akf, "vqa": vqa, "akf": akf}
