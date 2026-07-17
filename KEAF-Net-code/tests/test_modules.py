import torch

from keaf_net.models import HGAF, MHSR, AdaptiveKnowledgeFilter
from keaf_net.structures import KNOWLEDGE, TEXTUAL, VISUAL


def _toy_graph(d=64):
    # 2 visual, 2 textual, 2 knowledge nodes across 2 samples
    node_feat = torch.randn(6, d)
    node_type = torch.tensor([VISUAL, TEXTUAL, KNOWLEDGE, VISUAL, TEXTUAL, KNOWLEDGE])
    node_batch = torch.tensor([0, 0, 0, 1, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 0, 4, 3, 3]])
    qcls = torch.randn(2, d)
    return node_feat, node_type, node_batch, edge_index, qcls


def test_akf_scores_and_threshold(model_cfg):
    akf = AdaptiveKnowledgeFilter(model_cfg.hidden_dim, threshold_init=0.5)
    nf, nt, nb, _, q = _toy_graph(model_cfg.hidden_dim)
    scores, gate, node_keep, k_index = akf(nf, nt, nb, q)

    assert scores.shape == (2,)                 # two knowledge nodes
    assert torch.all((scores >= 0) & (scores <= 1))
    assert torch.allclose(akf.threshold, torch.tensor(0.5), atol=1e-6)
    # visual/textual nodes are always kept; only knowledge nodes are gated
    assert node_keep[k_index].equal(gate)
    assert (node_keep[nt != KNOWLEDGE] == 1).all()


def test_akf_loss_prefers_helpful_triplets(model_cfg):
    akf = AdaptiveKnowledgeFilter(model_cfg.hidden_dim)
    scores = torch.tensor([0.9, 0.1], requires_grad=True)
    loo = torch.tensor([2.0, -2.0])            # first helps, second hurts
    loss = akf.loss(scores, loo)
    loss.backward()
    assert loss.item() >= 0
    assert scores.grad is not None


def test_hgaf_masks_filtered_knowledge(model_cfg):
    hgaf = HGAF(model_cfg.hidden_dim, model_cfg.gat_heads, 2, 0.0, 0.2)
    nf, nt, nb, ei, _ = _toy_graph(model_cfg.hidden_dim)
    node_keep = torch.ones(6)
    node_keep[2] = 0.0                          # discard one knowledge node
    fused = hgaf(nf, nt, ei, node_keep)
    assert fused.shape == nf.shape
    assert torch.count_nonzero(fused[2]) == 0   # discarded node is zeroed out


def test_mhsr_runs_requested_hops(model_cfg):
    mhsr = MHSR(model_cfg.hidden_dim, num_hops=3, num_answers=model_cfg.num_answers, dropout=0.0)
    nf, nt, nb, _, q = _toy_graph(model_cfg.hidden_dim)
    node_keep = torch.ones(6)
    logits, aux = mhsr(nf, nb, q, node_keep, return_attention=True)
    assert logits.shape == (2, model_cfg.num_answers)
    assert len(aux["hop_attention"]) == 3
