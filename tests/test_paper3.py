import torch
from keaf_net.paper3 import KEAFNetPaper3, Paper3Config
from keaf_net.graph import build_paper3_graph


def test_paper3_end_to_end_and_loss():
    torch.manual_seed(42)
    cfg=Paper3Config(dim=32,num_answers=17,gat_heads=4,gat_layers=2,hops=3)
    model=KEAFNetPaper3(cfg)
    v=torch.randn(2,6,32); t=torch.randn(2,5,32); k=torch.randn(2,8,32)
    adj,edge=build_paper3_graph(v,t,k,semantic_threshold=0.0)
    out=model(v,t,k,adj,edge)
    assert out['logits'].shape==(2,17)
    assert out['knowledge_relevance'].shape==(2,8)
    assert len(out['graph_attention'])==2
    assert len(out['hop_attention'])==3
    targets=torch.randint(0,17,(2,))
    labels=torch.randint(0,2,(2,8)).float()
    losses=model.loss(out,targets,labels)
    losses['total'].backward()
    assert torch.isfinite(losses['total'])


def test_threshold_is_trainable_and_bounded():
    cfg=Paper3Config(dim=16,num_answers=3,gat_heads=4)
    model=KEAFNetPaper3(cfg)
    assert model.akf.threshold_logit.requires_grad
    assert 0 < model.akf.threshold.item() < 1
