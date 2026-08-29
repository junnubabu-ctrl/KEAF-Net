import torch

from keaf_net import KEAFNet, KEAFNetConfig


def test_forward_shapes():
    cfg = KEAFNetConfig(visual_dim=8, question_dim=9, knowledge_dim=10, hidden_dim=12, num_answers=7, reasoning_hops=3)
    model = KEAFNet(cfg)
    out = model(
        torch.randn(3, 8),
        torch.randn(3, 9),
        torch.randn(3, 10),
        torch.randn(3, 4, 10),
    )
    assert out["logits"].shape == (3, 7)
    assert out["fusion_gates"].shape == (3, 36)
    assert out["representation"].shape == (3, 12)


def test_backward_path():
    cfg = KEAFNetConfig(visual_dim=4, question_dim=4, knowledge_dim=4, hidden_dim=6, num_answers=5)
    model = KEAFNet(cfg)
    out = model(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 3, 4))
    loss = out["logits"].sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
