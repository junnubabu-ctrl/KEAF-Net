"""Dataset-free smoke test for the KEAF-Net tensor pipeline."""

import torch

from keaf_net import KEAFNet, KEAFNetConfig


def main():
    torch.manual_seed(7)
    cfg = KEAFNetConfig(visual_dim=32, question_dim=24, knowledge_dim=20, hidden_dim=16, num_answers=11)
    model = KEAFNet(cfg).eval()
    batch, nodes = 2, 5
    with torch.no_grad():
        out = model(
            visual=torch.randn(batch, cfg.visual_dim),
            question=torch.randn(batch, cfg.question_dim),
            knowledge=torch.randn(batch, cfg.knowledge_dim),
            graph_nodes=torch.randn(batch, nodes, cfg.knowledge_dim),
        )
    assert out["logits"].shape == (batch, cfg.num_answers)
    assert torch.isfinite(out["logits"]).all()
    print("KEAF-Net smoke test passed", tuple(out["logits"].shape))


if __name__ == "__main__":
    main()
