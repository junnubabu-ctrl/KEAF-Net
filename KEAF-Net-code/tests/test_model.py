import torch

from keaf_net.data.synthetic import random_batch
from keaf_net.models import KEAFNet


def test_forward_shapes(model_cfg):
    model = KEAFNet(model_cfg)
    batch = random_batch(model_cfg, batch_size=3)
    out = model(batch)
    assert out["logits"].shape == (3, model_cfg.num_answers)
    assert out["node_keep"].min() >= 0
    # every knowledge node received a score in [0, 1]
    assert torch.all((out["akf_scores"] >= 0) & (out["akf_scores"] <= 1))


def test_backward_updates_all_modules(model_cfg):
    model = KEAFNet(model_cfg)
    batch = random_batch(model_cfg, batch_size=2)
    out = model(batch)
    out["logits"].sum().backward()

    touched = {
        "akf": model.akf.score.weight.grad,
        "hgaf": model.hgaf.layers[0].attn.grad,
        "mhsr": model.mhsr.gru.weight_ih.grad,
        "classifier": model.mhsr.classifier.weight.grad,
        "region_proj": model.region_proj.weight.grad,
    }
    for name, grad in touched.items():
        assert grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(grad).all()


def test_reason_is_reusable(model_cfg):
    model = KEAFNet(model_cfg).eval()
    batch = model.assemble_graph(random_batch(model_cfg, batch_size=2))
    _, _, node_keep, _ = model.filter(batch)
    logits_a, _ = model.reason(batch, node_keep)
    logits_b, _ = model.reason(batch, node_keep)
    assert torch.allclose(logits_a, logits_b)   # deterministic in eval-style reuse
