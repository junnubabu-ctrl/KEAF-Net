"""Smoke tests for KEAF-Net modules and the end-to-end forward/backward pass."""
import torch

from keafnet.models import KEAFConfig, KEAFNet
from keafnet.models.akf import AdaptiveKnowledgeFilter
from keafnet.models.hgaf import HGAF
from keafnet.models.mhsr import MHSR
from keafnet.models.graph_builder import build_hetero_graph
from keafnet.data import SyntheticVQADataset, collate
from keafnet.utils import vqa_soft_accuracy


def test_akf_shapes():
    akf = AdaptiveKnowledgeFilter(dim=64)
    k = torch.randn(2, 10, 64)
    q = torch.randn(2, 64)
    v = torch.randn(2, 5, 64)
    filtered, alpha, keep = akf(k, q, v)
    assert filtered.shape == (2, 10, 64)
    assert alpha.shape == (2, 10)
    assert keep.shape == (2, 10)


def test_hgaf_shapes():
    hgaf = HGAF(dim=64, heads=4, layers=2)
    v = torch.randn(2, 5, 64)
    t = torch.randn(2, 6, 64)
    k = torch.randn(2, 4, 64)
    nodes, adj, tids = build_hetero_graph(v, t, k)
    out = hgaf(nodes, adj, tids)
    assert out.shape == nodes.shape


def test_mhsr_shapes():
    mhsr = MHSR(dim=64, hops=3)
    q0 = torch.randn(2, 64)
    nodes = torch.randn(2, 15, 64)
    q, ctx, attn = mhsr(q0, nodes)
    assert q.shape == (2, 64)
    assert ctx.shape == (2, 64)
    assert len(attn) == 3


def test_end_to_end_forward_backward():
    cfg = KEAFConfig(dim=64, num_answers=50, num_regions=8, region_dim=128,
                     triplet_dim=32, max_triplets=10, gat_heads=4,
                     gat_layers=2, mhsr_hops=2, pretrained=False)
    model = KEAFNet(cfg)
    ds = SyntheticVQADataset(n=4, num_answers=50, num_regions=8, region_dim=128,
                             triplet_dim=32, max_triplets=10, max_len=8)
    batch = collate([ds[i] for i in range(4)])
    loss, logs = model.compute_loss(batch, loo_sample_size=3)
    loss.backward()
    assert torch.isfinite(loss)
    assert logs["l_vqa"] >= 0


def test_vqa_soft_accuracy():
    preds = ["cat", "dog"]
    gts = [["cat"] * 3 + ["other"] * 7, ["fish"] * 10]
    acc = vqa_soft_accuracy(preds, gts)
    assert abs(acc - 0.5) < 1e-6  # first fully correct, second wrong


if __name__ == "__main__":
    test_akf_shapes()
    test_hgaf_shapes()
    test_mhsr_shapes()
    test_end_to_end_forward_backward()
    test_vqa_soft_accuracy()
    print("All tests passed.")
