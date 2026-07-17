import torch

from keaf_net.models.graph import box_iou, build_sample_graph
from keaf_net.structures import EDGE_TT, EDGE_VV


def test_box_iou_identity():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0],
                          [100.0, 100.0, 110.0, 110.0]])
    iou = box_iou(boxes)
    assert torch.allclose(iou.diagonal(), torch.ones(3))
    assert iou[0, 1] == 1.0          # identical boxes
    assert iou[0, 2] == 0.0          # disjoint boxes


def test_build_sample_graph_node_layout():
    d = 8
    v = torch.randn(5, d)
    boxes = torch.tensor([[0, 0, 10, 10]] * 5, dtype=torch.float)
    t = torch.randn(4, d)
    k = torch.randn(3, d)
    vk = torch.tensor([[0, 1]])
    tk = torch.tensor([[2, 0]])

    nf, nt, ei, et = build_sample_graph(v, boxes, t, k, vk, tk)
    assert nf.shape == (12, d)
    # node order is visual, then textual, then knowledge
    assert (nt[:5] == 0).all() and (nt[5:9] == 1).all() and (nt[9:] == 2).all()
    assert ei.shape[0] == 2 and ei.size(1) == et.numel()
    # overlapping boxes create visual-visual edges; token window creates t-t edges
    assert (et == EDGE_VV).any()
    assert (et == EDGE_TT).any()
    assert ei.max() < nf.size(0)


def test_build_sample_graph_handles_empty_knowledge():
    d = 8
    v = torch.randn(3, d)
    boxes = torch.tensor([[0, 0, 10, 10]] * 3, dtype=torch.float)
    t = torch.randn(2, d)
    k = torch.zeros(0, d)
    empty = torch.zeros(0, 2, dtype=torch.long)
    nf, nt, ei, et = build_sample_graph(v, boxes, t, k, empty, empty)
    assert nf.shape == (5, d)
    assert (nt == 2).sum() == 0
