"""Synthetic samples for unit tests and smoke runs.

Produces tensors with the same shapes and dtypes the real pipeline yields, so
the model, trainer and evaluator can be exercised end-to-end without the
benchmark datasets or pretrained backbones.
"""
from __future__ import annotations

import random

import torch

from ..config import ModelConfig


def random_sample(cfg: ModelConfig, num_grid: int = 16, num_regions: int = 8,
                  num_tokens: int = 10, num_triplets: int = 12,
                  num_answers: int | None = None, with_target: bool = True,
                  generator: torch.Generator | None = None) -> dict:
    d = cfg.hidden_dim
    g = generator
    box = lambda n: _random_boxes(n, g)

    triplets = num_triplets
    vk = _random_pairs(num_grid + num_regions, triplets, g, max_edges=triplets)
    tk = _random_pairs(num_tokens, triplets, g, max_edges=triplets)

    sample = {
        "grid": torch.randn(num_grid, d, generator=g),
        "grid_boxes": box(num_grid),
        "region": torch.randn(num_regions, cfg.region_feat_dim, generator=g),
        "region_boxes": box(num_regions),
        "tokens": torch.randn(num_tokens, d, generator=g),
        "qcls": torch.randn(d, generator=g),
        "triplets": torch.randn(triplets, cfg.knowledge_feat_dim, generator=g),
        "vk_pairs": vk,
        "tk_pairs": tk,
        "target": None,
        "meta": {"qid": random.randint(0, 10**9), "answers": ["yes", "yes", "no"]},
    }
    if with_target:
        n = num_answers or cfg.num_answers
        target = torch.zeros(n)
        target[torch.randint(0, n, (1,), generator=g)] = 1.0
        sample["target"] = target
    return sample


def random_batch(cfg: ModelConfig, batch_size: int = 4, **kwargs) -> list[dict]:
    return [random_sample(cfg, **kwargs) for _ in range(batch_size)]


def _random_boxes(n: int, g: torch.Generator | None) -> torch.Tensor:
    xy = torch.rand(n, 2, generator=g) * 200
    wh = torch.rand(n, 2, generator=g) * 100 + 20
    return torch.cat([xy, xy + wh], dim=1)


def _random_pairs(num_src: int, num_dst: int, g: torch.Generator | None,
                  max_edges: int) -> torch.Tensor:
    k = int(torch.randint(0, max_edges + 1, (1,), generator=g))
    if k == 0 or num_src == 0 or num_dst == 0:
        return torch.zeros(0, 2, dtype=torch.long)
    src = torch.randint(0, num_src, (k,), generator=g)
    dst = torch.randint(0, num_dst, (k,), generator=g)
    return torch.stack([src, dst], dim=1)
