"""Minimal scatter operations.

We keep the model free of a ``torch_geometric`` dependency: the heterogeneous
graph in HGAF and the hop attention in MHSR only need segment-wise softmax,
sum and mean over a flat (disjoint-batched) node/edge layout. Implementing the
three primitives here keeps the install light and the message passing easy to
read against Eq. (14)-(16) and Eq. (20).
"""
from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``."""
    out = src.new_zeros((dim_size, *src.shape[1:]))
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-pool ``src`` rows by ``index`` (empty buckets stay zero)."""
    summed = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, dim_size)
    return summed / count.clamp_min(1.0).view(-1, *([1] * (src.dim() - 1)))


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Numerically stable softmax over groups defined by ``index``.

    ``src`` carries one score per edge (HGAF: shape ``[E, heads]``) or per node
    (MHSR: shape ``[N]``); the softmax is taken within each destination group so
    that a node's incoming attention weights sum to one, matching Eq. (15).
    Masked-out entries should be set to a large negative value beforehand.
    """
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    max_per = src.new_full((dim_size, *src.shape[1:]), float("-inf"))
    max_per.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    centred = src - max_per[index]
    weight = centred.exp()
    denom = scatter_sum(weight, index, dim_size)[index]
    return weight / denom.clamp_min(1e-16)
