"""Answer loss (Section 3.7).

L_VQA is the soft-target binary cross-entropy over the answer vocabulary used by
the standard VQA setup: each class carries the soft accuracy of that answer
(``min(1, votes/3)``) so the model is rewarded for any answer the annotators
agreed on. The trainer also needs the *per-sample* loss for the leave-one-out
signal, hence the ``reduce`` switch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vqa_bce(logits: torch.Tensor, target: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Soft-target BCE summed over classes; mean over the batch when reduced."""
    per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)
    return per_sample.mean() if reduce else per_sample
