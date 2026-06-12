"""
Evaluation metrics.

Implements the standard VQA soft-accuracy metric used by OK-VQA and A-OKVQA:

    Acc(a) = min(1, (# humans that said a) / 3)

averaged over all 10-choose-9 subsets of the human annotators, which for a
single predicted answer reduces to min(1, matching_count / 3).

Reference: KEAF-Net, Section 5 (Evaluation Protocol).
"""
from __future__ import annotations

from typing import Sequence

import torch


def vqa_soft_accuracy(pred_answers: Sequence[str],
                      gt_answer_lists: Sequence[Sequence[str]]) -> float:
    """Compute mean VQA soft accuracy.

    Args:
        pred_answers:    list of predicted answer strings (length B).
        gt_answer_lists: list of lists of human answer strings (length B).
    Returns:
        Mean soft accuracy in [0, 1].
    """
    assert len(pred_answers) == len(gt_answer_lists)
    total = 0.0
    for pred, gts in zip(pred_answers, gt_answer_lists):
        pred_n = _normalise(pred)
        matches = sum(1 for g in gts if _normalise(g) == pred_n)
        total += min(1.0, matches / 3.0)
    return total / max(len(pred_answers), 1)


def accuracy_from_logits(logits: torch.Tensor, answer_scores: torch.Tensor) -> float:
    """Soft accuracy directly from logits and per-answer soft scores.

    Picks argmax answer and reads its soft score (already min(1, n/3) capped).
    """
    pred = logits.argmax(dim=-1)
    picked = answer_scores.gather(1, pred.unsqueeze(1)).squeeze(1)
    return picked.clamp(max=1.0).mean().item()


def _normalise(ans: str) -> str:
    ans = ans.lower().strip()
    # Light VQA-style normalisation: strip punctuation and articles.
    for art in (" a ", " an ", " the "):
        ans = ans.replace(art, " ")
    ans = "".join(ch for ch in ans if ch.isalnum() or ch.isspace())
    return " ".join(ans.split())
