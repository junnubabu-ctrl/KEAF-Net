"""VQA evaluation metrics."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from scipy.stats import wilcoxon


def vqa_soft_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """Standard VQA soft accuracy for one question, Eq. (25).

    A prediction scores ``min(1, n/3)`` where ``n`` is the number of the (ten)
    human annotators that gave the same answer. ``answers`` is the list of
    normalised ground-truth answers.
    """
    n = sum(1 for a in answers if a == prediction)
    return min(1.0, n / 3.0)


def accuracy_vector(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> list[float]:
    """Per-question soft accuracy; the mean is the reported VQA accuracy."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be aligned")
    return [vqa_soft_accuracy(p, refs) for p, refs in zip(predictions, references)]


def majority_answer(answers: Iterable[str]) -> str:
    """Most frequent normalised answer (used when building soft targets)."""
    return Counter(answers).most_common(1)[0][0]


def wilcoxon_pvalue(acc_a: Sequence[float], acc_b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank test on per-question correctness.

    Used in Section 6 to check that the gap between two configurations is
    significant at p < 0.05. Returns ``1.0`` when the vectors are identical
    (``wilcoxon`` raises on an all-zero difference).
    """
    diff = [a - b for a, b in zip(acc_a, acc_b)]
    if all(d == 0 for d in diff):
        return 1.0
    return float(wilcoxon(acc_a, acc_b, zero_method="wilcox").pvalue)
