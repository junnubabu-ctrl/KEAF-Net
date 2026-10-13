"""VQA evaluation metrics."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from scipy.stats import wilcoxon
from .text import normalize_answer


def vqa_soft_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """VQA/OK-VQA leave-one-annotator-out agreement on normalized strings.

    Hold out each annotation in turn, score against the remaining annotations,
    and average. For ten annotations, 0/1/2/3/4 matching answers give
    0/.3/.6/.9/1. This is the agreement formula in the official VQA evaluator;
    official dataset tools remain the final authority for benchmark submission.
    Empty references cannot establish an accuracy and raise instead of scoring 0.
    """
    if not answers:
        raise ValueError("VQA accuracy requires reference answers")
    n = sum(a == prediction for a in answers)
    return sum(min(1.0, (n - int(a == prediction)) / 3.0) for a in answers) / len(answers)


def aokvqa_direct_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """A-OKVQA DA agreement: exact strings, min(1, matches/3).

    The A-OKVQA protocol additionally excludes difficult_direct_answer records;
    the evaluator handles that population rule. Do not apply VQA normalization.
    """
    if not answers:
        raise ValueError("A-OKVQA accuracy requires reference answers")
    return min(1.0, sum(a == prediction for a in answers) / 3.0)


def vqa_official_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """Apply the official VQA evaluator's conditional text processing.

    The upstream evaluator cleans whitespace for every answer, and applies its
    punctuation/digit/article/contraction rules when reference strings differ.
    This preserves that behavior, including its homogeneous-reference case.
    """
    clean = lambda s: s.replace("\n", " ").replace("\t", " ").strip()
    refs = [clean(a) for a in answers]
    pred = clean(prediction)
    if len(set(refs)) > 1:
        refs = [normalize_answer(a) for a in refs]
        pred = normalize_answer(pred)
    return vqa_soft_accuracy(pred, refs)


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
    if len(acc_a) != len(acc_b) or not acc_a:
        raise ValueError("paired accuracy vectors must be nonempty and aligned")
    diff = [a - b for a, b in zip(acc_a, acc_b)]
    if all(d == 0 for d in diff):
        return 1.0
    return float(wilcoxon(acc_a, acc_b, zero_method="wilcox").pvalue)
