"""Answer vocabulary and soft-target construction (Section 5.4).

The classifier predicts over the most frequent normalised training answers
(3,129 classes in the reported runs). Soft targets give each present answer its
VQA accuracy ``min(1, votes/3)``, matching the soft-target BCE used for L_VQA.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, Sequence

import torch

from ..utils.text import normalize_answer


class AnswerVocab:
    def __init__(self, answers: Sequence[str]):
        self.itos = list(answers)
        self.stoi = {a: i for i, a in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, annotation_answers: Iterable[Sequence[str]], num_answers: int = 3129,
              min_count: int = 0) -> "AnswerVocab":
        """Pick the ``num_answers`` most frequent normalised answers."""
        counts: Counter[str] = Counter()
        for answers in annotation_answers:
            for a in answers:
                counts[normalize_answer(a)] += 1
        ranked = [a for a, c in counts.most_common() if c >= min_count and a]
        return cls(ranked[:num_answers])

    def soft_target(self, answers: Sequence[str]) -> torch.Tensor:
        """Soft label vector for one question's (typically ten) annotations."""
        target = torch.zeros(len(self))
        votes: Counter[str] = Counter(normalize_answer(a) for a in answers)
        for answer, n in votes.items():
            idx = self.stoi.get(answer)
            if idx is not None:
                target[idx] = min(1.0, n / 3.0)
        return target

    def decode(self, index: int) -> str:
        return self.itos[index]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.itos, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "AnswerVocab":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))
