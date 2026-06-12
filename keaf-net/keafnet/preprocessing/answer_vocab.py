"""
Answer-vocabulary construction for knowledge-VQA.

Builds the fixed answer vocabulary used by the classifier head, following the
standard VQA v2 / OK-VQA recipe: collect candidate answers, normalise them, keep
those that appear at least `min_occurrence` times, and assign each an index.

Also computes per-question soft target scores:

    score(a) = min(1, (# annotators who gave a) / 3)

Reference: KEAF-Net, Section 5 (Evaluation Protocol).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

import torch

# --- VQA answer normalisation (Antol et al. 2015 / Goyal et al. 2017) ---------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
    "isnt": "isn't", "its": "it's", "shouldnt": "shouldn't", "thats": "that's",
    "theres": "there's", "theyre": "they're", "wasnt": "wasn't", "werent": "weren't",
    "whats": "what's", "wont": "won't", "wouldnt": "wouldn't", "youre": "you're",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_PERIOD_STRIP = re.compile(r"(?<=\d)\.(?=\d)")  # keep decimals


def normalise_answer(ans: str) -> str:
    ans = ans.lower().strip().replace("\n", " ").replace("\t", " ")
    ans = _PUNCT.sub("", ans)
    words = []
    for w in ans.split():
        w = _MANUAL_MAP.get(w, w)
        w = _CONTRACTIONS.get(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(words)


class AnswerVocab:
    def __init__(self, answer2idx: dict[str, int]) -> None:
        self.answer2idx = answer2idx
        self.idx2answer = {i: a for a, i in answer2idx.items()}

    def __len__(self) -> int:
        return len(self.answer2idx)

    def encode_scores(self, annotator_answers: Iterable[str]) -> torch.Tensor:
        """Per-answer soft scores for one question's 10 annotator answers."""
        scores = torch.zeros(len(self.answer2idx))
        counts = Counter(normalise_answer(a) for a in annotator_answers)
        for ans, c in counts.items():
            idx = self.answer2idx.get(ans)
            if idx is not None:
                scores[idx] = min(1.0, c / 3.0)
        return scores

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.answer2idx, f)

    @classmethod
    def load(cls, path: str) -> "AnswerVocab":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def build(cls, all_annotator_answers: Iterable[Iterable[str]],
              min_occurrence: int = 8) -> "AnswerVocab":
        """Build vocab from the training annotations.

        Args:
            all_annotator_answers: iterable of per-question answer lists.
            min_occurrence: keep answers occurring at least this many times
                            (VQA v2 uses 9; OK-VQA commonly uses 8).
        """
        counter: Counter = Counter()
        for ans_list in all_annotator_answers:
            for a in ans_list:
                counter[normalise_answer(a)] += 1
        kept = [a for a, c in counter.items() if c >= min_occurrence and a]
        kept.sort()
        return cls({a: i for i, a in enumerate(kept)})
