"""Evaluation loop and VQA soft-accuracy scoring (Section 5.4)."""
from __future__ import annotations

import torch

from ..data.dataset import move_sample
from ..data.vocab import AnswerVocab
from ..models import KEAFNet
from ..utils.metrics import accuracy_vector
from ..utils.text import normalize_answer


class Evaluator:
    def __init__(self, model: KEAFNet, vocab: AnswerVocab, device: torch.device | str = "cpu"):
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        predictions: list[dict] = []
        pred_answers: list[str] = []
        references: list[list[str]] = []
        for batch in loader:
            samples = [move_sample(s, self.device) for s in batch]
            logits = self.model(samples)["logits"]
            for i, idx in enumerate(logits.argmax(dim=-1).tolist()):
                answer = self.vocab.decode(idx)
                pred_answers.append(answer)
                references.append([normalize_answer(a) for a in samples[i]["meta"]["answers"]])
                predictions.append({"qid": samples[i]["meta"]["qid"], "answer": answer})

        scored = references and all(len(r) for r in references)
        accuracy = sum(accuracy_vector(pred_answers, references)) / len(pred_answers) if scored else None
        return {"accuracy": accuracy, "predictions": predictions}
