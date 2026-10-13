"""Evaluation loop and VQA soft-accuracy scoring (Section 5.4)."""
from __future__ import annotations

import torch

from ..data.dataset import move_sample
from ..data.vocab import AnswerVocab
from ..models import KEAFNet
from ..utils.metrics import aokvqa_direct_accuracy, vqa_official_accuracy


class Evaluator:
    def __init__(self, model: KEAFNet, vocab: AnswerVocab, device: torch.device | str = "cpu",
                 dataset: str = "okvqa"):
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device)
        if dataset not in {"okvqa", "vqav2", "aokvqa"}:
            raise ValueError(f"no verified evaluation protocol for dataset {dataset!r}")
        self.dataset = dataset

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        predictions: list[dict] = []
        scores: list[float] = []
        missing_references = False
        seen_qids: set = set()
        for batch in loader:
            samples = [move_sample(s, self.device) for s in batch]
            logits = self.model(samples)["logits"]
            for i, idx in enumerate(logits.argmax(dim=-1).tolist()):
                answer = self.vocab.decode(idx)
                meta = samples[i]["meta"]
                if meta["qid"] in seen_qids:
                    raise ValueError(f"duplicate prediction question ID {meta['qid']!r}")
                seen_qids.add(meta["qid"])
                predictions.append({"qid": meta["qid"], "answer": answer})
                if self.dataset == "aokvqa":
                    if not isinstance(meta.get("difficult_direct_answer"), bool):
                        raise ValueError("A-OKVQA DA requires difficult_direct_answer metadata; rebuild annotations")
                    if meta["difficult_direct_answer"]:
                        continue
                    metric, refs, pred = aokvqa_direct_accuracy, meta["answers"], answer
                else:
                    metric, refs, pred = vqa_official_accuracy, meta["answers"], answer
                if not refs:
                    missing_references = True
                else:
                    scores.append(metric(pred, refs))

        accuracy = sum(scores) / len(scores) if scores and not missing_references else None
        return {"accuracy": accuracy, "predictions": predictions,
                "scored_questions": len(scores), "dataset": self.dataset,
                "protocol": "aokvqa_direct_exact" if self.dataset == "aokvqa" else "vqa_leave_one_out",
                "missing_references": missing_references}
