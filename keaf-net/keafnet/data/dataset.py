"""
Knowledge-VQA dataset.

Loads pre-extracted features (region features, triplet embeddings) and question
tokens for OK-VQA / A-OKVQA. The expected on-disk layout is described in
docs/DATA.md. When real data is unavailable, `SyntheticVQADataset` produces
shape-correct random tensors so the full training/eval loop can be exercised
(useful for CI and smoke tests).

Reference: KEAF-Net, Section 5 (Datasets).
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch.utils.data import Dataset


class KnowledgeVQADataset(Dataset):
    """Real-feature dataset.

    Each example directory / record is expected to provide:
        region_feats:  (M, region_dim) float
        boxes:         (M, 4) float (x1,y1,x2,y2)
        triplet_feats: (P, triplet_dim) float
        triplet_mask:  (P,) {0,1}
        input_ids:     (L,) long
        attention_mask:(L,) long
        answer_scores: (num_answers,) float  soft VQA scores
        answers:       list[str] (for soft-accuracy eval)
    """

    def __init__(self, annotations_file: str, features_dir: str,
                 num_answers: int = 3129, max_triplets: int = 50,
                 max_len: int = 64) -> None:
        with open(annotations_file) as f:
            self.records = json.load(f)
        self.features_dir = features_dir
        self.num_answers = num_answers
        self.max_triplets = max_triplets
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        feat_path = os.path.join(self.features_dir, f"{rec['id']}.pt")
        data = torch.load(feat_path, map_location="cpu")
        data["answers"] = rec.get("answers", [])
        return data


class SyntheticVQADataset(Dataset):
    """Random shape-correct data for smoke-testing the pipeline end-to-end."""

    def __init__(self, n: int = 64, num_answers: int = 3129, num_regions: int = 36,
                 region_dim: int = 2048, triplet_dim: int = 384,
                 max_triplets: int = 50, max_len: int = 20,
                 vocab_size: int = 30522, seed: int = 0) -> None:
        self.n = n
        self.num_answers = num_answers
        self.num_regions = num_regions
        self.region_dim = region_dim
        self.triplet_dim = triplet_dim
        self.max_triplets = max_triplets
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(i)
        region_feats = torch.randn(self.num_regions, self.region_dim, generator=g)
        boxes = torch.rand(self.num_regions, 4, generator=g)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] * 0.5  # ensure x2>x1, y2>y1
        triplet_feats = torch.randn(self.max_triplets, self.triplet_dim, generator=g)
        triplet_mask = (torch.rand(self.max_triplets, generator=g) > 0.2).float()
        input_ids = torch.randint(0, self.vocab_size, (self.max_len,), generator=g)
        input_ids[0] = 101  # [CLS]
        attention_mask = torch.ones(self.max_len, dtype=torch.long)

        answer_scores = torch.zeros(self.num_answers)
        gold = torch.randint(0, self.num_answers, (1,), generator=g).item()
        answer_scores[gold] = 1.0
        return {
            "region_feats": region_feats,
            "boxes": boxes,
            "triplet_feats": triplet_feats,
            "triplet_mask": triplet_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_scores": answer_scores,
            "answers": ["answer_%d" % gold] * 10,
            "gold_idx": gold,
        }


def collate(batch: list[dict]) -> dict[str, Any]:
    """Stack a list of examples into a batched dict."""
    out: dict[str, Any] = {}
    keys_tensor = ["region_feats", "boxes", "triplet_feats", "triplet_mask",
                   "input_ids", "attention_mask", "answer_scores"]
    for k in keys_tensor:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "answers" in batch[0]:
        out["answers"] = [b["answers"] for b in batch]
    if "gold_idx" in batch[0]:
        out["gold_idx"] = torch.tensor([b["gold_idx"] for b in batch])
    return out
