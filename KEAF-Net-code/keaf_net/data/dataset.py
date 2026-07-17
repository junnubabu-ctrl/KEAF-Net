"""VQA dataset over pre-extracted features and cached knowledge.

Expected on-disk layout (produced by ``scripts/extract_features.py`` and
``scripts/retrieve_knowledge.py``)::

    features/<split>/grid/<image_id>.pt       -> {feat:[Np,768], boxes:[Np,4]}
    features/<split>/region/<image_id>.pt     -> {feat:[M,2048], boxes:[M,4]}
    features/<split>/question/<qid>.pt        -> {tokens:[L,768], qcls:[768]}
    knowledge/cache/<split>/<qid>.pt          -> {triplets:[P,384],
                                                  vk_pairs:[*,2], tk_pairs:[*,2],
                                                  strings:[P]}
    data/<dataset>/<split>.json               -> [{qid, image_id, question, answers}]

Grid and region tensors are shared across the questions of an image, so they are
cached in memory. The model assembles the graph, so a sample is just the raw
per-stream tensors plus the entity-link index pairs.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import torch
from torch.utils.data import Dataset

from .vocab import AnswerVocab


class VQADataset(Dataset):
    def __init__(self, root: str, feature_root: str, knowledge_root: str,
                 dataset: str, split: str, vocab: AnswerVocab | None = None):
        self.feature_root = feature_root
        self.knowledge_root = knowledge_root
        self.split = split
        self.vocab = vocab
        with open(os.path.join(root, dataset, f"{split}.json"), "r", encoding="utf-8") as fh:
            self.records = json.load(fh)

    def __len__(self) -> int:
        return len(self.records)

    @lru_cache(maxsize=2048)
    def _load_image(self, image_id: str) -> tuple:
        grid = torch.load(os.path.join(self.feature_root, self.split, "grid", f"{image_id}.pt"))
        region = torch.load(os.path.join(self.feature_root, self.split, "region", f"{image_id}.pt"))
        return grid, region

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        grid, region = self._load_image(str(rec["image_id"]))
        q = torch.load(os.path.join(self.feature_root, self.split, "question", f"{rec['qid']}.pt"))
        kg = torch.load(os.path.join(self.knowledge_root, self.split, f"{rec['qid']}.pt"))

        sample = {
            "grid": grid["feat"], "grid_boxes": grid["boxes"],
            "region": region["feat"], "region_boxes": region["boxes"],
            "tokens": q["tokens"], "qcls": q["qcls"],
            "triplets": kg["triplets"],
            "vk_pairs": kg["vk_pairs"].long(),
            "tk_pairs": kg["tk_pairs"].long(),
            "target": None,
            "meta": {"qid": rec["qid"], "answers": rec.get("answers", []),
                     "question": rec.get("question", "")},
        }
        if self.vocab is not None and rec.get("answers"):
            sample["target"] = self.vocab.soft_target(rec["answers"])
        return sample


def move_sample(sample: dict, device: torch.device | str) -> dict:
    out = {}
    for key, value in sample.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def list_collate(batch: list[dict]) -> list[dict]:
    """KEAFNet consumes a list of per-sample dicts and batches them itself."""
    return batch
