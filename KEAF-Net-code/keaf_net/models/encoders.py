"""Dual-stream encoders for offline feature extraction (Section 3.2).

These wrap the frozen / partially-frozen backbones used to pre-extract features
once before training: ViT-B/16 grid patches, BERT-base question tokens and
Sentence-BERT triplet embeddings. They are deliberately *not* part of the
trainable :class:`KEAFNet`; the paper pre-extracts features so that training
focuses on the filter, graph fusion and reasoning. Region (Faster R-CNN)
features are produced by a standard bottom-up-attention detector and are loaded
from disk, so no detector wrapper is included here.

``transformers`` is imported lazily, so the rest of the package (and the unit
tests) run without the heavy dependencies installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def patch_boxes(image_size: int, patch_size: int = 16) -> torch.Tensor:
    """Bounding boxes (in pixels) for each ViT patch, ``[Np, 4]``.

    Grid patches tile the image, so each one has a well-defined box; this lets
    grid and region proposals share the visual-visual IoU rule in Algorithm 1.
    """
    side = image_size // patch_size
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    x1 = (xs * patch_size).flatten().float()
    y1 = (ys * patch_size).flatten().float()
    return torch.stack([x1, y1, x1 + patch_size, y1 + patch_size], dim=1)


class VisualEncoder(nn.Module):
    """ViT-B/16 (ImageNet-21K) grid features; layers 1-6 frozen."""

    def __init__(self, name: str = "google/vit-base-patch16-224-in21k", freeze_layers: int = 6):
        super().__init__()
        from transformers import ViTModel  # lazy

        self.vit = ViTModel.from_pretrained(name)
        for layer in self.vit.encoder.layer[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vit(pixel_values=pixel_values).last_hidden_state
        return out[:, 1:]  # drop [CLS], keep patch grid


class TextEncoder(nn.Module):
    """BERT-base-uncased question encoder; embedding layer frozen."""

    def __init__(self, name: str = "bert-base-uncased"):
        super().__init__()
        from transformers import BertModel, BertTokenizerFast  # lazy

        self.tokenizer = BertTokenizerFast.from_pretrained(name)
        self.bert = BertModel.from_pretrained(name)
        for p in self.bert.embeddings.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, questions: list[str], device: torch.device | str = "cpu"):
        enc = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt").to(device)
        out = self.bert(**enc).last_hidden_state
        return out, out[:, 0]  # token states, [CLS]


class TripletEncoder(nn.Module):
    """Sentence-BERT (all-MiniLM-L6-v2) over linearised triplets, Eq. (7)."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(name)

    @torch.no_grad()
    def forward(self, triplets: list[tuple[str, str, str]]) -> torch.Tensor:
        sentences = [f"{s} {r} {o}" for s, r, o in triplets]
        return self.model.encode(sentences, convert_to_tensor=True)
