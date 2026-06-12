"""
Visual feature extraction (offline).

Extracts the two visual streams KEAF-Net consumes:
  * region features + boxes from a Faster R-CNN detector (bottom-up attention),
  * grid features from ViT-B/16.

torchvision's Faster R-CNN provides a practical detector when the original
bottom-up-attention weights are unavailable; the interface returns the same
`(region_feats, boxes, labels)` tuple either way. timm provides ViT.

These extractors are meant to be run once during preprocessing and cached to
disk (see `keafnet/preprocessing/extract_features.py`).

Reference: KEAF-Net, Section 3.2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.ops import roi_align
    _HAS_TV = True
except Exception:  # pragma: no cover
    _HAS_TV = False

try:
    import timm
    _HAS_TIMM = True
except Exception:  # pragma: no cover
    _HAS_TIMM = False

# COCO category names for the torchvision detector (index 0 is background).
COCO_CLASSES = [
    "__bg__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class RegionExtractor(nn.Module):
    """Faster R-CNN region features + boxes + labels (top-M by score)."""

    def __init__(self, num_regions: int = 36, score_thresh: float = 0.2) -> None:
        super().__init__()
        self.num_regions = num_regions
        self.score_thresh = score_thresh
        if _HAS_TV:
            self.detector = fasterrcnn_resnet50_fpn(weights="DEFAULT")
            self.detector.eval()
            self.backbone = self.detector.backbone
        else:  # pragma: no cover
            self.detector = None

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """image: (3, H, W) float in [0,1]. Returns (feats, boxes, labels)."""
        if self.detector is None:  # pragma: no cover
            m = self.num_regions
            return (np.zeros((m, 2048), np.float32),
                    np.zeros((m, 4), np.float32),
                    ["object"] * m)

        out = self.detector([image])[0]
        boxes = out["boxes"][: self.num_regions]
        scores = out["scores"][: self.num_regions]
        labels = out["labels"][: self.num_regions]

        # Pool backbone features at the predicted boxes via RoIAlign.
        feat_map = self.backbone(image.unsqueeze(0))
        level = feat_map["0"] if isinstance(feat_map, dict) else feat_map
        spatial_scale = level.shape[-1] / image.shape[-1]
        if boxes.numel() == 0:
            pooled = torch.zeros(0, level.shape[1], 7, 7)
        else:
            pooled = roi_align(level, [boxes], output_size=(7, 7),
                               spatial_scale=spatial_scale)
        region_feats = pooled.flatten(1)  # (n, C*7*7)
        # Project to 2048 for a stable interface.
        region_feats = _fixed_project(region_feats, 2048)

        m = region_feats.shape[0]
        names = [COCO_CLASSES[i] if i < len(COCO_CLASSES) else "object"
                 for i in labels.tolist()]
        feats = _pad(region_feats.cpu().numpy(), self.num_regions, 2048)
        bxs = _pad(boxes.cpu().numpy(), self.num_regions, 4)
        names = (names + ["object"] * self.num_regions)[: self.num_regions]
        return feats, bxs, names


class GridExtractor(nn.Module):
    """ViT-B/16 grid (patch-token) features."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        if _HAS_TIMM:
            self.vit = timm.create_model("vit_base_patch16_224", pretrained=True,
                                         num_classes=0)
            self.vit.eval()
        else:  # pragma: no cover
            self.vit = None
        self.dim = dim

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> np.ndarray:
        if self.vit is None:  # pragma: no cover
            return np.zeros((197, self.dim), np.float32)
        tok = self.vit.forward_features(image.unsqueeze(0))  # (1, T, D)
        return tok.squeeze(0).cpu().numpy().astype(np.float32)


def _fixed_project(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Deterministic linear projection (no learnable params) to a fixed dim,
    so cached features are reproducible across runs."""
    in_dim = x.shape[1]
    g = torch.Generator(device=x.device).manual_seed(0)
    w = torch.randn(in_dim, out_dim, generator=g, device=x.device) / (in_dim ** 0.5)
    return x @ w


def _pad(arr: np.ndarray, n: int, d: int) -> np.ndarray:
    out = np.zeros((n, d), dtype=np.float32)
    m = min(arr.shape[0], n)
    if m > 0:
        out[:m] = arr[:m, :d]
    return out
