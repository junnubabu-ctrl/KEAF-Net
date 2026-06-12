"""
Dual-stream feature extraction.

Visual stream: ViT-B/16 grid features plus pre-extracted Faster R-CNN region
features (the detector runs offline; only its cached features are loaded here).
Textual stream: BERT-base token embeddings with the [CLS] vector as q_cls.

To keep the repository light and runnable without the full HuggingFace stack,
each encoder degrades gracefully to a lightweight learnable stub when
transformers/timm are unavailable (controlled by `pretrained=False`). The stub
preserves tensor shapes so the rest of the pipeline trains end-to-end.

Reference: KEAF-Net, Section 3.2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:  # optional heavy deps
    from transformers import BertModel, BertTokenizerFast
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False

try:
    import timm
    _HAS_TIMM = True
except Exception:  # pragma: no cover
    _HAS_TIMM = False


class VisualEncoder(nn.Module):
    """ViT-B/16 grid features fused with cached Faster R-CNN region features."""

    def __init__(self, dim: int = 768, num_regions: int = 36,
                 region_dim: int = 2048, pretrained: bool = True,
                 freeze_layers: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.num_regions = num_regions

        if pretrained and _HAS_TIMM:
            self.vit = timm.create_model("vit_base_patch16_224", pretrained=True,
                                         num_classes=0)
            vit_dim = self.vit.num_features
            # Freeze the first `freeze_layers` transformer blocks.
            blocks = getattr(self.vit, "blocks", [])
            for i, blk in enumerate(blocks):
                if i < freeze_layers:
                    for p in blk.parameters():
                        p.requires_grad = False
        else:
            self.vit = None
            vit_dim = dim

        self.grid_proj = nn.Linear(vit_dim, dim)
        self.region_proj = nn.Linear(region_dim, dim)

    def forward(self, images: torch.Tensor | None,
                region_feats: torch.Tensor | None,
                grid_feats: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            images:       (B, 3, 224, 224) raw images (used when vit is loaded).
            region_feats: (B, M, region_dim) cached Faster R-CNN features.
            grid_feats:   (B, G, vit_dim) precomputed grid features (optional).
        Returns:
            (B, G+M, dim) combined visual features V = [V_grid; V_region].
        """
        feats = []
        if grid_feats is not None:
            feats.append(self.grid_proj(grid_feats))
        elif self.vit is not None and images is not None:
            tokens = self.vit.forward_features(images)  # (B, T, vit_dim)
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(1)
            feats.append(self.grid_proj(tokens))
        if region_feats is not None:
            feats.append(self.region_proj(region_feats))
        if not feats:
            raise ValueError("VisualEncoder needs images, grid_feats, or region_feats")
        return torch.cat(feats, dim=1)


class TextEncoder(nn.Module):
    """BERT-base question encoder. Returns token states and the [CLS] vector."""

    def __init__(self, dim: int = 768, pretrained: bool = True,
                 freeze_embeddings: bool = True, vocab_size: int = 30522,
                 max_len: int = 64) -> None:
        super().__init__()
        self.dim = dim
        if pretrained and _HAS_HF:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            if freeze_embeddings:
                for p in self.bert.embeddings.parameters():
                    p.requires_grad = False
            bert_dim = self.bert.config.hidden_size
        else:
            # Lightweight stub: embedding + 2-layer transformer encoder.
            self.bert = None
            self.embed = nn.Embedding(vocab_size, dim)
            self.pos = nn.Embedding(max_len, dim)
            layer = nn.TransformerEncoderLayer(dim, nhead=8, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            bert_dim = dim
        self.proj = nn.Linear(bert_dim, dim) if bert_dim != dim else nn.Identity()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        """
        Returns:
            tokens: (B, L, dim) token embeddings T.
            q_cls:  (B, dim) [CLS] embedding.
        """
        if self.bert is not None:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq = self.proj(out.last_hidden_state)
        else:
            b, l = input_ids.shape
            pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
            x = self.embed(input_ids) + self.pos(pos)
            pad = None if attention_mask is None else (attention_mask == 0)
            seq = self.proj(self.encoder(x, src_key_padding_mask=pad))
        q_cls = seq[:, 0]
        return seq, q_cls
