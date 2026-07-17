"""Configuration dataclasses for KEAF-Net.

The defaults reproduce the reported configuration (Table 4 of the paper). A run
is fully described by a YAML file under ``configs/``; :func:`load_config` merges
it onto these defaults so that a config only needs to list what it overrides.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 768            # shared d for visual / textual / knowledge
    region_feat_dim: int = 2048      # Faster R-CNN (Visual Genome) region features
    knowledge_feat_dim: int = 384    # Sentence-BERT all-MiniLM-L6-v2 triplet embeddings

    # Adaptive Knowledge Filter (Section 3.4)
    max_triplets: int = 50          # P, retrieval cap
    akf_threshold_init: float = 0.5  # c_th initialisation (learned end-to-end)
    akf_temperature: float = 0.1     # T_temp in Eq. (12)
    akf_loo_samples: int = 10        # |S|, sampled leave-one-out triplets

    # Heterogeneous Graph Adaptive Fusion (Section 3.5)
    gat_heads: int = 8
    gat_layers: int = 2
    leaky_relu_slope: float = 0.2
    iou_threshold: float = 0.3       # visual-visual edges
    token_window: int = 3            # textual-textual edges
    visual_text_sim: float = 0.4     # visual-textual cosine edges

    # Multi-Hop Semantic Reasoning (Section 3.6)
    num_hops: int = 3                # T

    num_answers: int = 3129          # classifier vocabulary
    dropout: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 1e-4
    min_lr: float = 1e-6             # cosine anneal target
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 20
    batch_size: int = 64
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.05
    early_stop_patience: int = 3
    grad_clip: float = 5.0
    akf_loss_weight: float = 0.3     # w_AKF in Eq. (23)


@dataclass
class DataConfig:
    dataset: str = "okvqa"
    data_root: str = "data"
    feature_root: str = "features"
    knowledge_cache: str = "knowledge/cache"
    num_regions: int = 36            # M Faster R-CNN proposals
    num_workers: int = 4
    answer_vocab: str = "data/answer_vocab.json"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    experiment: str = "keaf_net"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(section: Any, override: dict[str, Any]) -> Any:
    valid = {f.name for f in fields(section)}
    for key, value in override.items():
        if key not in valid:
            raise KeyError(f"unknown config key '{key}' for {type(section).__name__}")
        setattr(section, key, value)
    return section


def load_config(path: str | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config`, optionally layering a YAML file and kwargs."""
    cfg = Config()
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    raw = {**raw, **overrides}
    for name, sub in (("model", cfg.model), ("optim", cfg.optim), ("data", cfg.data)):
        if name in raw:
            _merge(sub, raw.pop(name))
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise KeyError(f"unknown top-level config key '{key}'")
        setattr(cfg, key, value)
    return cfg
