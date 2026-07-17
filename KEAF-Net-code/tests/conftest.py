import pytest
import torch

from keaf_net.config import Config, ModelConfig


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model_cfg() -> ModelConfig:
    # small dimensions keep the CPU tests fast while exercising every path
    return ModelConfig(
        hidden_dim=64,
        region_feat_dim=128,
        knowledge_feat_dim=32,
        gat_heads=4,
        gat_layers=2,
        num_hops=3,
        num_answers=50,
        akf_loo_samples=4,
    )


@pytest.fixture
def cfg(model_cfg) -> Config:
    c = Config(model=model_cfg, device="cpu")
    c.optim.grad_clip = 5.0
    c.optim.akf_loss_weight = 0.3
    return c
