"""KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for VQA.

Reference implementation of the AKF / HGAF / MHSR pipeline described in
"KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for Visual Question
Answering with Multi-Hop Graph Reasoning".
"""
from .config import Config, load_config
from .models import KEAFNet
from .structures import KEAFBatch

__all__ = ["KEAFNet", "Config", "load_config", "KEAFBatch", "__version__"]

__version__ = "0.1.0"
