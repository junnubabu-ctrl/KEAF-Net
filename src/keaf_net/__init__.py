"""KEAF-Net research implementation.

Components are intentionally modular so paper-specific choices can be audited,
replaced, and reproduced without hiding assumptions.
"""

from .model import KEAFNet, KEAFNetConfig

__all__ = ["KEAFNet", "KEAFNetConfig"]
