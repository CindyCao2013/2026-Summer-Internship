"""Modality-aware fusion for F² Agent Lite.

Sklearn gate fusion is preserved in ``fusion_layer_sklearn.py``.
Default export is the PyTorch ``ModalityAwareFusion``.
"""

from __future__ import annotations

try:
    from sideprojects.f2_agent_lite.agents.torch_agents import ModalityAwareFusion
except Exception:  # pragma: no cover
    ModalityAwareFusion = None  # type: ignore

from .fusion_layer_sklearn import FusionLayer  # noqa: F401

__all__ = ["ModalityAwareFusion", "FusionLayer"]
