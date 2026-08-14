"""Abstract agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray | None = None) -> "BaseAgent":
        raise NotImplementedError

    @abstractmethod
    def encode(self, X: Any) -> np.ndarray:
        """Return (batch, d_model) float array."""
        raise NotImplementedError
