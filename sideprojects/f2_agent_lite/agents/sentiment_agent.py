"""Sentiment agent: scalar score -> fixed d_model via deterministic basis + scale fit."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    """Map daily sentiment in [-1,1] to d_model using polynomial + Fourier features."""

    name = "sentiment"

    def __init__(self, d_model: int = 32):
        self.d_model = d_model
        self.scaler = StandardScaler()
        self._fitted = False

    def _expand(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float).reshape(-1)
        s = np.clip(np.nan_to_num(s, nan=0.0), -1.0, 1.0)
        feats = [s, s ** 2, s ** 3, np.abs(s), (s > 0).astype(float), (s < 0).astype(float)]
        # Fourier-like bases to fill d_model
        k = 1
        while len(feats) < self.d_model:
            feats.append(np.sin(k * np.pi * s))
            if len(feats) >= self.d_model:
                break
            feats.append(np.cos(k * np.pi * s))
            k += 1
        mat = np.stack(feats[: self.d_model], axis=1)
        return mat

    def fit(self, X, y: np.ndarray | None = None) -> "SentimentAgent":
        mat = self._expand(X)
        self.scaler.fit(mat)
        self._fitted = True
        return self

    def encode(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SentimentAgent is not fitted")
        return self.scaler.transform(self._expand(X)).astype(float)
