"""Market / Tech agents: flatten window + SVD projection (sklearn Lite)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent


class _WindowSVDAgent(BaseAgent):
    def __init__(self, d_model: int = 32, name: str = "window_svd", random_state: int = 42):
        self.d_model = d_model
        self.name = name
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"{self.name} expects (N,T,F), got {X.shape}")
        return np.nan_to_num(X.reshape(X.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_WindowSVDAgent":
        flat = self._flatten(X)
        n_features = flat.shape[1]
        n_comp = int(min(self.d_model, n_features, max(1, flat.shape[0] - 1)))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        scaled = self.scaler.fit_transform(flat)
        self.svd.fit(scaled)
        self._fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError(f"{self.name} is not fitted")
        flat = self._flatten(X)
        z = self.svd.transform(self.scaler.transform(flat))
        # Pad if n_comp < d_model
        if z.shape[1] < self.d_model:
            pad = np.zeros((z.shape[0], self.d_model - z.shape[1]), dtype=float)
            z = np.concatenate([z, pad], axis=1)
        return z.astype(float)


class MarketAgent(_WindowSVDAgent):
    def __init__(self, d_model: int = 32, random_state: int = 42):
        super().__init__(d_model=d_model, name="market", random_state=random_state)


class TechAgent(_WindowSVDAgent):
    def __init__(self, d_model: int = 32, random_state: int = 42):
        super().__init__(d_model=d_model, name="tech", random_state=random_state)
