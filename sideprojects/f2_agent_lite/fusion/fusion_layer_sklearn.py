"""Fusion: concat or modality-gated weighted sum."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


class FusionLayer:
    def __init__(self, d_model: int = 32, mode: str = "gate", random_state: int = 42):
        if mode not in {"concat", "gate"}:
            raise ValueError("mode must be 'concat' or 'gate'")
        self.d_model = d_model
        self.mode = mode
        self.random_state = random_state
        self.gate_weights_: np.ndarray | None = None  # (4,)
        self.modality_names = ["market", "tech", "news", "sentiment"]
        self._fitted = False

    def fit(self, embeddings: Sequence[np.ndarray], y: np.ndarray) -> "FusionLayer":
        embs = [np.asarray(e, dtype=float) for e in embeddings]
        if len(embs) != 4:
            raise ValueError("expected 4 modality embeddings")
        if self.mode == "concat":
            self.gate_weights_ = np.ones(4) / 4.0
            self._fitted = True
            return self

        # Gate importance from per-modality logistic |coef| mass
        scores = []
        for e in embs:
            clf = LogisticRegression(
                max_iter=200,
                random_state=self.random_state,
                solver="lbfgs",
            )
            try:
                clf.fit(e, y)
                scores.append(float(np.mean(np.abs(clf.coef_))))
            except Exception:
                scores.append(1e-6)
        w = np.asarray(scores, dtype=float)
        w = np.maximum(w, 1e-8)
        # softmax
        w = np.exp(w - np.max(w))
        w = w / w.sum()
        self.gate_weights_ = w
        self._fitted = True
        return self

    def transform(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if not self._fitted or self.gate_weights_ is None:
            raise RuntimeError("FusionLayer is not fitted")
        embs = [np.asarray(e, dtype=float) for e in embeddings]
        if self.mode == "concat":
            return np.concatenate(embs, axis=1)

        w = self.gate_weights_.reshape(1, 4, 1)
        stacked = np.stack(embs, axis=1)  # (N, 4, d)
        return np.sum(stacked * w, axis=1)

    def gate_report(self) -> Dict[str, float]:
        if self.gate_weights_ is None:
            return {}
        return {k: float(v) for k, v in zip(self.modality_names, self.gate_weights_)}
