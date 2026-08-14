"""Decision MLP classifier (sklearn)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class Classifier:
    def __init__(
        self,
        hidden: Tuple[int, ...] = (64, 32),
        max_iter: int = 200,
        learning_rate_init: float = 1e-3,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.model.fit(Xs, np.asarray(y, dtype=int))
        self._fitted = True
        return self

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        proba = self.model.predict_proba(Xs)
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(Xs), dtype=float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba_up(X) >= threshold).astype(int)
