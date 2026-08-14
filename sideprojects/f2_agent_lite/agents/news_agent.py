"""News agent: Chinese char TF-IDF + TruncatedSVD -> d_model."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_agent import BaseAgent


class NewsAgent(BaseAgent):
    name = "news"

    def __init__(
        self,
        d_model: int = 32,
        max_features: int = 2048,
        random_state: int = 42,
    ):
        self.d_model = d_model
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            min_df=1,
        )
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, X, y: np.ndarray | None = None) -> "NewsAgent":
        texts = [str(t) if str(t).strip() else " " for t in np.asarray(X).tolist()]
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = int(min(self.d_model, max(1, tfidf.shape[1] - 1), max(1, tfidf.shape[0] - 1)))
        # TruncatedSVD requires n_components < n_features for sparse in some versions
        n_comp = max(1, min(n_comp, tfidf.shape[1] - 1 if tfidf.shape[1] > 1 else 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def encode(self, X) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError("NewsAgent is not fitted")
        texts = [str(t) if str(t).strip() else " " for t in np.asarray(X).tolist()]
        tfidf = self.vectorizer.transform(texts)
        z = self.svd.transform(tfidf)
        if z.shape[1] < self.d_model:
            pad = np.zeros((z.shape[0], self.d_model - z.shape[1]), dtype=float)
            z = np.concatenate([z, pad], axis=1)
        return z.astype(float)
