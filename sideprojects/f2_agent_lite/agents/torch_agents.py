"""PyTorch Transformer agents + multimodal F² model (Lite)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformerAgent(nn.Module):
    """Shared temporal Transformer for Market / Tech windows."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.pool_proj(x)


class NewsTfidfTorchAgent(nn.Module):
    """Char TF-IDF (fit offline) + learnable projection to d_model.

    sentence-transformers is unavailable offline; this keeps news modality
    differentiable after the sparse TF-IDF transform.
    """

    def __init__(
        self,
        d_model: int = 32,
        max_features: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            min_df=1,
        )
        self.proj: Optional[nn.Linear] = None
        self.dropout = nn.Dropout(dropout)
        self._fitted = False
        self._n_features = 0
        self._empty_mode = False

    def fit_vectorizer(self, texts: Sequence[str]) -> "NewsTfidfTorchAgent":
        clean = [str(t) if str(t).strip() else " " for t in texts]
        non_empty = [t for t in clean if t.strip() and t != " "]
        if len(non_empty) == 0:
            # No titles available (common when fetch_news_titles=False)
            self._empty_mode = True
            self._n_features = 1
            self.proj = nn.Linear(1, self.d_model)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self._fitted = True
            return self
        self._empty_mode = False
        mat = self.vectorizer.fit_transform(clean)
        self._n_features = int(mat.shape[1])
        if self._n_features <= 0:
            self._empty_mode = True
            self._n_features = 1
            self.proj = nn.Linear(1, self.d_model)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self._fitted = True
            return self
        self.proj = nn.Linear(self._n_features, self.d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self._fitted = True
        return self

    def transform_numpy(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("NewsTfidfTorchAgent vectorizer not fitted")
        n = len(list(texts))
        if self._empty_mode:
            return np.ones((n, 1), dtype=np.float32)
        clean = [str(t) if str(t).strip() else " " for t in texts]
        return self.vectorizer.transform(clean).astype(np.float32).toarray()

    def forward(self, tfidf_dense: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            raise RuntimeError("News projection layer not built; call fit_vectorizer")
        return self.dropout(torch.relu(self.proj(tfidf_dense)))


class SentimentTorchAgent(nn.Module):
    """Expand scalar sentiment to d_model via a small MLP."""

    def __init__(self, d_model: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def expand_features(scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        s = np.clip(np.nan_to_num(s, nan=0.0), -1.0, 1.0)
        return np.stack(
            [s, s ** 2, s ** 3, np.abs(s), (s > 0).astype(np.float32), (s < 0).astype(np.float32)],
            axis=1,
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class ModalityAwareFusion(nn.Module):
    """Cross-modal attention fusion (F²-style modality-aware attention, Lite)."""

    def __init__(self, d_model: int, num_modalities: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, modality_embeddings: Sequence[torch.Tensor]):
        stacked = torch.stack(list(modality_embeddings), dim=1)  # (B, M, D)
        q = self.query_proj(stacked.mean(dim=1, keepdim=True))  # (B, 1, D)
        k = self.key_proj(stacked)
        v = self.value_proj(stacked)
        attn = torch.matmul(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        fused = torch.matmul(attn, v).squeeze(1)
        return self.out_proj(fused), attn.squeeze(1)


class F2AgentModel(nn.Module):
    """End-to-end Market/Tech Transformer + News/Sentiment + fusion + 3-class head."""

    def __init__(
        self,
        market_dim: int = 5,
        tech_dim: int = 7,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_classes: int = 3,
        tfidf_max_features: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.market_agent = TimeSeriesTransformerAgent(
            market_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )
        self.tech_agent = TimeSeriesTransformerAgent(
            tech_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )
        self.news_agent = NewsTfidfTorchAgent(
            d_model=d_model, max_features=tfidf_max_features, dropout=dropout
        )
        self.sentiment_agent = SentimentTorchAgent(d_model=d_model, dropout=dropout)
        self.fusion = ModalityAwareFusion(d_model=d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def fit_news_vectorizer(self, texts: Sequence[str]) -> None:
        self.news_agent.fit_vectorizer(texts)
        # Projection Linear is registered on news_agent via Module.__setattr__.

    def forward(
        self,
        market: torch.Tensor,
        tech: torch.Tensor,
        news_tfidf: torch.Tensor,
        sentiment_feats: torch.Tensor,
    ):
        e_m = self.market_agent(market)
        e_t = self.tech_agent(tech)
        e_n = self.news_agent(news_tfidf)
        e_s = self.sentiment_agent(sentiment_feats)
        fused, attn = self.fusion([e_m, e_t, e_n, e_s])
        logits = self.classifier(fused)
        return logits, attn


class CrossSectionalF2Model(nn.Module):
    """Scheme B: shared temporal Transformer + cross-stock attention.

    Input:  (batch, n_stocks, seq_len, n_features)
    Output: logits (batch, n_stocks, n_classes), cross_attn (batch, n_heads, n_stocks, n_stocks)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 3,
        n_industries: int = 0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_industries = int(n_industries)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.industry_emb = (
            nn.Embedding(self.n_industries, d_model) if self.n_industries > 0 else None
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor, industry_ids: torch.Tensor = None):
        # x: (B, S, T, F) ; industry_ids: (B, S) or (S,)
        bsz, n_stocks, seq_len, n_feat = x.shape
        flat = x.reshape(bsz * n_stocks, seq_len, n_feat)
        h = self.input_proj(flat)
        h = self.pos_encoder(h)
        h = self.seq_encoder(h)
        stock_emb = h.mean(dim=1).reshape(bsz, n_stocks, self.d_model)
        if self.industry_emb is not None and industry_ids is not None:
            if industry_ids.dim() == 1:
                industry_ids = industry_ids.unsqueeze(0).expand(bsz, -1)
            stock_emb = stock_emb + self.industry_emb(industry_ids.long())
        cross_out, attn_w = self.cross_attn(
            stock_emb, stock_emb, stock_emb, need_weights=True, average_attn_weights=False
        )
        fused = self.norm(stock_emb + cross_out)
        logits = self.classifier(fused)
        return logits, attn_w

    def predict_proba(self, x: torch.Tensor, industry_ids: torch.Tensor = None) -> torch.Tensor:
        logits, _ = self.forward(x, industry_ids=industry_ids)
        return torch.softmax(logits, dim=-1)
