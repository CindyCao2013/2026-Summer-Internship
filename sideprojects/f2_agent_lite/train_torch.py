"""PyTorch training / inference helpers for F² Agent Lite."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .agents.torch_agents import F2AgentModel, SentimentTorchAgent
from .config import Config
from .data.data_loader import WindowDataset


# ========== Pairwise Ranking Loss (shared; Scheme B uses train_cross_sectional) ==========
class PairwiseRankingLoss(nn.Module):
    """
    成对排序损失：让 "多头票" 的得分 > "空头票" 的得分
    输入:
        scores: (batch, n_stocks) 每只票的 P(LONG) - P(SHORT) / logit diff
        long_mask: (batch, n_stocks) 哪些票是真正上涨的（标签LONG）
        short_mask: (batch, n_stocks) 哪些票是真正下跌的（标签SHORT）
    """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores, long_mask, short_mask):
        batch_loss = scores.new_zeros(())
        count = 0

        for b in range(scores.shape[0]):
            long_idx = torch.where(long_mask[b])[0]
            short_idx = torch.where(short_mask[b])[0]

            if len(long_idx) == 0 or len(short_idx) == 0:
                continue

            s_b = scores[b]
            diff = s_b[long_idx].unsqueeze(1) - s_b[short_idx].unsqueeze(0)
            hinge = torch.clamp(self.margin - diff, min=0)
            batch_loss = batch_loss + hinge.sum()
            count += int(hinge.numel())

        if count > 0:
            return batch_loss / count
        return scores.new_zeros(())


class MultimodalTorchDataset(Dataset):
    def __init__(
        self,
        ds: WindowDataset,
        news_tfidf: np.ndarray,
        sentiment_feats: np.ndarray,
    ):
        self.market = ds.market.astype(np.float32)
        self.tech = ds.tech.astype(np.float32)
        self.news = news_tfidf.astype(np.float32)
        self.sent = sentiment_feats.astype(np.float32)
        self.y = ds.y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.market[idx],
            self.tech[idx],
            self.news[idx],
            self.sent[idx],
            self.y[idx],
        )


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def evaluate(model: F2AgentModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()
    attn_sum = None
    for market, tech, news, sent, y in loader:
        market = market.to(device)
        tech = tech.to(device)
        news = news.to(device)
        sent = sent.to(device)
        y = y.to(device)
        logits, attn = model(market, tech, news, sent)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits, y) * bs
        n += bs
        if attn_sum is None:
            attn_sum = attn.sum(dim=0)
        else:
            attn_sum = attn_sum + attn.sum(dim=0)
    out = {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
        "n": float(n),
    }
    if attn_sum is not None and n > 0:
        w = (attn_sum / n).detach().cpu().numpy()
        names = ["market", "tech", "news", "sentiment"]
        out["attn_mean"] = {names[i]: float(w[i]) for i in range(len(names))}
    return out


def train_f2_model(
    config: Config,
    train_ds: WindowDataset,
    val_ds: WindowDataset,
) -> Tuple[F2AgentModel, Dict[str, object], torch.device]:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[torch] device =", device)

    model = F2AgentModel(
        market_dim=train_ds.market.shape[-1],
        tech_dim=train_ds.tech.shape[-1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
        n_classes=config.n_classes,
        tfidf_max_features=config.tfidf_max_features,
    )
    model.fit_news_vectorizer(train_ds.news_text.tolist())
    model = model.to(device)

    news_train = model.news_agent.transform_numpy(train_ds.news_text.tolist())
    news_val = (
        model.news_agent.transform_numpy(val_ds.news_text.tolist())
        if len(val_ds.y) > 0
        else np.zeros((0, news_train.shape[1]), dtype=np.float32)
    )
    sent_train = SentimentTorchAgent.expand_features(train_ds.sentiment)
    sent_val = (
        SentimentTorchAgent.expand_features(val_ds.sentiment)
        if len(val_ds.y) > 0
        else np.zeros((0, 6), dtype=np.float32)
    )

    train_loader = DataLoader(
        MultimodalTorchDataset(train_ds, news_train, sent_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = None
    if len(val_ds.y) > 0:
        val_loader = DataLoader(
            MultimodalTorchDataset(val_ds, news_val, sent_val),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    counts = np.bincount(train_ds.y.astype(int), minlength=config.n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = counts.sum() / (config.n_classes * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_w, dtype=torch.float32, device=device)
    )
    print("[train] class_counts={} class_weights={}".format(counts.tolist(), class_w.tolist()))

    history = []
    best_val = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        running_acc = 0.0
        seen = 0
        for market, tech, news, sent, y in train_loader:
            market = market.to(device)
            tech = tech.to(device)
            news = news.to(device)
            sent = sent.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(market, tech, news, sent)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = y.size(0)
            running += float(loss.item()) * bs
            running_acc += _accuracy(logits.detach(), y) * bs
            seen += bs

        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "train_acc": running_acc / max(seen, 1),
        }
        if val_loader is not None:
            val_m = evaluate(model, val_loader, device)
            row.update({"val_loss": val_m["loss"], "val_acc": val_m["accuracy"], "attn_mean": val_m.get("attn_mean")})
            print(
                "Epoch {:02d} | train_acc={:.4f} val_acc={:.4f} attn={}".format(
                    epoch, row["train_acc"], row["val_acc"], row.get("attn_mean")
                )
            )
            if row["val_acc"] >= best_val:
                best_val = row["val_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            print("Epoch {:02d} | train_acc={:.4f}".format(epoch, row["train_acc"]))
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[train] restored best val_acc={:.4f}".format(best_val))

    metrics = {
        "best_val_acc": best_val if best_val >= 0 else None,
        "history": history,
        "device": str(device),
    }
    if val_loader is not None:
        metrics["final_val"] = evaluate(model, val_loader, device)
    return model, metrics, device


@torch.no_grad()
def predict_signals(
    model: F2AgentModel,
    ds: WindowDataset,
    device: torch.device,
    batch_size: int = 128,
):
    """Return class_id, trade_signal, attn, probs(N,3), score=P(long)-P(short)."""
    model.eval()
    news = model.news_agent.transform_numpy(ds.news_text.tolist())
    sent = SentimentTorchAgent.expand_features(ds.sentiment)
    loader = DataLoader(
        MultimodalTorchDataset(ds, news, sent),
        batch_size=batch_size,
        shuffle=False,
    )
    classes = []
    attns = []
    probs = []
    for market, tech, news_b, sent_b, _y in loader:
        logits, attn = model(
            market.to(device),
            tech.to(device),
            news_b.to(device),
            sent_b.to(device),
        )
        p = torch.softmax(logits, dim=1)
        classes.append(logits.argmax(dim=1).cpu().numpy())
        attns.append(attn.cpu().numpy())
        probs.append(p.cpu().numpy())
    class_id = np.concatenate(classes, axis=0)
    attn_arr = np.concatenate(attns, axis=0)
    prob_arr = np.concatenate(probs, axis=0)
    trade_signal = class_id.astype(np.int64) - 1  # 0,1,2 -> -1,0,1
    # classes: 0=SHORT, 1=HOLD, 2=LONG
    score = prob_arr[:, 2] - prob_arr[:, 0]
    return class_id, trade_signal, attn_arr, prob_arr, score
