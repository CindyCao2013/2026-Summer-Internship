"""Train / infer Scheme-B cross-sectional F² model."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .agents.torch_agents import CrossSectionalF2Model
from .config import Config
from .data.cross_sectional_dataset import CrossSectionalDayDataset


# ========== Pairwise Ranking Loss (Scheme B primary objective) ==========
class PairwiseRankingLoss(nn.Module):
    """
    成对排序损失：让 "多头票" 的得分 > "空头票" 的得分
    输入:
        scores: (batch, n_stocks) 每只票的 logit_LONG - logit_SHORT
        long_mask: (batch, n_stocks) 哪些票是真正上涨的（标签 LONG=2）
        short_mask: (batch, n_stocks) 哪些票是真正下跌的（标签 SHORT=0）
    """

    def __init__(self, margin: float = 0.1):
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
            # Vectorized pairwise: (n_long, n_short)
            diff = s_b[long_idx].unsqueeze(1) - s_b[short_idx].unsqueeze(0)
            hinge = torch.clamp(self.margin - diff, min=0)
            batch_loss = batch_loss + hinge.sum()
            count += int(hinge.numel())

        if count > 0:
            return batch_loss / count
        return scores.new_zeros(())


def _accuracy(logits: torch.Tensor, y: torch.Tensor, tradable: torch.Tensor = None) -> float:
    pred = logits.argmax(dim=-1)
    if tradable is None:
        return float((pred == y).float().mean().item())
    mask = tradable > 0.5
    if mask.sum() == 0:
        return 0.0
    return float((pred[mask] == y[mask]).float().mean().item())


def score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """P(LONG) - P(SHORT) ranking score, shape (B, S)."""
    probs = F.softmax(logits, dim=-1)
    return probs[..., 2] - probs[..., 0]


def score_logit_spread(logits: torch.Tensor) -> torch.Tensor:
    """logit_LONG - logit_SHORT (Prompt variant; less stable for CE mix)."""
    return logits[..., 2] - logits[..., 0]


def pairwise_topbottom_loss(
    scores: torch.Tensor,
    fwd_ret: torch.Tensor,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
) -> torch.Tensor:
    """Push scores of top-return names above bottom-return names (per day)."""
    bsz, n = scores.shape
    k_top = max(1, int(round(n * top_frac)))
    k_bot = max(1, int(round(n * bottom_frac)))
    losses = []
    for i in range(bsz):
        order = torch.argsort(fwd_ret[i], descending=True)
        long_idx = order[:k_top]
        short_idx = order[-k_bot:]
        margin = scores[i, long_idx].mean() - scores[i, short_idx].mean()
        losses.append(F.softplus(-margin))
    return torch.stack(losses).mean()


@torch.no_grad()
def ranking_hit_rate(scores: torch.Tensor, fwd_ret: torch.Tensor) -> float:
    """Fraction of stock pairs where score order matches return order."""
    bsz, n = scores.shape
    hits = 0.0
    tot = 0.0
    for i in range(bsz):
        s = scores[i]
        r = fwd_ret[i]
        for a in range(n):
            for b in range(a + 1, n):
                dr = r[a] - r[b]
                if float(dr.abs()) < 1e-12:
                    continue
                tot += 1.0
                if float((s[a] - s[b]) * dr) > 0:
                    hits += 1.0
    return hits / max(tot, 1.0)


def combined_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    fwd_ret: torch.Tensor,
    ce_criterion: nn.Module,
    ranking_criterion: PairwiseRankingLoss,
    ce_weight: float,
    rank_weight: float,
    top_frac: float,
    bottom_frac: float,
    use_label_pairwise: bool = True,
    use_ret_topbottom: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    scores = score_from_logits(logits)
    rank_parts = []
    label_rank = scores.new_zeros(())
    ret_rank = scores.new_zeros(())
    if use_label_pairwise:
        label_rank = ranking_criterion(scores, y == 2, y == 0)
        rank_parts.append(label_rank)
    if use_ret_topbottom:
        ret_rank = pairwise_topbottom_loss(scores, fwd_ret, top_frac=top_frac, bottom_frac=bottom_frac)
        rank_parts.append(ret_rank)
    rank = torch.stack(rank_parts).mean() if rank_parts else scores.new_zeros(())
    ce = ce_criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss = float(rank_weight) * rank + float(ce_weight) * ce
    return loss, {
        "ce": float(ce.item()),
        "rank": float(rank.item()),
        "rank_label": float(label_rank.item()) if use_label_pairwise else 0.0,
        "rank_ret": float(ret_rank.item()) if use_ret_topbottom else 0.0,
    }


@torch.no_grad()
def evaluate_cs(
    model: CrossSectionalF2Model,
    loader: DataLoader,
    device: torch.device,
    ce_criterion: nn.Module,
    ranking_criterion: PairwiseRankingLoss,
    ce_weight: float,
    rank_weight: float,
    top_frac: float,
    bottom_frac: float,
    use_label_pairwise: bool,
    use_ret_topbottom: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_rank_hit = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        fwd = batch["fwd_ret"].to(device)
        trad = batch["tradable"].to(device)
        ind = batch.get("industry_ids")
        if ind is not None:
            ind = ind.to(device)
        logits, _ = model(x, industry_ids=ind)
        loss, _ = combined_loss(
            logits,
            y,
            fwd,
            ce_criterion,
            ranking_criterion,
            ce_weight,
            rank_weight,
            top_frac,
            bottom_frac,
            use_label_pairwise=use_label_pairwise,
            use_ret_topbottom=use_ret_topbottom,
        )
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits, y, trad) * bs
        total_rank_hit += ranking_hit_rate(score_from_logits(logits), fwd) * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
        "rank_hit": total_rank_hit / max(n, 1),
        "n": float(n),
    }


def train_cs_model(
    config: Config,
    train_ds: CrossSectionalDayDataset,
    val_ds: CrossSectionalDayDataset,
) -> Tuple[CrossSectionalF2Model, Dict[str, object], torch.device]:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[cs-torch] device =", device)

    use_ind = bool(getattr(config, "use_industry_embed", True))
    n_ind = int(len(train_ds.industry_vocab)) if use_ind else 0
    model = CrossSectionalF2Model(
        n_features=train_ds.n_features,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=max(2, int(config.num_layers)),
        dropout=config.dropout,
        n_classes=config.n_classes,
        n_industries=n_ind,
    ).to(device)
    print("[cs-train] d_model={} layers={} n_industries={}".format(config.d_model, config.num_layers, n_ind))

    counts = np.bincount(train_ds.y.reshape(-1).astype(int), minlength=config.n_classes).astype(
        np.float64
    )
    counts = np.maximum(counts, 1.0)
    class_w = counts.sum() / (config.n_classes * counts)
    long_boost = float(getattr(config, "long_class_boost", 3.0))
    class_w[2] *= long_boost
    class_w = class_w / class_w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))
    print(
        "[cs-train] class_counts={} weights={} long_boost={}".format(
            counts.tolist(), class_w.tolist(), long_boost
        )
    )

    ce_weight = float(getattr(config, "ce_loss_weight", 0.3))
    rank_weight = float(getattr(config, "rank_loss_weight", 1.0))
    margin = float(getattr(config, "ranking_margin", 0.1))
    top_frac = float(getattr(config, "rotation_top_frac", 0.2))
    bottom_frac = float(getattr(config, "rotation_bottom_frac", 0.2))
    use_label_pairwise = bool(getattr(config, "use_label_pairwise", False))
    use_ret_topbottom = bool(getattr(config, "use_ret_topbottom", True))
    ranking_criterion = PairwiseRankingLoss(margin=margin)
    print(
        "[cs-train] loss=Rank*{:.2f}(label={}, retTB={}, margin={:.2f})+CE*{:.2f}".format(
            rank_weight, use_label_pairwise, use_ret_topbottom, margin, ce_weight
        )
    )

    batch_size = min(int(config.batch_size), 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if len(val_ds) > 0
        else None
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    epochs = int(getattr(config, "cs_epochs", config.epochs))

    history = []
    best_val = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_acc = 0.0
        running_rank = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            fwd = batch["fwd_ret"].to(device)
            trad = batch["tradable"].to(device)
            ind = batch.get("industry_ids")
            if ind is not None:
                ind = ind.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, industry_ids=ind)
            loss, parts = combined_loss(
                logits,
                y,
                fwd,
                criterion,
                ranking_criterion,
                ce_weight,
                rank_weight,
                top_frac,
                bottom_frac,
                use_label_pairwise=use_label_pairwise,
                use_ret_topbottom=use_ret_topbottom,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = x.size(0)
            running += float(loss.item()) * bs
            running_acc += _accuracy(logits.detach(), y, trad) * bs
            running_rank += parts["rank"] * bs
            seen += bs
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "train_acc": running_acc / max(seen, 1),
            "train_rank_loss": running_rank / max(seen, 1),
        }
        if val_loader is not None:
            vm = evaluate_cs(
                model,
                val_loader,
                device,
                criterion,
                ranking_criterion,
                ce_weight,
                rank_weight,
                top_frac,
                bottom_frac,
                use_label_pairwise,
                use_ret_topbottom,
            )
            row["val_loss"] = vm["loss"]
            row["val_acc"] = vm["accuracy"]
            row["val_rank_hit"] = vm["rank_hit"]
            score = float(vm["rank_hit"])
            if score >= best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        print(
            "[cs-epoch {:02d}] train_loss={:.4f} train_acc={:.3f} rankL={:.4f} | "
            "val_loss={} val_acc={} val_rank_hit={}".format(
                epoch,
                row["train_loss"],
                row["train_acc"],
                row["train_rank_loss"],
                "{:.4f}".format(row["val_loss"]) if "val_loss" in row else "n/a",
                "{:.3f}".format(row["val_acc"]) if "val_acc" in row else "n/a",
                "{:.3f}".format(row["val_rank_hit"]) if "val_rank_hit" in row else "n/a",
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[cs-train] restored best val_rank_hit={:.4f}".format(best_val))

    meta = {
        "history": history,
        "best_val_rank_hit": best_val,
        "device": str(device),
        "class_weights": class_w.tolist(),
        "long_class_boost": long_boost,
        "ce_loss_weight": ce_weight,
        "rank_loss_weight": rank_weight,
        "ranking_margin": margin,
        "use_label_pairwise": use_label_pairwise,
        "use_ret_topbottom": use_ret_topbottom,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    return model, meta, device


@torch.no_grad()
def predict_cs_scores(
    model: CrossSectionalF2Model,
    ds: CrossSectionalDayDataset,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Return soft long-short scores + class probs for each day/stock."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_scores = []
    all_probs = []
    all_pred = []
    for batch in loader:
        x = batch["x"].to(device)
        ind = batch.get("industry_ids")
        if ind is not None:
            ind = ind.to(device)
        logits, _ = model(x, industry_ids=ind)
        probs = F.softmax(logits, dim=-1)
        scores = probs[..., 2] - probs[..., 0]
        all_scores.append(scores.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_pred.append(logits.argmax(dim=-1).cpu().numpy())
    return {
        "scores": np.concatenate(all_scores, axis=0),  # (N, S)
        "probs": np.concatenate(all_probs, axis=0),  # (N, S, 3)
        "pred": np.concatenate(all_pred, axis=0),  # (N, S)
        "dates": ds.dates,
        "next_dates": ds.next_dates,
        "open_px": ds.open_px,
        "next_close_px": ds.next_close_px,
        "tradable_exec": ds.tradable_exec,
        "symbols": list(ds.symbols),
    }


@torch.no_grad()
def predict_cs_proba(
    model: CrossSectionalF2Model,
    ds: CrossSectionalDayDataset,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Return class probabilities shaped (N, S, 3)."""
    return predict_cs_scores(model, ds, device, batch_size=batch_size)["probs"]

