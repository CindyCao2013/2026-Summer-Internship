"""Half-year rolling train / predict. Final factor = mean over ``n_seeds`` runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from alphanet.config import AlphaNetConfig, MAX_WORKERS, TrainConfig
from alphanet.data import MarketPanel, sample_dates
from alphanet.dataset import AlphaNetDataset, in_sample_window, split_train_val_dates
from alphanet.paths import FACTORS, MODELS, ensure_result_dirs
from alphanet.train import make_loader, train_one


@dataclass
class RollingFold:
    asof: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def retrain_asofs(
    calendar: pd.DatetimeIndex,
    start,
    end,
    months: int,
    warmup_days: int = 0,
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(calendar)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = idx[(idx >= start_ts) & (idx <= end_ts)]
    if dates.empty:
        return dates
    if warmup_days:
        warm_loc = idx.searchsorted(dates[0], side="left") + int(warmup_days) - 1
        if warm_loc >= len(idx):
            return pd.DatetimeIndex([])
        first_ok = idx[warm_loc]
        dates = dates[dates >= first_ok]
    if dates.empty:
        return dates
    marks = [dates[0]]
    cursor = dates[0]
    while True:
        nxt = cursor + pd.DateOffset(months=int(months))
        later = dates[dates >= nxt]
        if later.empty:
            break
        marks.append(later[0])
        cursor = later[0]
    return pd.DatetimeIndex(marks)


def build_folds(calendar: pd.DatetimeIndex, cfg: AlphaNetConfig) -> List[RollingFold]:
    idx = pd.DatetimeIndex(calendar)
    asofs = retrain_asofs(
        idx,
        cfg.start,
        cfg.end,
        cfg.train.retrain_months,
        warmup_days=cfg.train.in_sample_days,
    )
    folds: List[RollingFold] = []
    for i, asof in enumerate(asofs):
        is_start, is_end = in_sample_window(idx, asof, cfg.train.in_sample_days)
        train_dates, val_dates = split_train_val_dates(
            idx, cfg.train.sample_every, is_start, is_end, train_frac=cfg.train.train_frac
        )
        if len(train_dates) == 0 or len(val_dates) == 0:
            continue
        oos_start = asof
        oos_end = asofs[i + 1] - pd.Timedelta(days=1) if i + 1 < len(asofs) else pd.Timestamp(cfg.end)
        oos_end = min(oos_end, idx[-1])
        folds.append(
            RollingFold(
                asof=pd.Timestamp(asof),
                train_start=pd.Timestamp(train_dates[0]),
                train_end=pd.Timestamp(train_dates[-1]),
                val_start=pd.Timestamp(val_dates[0]),
                oos_start=pd.Timestamp(oos_start),
                oos_end=pd.Timestamp(oos_end),
            )
        )
    return folds


def assert_fold_no_leak(fold: RollingFold) -> None:
    if fold.train_end > fold.val_start:
        raise AssertionError("train_end {} > val_start {}".format(fold.train_end, fold.val_start))
    if fold.train_end > fold.oos_start:
        raise AssertionError("train uses dates after OOS start")


def _predict_panel(
    model,
    panel: MarketPanel,
    dates,
    lookback: int,
    device: str,
    train_cfg: TrainConfig,
) -> pd.DataFrame:
    import torch

    ds = AlphaNetDataset(
        panel,
        dates,
        train_cfg=train_cfg,
        lookback=lookback,
        require_label=False,
        feature_names=panel.meta.get("feature_names"),
    )
    if len(ds) == 0:
        return pd.DataFrame()
    model.eval()
    rows = []
    batch = 512
    with torch.no_grad():
        for start in range(0, len(ds), batch):
            sl = slice(start, start + batch)
            xs = torch.stack([ds[i][0] for i in range(sl.start, min(sl.stop, len(ds)))]).to(device)
            pred = model(xs).detach().cpu().numpy().reshape(-1)
            for j, p in enumerate(pred):
                rec = ds.index[start + j]
                rows.append((rec.date, rec.symbol, float(p)))
    out = pd.DataFrame(rows, columns=["date", "symbol", "value"])
    return out.pivot(index="date", columns="symbol", values="value").sort_index()


def rolling_predict(
    panel: MarketPanel,
    cfg: AlphaNetConfig,
    *,
    n_seeds: Optional[int] = None,
    device: Optional[str] = None,
    max_folds: Optional[int] = None,
) -> pd.DataFrame:
    """Train on each fold, predict OOS, average seeds. Returns factor wide table."""
    import torch

    ensure_result_dirs()
    n_seeds = int(n_seeds or cfg.train.n_seeds)
    if n_seeds > MAX_WORKERS:
        n_seeds = MAX_WORKERS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    folds = build_folds(panel.calendar, cfg)
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    panel.meta["train_cfg"] = cfg.train
    panel.meta["feature_names"] = cfg.model.feature_names
    acc: List[pd.DataFrame] = []
    for fold in folds:
        assert_fold_no_leak(fold)
        is_start, is_end = in_sample_window(panel.calendar, fold.asof, cfg.train.in_sample_days)
        train_dates, val_dates = split_train_val_dates(
            panel.calendar, cfg.train.sample_every, is_start, is_end, train_frac=cfg.train.train_frac
        )
        train_ds = AlphaNetDataset(
            panel,
            train_dates,
            cfg.train,
            lookback=cfg.model.lookback,
            feature_names=cfg.model.feature_names,
        )
        val_ds = AlphaNetDataset(
            panel,
            val_dates,
            cfg.train,
            lookback=cfg.model.lookback,
            feature_names=cfg.model.feature_names,
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue
        oos_dates = sample_dates(
            panel.calendar, cfg.eval.rebalance_every, start=fold.oos_start, end=fold.oos_end
        )
        seed_panels = []
        for k in range(n_seeds):
            seed = cfg.train.seed0 + k
            result = train_one(
                make_loader(train_ds, cfg.train.batch_size, True),
                make_loader(val_ds, cfg.train.batch_size, False),
                model_cfg=cfg.model,
                train_cfg=cfg.train,
                seed=seed,
                device=device,
                ckpt_dir=MODELS / cfg.variant / str(fold.asof.date()),
                tag=cfg.variant,
            )
            pred = _predict_panel(
                result.model,
                panel,
                oos_dates,
                cfg.model.lookback,
                device,
                cfg.train,
            )
            seed_panels.append(pred)
        if not seed_panels:
            continue
        stacked = pd.concat(seed_panels, keys=range(len(seed_panels)))
        mean = stacked.groupby(level=1).mean()
        acc.append(mean)
    if not acc:
        return pd.DataFrame()
    factor = pd.concat(acc).sort_index()
    factor = factor[~factor.index.duplicated(keep="last")]
    out = FACTORS / "{}_factor.parquet".format(cfg.variant)
    factor.to_parquet(out)
    return factor
