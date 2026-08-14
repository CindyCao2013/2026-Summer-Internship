"""Cross-sectional day dataset: one sample = all stocks on one day.

Shape per sample:
  x: (n_stocks, lookback, n_features)
  y: (n_stocks,) int labels in {0,1,2}
  fwd_ret: (n_stocks,) open->next_close return for ranking loss
  meta for backtest: dates, open/close/tradable panels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_loader import DataLoader, SymbolPanel, _zscore_window, resolve_feature_cols
from .industry import SYMBOL_INDUSTRY, industry_ids_for_symbols


# Default (no alpha flags) — prefer resolve_feature_cols(config) at runtime
FEATURE_COLS = resolve_feature_cols(None)


@dataclass
class CSSplit:
    train: "CrossSectionalDayDataset"
    val: "CrossSectionalDayDataset"
    test: "CrossSectionalDayDataset"
    symbols: List[str]


class CrossSectionalDayDataset(Dataset):
    """Each index is a signal day; tensors cover the full stock universe."""

    def __init__(
        self,
        symbols: Sequence[str],
        X: np.ndarray,
        y: np.ndarray,
        fwd_ret: np.ndarray,
        dates: np.ndarray,
        next_dates: np.ndarray,
        open_px: np.ndarray,
        next_close_px: np.ndarray,
        tradable_exec: np.ndarray,
        feature_cols: Sequence[str],
        industry_ids: Optional[Sequence[int]] = None,
        industry_names: Optional[Sequence[str]] = None,
        close_px: Optional[np.ndarray] = None,
    ):
        self.symbols = list(symbols)
        self.X = X.astype(np.float32)  # (N, S, T, F)
        self.y = y.astype(np.int64)  # (N, S)
        self.fwd_ret = fwd_ret.astype(np.float32)  # (N, S)
        self.dates = np.asarray(dates)
        self.next_dates = np.asarray(next_dates)
        self.open_px = open_px.astype(np.float32)  # (N, S)
        self.next_close_px = next_close_px.astype(np.float32)
        self.close_px = (
            close_px.astype(np.float32)
            if close_px is not None
            else np.full_like(self.open_px, np.nan)
        )
        self.tradable_exec = tradable_exec.astype(bool)
        self.feature_cols = list(feature_cols)
        self.n_stocks = len(self.symbols)
        self.n_features = len(self.feature_cols)
        if industry_ids is None:
            ids, vocab = industry_ids_for_symbols(self.symbols)
            self.industry_ids = np.asarray(ids, dtype=np.int64)
            self.industry_vocab = list(vocab)
        else:
            self.industry_ids = np.asarray(industry_ids, dtype=np.int64)
            self.industry_vocab = list(industry_names) if industry_names is not None else []
        self.industry_names = [SYMBOL_INDUSTRY.get(s, "其他") for s in self.symbols]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return {
            "x": torch.from_numpy(self.X[idx]),
            "y": torch.from_numpy(self.y[idx]),
            "fwd_ret": torch.from_numpy(self.fwd_ret[idx]),
            "tradable": torch.from_numpy(self.tradable_exec[idx].astype(np.float32)),
            "industry_ids": torch.from_numpy(self.industry_ids),
        }

    def slice_range(self, start: int, end: int) -> "CrossSectionalDayDataset":
        return CrossSectionalDayDataset(
            self.symbols,
            self.X[start:end],
            self.y[start:end],
            self.fwd_ret[start:end],
            self.dates[start:end],
            self.next_dates[start:end],
            self.open_px[start:end],
            self.next_close_px[start:end],
            self.tradable_exec[start:end],
            self.feature_cols,
            industry_ids=self.industry_ids,
            industry_names=self.industry_vocab,
            close_px=self.close_px[start:end],
        )


def build_cs_arrays_from_panels(
    panels: Dict[str, SymbolPanel],
    symbols: Sequence[str],
    lookback: int,
    start,
    end,
    feature_cols: Optional[Sequence[str]] = None,
) -> Optional[CrossSectionalDayDataset]:
    """Build CS tensors for dates in [start, end] (inclusive)."""
    symbols = list(symbols)
    feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    common = None
    for sym in symbols:
        idx = panels[sym].daily.index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        return None
    common = pd.DatetimeIndex(sorted(common))

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    feat = np.full((len(common), len(symbols), len(feature_cols)), np.nan, dtype=float)
    label = np.full((len(common), len(symbols)), np.nan, dtype=float)
    fwd = np.full((len(common), len(symbols)), np.nan, dtype=float)
    open_next = np.full((len(common), len(symbols)), np.nan, dtype=float)
    close_now = np.full((len(common), len(symbols)), np.nan, dtype=float)
    close_next = np.full((len(common), len(symbols)), np.nan, dtype=float)
    next_date = np.empty(len(common), dtype="datetime64[ns]")
    next_date[:] = np.datetime64("NaT")
    tradable_next = np.zeros((len(common), len(symbols)), dtype=bool)

    for j, sym in enumerate(symbols):
        daily = panels[sym].daily.reindex(common)
        for k, col in enumerate(feature_cols):
            feat[:, j, k] = pd.to_numeric(daily[col], errors="coerce").to_numpy(dtype=float)
        label[:, j] = pd.to_numeric(daily["label"], errors="coerce").to_numpy(dtype=float)
        fwd[:, j] = pd.to_numeric(daily["fwd_ret"], errors="coerce").to_numpy(dtype=float)
        open_next[:, j] = pd.to_numeric(daily["next_open"], errors="coerce").to_numpy(dtype=float)
        close_now[:, j] = pd.to_numeric(daily["close"], errors="coerce").to_numpy(dtype=float)
        close_next[:, j] = pd.to_numeric(daily["next_close"], errors="coerce").to_numpy(dtype=float)
        tradable_next[:, j] = daily["next_tradable"].notna().to_numpy()
        nd = pd.to_datetime(daily["next_date"], errors="coerce")
        for i, v in enumerate(nd):
            if pd.notna(v) and np.isnat(next_date[i]):
                next_date[i] = np.datetime64(pd.Timestamp(v).to_datetime64())

    rows_x, rows_y, rows_fwd = [], [], []
    rows_dates, rows_next = [], []
    rows_open, rows_close, rows_close_now, rows_trad = [], [], [], []

    for i in range(lookback - 1, len(common)):
        d = pd.Timestamp(common[i]).normalize()
        if d < start_ts or d > end_ts:
            continue
        if np.isnan(label[i]).any() or np.isnan(fwd[i]).any():
            continue
        if np.isnan(open_next[i]).any() or np.isnan(close_next[i]).any():
            continue
        if np.isnat(next_date[i]):
            continue
        win = feat[i - lookback + 1 : i + 1]
        if np.isnan(win).any():
            win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        xz = np.zeros_like(win)
        for j in range(len(symbols)):
            xz[:, j, :] = _zscore_window(win[:, j, :])
        rows_x.append(np.transpose(xz, (1, 0, 2)))
        rows_y.append(label[i].astype(np.int64))
        rows_fwd.append(fwd[i].astype(np.float32))
        rows_dates.append(d.to_datetime64())
        rows_next.append(next_date[i])
        rows_open.append(open_next[i])
        rows_close.append(close_next[i])
        rows_close_now.append(close_now[i])
        rows_trad.append(tradable_next[i])

    if not rows_x:
        return None

    return CrossSectionalDayDataset(
        symbols=symbols,
        X=np.stack(rows_x, axis=0),
        y=np.stack(rows_y, axis=0),
        fwd_ret=np.stack(rows_fwd, axis=0),
        dates=np.asarray(rows_dates),
        next_dates=np.asarray(rows_next),
        open_px=np.stack(rows_open, axis=0),
        next_close_px=np.stack(rows_close, axis=0),
        tradable_exec=np.stack(rows_trad, axis=0),
        feature_cols=feature_cols,
        close_px=np.stack(rows_close_now, axis=0),
    )


def prepare_cs_splits(config, loader: Optional[DataLoader] = None) -> CSSplit:
    """Load panels (via cache-aware prepare_all) and build train/val/test CS sets."""
    loader = loader or DataLoader(config)
    packs = loader.prepare_all()
    panels: Dict[str, SymbolPanel] = packs["panels"]
    symbols = [s for s in config.symbols if s in panels]
    if len(symbols) < 2:
        raise RuntimeError("Need at least 2 symbols for cross-sectional model")

    feature_cols = resolve_feature_cols(config)
    print("[cs-data] feature_cols={} ({})".format(len(feature_cols), feature_cols), flush=True)
    train_full = build_cs_arrays_from_panels(
        panels,
        symbols,
        config.lookback_window,
        config.train_start,
        config.train_end,
        feature_cols=feature_cols,
    )
    test = build_cs_arrays_from_panels(
        panels,
        symbols,
        config.lookback_window,
        config.test_start,
        config.test_end,
        feature_cols=feature_cols,
    )
    if train_full is None or len(train_full) < 20:
        raise RuntimeError("Insufficient train CS samples")
    if test is None or len(test) == 0:
        raise RuntimeError("Insufficient test CS samples")

    n = len(train_full)
    n_val = max(1, int(round(n * float(config.val_ratio))))
    n_train = n - n_val
    train = train_full.slice_range(0, n_train)
    val = train_full.slice_range(n_train, n)
    print(
        "[cs-data] symbols={} train={} val={} test={} features={} horizon={} industries={}".format(
            len(symbols),
            len(train),
            len(val),
            len(test),
            train.n_features,
            getattr(config, "pred_horizon", 1),
            train.industry_vocab,
        )
    )
    return CSSplit(train=train, val=val, test=test, symbols=symbols)


def cs_predictions_to_signal_frame(
    ds: CrossSectionalDayDataset,
    probs: np.ndarray,
    industry_neutral: bool = False,
) -> pd.DataFrame:
    """Flatten (N,S,3) probs into long signal table for RotationBacktester."""
    rows = []
    for i in range(len(ds)):
        scores = probs[i, :, 2] - probs[i, :, 0]
        if industry_neutral:
            # demean within industry on this day
            adj = scores.copy()
            for ind in np.unique(ds.industry_ids):
                mask = ds.industry_ids == ind
                if mask.sum() >= 2:
                    adj[mask] = scores[mask] - scores[mask].mean()
                else:
                    adj[mask] = scores[mask]  # singleton industry: keep raw
            scores = adj
        for j, sym in enumerate(ds.symbols):
            p = probs[i, j]
            rows.append(
                {
                    "date": pd.Timestamp(ds.dates[i]),
                    "next_date": pd.Timestamp(ds.next_dates[i]),
                    "symbol": sym,
                    "industry": ds.industry_names[j],
                    "prob_short": float(p[0]),
                    "prob_hold": float(p[1]),
                    "prob_long": float(p[2]),
                    "score": float(scores[j]),
                    "class_id": int(np.argmax(p)),
                    "signal": int(np.argmax(p)) - 1,
                    "open_px": float(ds.open_px[i, j]),
                    "close_px": float(ds.close_px[i, j]),
                    "next_close_px": float(ds.next_close_px[i, j]),
                    "tradable_exec": bool(ds.tradable_exec[i, j]),
                    "y": int(ds.y[i, j]),
                }
            )
    return pd.DataFrame(rows)


def apply_score_smoothing(
    signals: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Causal per-symbol score smoothing: rolling mean then shift(1) (no look-ahead)."""
    out = signals.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["score_raw"] = out["score"].astype(float)
    w = max(1, int(window))
    out["score"] = (
        out.groupby("symbol", group_keys=False)["score_raw"]
        .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
    )
    # Warm-up: first day per symbol has NaN after shift → fall back to raw
    out["score"] = out["score"].fillna(out["score_raw"])
    return out
