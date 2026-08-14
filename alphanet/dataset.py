"""Torch Dataset of AlphaNet (9, T) pictures plus CS-zscored forward returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphanet.config import FEATURE_NAMES, TrainConfig
from alphanet.data import MarketPanel, cs_zscore, forward_return, sample_dates
from alphanet.synthetic import stack_images
from alphanet.universe import apply_mask

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


@dataclass
class SampleIndex:
    date: pd.Timestamp
    symbol: str
    row: int


class AlphaNetDataset(Dataset):
    """Materializes images for a list of section dates.

    Labels are cross-section z-scores of the m-day forward return on that date.
    CS stats use only stocks present on the same date (no future dates).
    """

    def __init__(
        self,
        panel: MarketPanel,
        dates: Sequence[pd.Timestamp],
        train_cfg: TrainConfig,
        lookback: int = 30,
        require_label: bool = True,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.panel = panel
        self.lookback = int(lookback)
        self.train_cfg = train_cfg
        self.require_label = bool(require_label)
        self.feature_names = tuple(
            feature_names
            if feature_names is not None
            else panel.meta.get("feature_names", FEATURE_NAMES)
        )
        self.images: List[np.ndarray] = []
        self.labels: List[float] = []
        self.index: List[SampleIndex] = []
        self._build(dates)

    def _build(self, dates: Sequence[pd.Timestamp]) -> None:
        horizon = int(self.train_cfg.horizon)
        fwd = forward_return(self.panel.ret_1d, horizon, self.train_cfg.execution)
        min_obs = min(int(self.train_cfg.label_min_obs), max(8, int(self.panel.ret_1d.shape[1]) // 2))
        if self.train_cfg.label_cs_zscore:
            y_panel = cs_zscore(apply_mask(fwd, self.panel.tradable), min_obs=min_obs)
        else:
            y_panel = apply_mask(fwd, self.panel.tradable)
        calendar = self.panel.calendar
        for date in dates:
            ts = pd.Timestamp(date)
            if ts not in calendar:
                continue
            loc = calendar.get_loc(ts)
            if loc + 1 < self.lookback:
                continue
            images, symbols = stack_images(
                self.panel.features,
                ts,
                self.lookback,
                symbols=self.panel.symbols,
                feature_names=self.feature_names,
            )
            if self.panel.tradable is not None:
                mask_row = self.panel.tradable.loc[ts].reindex(symbols)
                keep = mask_row.eq(1).fillna(False).to_numpy()
                images = images[keep]
                symbols = symbols[keep]
            if images.shape[0] == 0:
                continue
            y = y_panel.loc[ts].reindex(symbols)
            if self.require_label:
                ok = y.notna().to_numpy()
                images = images[ok]
                symbols = symbols[ok]
                y = y[ok]
            for i, sym in enumerate(symbols):
                label = float(y.iloc[i]) if self.require_label or pd.notna(y.iloc[i]) else np.nan
                if self.require_label and not np.isfinite(label):
                    continue
                self.images.append(images[i])
                self.labels.append(label)
                self.index.append(SampleIndex(date=ts, symbol=str(sym), row=len(self.images) - 1))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = self.images[i]
        y = np.float32(self.labels[i])
        if torch is None:
            return x, y
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y)


def split_train_val_dates(
    calendar: pd.DatetimeIndex,
    every: int,
    start,
    end,
    train_frac: float = 0.5,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Time-ordered split of sampled section dates (train first, val last)."""
    dates = sample_dates(calendar, every, start=start, end=end)
    if len(dates) < 2:
        return dates, dates[:0]
    frac = min(max(float(train_frac), 1e-6), 1.0 - 1e-6)
    n_train = int(round(len(dates) * frac))
    n_train = min(max(n_train, 1), len(dates) - 1)
    return dates[:n_train], dates[n_train:]


def in_sample_window(
    calendar: pd.DatetimeIndex,
    asof,
    lookback_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive [start, end] covering the last ``lookback_days`` sessions at asof."""
    idx = pd.DatetimeIndex(calendar)
    asof_ts = pd.Timestamp(asof)
    loc = idx.searchsorted(asof_ts, side="right") - 1
    if loc < 0:
        raise ValueError("asof {} precedes calendar".format(asof))
    start_loc = max(0, loc - int(lookback_days) + 1)
    return idx[start_loc], idx[loc]
