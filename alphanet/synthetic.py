"""Synthetic (9, T) panels for smoke tests and leakage checks. No database."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphanet.config import FEATURE_NAMES, N_FEATURES
from alphanet.ratios import add_ratio_features


@dataclass
class SyntheticPanel:
    features: Dict[str, pd.DataFrame]
    ret_1d: pd.DataFrame
    industry: pd.DataFrame
    log_mcap: pd.DataFrame
    tradable: pd.DataFrame
    calendar: pd.DatetimeIndex
    symbols: pd.Index


def make_synthetic_panel(
    n_days: int = 80,
    n_stocks: int = 40,
    n_industries: int = 5,
    seed: int = 0,
    start: str = "2018-01-02",
) -> SyntheticPanel:
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range(start, periods=n_days)
    symbols = pd.Index(["{:06d}.SZ".format(i) for i in range(n_stocks)])
    eps = rng.normal(0.0, 0.02, size=(n_days, n_stocks))
    ret_1d = pd.DataFrame(eps, index=calendar, columns=symbols)
    close = (1.0 + ret_1d).cumprod() * 10.0
    open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.005, close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.005, close.shape)))
    volume = pd.DataFrame(rng.lognormal(10, 0.3, close.shape), index=calendar, columns=symbols)
    turn = pd.DataFrame(np.abs(rng.normal(0.02, 0.01, close.shape)), index=calendar, columns=symbols)
    free_turn = turn * rng.uniform(0.6, 1.0, size=close.shape)
    vwap = (high + low + close) / 3.0
    features = {
        "return1": ret_1d.copy(),
        "open": open_,
        "close": close,
        "high": pd.DataFrame(high, index=calendar, columns=symbols),
        "low": pd.DataFrame(low, index=calendar, columns=symbols),
        "vwap": vwap,
        "volume": volume,
        "turn": turn,
        "free_turn": pd.DataFrame(free_turn, index=calendar, columns=symbols),
    }
    for name in FEATURE_NAMES:
        if name not in features:
            raise AssertionError(name)
    features = add_ratio_features(features)
    ind_codes = rng.integers(0, n_industries, size=n_stocks)
    industry = pd.DataFrame(
        np.broadcast_to(ind_codes, (n_days, n_stocks)),
        index=calendar,
        columns=symbols,
    )
    log_mcap = pd.DataFrame(
        np.log(rng.uniform(1e5, 1e7, size=(n_days, n_stocks))),
        index=calendar,
        columns=symbols,
    )
    tradable = pd.DataFrame(1.0, index=calendar, columns=symbols)
    # punch a few holes so mask logic is exercised
    tradable.iloc[::17, ::11] = np.nan
    return SyntheticPanel(
        features=features,
        ret_1d=ret_1d,
        industry=industry,
        log_mcap=log_mcap,
        tradable=tradable,
        calendar=calendar,
        symbols=symbols,
    )


def stack_images(
    features: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    lookback: int,
    symbols: Optional[pd.Index] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, pd.Index]:
    names = tuple(feature_names) if feature_names is not None else FEATURE_NAMES
    calendar = next(iter(features.values())).index
    loc = calendar.get_loc(pd.Timestamp(date))
    if isinstance(loc, slice):
        raise KeyError(date)
    if loc + 1 < lookback:
        raise ValueError("not enough history at {}".format(date))
    window_idx = calendar[loc + 1 - lookback : loc + 1]
    cols = symbols if symbols is not None else next(iter(features.values())).columns
    images = np.empty((len(cols), len(names), lookback), dtype=np.float32)
    valid = np.ones(len(cols), dtype=bool)
    for f_i, name in enumerate(names):
        block = features[name].reindex(index=window_idx, columns=cols).to_numpy(dtype=np.float64)
        images[:, f_i, :] = np.nan_to_num(block.T, nan=0.0)
        valid &= np.isfinite(block).all(axis=0)
    keep = pd.Index(cols)[valid]
    return images[valid], keep
