"""L2 / minute microstructure bricks + EOD proxy bricks.

Bricks are market-state primitives, NOT alpha signals.
When L2 is unavailable, OHLCV proxies approximate intraday mechanics.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

_EPS = 1e-6

BRICK_NAMES = [
    "ofi_proxy",
    "price_impact_proxy",
    "liquidity_consumption_proxy",
    "spread_proxy",
    "aggressiveness_proxy",
    "volume_burst_proxy",
]


@dataclass
class EODMicrostructureInputs:
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame


def extract_eod_proxy_bricks(data: EODMicrostructureInputs) -> Dict[str, pd.DataFrame]:
    """Daily bricks from EOD OHLCV (one row per date per stock)."""
    close, open_, high, low = data.close, data.open, data.high, data.low
    volume, amount = data.volume, data.amount
    ret_1d = close / close.shift(1) - 1

    ofi_proxy = np.sign(close - open_) * volume
    price_impact = ret_1d.abs() / volume.replace(0, np.nan)
    spread_proxy = (high - low) / close.replace(0, np.nan)
    depth_proxy = amount.replace(0, np.nan)
    liquidity_consumption = volume / depth_proxy
    aggressiveness = (close - low) / (high - low + _EPS)
    vol_mean_20 = volume.rolling(20, min_periods=10).mean()
    volume_burst = volume / vol_mean_20.replace(0, np.nan)

    return {
        "ofi_proxy": ofi_proxy,
        "price_impact_proxy": price_impact,
        "liquidity_consumption_proxy": liquidity_consumption,
        "spread_proxy": spread_proxy,
        "aggressiveness_proxy": aggressiveness,
        "volume_burst_proxy": volume_burst,
    }


def extract_minute_bricks(
    minute_df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    bid_col: Optional[str] = None,
    ask_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract intraday bricks from narrow minute table.

    Expected columns: tradetime, symbol, + price/volume fields.
    Returns long table with brick columns per row.
    """
    df = minute_df.copy()
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df = df.sort_values(["symbol", "tradetime"])

    grp = df.groupby("symbol", group_keys=False)
    df["ret_1m"] = grp[price_col].pct_change()
    df["ofi"] = np.nan
    if bid_col and ask_col and bid_col in df.columns and ask_col in df.columns:
        mid = (df[bid_col] + df[ask_col]) / 2
        df["ofi"] = np.sign(df[price_col] - mid) * df[volume_col]
    else:
        df["ofi"] = np.sign(df["ret_1m"]) * df[volume_col]

    df["price_impact"] = df["ret_1m"].abs() / df[volume_col].replace(0, np.nan)
    if bid_col and ask_col:
        df["spread"] = (df[ask_col] - df[bid_col]) / df[price_col].replace(0, np.nan)
    else:
        df["spread"] = np.nan

    df["volume_burst"] = df[volume_col] / grp[volume_col].transform(
        lambda x: x.rolling(30, min_periods=5).mean()
    )
    return df


def bricks_to_daily_wide(
    minute_bricks: pd.DataFrame,
    brick_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """Compress intraday brick column to daily wide (Date x Symbol)."""
    df = minute_bricks.copy()
    df["date"] = pd.to_datetime(df["tradetime"]).dt.normalize()
    if agg == "mean":
        daily = df.groupby(["date", "symbol"])[brick_col].mean()
    elif agg == "sum":
        daily = df.groupby(["date", "symbol"])[brick_col].sum()
    else:
        raise ValueError(f"Unknown agg: {agg}")
    return daily.unstack("symbol")
