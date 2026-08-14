"""Return distribution controls (Temporal Feature Layer Stage 2.5).

Explicitly separates amplitude statistics used by Stage-2 residualization:

  Rū = mean(r_t | r_t > 0)   — NOT mean of all minute returns
  Rd̄ = mean(r_t | r_t < 0)

Also exposes:
  zero_return_count  — # of finite minute bars with r == 0
  n_up / n_down
  limit_hit_flag     — optional (False unless limit columns provided)

These are controls / attribution helpers for TGD; not the TGD factor itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.l2_features.return_timing import compute_minute_returns


@dataclass(frozen=True)
class DailyReturnDistribution:
    date: Optional[pd.Timestamp]
    symbol: Optional[str]
    avg_up_return: float
    avg_down_return: float
    n_up: int
    n_down: int
    zero_return_count: int
    n_missing: int
    session_ret: float
    limit_hit_flag: bool


def compute_return_distribution(
    returns: np.ndarray,
    *,
    date: Optional[pd.Timestamp] = None,
    symbol: Optional[str] = None,
    limit_hit: bool = False,
) -> DailyReturnDistribution:
    """Sample means over up / down minutes only (+ zero-count diagnostics)."""
    r = np.asarray(returns, dtype=float)
    finite = np.isfinite(r)
    n_missing = int((~finite).sum())
    rf = r[finite]
    up = rf[rf > 0]
    down = rf[rf < 0]
    zero = rf[rf == 0]
    return DailyReturnDistribution(
        date=pd.Timestamp(date) if date is not None else None,
        symbol=symbol,
        avg_up_return=float(up.mean()) if up.size else float("nan"),
        avg_down_return=float(down.mean()) if down.size else float("nan"),
        n_up=int(up.size),
        n_down=int(down.size),
        zero_return_count=int(zero.size),
        n_missing=n_missing,
        session_ret=float(np.nansum(rf)) if finite.any() else float("nan"),
        limit_hit_flag=bool(limit_hit),
    )


def compute_return_distribution_daily(
    minute_df: pd.DataFrame,
    *,
    close_col: str = "close",
    bartime_col: str = "bartime",
    date_col: str = "date",
    symbol_col: str = "symbol",
    precomputed_returns_col: Optional[str] = None,
    limit_col: Optional[str] = None,
) -> pd.DataFrame:
    """Long minute bars → daily Rū / Rd̄ / zero_return_count panel."""
    required = {date_col, symbol_col, bartime_col}
    missing = required - set(minute_df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if precomputed_returns_col is None and close_col not in minute_df.columns:
        raise ValueError(f"Need `{close_col}` or precomputed_returns_col")

    rows = []
    for (date, symbol), g in minute_df.groupby([date_col, symbol_col], sort=False):
        g = g.sort_values(bartime_col)
        if precomputed_returns_col is not None:
            rets = g[precomputed_returns_col].to_numpy(dtype=float)
        else:
            rets = compute_minute_returns(g[close_col].to_numpy(dtype=float))
        hit = False
        if limit_col and limit_col in g.columns:
            hit = bool(np.nanmax(g[limit_col].to_numpy(dtype=float)) > 0)
        dist = compute_return_distribution(
            rets, date=pd.Timestamp(date), symbol=str(symbol), limit_hit=hit
        )
        rows.append(
            {
                "date": dist.date,
                "symbol": dist.symbol,
                "avg_up_return": dist.avg_up_return,
                "avg_down_return": dist.avg_down_return,
                "n_up": dist.n_up,
                "n_down": dist.n_down,
                "zero_return_count": dist.zero_return_count,
                "n_missing": dist.n_missing,
                "session_ret": dist.session_ret,
                "limit_hit_flag": dist.limit_hit_flag,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "avg_up_return",
                "avg_down_return",
                "n_up",
                "n_down",
                "zero_return_count",
                "n_missing",
                "session_ret",
                "limit_hit_flag",
            ]
        )
    return pd.DataFrame(rows)


def enrich_centers_with_distribution(
    centers: pd.DataFrame,
    distribution: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Merge Stage-1 Gu/Gd with Stage-2.5 amplitude controls (prefer distribution cols)."""
    left = centers.copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right = distribution.copy()
    right[date_col] = pd.to_datetime(right[date_col])
    keep = [
        date_col,
        symbol_col,
        "avg_up_return",
        "avg_down_return",
        "zero_return_count",
        "limit_hit_flag",
    ]
    keep = [c for c in keep if c in right.columns]
    out = left.merge(right[keep], on=[date_col, symbol_col], how="left", suffixes=("", "_dist"))
    # drop Stage-1 mean_up/mean_down if avg_* present (avoid dual truth)
    if "avg_up_return" in out.columns and "mean_up" in out.columns:
        out = out.drop(columns=["mean_up"], errors="ignore")
    if "avg_down_return" in out.columns and "mean_down" in out.columns:
        out = out.drop(columns=["mean_down"], errors="ignore")
    return out
