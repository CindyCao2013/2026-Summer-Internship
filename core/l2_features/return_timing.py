"""Intraday return-timing primitives (Temporal Feature Layer v1).

Implements only Gu / Gd — the time centers of positive / negative minute returns.

NOT in this module (later stages):
  - residualization vs (Ru_bar, R1, R2, Rovernight)  → timing_residual.py (Stage 2)
  - εd ~ εu + MA20 / TGD20                            → tgd.py (Stage 3)
  - any change to the Flow Density pipeline

Definitions (one trading day, one symbol):
  r_t  = minute close-to-close return at trading-minute index t ∈ {0..T-1}
  Gu   = Σ_{r>0} t · r  /  Σ_{r>0} r     (up-return time center)
  Gd   = Σ_{r<0} t · |r| / Σ_{r<0} |r|   (down-return time center)

Missing / flat / one-sided days return NaN for the undefined center(s).

Interface for future TGD:
  DailyTimingCenters → residual layer → thin tgd builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# A-share continuous auction minutes used for default t-index (excludes lunch gap).
# Morning 09:31–11:30 (120) + Afternoon 13:01–15:00 (120) = 240 minutes.
DEFAULT_AM_START = (9, 31)
DEFAULT_AM_END = (11, 30)
DEFAULT_PM_START = (13, 1)
DEFAULT_PM_END = (15, 0)


def trading_minute_index(
    bartime: Union[pd.Series, Sequence],
    *,
    am_start: Tuple[int, int] = DEFAULT_AM_START,
    am_end: Tuple[int, int] = DEFAULT_AM_END,
    pm_start: Tuple[int, int] = DEFAULT_PM_START,
    pm_end: Tuple[int, int] = DEFAULT_PM_END,
) -> np.ndarray:
    """Map clock times to contiguous trading-minute indices (0-based).

    Lunch break is skipped so t is continuous across the session.
    Times outside the session map to NaN.
    """
    bt = pd.to_datetime(pd.Series(bartime))
    minutes = bt.dt.hour.to_numpy() * 60 + bt.dt.minute.to_numpy()

    am0 = am_start[0] * 60 + am_start[1]
    am1 = am_end[0] * 60 + am_end[1]
    pm0 = pm_start[0] * 60 + pm_start[1]
    pm1 = pm_end[0] * 60 + pm_end[1]
    am_len = am1 - am0 + 1

    out = np.full(len(minutes), np.nan, dtype=float)
    am_mask = (minutes >= am0) & (minutes <= am1)
    pm_mask = (minutes >= pm0) & (minutes <= pm1)
    out[am_mask] = minutes[am_mask] - am0
    out[pm_mask] = am_len + (minutes[pm_mask] - pm0)
    return out


def compute_minute_returns(close: np.ndarray) -> np.ndarray:
    """Close-to-close minute returns; first bar NaN."""
    c = np.asarray(close, dtype=float)
    out = np.full_like(c, np.nan, dtype=float)
    if c.size < 2:
        return out
    prev = c[:-1]
    curr = c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = curr / prev - 1.0
    r[~np.isfinite(r)] = np.nan
    # zero / negative prices → NaN
    bad = (prev <= 0) | (curr <= 0) | ~np.isfinite(prev) | ~np.isfinite(curr)
    r[bad] = np.nan
    out[1:] = r
    return out


def compute_up_time_center(
    returns: np.ndarray,
    time_index: np.ndarray,
) -> float:
    """Gu: return-weighted average of t where r > 0. NaN if no up minutes."""
    r = np.asarray(returns, dtype=float)
    t = np.asarray(time_index, dtype=float)
    mask = np.isfinite(r) & np.isfinite(t) & (r > 0)
    if not mask.any():
        return float("nan")
    w = r[mask]
    tw = t[mask]
    denom = w.sum()
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(tw, w) / denom)


def compute_down_time_center(
    returns: np.ndarray,
    time_index: np.ndarray,
) -> float:
    """Gd: |return|-weighted average of t where r < 0. NaN if no down minutes."""
    r = np.asarray(returns, dtype=float)
    t = np.asarray(time_index, dtype=float)
    mask = np.isfinite(r) & np.isfinite(t) & (r < 0)
    if not mask.any():
        return float("nan")
    w = -r[mask]  # positive weights
    tw = t[mask]
    denom = w.sum()
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(tw, w) / denom)


@dataclass(frozen=True)
class DailyTimingCenters:
    """One symbol × date return-timing snapshot.

    Future residual / TGD layers should consume this object (or its panel form)
    rather than re-parsing minute bars.
    """

    date: Optional[pd.Timestamp]
    symbol: Optional[str]
    Gu: float
    Gd: float
    n_up: int
    n_down: int
    n_flat: int
    n_missing: int
    mean_up: float
    mean_down: float
    session_ret: float

    @property
    def up_return_center(self) -> float:
        """Alias for Gu (API clarity)."""
        return self.Gu

    @property
    def down_return_center(self) -> float:
        """Alias for Gd (API clarity)."""
        return self.Gd


def _session_stats(returns: np.ndarray) -> Tuple[int, int, int, int, float, float, float]:
    r = np.asarray(returns, dtype=float)
    finite = np.isfinite(r)
    n_missing = int((~finite).sum())
    rf = r[finite]
    n_up = int((rf > 0).sum())
    n_down = int((rf < 0).sum())
    n_flat = int((rf == 0).sum())
    mean_up = float(rf[rf > 0].mean()) if n_up else float("nan")
    mean_down = float(rf[rf < 0].mean()) if n_down else float("nan")
    session_ret = float(np.nansum(rf)) if finite.any() else float("nan")
    return n_up, n_down, n_flat, n_missing, mean_up, mean_down, session_ret


def compute_timing_centers_from_arrays(
    returns: np.ndarray,
    time_index: np.ndarray,
    *,
    date: Optional[pd.Timestamp] = None,
    symbol: Optional[str] = None,
) -> DailyTimingCenters:
    """Compute Gu/Gd (+ diagnostics) for one day from aligned arrays."""
    Gu = compute_up_time_center(returns, time_index)
    Gd = compute_down_time_center(returns, time_index)
    n_up, n_down, n_flat, n_missing, mean_up, mean_down, session_ret = _session_stats(returns)
    return DailyTimingCenters(
        date=pd.Timestamp(date) if date is not None else None,
        symbol=symbol,
        Gu=Gu,
        Gd=Gd,
        n_up=n_up,
        n_down=n_down,
        n_flat=n_flat,
        n_missing=n_missing,
        mean_up=mean_up,
        mean_down=mean_down,
        session_ret=session_ret,
    )


def compute_timing_centers_daily(
    minute_df: pd.DataFrame,
    *,
    close_col: str = "close",
    bartime_col: str = "bartime",
    date_col: str = "date",
    symbol_col: str = "symbol",
    precomputed_returns_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate long minute bars → daily Gu/Gd panel (long format).

    Expected columns: date, symbol, bartime, close
    (or precomputed_returns_col instead of deriving from close).

    Returns DataFrame with columns:
      date, symbol, Gu, Gd, n_up, n_down, n_flat, n_missing,
      mean_up, mean_down, session_ret
    """
    required = {date_col, symbol_col, bartime_col}
    missing = required - set(minute_df.columns)
    if missing:
        raise ValueError(f"minute_df missing columns: {sorted(missing)}")
    if precomputed_returns_col is None and close_col not in minute_df.columns:
        raise ValueError(f"Need `{close_col}` or precomputed_returns_col")

    rows = []
    group_cols = [date_col, symbol_col]
    for (date, symbol), g in minute_df.groupby(group_cols, sort=False):
        g = g.sort_values(bartime_col)
        t_idx = trading_minute_index(g[bartime_col])
        if precomputed_returns_col is not None:
            rets = g[precomputed_returns_col].to_numpy(dtype=float)
        else:
            rets = compute_minute_returns(g[close_col].to_numpy(dtype=float))
        ctr = compute_timing_centers_from_arrays(
            rets, t_idx, date=pd.Timestamp(date), symbol=str(symbol)
        )
        rows.append(
            {
                "date": ctr.date,
                "symbol": ctr.symbol,
                "Gu": ctr.Gu,
                "Gd": ctr.Gd,
                "n_up": ctr.n_up,
                "n_down": ctr.n_down,
                "n_flat": ctr.n_flat,
                "n_missing": ctr.n_missing,
                "mean_up": ctr.mean_up,
                "mean_down": ctr.mean_down,
                "session_ret": ctr.session_ret,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "Gu",
                "Gd",
                "n_up",
                "n_down",
                "n_flat",
                "n_missing",
                "mean_up",
                "mean_down",
                "session_ret",
            ]
        )
    return pd.DataFrame(rows)


# --- Future TGD hook (stub only) -------------------------------------------------

def prepare_tgd_inputs(daily_centers: pd.DataFrame) -> pd.DataFrame:
    """Pass-through placeholder for Stage-2 residual / TGD builders.

    Expected future columns on top of Gu/Gd:
      Ru_bar, Rd_bar, R1, R2, Rovernight → εu, εd → TGD
    """
    need = {"date", "symbol", "Gu", "Gd"}
    if not need.issubset(daily_centers.columns):
        raise ValueError(f"prepare_tgd_inputs needs columns {need}")
    out = daily_centers.copy()
    out.attrs["tgd_stage"] = "centers_only"
    return out
