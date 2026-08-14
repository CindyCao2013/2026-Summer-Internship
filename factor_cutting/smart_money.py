"""Smart Money factor — Kaiyuan 聪明钱 (Option B rolling 10d).

Paper 步骤1: 回溯过去 10 个交易日分钟 → S 降序 → cumvol top 20% → Q.

Locked identity: SmartMoney10d, β=0.25 (Stage-0). No Active_* fields.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from factor_cutting.engine import CuttingSpec, KnifeSpec, ObjectSpec, OutputSpec

LOOKBACK_DAYS = 10
TOP_CUMVOL_PCT = 0.20
BETA_DEFAULT = 0.25
MIN_MINUTES_DEFAULT = 50

SMART_MONEY_SPEC = CuttingSpec(
    name="smart_money",
    paper="聪明钱因子模型",
    direction_paper="negative_ic",
    status="implemented_minute_optB",
    object=ObjectSpec(variable="minute_vwap", additive=False),
    knife=KnifeSpec(
        variable="smart_score",
        method="cumvol_top_pct",
        window=LOOKBACK_DAYS,
        formula=f"abs(ret_1m) / volume_1m ** {BETA_DEFAULT}",
    ),
    output=OutputSpec(op="ratio", formula="VWAP_smart / VWAP_all"),
)


def _q_one_window(
    score: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    close: np.ndarray,
    *,
    top_cumvol_pct: float,
) -> float:
    """Cumvol top-pct VWAP ratio for one (symbol, date) window."""
    if len(score) == 0:
        return np.nan
    valid = (
        np.isfinite(score)
        & np.isfinite(volume)
        & (volume > 0)
        & np.isfinite(close)
    )
    if not np.any(valid):
        return np.nan

    sc = score[valid]
    vol = volume[valid]
    am = np.asarray(amount[valid], dtype=float)
    cl = close[valid]
    # Amount preferred; fallback Close*Volume
    need_fb = (~np.isfinite(am)) | (am <= 0)
    am = np.where(need_fb, cl * vol, am)
    ok = np.isfinite(am) & (am > 0)
    if not np.any(ok):
        return np.nan
    sc, vol, am = sc[ok], vol[ok], am[ok]

    order = np.lexsort((-vol, -sc))  # S desc, then volume desc
    vol_s = vol[order]
    am_s = am[order]
    v_tot = float(vol_s.sum())
    if v_tot <= 0:
        return np.nan

    cum = np.cumsum(vol_s)
    cutoff = top_cumvol_pct * v_tot
    prev = np.concatenate([[0.0], cum[:-1]])
    take = prev < cutoff  # include crossing bar

    v_smart = float(vol_s[take].sum())
    a_smart = float(am_s[take].sum())
    if v_smart <= 0:
        return np.nan
    vwap_smart = a_smart / v_smart
    vwap_all = float(am_s.sum()) / v_tot
    if vwap_all <= 0 or not np.isfinite(vwap_smart) or not np.isfinite(vwap_all):
        return np.nan
    return float(vwap_smart / vwap_all)


def compute_daily_smart_money_q(
    minutes: pd.DataFrame,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    top_cumvol_pct: float = TOP_CUMVOL_PCT,
    min_minutes: int = MIN_MINUTES_DEFAULT,
    dates: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Option B: for each (symbol, date T), pool last `lookback_days` trading days.

    Required columns: date, symbol, bartime, close, volume, amount, smart_score.
    Returns long [date, symbol, Q].
    """
    need = {"date", "symbol", "volume", "amount", "close", "smart_score"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"minutes missing columns: {missing}")

    df = minutes.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    sort_cols = ["symbol", "date"]
    if "bartime" in df.columns:
        sort_cols.append("bartime")
    df = df.sort_values(sort_cols)

    want = None
    if dates is not None:
        want = set(pd.Timestamp(x) for x in pd.to_datetime(list(dates)))

    rows: list[dict] = []
    for sym, g in df.groupby("symbol", sort=False):
        by_date = {pd.Timestamp(d): sub for d, sub in g.groupby("date", sort=True)}
        sym_dates = sorted(by_date.keys())
        if len(sym_dates) < lookback_days:
            continue
        for i in range(lookback_days - 1, len(sym_dates)):
            t = sym_dates[i]
            if want is not None and t not in want:
                continue
            window_dates = sym_dates[i - lookback_days + 1 : i + 1]
            w = pd.concat([by_date[d] for d in window_dates], ignore_index=True)
            if len(w) < min_minutes:
                rows.append({"date": t, "symbol": sym, "Q": np.nan})
                continue
            q = _q_one_window(
                w["smart_score"].to_numpy(dtype=float),
                w["volume"].to_numpy(dtype=float),
                w["amount"].to_numpy(dtype=float),
                w["close"].to_numpy(dtype=float),
                top_cumvol_pct=top_cumvol_pct,
            )
            rows.append({"date": t, "symbol": sym, "Q": q})

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "Q"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def compute_smart_money(*_args, **_kwargs):
    """Prefer panel builder / compute_daily_smart_money_q."""
    raise NotImplementedError(
        "Use core.l2_features.smart_money_panel_builder.build_smart_money10d_panel "
        "or compute_daily_smart_money_q on minute_feature frames. "
        "Do not approximate with daily OHLCV; do not use Active_*."
    )
