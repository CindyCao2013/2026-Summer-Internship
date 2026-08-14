"""SmartMoneyActiveV2 — 主动大单集中度（聪明钱 2.0）.

Distinct from SmartMoney10d (Kaiyuan VWAP / S-score). Uses Active_* amount
concentration within each trading day, then EWM smooth, long − short.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from factor_cutting.engine import CuttingSpec, KnifeSpec, ObjectSpec, OutputSpec

TOP_AMT_PCT = 0.20  # top 20% → quantile threshold = 0.80
EWM_SPAN = 10
EWM_MIN_PERIODS = 5
MIN_MINUTES_DEFAULT = 30
MAD_NSIG = 5
MIN_CS_N = 30
MAX_HALT_FILL_DAYS = 20
FFILL_MAX_DAYS = 3

FORMULA_VERSION = "sm_active_v2_ewm10_top20pct"

SMART_MONEY_ACTIVE_V2_SPEC = CuttingSpec(
    name="smart_money_active_v2",
    paper="聪明钱因子2.0（主动大单集中度·研究版）",
    direction_paper="positive_ic",
    status="implemented_minute_active_conc",
    object=ObjectSpec(variable="active_buy_sell_amount", additive=False),
    knife=KnifeSpec(
        variable="active_amount",
        method="quantile_split",
        window=EWM_SPAN,
        formula="sum(top20% Active_*) / sum(all Active_*)",
    ),
    output=OutputSpec(
        op="difference",
        formula="ewm(smart_long) - ewm(smart_short)",
    ),
)


def _concentration_one_day(
    amounts: np.ndarray,
    *,
    top_pct: float = TOP_AMT_PCT,
    size_ok: Optional[np.ndarray] = None,
) -> float:
    """Fraction of total amount contributed by top-pct bars (by amount)."""
    am = np.asarray(amounts, dtype=float)
    if am.size == 0:
        return np.nan
    valid = np.isfinite(am) & (am >= 0)
    if size_ok is not None:
        valid = valid & np.asarray(size_ok, dtype=bool)
    am = am[valid]
    if am.size == 0:
        return np.nan
    total = float(am.sum())
    if total <= 0 or not np.isfinite(total):
        return np.nan
    q = 1.0 - top_pct
    thr = float(np.nanquantile(am, q))
    big = am[am >= thr]
    if big.size == 0:
        return np.nan
    return float(big.sum() / total)


def compute_daily_smart_active(
    minutes: pd.DataFrame,
    *,
    top_pct: float = TOP_AMT_PCT,
    min_minutes: int = MIN_MINUTES_DEFAULT,
    require_large_avg_size: bool = False,
) -> pd.DataFrame:
    """Per (symbol, date) smart_long / smart_short concentration.

    Required columns: date, symbol, active_buy_amt, active_sell_amt.
    Optional: avg_buy_size, avg_sell_size (for require_large_avg_size).
    Returns long [date, symbol, smart_long, smart_short].
    """
    need = {"date", "symbol", "active_buy_amt", "active_sell_amt"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"minutes missing columns: {missing}")

    df = minutes.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)

    rows: list[dict] = []
    for (sym, d), g in df.groupby(["symbol", "date"], sort=False):
        if len(g) < min_minutes:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "smart_long": np.nan,
                    "smart_short": np.nan,
                }
            )
            continue

        buy = g["active_buy_amt"].to_numpy(dtype=float)
        sell = g["active_sell_amt"].to_numpy(dtype=float)

        buy_ok = None
        sell_ok = None
        if require_large_avg_size:
            if "avg_buy_size" not in g.columns or "avg_sell_size" not in g.columns:
                raise ValueError(
                    "require_large_avg_size needs avg_buy_size / avg_sell_size"
                )
            buy_sz = g["avg_buy_size"].to_numpy(dtype=float)
            sell_sz = g["avg_sell_size"].to_numpy(dtype=float)
            med_b = np.nanmedian(buy_sz)
            med_s = np.nanmedian(sell_sz)
            buy_ok = np.isfinite(buy_sz) & (buy_sz >= med_b)
            sell_ok = np.isfinite(sell_sz) & (sell_sz >= med_s)

        rows.append(
            {
                "date": d,
                "symbol": sym,
                "smart_long": _concentration_one_day(
                    buy, top_pct=top_pct, size_ok=buy_ok
                ),
                "smart_short": _concentration_one_day(
                    sell, top_pct=top_pct, size_ok=sell_ok
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["date", "symbol", "smart_long", "smart_short"]
        )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def compute_daily_smart_active_fast(
    minutes: pd.DataFrame,
    *,
    top_pct: float = TOP_AMT_PCT,
    min_minutes: int = MIN_MINUTES_DEFAULT,
    require_large_avg_size: bool = False,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Faster path: pre-group by symbol, less pandas overhead."""
    need = {"date", "symbol", "active_buy_amt", "active_sell_amt"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"minutes missing columns: {missing}")

    df = minutes
    rows: list[dict] = []
    symbols = df["symbol"].astype(str).unique()
    n_sym = len(symbols)

    for si, (sym, g) in enumerate(df.groupby(df["symbol"].astype(str), sort=False)):
        if progress_every and si > 0 and si % progress_every == 0:
            print(f"  smart_active daily {si}/{n_sym} symbols ...", flush=True)
        for d, sub in g.groupby("date", sort=True):
            if len(sub) < min_minutes:
                rows.append(
                    {
                        "date": d,
                        "symbol": sym,
                        "smart_long": np.nan,
                        "smart_short": np.nan,
                    }
                )
                continue
            buy = sub["active_buy_amt"].to_numpy(dtype=float)
            sell = sub["active_sell_amt"].to_numpy(dtype=float)
            buy_ok = sell_ok = None
            if require_large_avg_size:
                buy_sz = sub["avg_buy_size"].to_numpy(dtype=float)
                sell_sz = sub["avg_sell_size"].to_numpy(dtype=float)
                med_b = np.nanmedian(buy_sz)
                med_s = np.nanmedian(sell_sz)
                buy_ok = np.isfinite(buy_sz) & (buy_sz >= med_b)
                sell_ok = np.isfinite(sell_sz) & (sell_sz >= med_s)
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "symbol": sym,
                    "smart_long": _concentration_one_day(
                        buy, top_pct=top_pct, size_ok=buy_ok
                    ),
                    "smart_short": _concentration_one_day(
                        sell, top_pct=top_pct, size_ok=sell_ok
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["date", "symbol", "smart_long", "smart_short"]
        )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def ewm_smooth_daily(
    daily: pd.DataFrame,
    *,
    span: int = EWM_SPAN,
    min_periods: int = EWM_MIN_PERIODS,
) -> pd.DataFrame:
    """Add smart_long_ewm, smart_short_ewm, smart_raw (= long − short)."""
    need = {"date", "symbol", "smart_long", "smart_short"}
    missing = need - set(daily.columns)
    if missing:
        raise ValueError(f"daily missing columns: {missing}")

    out = daily.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)
    out["smart_long_ewm"] = g["smart_long"].transform(
        lambda x: x.ewm(span=span, min_periods=min_periods).mean()
    )
    out["smart_short_ewm"] = g["smart_short"].transform(
        lambda x: x.ewm(span=span, min_periods=min_periods).mean()
    )
    out["smart_raw"] = out["smart_long_ewm"] - out["smart_short_ewm"]
    return out


def apply_minute_qc(
    minutes: pd.DataFrame,
    *,
    max_abs_ret: float = 0.20,
    min_adj_price: float = 0.01,
) -> pd.DataFrame:
    """Adjfactor adjust + drop bad bars.

    Normalized expected columns: open/high/low/close, amount, active_buy_amt,
    active_sell_amt, volume, adjfactor; optional active_buy_count/sell_count.
    """
    df = minutes.copy()
    if "adjfactor" not in df.columns:
        df["adjfactor"] = 1.0
    adj = df["adjfactor"].replace(0, np.nan).fillna(1.0)

    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = df[c] * adj
    for c in ("amount", "active_buy_amt", "active_sell_amt"):
        if c in df.columns:
            df[c] = df[c] * adj

    if "close" in df.columns:
        sort_cols = ["symbol", "date"]
        if "bartime" in df.columns:
            sort_cols.append("bartime")
        df = df.sort_values(sort_cols)
        prev = df.groupby(["symbol", "date"], sort=False)["close"].shift(1)
        ret = df["close"] / prev - 1.0
        bad_px = (~np.isfinite(df["close"])) | (df["close"] < min_adj_price)
        bad_ret = np.isfinite(ret) & (ret.abs() > max_abs_ret)
        df = df.loc[~(bad_px | bad_ret)].copy()

    if "active_buy_count" in df.columns:
        cnt = df["active_buy_count"].replace(0, np.nan)
        df["avg_buy_size"] = df["active_buy_amt"] / cnt
    if "active_sell_count" in df.columns:
        cnt = df["active_sell_count"].replace(0, np.nan)
        df["avg_sell_size"] = df["active_sell_amt"] / cnt

    df["net_active"] = df["active_buy_amt"] - df["active_sell_amt"]
    if "amount" in df.columns:
        df["active_ratio"] = (df["active_buy_amt"] + df["active_sell_amt"]) / df[
            "amount"
        ].replace(0, np.nan)

    return df


def mad_winsorize_cs(
    factor: pd.DataFrame,
    *,
    n_sig: float = MAD_NSIG,
) -> pd.DataFrame:
    """Cross-sectional MAD clip: median ± n_sig * MAD (raw MAD, no 1.4826)."""

    def _one(row: pd.Series) -> pd.Series:
        x = row.astype(float)
        med = x.median()
        mad = (x - med).abs().median()
        if not np.isfinite(mad) or mad == 0:
            return x
        return x.clip(med - n_sig * mad, med + n_sig * mad)

    return factor.apply(_one, axis=1)


def cs_zscore_min_n(factor: pd.DataFrame, *, min_n: int = MIN_CS_N) -> pd.DataFrame:
    """Row-wise z-score; rows with fewer than min_n finite values → NaN."""

    def _one(row: pd.Series) -> pd.Series:
        x = row.astype(float)
        n = int(np.isfinite(x.to_numpy()).sum())
        if n < min_n:
            return pd.Series(np.nan, index=x.index)
        mu = x.mean()
        sd = x.std()
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=x.index)
        return (x - mu) / sd

    return factor.apply(_one, axis=1)


def fill_industry_then_market(
    factor: pd.DataFrame,
    industry: pd.DataFrame,
    *,
    halt_mask: Optional[pd.DataFrame] = None,
    max_halt_days: int = MAX_HALT_FILL_DAYS,
) -> pd.DataFrame:
    """Fill NaN by industry median, then market median.

    When halt_mask is provided (1 = trading), consecutive missing / halt stretches
    longer than max_halt_days are left unfilled.
    """
    fac = factor.copy().astype(float)
    ind = industry.reindex_like(fac)

    if halt_mask is not None:
        hm = halt_mask.reindex_like(fac)
        miss = fac.isna() | hm.isna() | (hm != 1)
        long_gap = pd.DataFrame(False, index=fac.index, columns=fac.columns)
        for col in fac.columns:
            m = miss[col].to_numpy(dtype=bool)
            run = 0
            flags = np.zeros(len(m), dtype=bool)
            for i, v in enumerate(m):
                if v:
                    run += 1
                    if run > max_halt_days:
                        flags[i] = True
                else:
                    run = 0
            long_gap[col] = flags
    else:
        long_gap = pd.DataFrame(False, index=fac.index, columns=fac.columns)

    out = fac.copy()
    for dt_idx in out.index:
        row = out.loc[dt_idx]
        ind_row = ind.loc[dt_idx]
        filled = row.copy()
        for _ind_code, idxs in ind_row.groupby(ind_row).groups.items():
            if pd.isna(_ind_code):
                continue
            block = filled.loc[idxs]
            med = block.median()
            if np.isfinite(med):
                filled.loc[idxs] = block.fillna(med)
        mkt = filled.median()
        if np.isfinite(mkt):
            filled = filled.fillna(mkt)
        gap = long_gap.loc[dt_idx]
        filled = filled.where(~gap, np.nan)
        out.loc[dt_idx] = filled
    return out


def ffill_limited(
    factor: pd.DataFrame,
    *,
    max_days: int = FFILL_MAX_DAYS,
) -> pd.DataFrame:
    """Forward-fill each column at most max_days."""
    return factor.ffill(limit=max_days)
