"""P0 SUE factor formulas — earliest-known announcement timeline.

Factors (raw event surprises, then transformed by hold/decay):
  - sue_np_yoy_z
  - sue_eps_consensus
  - analyst_np_revision_20d
  - unexpected_profit_notice_surprise_20d  (required ace candidate)
  - profit_notice_mid_surprise

Hard rules:
  - known_date = earliest among notice / express / income
  - no peeking past known_date
  - neutralize industry + ln_mktcap before Base3 residual IC (in runner)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from factor_attribution import cs_zscore
from industry_neutral import panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual

SUE_FACTOR_LIST = [
    "unexpected_profit_notice_surprise_20d",
    "sue_np_yoy_z",
    "sue_eps_consensus",
    "analyst_np_revision_20d",
    "profit_notice_mid_surprise",
]

HOLD_DAYS_DEFAULT = 20
DECAY_HALFLIFE_DEFAULT = 5


def _period_shift_years(period: str, years: int = 1) -> Optional[str]:
    """'20231231' -> '20221231'."""
    if not isinstance(period, str) or len(period) < 8:
        return None
    try:
        y = int(period[:4]) - years
        return f"{y}{period[4:8]}"
    except ValueError:
        return None


def _sparse_event_wide(
    events: pd.DataFrame,
    trade_index: pd.DatetimeIndex,
    columns: pd.Index,
    value_col: str,
    date_col: str = "known_date",
) -> pd.DataFrame:
    """Pivot event surprises onto trading calendar (NaN elsewhere)."""
    out = pd.DataFrame(np.nan, index=trade_index, columns=columns)
    if events is None or events.empty:
        return out
    ev = events.dropna(subset=[date_col, value_col, "symbol"]).copy()
    ev[date_col] = pd.to_datetime(ev[date_col])
    ev = ev[ev["symbol"].isin(columns)]
    if ev.empty:
        return out
    idx = trade_index.sort_values()
    pos = idx.searchsorted(ev[date_col].to_numpy())
    valid = pos < len(idx)
    ev = ev.iloc[np.asarray(valid)].copy()
    pos = pos[valid]
    ev["_td"] = idx[pos]
    wide = ev.pivot_table(index="_td", columns="symbol", values=value_col, aggfunc="last")
    return wide.reindex(index=trade_index, columns=columns)


def apply_event_hold(raw_events: pd.DataFrame, hold_days: int = HOLD_DAYS_DEFAULT) -> pd.DataFrame:
    """方式 A：事件日写入，之后最多 hold_days 个交易日 ffill。"""
    return raw_events.ffill(limit=hold_days)


def apply_daily_decay(
    raw_events: pd.DataFrame,
    half_life: int = DECAY_HALFLIFE_DEFAULT,
    horizon: int = HOLD_DAYS_DEFAULT,
) -> pd.DataFrame:
    """方式 B：事件日后指数衰减，超过 horizon 清零。

    Optimized: only iterate columns that ever have an event.
    """
    decay = 0.5 ** (1.0 / max(half_life, 1))
    active_cols = raw_events.columns[raw_events.notna().any(axis=0)]
    out = pd.DataFrame(np.nan, index=raw_events.index, columns=raw_events.columns)
    if len(active_cols) == 0:
        return out
    sub = raw_events[active_cols]
    state = np.full(len(active_cols), np.nan)
    age = np.zeros(len(active_cols), dtype=int)
    arr = sub.to_numpy(dtype=float)
    result = np.full(arr.shape, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        hit = ~np.isnan(row)
        if hit.any():
            state[hit] = row[hit]
            age[hit] = 0
        active = ~np.isnan(state)
        age[active & ~hit] += 1
        expired = age > horizon
        state[expired] = np.nan
        age[expired] = 0
        decay_mask = (~np.isnan(state)) & (~hit)
        state[decay_mask] *= decay
        result[i] = state
    out[active_cols] = result
    return out


def neutralize_size_industry(
    raw: pd.DataFrame,
    industry: pd.DataFrame,
    float_mktcap: pd.DataFrame,
) -> pd.DataFrame:
    """正交化市值 + 行业（硬要求 3）。"""
    ind = panel_industry_demean(raw, industry)
    log_size = np.log(float_mktcap.replace(0, np.nan))
    log_size = log_size.reindex(index=ind.index, columns=ind.columns)
    return panel_cross_sectional_residual(ind, [log_size])


def _yoy_sue_events(events: pd.DataFrame) -> pd.DataFrame:
    """On each disclosure that updates NP for period T, SUE = (NP_T - NP_{T-1y}) / σ(Δ)."""
    if events is None or events.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])

    ev = events.dropna(subset=["np_mid"]).sort_values(["symbol", "known_date"])
    best_np: Dict[Tuple[str, str], float] = {}
    hist_delta: Dict[str, list] = {}
    rows = []

    for sym, per, np_val, kd in zip(
        ev["symbol"].to_numpy(),
        ev["report_period"].astype(str).to_numpy(),
        ev["np_mid"].astype(float).to_numpy(),
        ev["known_date"].to_numpy(),
    ):
        best_np[(sym, per)] = np_val
        prev = _period_shift_years(per, 1)
        if prev is None:
            continue
        np_prev = best_np.get((sym, prev))
        if np_prev is None or not np.isfinite(np_prev):
            continue
        delta = np_val - np_prev
        hist = hist_delta.setdefault(sym, [])
        if len(hist) >= 4:
            sigma = float(np.std(hist[-8:], ddof=1))
        else:
            sigma = np.nan
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = max(abs(np_prev), 1.0)
        rows.append(
            {"symbol": sym, "known_date": kd, "surprise": delta / sigma, "report_period": per}
        )
        hist.append(delta)

    return pd.DataFrame(rows)


def _notice_unexpected_events(notice: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """unexpected_profit_notice_surprise: (notice_mid - mean past 4Q NP) / std past NP."""
    if notice is None or notice.empty or events is None or events.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])

    hist_last = (
        events.dropna(subset=["np_mid"])[["symbol", "report_period", "known_date", "np_mid"]]
        .assign(report_period=lambda x: x["report_period"].astype(str))
        .sort_values("known_date")
        .groupby(["symbol", "report_period"], as_index=False)
        .last()
    )
    hist_by_sym = {sym: g for sym, g in hist_last.groupby("symbol", sort=False)}

    n = notice.dropna(subset=["np_mid", "known_date"]).copy()
    n["report_period"] = n["report_period"].astype(str)
    rows = []
    for sym, ng in n.groupby("symbol", sort=False):
        past_all = hist_by_sym.get(sym)
        if past_all is None or past_all.empty:
            continue
        kd_arr = past_all["known_date"].to_numpy()
        per_arr = past_all["report_period"].to_numpy()
        np_arr = past_all["np_mid"].astype(float).to_numpy()
        for kd, per, np_mid in zip(
            ng["known_date"].to_numpy(),
            ng["report_period"].to_numpy(),
            ng["np_mid"].astype(float).to_numpy(),
        ):
            mask = (kd_arr < kd) & (per_arr != per)
            if mask.sum() < 2:
                continue
            # take last 4 by period string order among masked
            order = np.argsort(per_arr[mask])
            vals = np_arr[mask][order][-4:]
            if len(vals) < 2:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            if not np.isfinite(sd) or sd < 1e-6:
                sd = max(abs(mu), 1.0)
            rows.append(
                {
                    "symbol": sym,
                    "known_date": kd,
                    "surprise": (np_mid - mu) / sd,
                    "report_period": per,
                }
            )
    return pd.DataFrame(rows)


def _notice_mid_yoy_events(notice: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """profit_notice_mid vs same-period-last-year NP."""
    if notice is None or notice.empty or events is None or events.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])

    hist = (
        events.dropna(subset=["np_mid"])
        .assign(report_period=lambda x: x["report_period"].astype(str))
        .sort_values("known_date")
        .groupby(["symbol", "report_period"], as_index=False)
        .last()
    )
    hist = hist.rename(columns={"known_date": "prev_date", "np_mid": "prev_np"})
    n = notice.dropna(subset=["np_mid", "known_date"]).copy()
    n["report_period"] = n["report_period"].astype(str)
    n["prev_period"] = n["report_period"].map(lambda p: _period_shift_years(p, 1))
    merged = n.merge(
        hist[["symbol", "report_period", "prev_date", "prev_np"]].rename(
            columns={"report_period": "prev_period"}
        ),
        on=["symbol", "prev_period"],
        how="inner",
    )
    merged = merged[merged["prev_date"] < merged["known_date"]]
    if merged.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])
    denom = merged["prev_np"].abs().clip(lower=1.0)
    merged["surprise"] = (merged["np_mid"].astype(float) - merged["prev_np"].astype(float)) / denom
    return merged[["symbol", "known_date", "surprise", "report_period"]]


def _consensus_eps_sue_events(timeline: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """(actual_eps - consensus_eps) / |consensus| using consensus strictly before known_date."""
    if timeline is None or timeline.empty or consensus is None or consensus.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])
    tl = timeline.dropna(subset=["eps", "known_date"]).copy()
    tl["report_period"] = tl["report_period"].astype(str)
    # Prefer annual / quarterly periods that have EPS
    cons = consensus.dropna(subset=["est_dt", "eps_avg"]).copy()
    cons["report_period"] = cons["report_period"].astype(str)
    periods = set(tl["report_period"].unique())
    cons = cons[cons["report_period"].isin(periods)]
    # Keep last consensus per (symbol, period, est_dt) then join
    cons = cons.sort_values("est_dt").drop_duplicates(
        ["symbol", "report_period", "est_dt"], keep="last"
    )
    merged = tl.merge(
        cons[["symbol", "report_period", "est_dt", "eps_avg"]],
        on=["symbol", "report_period"],
        how="inner",
    )
    merged = merged[merged["est_dt"] < merged["known_date"]]
    if merged.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])
    idx = merged.groupby(["symbol", "report_period", "known_date"])["est_dt"].idxmax()
    best = merged.loc[idx]
    eps_c = best["eps_avg"].astype(float)
    mask = eps_c.notna() & (eps_c.abs() >= 1e-6)
    best = best.loc[mask].copy()
    best["surprise"] = (best["eps"].astype(float) - eps_c.loc[mask]) / eps_c.loc[mask].abs()
    return best[["symbol", "known_date", "surprise", "report_period"]]


def _analyst_revision_events(consensus: pd.DataFrame, window_days: int = 20) -> pd.DataFrame:
    """20d change in consensus NP — month-end snapshots to keep turnover low / compute tractable."""
    if consensus is None or consensus.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])
    c = (
        consensus.dropna(subset=["est_dt", "np_avg"])
        .assign(report_period=lambda x: x["report_period"].astype(str))
        .sort_values(["symbol", "report_period", "est_dt"])
        .drop_duplicates(["symbol", "report_period", "est_dt"], keep="last")
    )
    # Month-end est only (static / low-turnover spirit)
    c["_ym"] = pd.to_datetime(c["est_dt"]).dt.to_period("M")
    c = (
        c.sort_values("est_dt")
        .groupby(["symbol", "report_period", "_ym"], as_index=False)
        .last()
    )
    rows = []
    for (sym, per), grp in c.groupby(["symbol", "report_period"], sort=False):
        if len(grp) < 2:
            continue
        times = pd.to_datetime(grp["est_dt"]).to_numpy()
        vals = grp["np_avg"].to_numpy(dtype=float)
        for i in range(len(times)):
            target = times[i] - np.timedelta64(window_days, "D")
            j = np.searchsorted(times, target, side="right") - 1
            if j < 0:
                continue
            old, new = vals[j], vals[i]
            if not np.isfinite(old) or abs(old) < 1.0:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "known_date": pd.Timestamp(times[i]),
                    "surprise": (new - old) / abs(old),
                    "report_period": str(per),
                }
            )
    return pd.DataFrame(rows)


def build_sue_event_tables(bundle: dict) -> Dict[str, pd.DataFrame]:
    """Return dict name -> long events with columns symbol, known_date, surprise."""
    print("  building unexpected_profit_notice_surprise_20d ...", flush=True)
    notice_u = _notice_unexpected_events(bundle.get("notice"), bundle.get("events"))
    print(f"    -> {len(notice_u):,}", flush=True)
    print("  building sue_np_yoy_z ...", flush=True)
    yoy = _yoy_sue_events(bundle.get("events"))
    print(f"    -> {len(yoy):,}", flush=True)
    print("  building sue_eps_consensus ...", flush=True)
    cons = _consensus_eps_sue_events(bundle.get("timeline"), bundle.get("consensus"))
    print(f"    -> {len(cons):,}", flush=True)
    print("  building analyst_np_revision_20d ...", flush=True)
    rev = _analyst_revision_events(bundle.get("consensus"))
    print(f"    -> {len(rev):,}", flush=True)
    print("  building profit_notice_mid_surprise ...", flush=True)
    mid = _notice_mid_yoy_events(bundle.get("notice"), bundle.get("events"))
    print(f"    -> {len(mid):,}", flush=True)
    return {
        "unexpected_profit_notice_surprise_20d": notice_u,
        "sue_np_yoy_z": yoy,
        "sue_eps_consensus": cons,
        "analyst_np_revision_20d": rev,
        "profit_notice_mid_surprise": mid,
    }


def build_sue_panels(
    event_tables: Dict[str, pd.DataFrame],
    trade_index: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    mode: str = "hold",
    hold_days: int = HOLD_DAYS_DEFAULT,
    half_life: int = DECAY_HALFLIFE_DEFAULT,
) -> Dict[str, pd.DataFrame]:
    """mode: 'hold' | 'decay' | 'raw'."""
    out = {}
    for name, ev in event_tables.items():
        raw = _sparse_event_wide(ev, trade_index, columns, "surprise")
        if mode == "raw":
            panel = raw
        elif mode == "decay":
            panel = apply_daily_decay(raw, half_life=half_life, horizon=hold_days)
        else:
            panel = apply_event_hold(raw, hold_days=hold_days)
        out[name] = cs_zscore(panel)
    return out
