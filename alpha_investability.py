"""Investability filters — net cost Sharpe, tradability, stability & capacity."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from factor_attribution import align_signal, cs_zscore

# A-share round-trip cost: stamp 0.1% (sell) + commission/fees ~0.05% bilateral ≈ 0.15%
DEFAULT_ROUND_TRIP_COST = 0.0015
TRADING_DAYS_YEAR = 250
CAPACITY_PARTICIPATION = 0.05  # max fraction of ADV per name


def apply_tradability_mask(
    signal: pd.DataFrame,
    *,
    df_not_limit: Optional[pd.DataFrame] = None,
    df_not_st: Optional[pd.DataFrame] = None,
    df_trade_status: Optional[pd.DataFrame] = None,
    min_listing_days: int = 60,
    close: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Zero-out / NaN non-tradable names. Returns (masked_signal, daily_coverage).

    Coverage = tradable count / candidate count (candidates = signal notna before mask).
    """
    out = signal.copy()
    candidate = out.notna()

    if df_not_limit is not None:
        m = df_not_limit.reindex_like(out)
        out = out.where(m == 1)
    if df_not_st is not None:
        m = df_not_st.reindex_like(out)
        out = out.where(m == 1)
    if df_trade_status is not None:
        m = df_trade_status.reindex_like(out)
        out = out.where(m == 1)

    if close is not None and min_listing_days > 0:
        # Proxy for IPO seasoning: require ≥ min_listing_days of non-NaN closes to date
        listed = close.reindex_like(out).notna().astype(float).cumsum()
        out = out.where(listed >= min_listing_days)

    tradable = out.notna()
    coverage = tradable.sum(axis=1) / candidate.sum(axis=1).replace(0, np.nan)
    return out, coverage


def build_long_short_weights(
    signal: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight long (top) / short (bottom) portfolios from cross-sectional ranks."""
    ranks = signal.rank(axis=1, pct=True, method="first")
    long_mask = ranks >= (1.0 - top_frac)
    short_mask = ranks <= bottom_frac
    long_n = long_mask.sum(axis=1).replace(0, np.nan)
    short_n = short_mask.sum(axis=1).replace(0, np.nan)
    w_long = long_mask.div(long_n, axis=0)
    w_short = short_mask.div(short_n, axis=0)
    return w_long, w_short


def daily_hl_pnl_and_turnover(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
    signal_shift: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Gross H-L daily PnL and combined long+short turnover (L1 weight change).

    signal_shift=1: use T signal for T+1 return (standard).
    """
    sig = align_signal(signal, signal_shift)
    r = ret.reindex_like(sig)
    w_long, w_short = build_long_short_weights(sig, top_frac=top_frac, bottom_frac=bottom_frac)

    long_pnl = w_long.mul(r).sum(axis=1)
    short_pnl = w_short.mul(r).sum(axis=1)
    gross = long_pnl - short_pnl

    w_ls = w_long.fillna(0) - w_short.fillna(0)
    turnover = w_ls.diff().abs().sum(axis=1)
    turnover.iloc[0] = w_ls.iloc[0].abs().sum()
    return gross, turnover


def long_book_excess_performance(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    top_frac: float = 0.10,
    signal_shift: int = 1,
    direction: Optional[int] = None,
) -> dict:
    """Economic long book excess return versus the exact daily test-universe EW.

    ``direction=1`` selects the highest factor quantile; ``direction=-1`` selects
    the lowest.  If omitted, direction is inferred from the full-sample H-L mean.
    The benchmark is the mean return of every stock with both a valid aligned
    signal and a valid return on that date -- not the mean of decile returns.
    """
    sig = align_signal(signal, signal_shift)
    r = ret.reindex_like(sig)
    valid = sig.notna() & r.notna()
    ranks = sig.rank(axis=1, pct=True, method="first")

    high_mask = valid & (ranks > 1.0 - top_frac)
    low_mask = valid & (ranks <= top_frac)
    high_ret = r.where(high_mask).mean(axis=1)
    low_ret = r.where(low_mask).mean(axis=1)

    if direction is None:
        direction = 1 if (high_ret - low_ret).mean() >= 0 else -1
    if direction not in (-1, 1):
        raise ValueError("direction must be +1, -1, or None")

    long_group = 10 if direction == 1 else 1
    long_ret = high_ret if direction == 1 else low_ret
    universe_ew_ret = r.where(valid).mean(axis=1)
    excess_ret = long_ret - universe_ew_ret

    selected_mask = high_mask if direction == 1 else low_mask
    selected_count = selected_mask.sum(axis=1)
    universe_count = valid.sum(axis=1)

    perf = series_performance(excess_ret)
    return {
        "long_group": long_group,
        "direction": direction,
        "top_frac": top_frac,
        "excess_sharpe": perf["sharpe"],
        "excess_annu_ret": perf["annu_ret"],
        "excess_max_drawdown": perf["max_drawdown"],
        "excess_calmar": perf["calmar"],
        "n_days": perf["n_days"],
        "selected_count_mean": float(selected_count.mean()),
        "universe_count_mean": float(universe_count.mean()),
        "_long_ret": long_ret,
        "_universe_ew_ret": universe_ew_ret,
        "_excess_ret": excess_ret,
    }


def net_pnl_series(
    gross: pd.Series,
    turnover: pd.Series,
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
) -> pd.Series:
    """Apply round-trip cost on combined long+short turnover; flip if mean gross < 0."""
    direction = 1 if gross.mean() >= 0 else -1
    return direction * gross - round_trip_cost * turnover


def series_performance(pnl: pd.Series) -> dict:
    s = pnl.dropna()
    if len(s) < 50:
        return {
            "n_days": len(s),
            "annu_ret": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }
    annu = float(Factor_Dev_Lib.calAnnuRet(s))
    sharpe = float(Factor_Dev_Lib.calSharpe(s))
    mdd, _ = Factor_Dev_Lib.calMDD(s)
    mdd = float(mdd)
    calmar = annu / abs(mdd) if mdd and abs(mdd) > 1e-12 else np.nan
    return {
        "n_days": int(len(s)),
        "annu_ret": annu,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
    }


def annualized_turnover(daily_turnover: pd.Series) -> float:
    """Single-side annualized turnover ≈ mean(daily L1 TO of LS book) * 250 / 2."""
    # H-L combined TO counts both sides; report one-way as half
    return float(daily_turnover.mean() * TRADING_DAYS_YEAR / 2.0)


def evaluate_investability(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    df_not_limit: Optional[pd.DataFrame] = None,
    df_not_st: Optional[pd.DataFrame] = None,
    df_trade_status: Optional[pd.DataFrame] = None,
    close: Optional[pd.DataFrame] = None,
    amount: Optional[pd.DataFrame] = None,
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
    top_frac: float = 0.2,
    apply_tradability: bool = True,
    min_listing_days: int = 60,
    signal_shift: int = 1,
) -> dict:
    """Full investability pack for one signal on a given window."""
    if apply_tradability:
        sig_t, coverage = apply_tradability_mask(
            signal,
            df_not_limit=df_not_limit,
            df_not_st=df_not_st,
            df_trade_status=df_trade_status,
            min_listing_days=min_listing_days,
            close=close,
        )
    else:
        sig_t = signal
        coverage = signal.notna().sum(axis=1) / signal.notna().sum(axis=1).replace(0, np.nan)

    # Gross / net with tradability
    gross_t, to_t = daily_hl_pnl_and_turnover(
        sig_t, ret, top_frac=top_frac, bottom_frac=top_frac, signal_shift=signal_shift
    )
    net_t = net_pnl_series(gross_t, to_t, round_trip_cost)
    perf_gross_t = series_performance(
        (gross_t if gross_t.mean() >= 0 else -gross_t).dropna()
    )
    perf_net_t = series_performance(net_t)

    # Without tradability (cost still applied) for delta
    gross_u, to_u = daily_hl_pnl_and_turnover(
        signal, ret, top_frac=top_frac, bottom_frac=top_frac, signal_shift=signal_shift
    )
    net_u = net_pnl_series(gross_u, to_u, round_trip_cost)
    perf_net_u = series_performance(net_u)

    # IC after tradability mask
    sig_aligned = align_signal(sig_t, signal_shift)
    ic_daily = daily_rank_ic_series(sig_t, ret, signal_shift=signal_shift)
    # Also IC on unmasked for comparison
    ic_daily_raw = daily_rank_ic_series(signal, ret, signal_shift=signal_shift)

    # Capacity: ADV of names in long+short book × participation / 2 (gross book)
    capacity = estimate_capacity(sig_t, amount, top_frac=top_frac, signal_shift=signal_shift)

    return {
        "coverage_mean": float(coverage.mean()) if coverage.notna().any() else np.nan,
        "coverage_p10": float(coverage.quantile(0.1)) if coverage.notna().any() else np.nan,
        "rank_ic_raw": float(ic_daily_raw.mean()),
        "rank_ic_tradable": float(ic_daily.mean()),
        "icir_tradable": icir_from_daily(ic_daily),
        "gross_sharpe_tradable": perf_gross_t["sharpe"],
        "net_sharpe_tradable": perf_net_t["sharpe"],
        "net_annu_ret_tradable": perf_net_t["annu_ret"],
        "net_max_drawdown_tradable": perf_net_t["max_drawdown"],
        "net_calmar_tradable": perf_net_t["calmar"],
        "net_sharpe_no_tradability_filter": perf_net_u["sharpe"],
        "net_sharpe_delta_tradability": (
            float(perf_net_t["sharpe"] - perf_net_u["sharpe"])
            if pd.notna(perf_net_t["sharpe"]) and pd.notna(perf_net_u["sharpe"])
            else np.nan
        ),
        "daily_turnover_mean_ls": float(to_t.mean()),
        "annu_one_way_turnover": annualized_turnover(to_t),
        "round_trip_cost": round_trip_cost,
        "capacity_cny_approx": capacity,
        "n_days": perf_net_t["n_days"],
        "_net_pnl": net_t,
        "_gross_pnl": gross_t,
        "_turnover": to_t,
        "_ic_daily": ic_daily,
        "_coverage": coverage,
    }


def estimate_capacity(
    signal: pd.DataFrame,
    amount: Optional[pd.DataFrame],
    *,
    top_frac: float = 0.2,
    signal_shift: int = 1,
    participation: float = CAPACITY_PARTICIPATION,
) -> float:
    """
    Rough single-product capacity (CNY): median daily ADV of LS names × participation
    × typical book size (n_long + n_short), then / 2 for one-way.

    amount is assumed in 千元 (Wind); convert to 元 ×1000.
    """
    if amount is None:
        return np.nan
    sig = align_signal(signal, signal_shift)
    amt = amount.reindex_like(sig)
    ranks = sig.rank(axis=1, pct=True)
    in_book = (ranks >= 1 - top_frac) | (ranks <= top_frac)
    # Daily ADV of book names (千元 → 元)
    daily_adv = amt.where(in_book).mean(axis=1) * 1000.0
    n_book = in_book.sum(axis=1)
    # Capacity ≈ median(ADV_per_name * n_book * participation)
    cap_series = daily_adv * n_book * participation
    return float(cap_series.median()) if cap_series.notna().any() else np.nan


def classify_market_regimes(
    index_ret: pd.Series,
    *,
    window: int = 60,
    bull_thresh: float = 0.08,
    bear_thresh: float = -0.08,
) -> pd.Series:
    """Bull / bear / sideways from rolling cumulative index return."""
    cum = index_ret.rolling(window, min_periods=max(20, window // 2)).sum()
    regime = pd.Series("sideways", index=index_ret.index)
    regime = regime.mask(cum >= bull_thresh, "bull")
    regime = regime.mask(cum <= bear_thresh, "bear")
    return regime


def regime_net_sharpes(net_pnl: pd.Series, regime: pd.Series) -> dict:
    out = {}
    for label in ["bull", "bear", "sideways"]:
        mask = regime.reindex(net_pnl.index) == label
        sub = net_pnl[mask]
        if sub.dropna().shape[0] < 40:
            out[label] = {"n_days": int(sub.dropna().shape[0]), "net_sharpe": np.nan}
        else:
            perf = series_performance(sub)
            out[label] = {"n_days": perf["n_days"], "net_sharpe": perf["sharpe"]}
    return out


def yearly_net_sharpes(net_pnl: pd.Series) -> dict:
    out = {}
    for year, sub in net_pnl.groupby(net_pnl.index.year):
        if sub.dropna().shape[0] < 40:
            out[str(year)] = np.nan
        else:
            out[str(year)] = series_performance(sub)["sharpe"]
    return out


def weight_perturbation_stability(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    base_weights: Optional[Dict[str, float]] = None,
    scale: float = 0.30,
    tradability_kwargs: Optional[dict] = None,
) -> dict:
    """
    Perturb Base3 relative weights by ±scale and report net Sharpe distribution.
    panels keys = factor names; equal-weight baseline if base_weights is None.
    """
    names = list(panels.keys())
    if base_weights is None:
        base_weights = {n: 1.0 / len(names) for n in names}

    def blend(wdict: Dict[str, float]) -> pd.DataFrame:
        parts = []
        for n, w in wdict.items():
            parts.append(w * cs_zscore(panels[n]))
        return sum(parts)

    tradability_kwargs = tradability_kwargs or {}
    configs = {"baseline": base_weights}
    for n in names:
        up = dict(base_weights)
        down = dict(base_weights)
        up[n] = base_weights[n] * (1 + scale)
        down[n] = base_weights[n] * (1 - scale)
        # renormalize
        s_up = sum(up.values())
        s_dn = sum(down.values())
        configs[f"{n}_up"] = {k: v / s_up for k, v in up.items()}
        configs[f"{n}_down"] = {k: v / s_dn for k, v in down.items()}

    sharpes = {}
    for label, wdict in configs.items():
        sig = blend(wdict)
        inv = evaluate_investability(sig, ret, **tradability_kwargs)
        sharpes[label] = inv["net_sharpe_tradable"]

    vals = [v for v in sharpes.values() if pd.notna(v)]
    return {
        "perturbation_scale": scale,
        "net_sharpe_by_config": sharpes,
        "net_sharpe_min": float(min(vals)) if vals else np.nan,
        "net_sharpe_max": float(max(vals)) if vals else np.nan,
        "sign_flip": bool(vals) and (min(vals) < 0 < max(vals) or min(vals) * max(vals) < 0),
        "stable": bool(vals) and min(vals) > 0,
    }


def lambda_stability(
    base: pd.DataFrame,
    enhancer: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    center_lambda: float = 0.2,
    scale: float = 0.30,
    tradability_kwargs: Optional[dict] = None,
) -> dict:
    """Perturb enhancer λ by ±scale around center; report net Sharpe grid."""
    from alpha_d4_expansion_stack import build_satellite_stack

    tradability_kwargs = tradability_kwargs or {}
    lambdas = [
        center_lambda * (1 - scale),
        center_lambda,
        center_lambda * (1 + scale),
    ]
    grid = {}
    for lam in lambdas:
        sig = build_satellite_stack(
            base, {"enh": enhancer}, lam, satellite_factors=["enh"]
        )
        # build_satellite_stack already shifts; evaluate with signal_shift=0
        kw = dict(tradability_kwargs)
        kw["signal_shift"] = 0
        inv = evaluate_investability(sig, ret, **kw)
        grid[f"lambda_{lam:.3f}"] = inv["net_sharpe_tradable"]

    vals = [v for v in grid.values() if pd.notna(v)]
    return {
        "center_lambda": center_lambda,
        "perturbation_scale": scale,
        "net_sharpe_by_lambda": grid,
        "net_sharpe_min": float(min(vals)) if vals else np.nan,
        "net_sharpe_max": float(max(vals)) if vals else np.nan,
        "stable": bool(vals) and min(vals) > 0,
    }


def strip_internal(inv: dict) -> dict:
    """Drop private series keys before JSON serialization."""
    return {k: v for k, v in inv.items() if not k.startswith("_")}
