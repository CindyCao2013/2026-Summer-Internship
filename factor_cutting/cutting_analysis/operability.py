"""Operability diagnostics — long-only excess, turnover, size, MDD.

IC/ICIR measures information purity. This module asks whether the signal
is tradable under A-share constraints (long-biased, limit filter, size).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_attribution import align_signal, cs_zscore
from factor_cutting.cutting_analysis.knife_ic import ic_stats


def _silent_group_test(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    n: int = 10,
    fee: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import Factor_Dev_Lib

    sig = align_signal(cs_zscore(signal), 1)
    r = ret.reindex_like(sig)
    old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        signal_rank, group_pnl, group_to = Factor_Dev_Lib.groupTest(
            sig, r, n=n, fee=fee, info="silent"
        )
    finally:
        plt.show = old_show
    return signal_rank, group_pnl, group_to


def _series_stats(pnl: pd.Series, label: str) -> dict:
    import Factor_Dev_Lib

    s = pnl.dropna()
    if len(s) < 50:
        return {
            "label": label,
            "ann_ret": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "mdd_start": None,
            "mdd_end": None,
            "mdd_days": np.nan,
            "win_rate": np.nan,
            "n_days": int(len(s)),
        }
    ann = Factor_Dev_Lib.calAnnuRet(s)
    sharpe = Factor_Dev_Lib.calSharpe(s)
    mdd, (m0, m1) = Factor_Dev_Lib.calMDD(s)
    mdd_days = np.nan
    if pd.notna(m0) and pd.notna(m1):
        mdd_days = int((pd.Timestamp(m1) - pd.Timestamp(m0)).days)
    return {
        "label": label,
        "ann_ret": float(ann) if pd.notna(ann) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "mdd": float(mdd) if pd.notna(mdd) else np.nan,
        "mdd_start": str(pd.Timestamp(m0).date()) if pd.notna(m0) else None,
        "mdd_end": str(pd.Timestamp(m1).date()) if pd.notna(m1) else None,
        "mdd_days": mdd_days,
        "win_rate": float((s > 0).mean()),
        "n_days": int(len(s)),
    }


def long_short_groups(group_pnl: pd.DataFrame, n: int = 10) -> Tuple[int, int, int]:
    """Pick long/short group ids for the profitable H-L direction.

    groupTest: group 1 = lowest factor, group n = highest.
    If mean(H-L)=mean(n-1) < 0 → short high / long low → long=1, short=n.
    """
    hl = group_pnl[n] - group_pnl[1]
    if hl.mean() >= 0:
        return n, 1, 1  # long high, short low, direction +1
    return 1, n, -1


def size_book_stats(
    signal_rank: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    *,
    long_g: int,
    n_groups: int = 10,
) -> dict:
    """Cross-sectional size percentile of the long book each day."""
    mkt = float_mktcap.reindex_like(signal_rank)
    rows = []
    for dt_ in signal_rank.index:
        rank_row = signal_rank.loc[dt_]
        mkt_row = mkt.loc[dt_]
        mask = (rank_row == long_g) & mkt_row.notna()
        if mask.sum() < 5:
            continue
        # size percentile within day's traded universe
        uni = mkt_row.dropna()
        if len(uni) < 50:
            continue
        pct = uni.rank(pct=True)
        book_pct = pct.reindex(rank_row.index)[mask]
        rows.append(
            {
                "date": dt_,
                "median_size_pctile": float(book_pct.median()),
                "mean_size_pctile": float(book_pct.mean()),
                "frac_bottom_20": float((book_pct <= 0.20).mean()),
                "frac_bottom_10": float((book_pct <= 0.10).mean()),
                "n_names": int(mask.sum()),
            }
        )
    if not rows:
        return {
            "median_size_pctile": np.nan,
            "mean_size_pctile": np.nan,
            "frac_bottom_20": np.nan,
            "frac_bottom_10": np.nan,
            "n_days": 0,
        }
    df = pd.DataFrame(rows)
    return {
        "median_size_pctile": float(df["median_size_pctile"].mean()),
        "mean_size_pctile": float(df["mean_size_pctile"].mean()),
        "frac_bottom_20": float(df["frac_bottom_20"].mean()),
        "frac_bottom_10": float(df["frac_bottom_10"].mean()),
        "n_days": int(len(df)),
        "daily": df,
    }


def market_ew_return(ret: pd.DataFrame, signal_rank: pd.DataFrame) -> pd.Series:
    """Equal-weight return of names with a valid group assignment."""
    r = ret.reindex_like(signal_rank)
    valid = signal_rank.notna() & r.notna()
    return r.where(valid).mean(axis=1)


def operability_report(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    float_mktcap: Optional[pd.DataFrame] = None,
    n_groups: int = 10,
    fee: float = 0.0,
    label: str = "factor",
    mode: str = "raw",
) -> dict:
    """Full operability pack for one factor panel."""
    import Factor_Dev_Lib

    ic = ic_stats(cs_zscore(factor), ret)
    signal_rank, group_pnl, group_to = _silent_group_test(
        factor, ret, n=n_groups, fee=fee
    )
    long_g, short_g, direction = long_short_groups(group_pnl, n=n_groups)

    long_pnl = group_pnl[long_g]
    short_pnl = group_pnl[short_g]
    hl_pnl = (group_pnl[n_groups] - group_pnl[1]) * direction  # signed to + mean
    mkt = market_ew_return(ret, signal_rank)
    long_excess = long_pnl - mkt

    long_to = group_to[long_g]
    short_to = group_to[short_g]
    hl_to = group_to[n_groups] + group_to[1]  # bilateral

    # Daily turnover → monthly / annual summaries
    # groupTest turnover is |Δw| sum ≈ one-way for a group (0–2 scale typically ~0.3–1)
    daily_long_to = float(long_to.mean())
    daily_hl_to = float(hl_to.mean())
    # Approximate "monthly turnover" under daily rebalance: mean daily × ~21
    monthly_long_to = daily_long_to * 21
    monthly_hl_to = daily_hl_to * 21
    implied_annu_fee = Factor_Dev_Lib.implied_annu_fee(daily_hl_to)

    size = {}
    if float_mktcap is not None:
        size = size_book_stats(
            signal_rank, float_mktcap, long_g=long_g, n_groups=n_groups
        )
        size.pop("daily", None)  # keep summary only in dict; save daily separately

    group_means = {str(c): float(group_pnl[c].mean()) for c in group_pnl.columns if c != "H-L"}
    if "H-L" in group_pnl.columns:
        group_means["H-L"] = float(group_pnl["H-L"].mean())

    out = {
        "label": label,
        "mode": mode,
        "rank_ic": ic["rank_ic"],
        "icir": ic["icir"],
        "n_ic_days": ic["n_days"],
        "long_group": int(long_g),
        "short_group": int(short_g),
        "direction": int(direction),
        "long": _series_stats(long_pnl, "long"),
        "long_excess": _series_stats(long_excess, "long_excess_vs_ew"),
        "hl": _series_stats(hl_pnl, "hl_signed"),
        "short": _series_stats(short_pnl, "short"),
        "turnover": {
            "daily_long": daily_long_to,
            "daily_hl_bilateral": daily_hl_to,
            "monthly_long_approx": monthly_long_to,
            "monthly_hl_bilateral_approx": monthly_hl_to,
            "ann_long_approx": daily_long_to * 250,
            "ann_hl_bilateral_approx": daily_hl_to * 250,
            "implied_annu_fee_7p5bps": implied_annu_fee,
        },
        "size_long": size,
        "group_mean_daily_ret": group_means,
        "panels": {
            "group_pnl": group_pnl,
            "group_to": group_to,
            "signal_rank": signal_rank,
            "long_pnl": long_pnl,
            "long_excess": long_excess,
            "hl_pnl": hl_pnl,
        },
    }
    return out


def flatten_operability(rep: dict) -> dict:
    """One-row summary without heavy panels."""
    row = {
        "label": rep["label"],
        "mode": rep["mode"],
        "rank_ic": rep["rank_ic"],
        "icir": rep["icir"],
        "long_group": rep["long_group"],
        "direction": rep["direction"],
        "long_ann_ret": rep["long"]["ann_ret"],
        "long_sharpe": rep["long"]["sharpe"],
        "long_mdd": rep["long"]["mdd"],
        "long_excess_ann": rep["long_excess"]["ann_ret"],
        "long_excess_sharpe": rep["long_excess"]["sharpe"],
        "long_excess_mdd": rep["long_excess"]["mdd"],
        "long_excess_win_rate": rep["long_excess"]["win_rate"],
        "hl_ann_ret": rep["hl"]["ann_ret"],
        "hl_sharpe": rep["hl"]["sharpe"],
        "hl_mdd": rep["hl"]["mdd"],
        "hl_mdd_days": rep["hl"]["mdd_days"],
        "daily_to_long": rep["turnover"]["daily_long"],
        "daily_to_hl": rep["turnover"]["daily_hl_bilateral"],
        "monthly_to_long_approx": rep["turnover"]["monthly_long_approx"],
        "monthly_to_hl_approx": rep["turnover"]["monthly_hl_bilateral_approx"],
        "implied_annu_fee": rep["turnover"].get("implied_annu_fee_7p5bps", np.nan),
        "size_median_pctile": rep["size_long"].get("median_size_pctile", np.nan),
        "size_frac_bottom_20": rep["size_long"].get("frac_bottom_20", np.nan),
        "size_frac_bottom_10": rep["size_long"].get("frac_bottom_10", np.nan),
    }
    return row


def write_operability_markdown(path: Path, rows: pd.DataFrame, notes: list[str]) -> None:
    lines = [
        "# Factor Operability Report",
        "",
        "IC/ICIR = information purity. Below = **tradability** under daily rebalance.",
        "",
        "## Summary",
        "",
        "| Factor | Mode | RankIC | ICIR | Long excess ann | Long excess Sharpe | "
        "HL ann | HL Sharpe | MDD | Monthly TO (HL bilat≈) | Long size pctile | "
        "Frac size≤20% |",
        "|--------|------|--------|------|-----------------|--------------------|"
        "--------|-----------|-----|------------------------|------------------|"
        "---------------|",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| `{r['label']}` | `{r['mode']}` | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{r['long_excess_ann']:.2%} | {r['long_excess_sharpe']:.2f} | "
            f"{r['hl_ann_ret']:.2%} | {r['hl_sharpe']:.2f} | {r['hl_mdd']:.2%} | "
            f"{r['monthly_to_hl_approx']:.2f} | {r['size_median_pctile']:.2f} | "
            f"{r['size_frac_bottom_20']:.2%} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Long group = profitable decile given factor sign (neg IC → long group 1 / low factor).",
        "- Long excess = long-book EW − universe EW (same day, same coverage).",
        "- Turnover from `Factor_Dev_Lib.groupTest` (|Δw|); monthly ≈ daily × 21 under daily rebalance.",
        "- `filter_signal` = mask limit-up/down names on finished factor via `get_EOD_Not_Limit`.",
        "- Size pctile: 0=smallest, 1=largest within day's universe.",
        "",
    ]
    if notes:
        lines += [f"- {n}" for n in notes] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
