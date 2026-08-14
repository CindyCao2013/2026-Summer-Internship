"""Proxy knife matching for pre-L2 ideal_reversal backfill.

Problem: ats_trade_count needs L2 trade_count (~2018+).
Solution: on overlap, pick daily-OHLCV knife whose W-cut IC series
correlates best with true ATS cut; then run that knife on full history.

Honesty: label factor as ideal_reversal_proxy_<knife>, never claim paper ATS.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import daily_rank_ic_series
from factor_cutting.cutting_analysis.knife_ic import ic_stats, monthly_ic_stats
from factor_cutting.knives import build_knife
from factor_cutting.w_cut import w_cut

# Proxies available on Wind Oracle OHLCV (no L2)
PROXY_KNIFE_CANDIDATES = [
    "ats_volume",  # amount/volume ≈ avg trade price × lot; closest ATS analogue
    "avg_price",  # alias → ats_volume
    "volume",
    "amount",
    "turnover_proxy",  # amount / float_mktcap when TURN missing
]

DEFAULT_MATCH_START = "2018-11-27"  # L2 trade_count availability
CORR_ACCEPT = 0.80


def resolve_proxy_name(name: str) -> str:
    if name == "avg_price":
        return "ats_volume"
    return name


def build_proxy_knife(
    name: str,
    *,
    amount: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    name = resolve_proxy_name(name)
    if name == "turnover_proxy":
        if turnover is not None:
            return turnover.astype(float)
        if float_mktcap is None:
            raise ValueError("turnover_proxy needs float_mktcap or turnover")
        return amount / float_mktcap.replace(0, np.nan)
    return build_knife(name, amount=amount, volume=volume, turnover=turnover)


def ic_series_corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(aligned) < 60:
        return float("nan")
    return float(aligned["a"].corr(aligned["b"]))


def match_proxies_to_ats(
    ret_1d: pd.DataFrame,
    ret_fwd: pd.DataFrame,
    *,
    amount: pd.DataFrame,
    volume: pd.DataFrame,
    trade_count: pd.DataFrame,
    float_mktcap: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    proxy_names: Optional[Sequence[str]] = None,
    window: int = 20,
    match_start: Optional[pd.Timestamp] = None,
    match_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    On overlap where trade_count exists, compare each proxy W-cut to ATS W-cut.

    Returns:
      ranking_df — one row per proxy (ic corr, rank_ic, etc.)
      cut_panels — knife_name → factor panel (full input index)
      ats_factor — true ATS cut panel
    """
    match_start = match_start or pd.Timestamp(DEFAULT_MATCH_START)
    names = list(proxy_names or ["ats_volume", "volume", "amount", "turnover_proxy"])
    # dedupe aliases
    resolved = []
    seen = set()
    for n in names:
        r = resolve_proxy_name(n)
        if r not in seen:
            seen.add(r)
            resolved.append(r)

    ats_knife = build_knife(
        "ats_trade_count", amount=amount, trade_count=trade_count
    )
    ats_factor = w_cut(ret_1d, ats_knife, window=window)

    # restrict IC comparison to overlap
    mask_tc = trade_count.notna().sum(axis=1) >= 200
    overlap_idx = ats_factor.index[mask_tc]
    if match_start is not None:
        overlap_idx = overlap_idx[overlap_idx >= match_start]
    if match_end is not None:
        overlap_idx = overlap_idx[overlap_idx <= match_end]

    ats_ic = daily_rank_ic_series(ats_factor.loc[overlap_idx], ret_fwd.loc[overlap_idx])
    ats_stats = ic_stats(ats_factor.loc[overlap_idx], ret_fwd.loc[overlap_idx])

    cut_panels: Dict[str, pd.DataFrame] = {"ats_trade_count": ats_factor}
    rows = [
        {
            "knife": "ats_trade_count",
            "role": "benchmark",
            "ic_series_corr_vs_ats": 1.0,
            "rank_ic": ats_stats["rank_ic"],
            "icir": ats_stats["icir"],
            "n_overlap_days": int(ats_ic.dropna().shape[0]),
        }
    ]

    for name in resolved:
        try:
            knife = build_proxy_knife(
                name,
                amount=amount,
                volume=volume,
                float_mktcap=float_mktcap,
                turnover=turnover,
            )
        except ValueError:
            continue
        fac = w_cut(ret_1d, knife, window=window)
        cut_panels[name] = fac
        ic = daily_rank_ic_series(fac.loc[overlap_idx], ret_fwd.loc[overlap_idx])
        st = ic_stats(fac.loc[overlap_idx], ret_fwd.loc[overlap_idx])
        corr = ic_series_corr(ats_ic, ic)
        rows.append(
            {
                "knife": name,
                "role": "proxy",
                "ic_series_corr_vs_ats": corr,
                "rank_ic": st["rank_ic"],
                "icir": st["icir"],
                "n_overlap_days": int(ic.dropna().shape[0]),
            }
        )

    ranking = pd.DataFrame(rows)
    # sort proxies by |corr| then put benchmark first
    bench = ranking[ranking["role"] == "benchmark"]
    prox = ranking[ranking["role"] == "proxy"].copy()
    prox["_abs_corr"] = prox["ic_series_corr_vs_ats"].abs()
    prox = prox.sort_values("_abs_corr", ascending=False).drop(columns=["_abs_corr"])
    ranking = pd.concat([bench, prox], ignore_index=True)
    return ranking, cut_panels, ats_factor


def stitch_ats_with_proxy(
    ats_factor: pd.DataFrame,
    proxy_factor: pd.DataFrame,
    trade_count: pd.DataFrame,
    *,
    min_names: int = 200,
) -> pd.DataFrame:
    """Use ATS where trade_count coverage is OK; else proxy."""
    cover = trade_count.notna().sum(axis=1)
    use_ats = cover >= min_names
    out = proxy_factor.copy()
    out.loc[use_ats] = ats_factor.loc[use_ats]
    return out


def write_proxy_match_report(
    path,
    ranking: pd.DataFrame,
    *,
    best_knife: str,
    best_corr: float,
    accept: float = CORR_ACCEPT,
    full_stats: Optional[dict] = None,
) -> None:
    lines = [
        "# Proxy Knife Match — ideal_reversal pre-L2 backfill",
        "",
        f"Acceptance threshold: IC-series corr vs ATS ≥ **{accept:.2f}**",
        "",
        "| Knife | Role | Corr vs ATS | RankIC (overlap) | ICIR |",
        "|-------|------|-------------|------------------|------|",
    ]
    for _, r in ranking.iterrows():
        c = r["ic_series_corr_vs_ats"]
        lines.append(
            f"| `{r['knife']}` | {r['role']} | {c:.3f} | {r['rank_ic']:.4f} | {r['icir']:.2f} |"
        )
    lines += [
        "",
        f"**Best proxy:** `{best_knife}` (corr={best_corr:.3f})",
        "",
    ]
    if best_corr >= accept:
        lines.append(
            f"PASS — use `{best_knife}` for 2010–2017 (and stitch with ATS on 2018+)."
        )
    else:
        lines.append(
            f"FAIL threshold — best corr {best_corr:.3f} < {accept:.2f}. "
            "Still report best proxy with honest label; do not claim paper ATS."
        )
    if full_stats:
        lines += [
            "",
            "## Full-history proxy factor (stitched or pure proxy)",
            "",
            f"- knife: `{full_stats.get('knife')}`",
            f"- mode: `{full_stats.get('mode')}`",
            f"- RankIC: `{full_stats.get('rank_ic'):.4f}`",
            f"- ICIR: `{full_stats.get('icir'):.2f}`",
            f"- monthly RankIC: `{full_stats.get('monthly_rank_ic'):.4f}`",
            f"- n_days: `{full_stats.get('n_days')}`",
            "",
        ]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
