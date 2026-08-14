#!/usr/bin/env python3
"""Research-grade SKEW / IdioSKEW validation (P0).

Pipeline:
  Discovery formula (core/factors/skew)
  → IC / quantile / long-short
  → size / industry neutralization ladder
  → lottery mechanism diagnostics
  → TGD20 interaction (if panel cache present)
  → Pack + delivery card

Usage:
  OMP_NUM_THREADS=1 python run_skew_validation_v1.py
  OMP_NUM_THREADS=1 python run_skew_validation_v1.py --skip-tgd
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_investability import series_performance
from alpha_research_report import build_factor_report, publish_factor_report, report_summary_row
from core.factors.skew import build_idio_skew, build_total_skew
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel, panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual
from research.extreme_return_study.src.data_loader import load_csi300_index_return

ANNUAL_DAYS = 250
PACK = ROOT / "research/reports/factors/SKEW"
EXP = ROOT / "research/reports/skew_v1"
CACHE_DIR = ROOT / "research/cache/skew_panels"
DELIVERY = ROOT / "research_delivery/factors/SKEW"
TGD_PANEL = ROOT / "research/cache/tgd_panels/TGD20_20200101_20251231_w20.parquet"

# Pre-registered windows — do not cherry-pick after seeing results.
P0_RAW = ("SKEW20", "IdioSKEW60")
P0_ALPHA = ("AlphaSKEW20", "AlphaIdioSKEW60")
HEADLINE = "AlphaIdioSKEW60"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for p in (
        PACK,
        PACK / "ic_analysis",
        PACK / "quantile_analysis",
        PACK / "stability",
        PACK / "execution",
        PACK / "mechanism",
        PACK / "figures",
        PACK / "tables",
        EXP,
        CACHE_DIR,
        DELIVERY / "plots",
    ):
        p.mkdir(parents=True, exist_ok=True)


def rank_ic(signal: pd.DataFrame, forward_ret: pd.DataFrame) -> pd.Series:
    common_i = signal.index.intersection(forward_ret.index)
    common_c = signal.columns.intersection(forward_ret.columns)
    s = signal.loc[common_i, common_c]
    r = forward_ret.loc[common_i, common_c]
    return s.rank(axis=1, pct=True).corrwith(r.rank(axis=1, pct=True), axis=1)


def summarize_ic(daily_ic: pd.Series, label: str) -> Dict[str, float]:
    s = daily_ic.dropna()
    mean = float(s.mean()) if len(s) else np.nan
    std = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return {
        "label": label,
        "n_days": int(len(s)),
        "mean_rank_ic": mean,
        "ic_std": std,
        "icir_annualized": mean / std * math.sqrt(ANNUAL_DAYS) if std and std > 0 else np.nan,
        "positive_ic_ratio": float((s > 0).mean()) if len(s) else np.nan,
        "t_stat": mean / std * math.sqrt(len(s)) if std and std > 0 else np.nan,
    }


def neutralize_size_only(raw: pd.DataFrame, float_mktcap: pd.DataFrame) -> pd.DataFrame:
    log_size = np.log(float_mktcap.replace(0, np.nan))
    log_size = log_size.reindex(index=raw.index, columns=raw.columns)
    return panel_cross_sectional_residual(raw, [log_size])


def cs_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    a, b = a.align(b, join="inner")
    return a.rank(axis=1, pct=True).corrwith(b.rank(axis=1, pct=True), axis=1)


def decile_long_short(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    n_groups: int = 10,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Equal-weight decile returns; long highest alpha, short lowest."""
    sig = align_signal(signal, 1)
    r = ret.reindex_like(sig)
    group_rets = {f"G{i}": [] for i in range(1, n_groups + 1)}
    dates = []
    ls = []
    turnover = []
    prev_long = prev_short = None
    for dt_ in sig.index:
        frame = pd.DataFrame({"s": sig.loc[dt_], "r": r.loc[dt_]}).dropna()
        if len(frame) < n_groups * 20:
            continue
        ranks = frame["s"].rank(method="first")
        bins = pd.qcut(ranks, n_groups, labels=False, duplicates="drop") + 1
        if bins.nunique() < n_groups:
            continue
        dates.append(dt_)
        day_g = {}
        for g in range(1, n_groups + 1):
            day_g[g] = float(frame.loc[bins == g, "r"].mean())
            group_rets[f"G{g}"].append(day_g[g])
        long = set(frame.index[bins == n_groups])
        short = set(frame.index[bins == 1])
        ls.append(day_g[n_groups] - day_g[1])
        if prev_long is not None:
            to = 0.5 * (
                1 - len(long & prev_long) / max(len(long), 1)
                + 1 - len(short & prev_short) / max(len(short), 1)
            )
            turnover.append(to)
        else:
            turnover.append(np.nan)
        prev_long, prev_short = long, short
    gdf = pd.DataFrame(group_rets, index=pd.DatetimeIndex(dates))
    return gdf, pd.Series(ls, index=gdf.index, name="LS"), pd.Series(
        turnover, index=gdf.index, name="turnover"
    )


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_ic(daily_ic: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    s = daily_ic.dropna()
    ax.plot(s.index, s.values, lw=0.8, alpha=0.7, label="daily RankIC")
    ax.plot(s.index, s.rolling(60, min_periods=40).mean(), lw=1.5, label="MA60")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.legend()
    save_fig(path)


def plot_decile(gdf: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    means = gdf.mean()
    means.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel("Mean daily return")
    save_fig(path)


def plot_cum(ls: pd.Series, path: Path, title: str) -> None:
    """Deprecated single-line LS plot — prefer save_grouptest_cum_figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    cum = (1.0 + ls.fillna(0)).cumprod() - 1.0
    ax.plot(cum.index, cum.values, color="darkgreen")
    ax.set_title(title)
    ax.set_ylabel("Cumulative LS return")
    ax.axhline(0, color="black", lw=0.8)
    save_fig(path)


def save_grouptest_cum_figure(
    group_pnl_df: pd.DataFrame,
    *,
    factor_name: str,
    stats_title: str,
    out_path: Path,
) -> None:
    """10-group + H-L cumulative plot matching Factor_Dev_Lib.groupTest."""
    from alpha_research_report import save_cumulative_decile_figure

    save_cumulative_decile_figure(
        group_pnl_df.cumsum(),
        factor_name=factor_name,
        stats_title=stats_title,
        out_path=out_path,
    )


def build_factor_panels(
    ret_1d: pd.DataFrame, market_ret: pd.Series
) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}
    panels.update(build_total_skew(ret_1d, windows=(20, 60, 120), as_alpha=False))
    panels.update(build_total_skew(ret_1d, windows=(20, 60, 120), as_alpha=True))
    panels.update(build_idio_skew(ret_1d, market_ret, windows=(60, 120), as_alpha=False))
    panels.update(build_idio_skew(ret_1d, market_ret, windows=(60, 120), as_alpha=True))
    return panels


def mechanism_lottery(
    raw_skew: pd.DataFrame,
    ret_1d: pd.DataFrame,
    turnover: Optional[pd.DataFrame],
) -> pd.DataFrame:
    max20 = ret_1d.rolling(20, min_periods=10).max()
    vol20 = ret_1d.rolling(20, min_periods=10).std()
    rows = []
    for name, other in (
        ("MAX_return_20d", max20),
        ("volatility_20d", vol20),
    ):
        corr = cs_spearman(raw_skew, other).dropna()
        rows.append(
            {
                "vs": name,
                "mean_cs_spearman": float(corr.mean()),
                "median_cs_spearman": float(corr.median()),
                "n_days": int(len(corr)),
            }
        )
    if turnover is not None:
        to20 = turnover.rolling(20, min_periods=10).mean()
        corr = cs_spearman(raw_skew, to20).dropna()
        rows.append(
            {
                "vs": "turnover_mean_20d",
                "mean_cs_spearman": float(corr.mean()),
                "median_cs_spearman": float(corr.median()),
                "n_days": int(len(corr)),
            }
        )
    # Tail frequency by skew decile
    sig = raw_skew.copy()
    up_extreme = (ret_1d > ret_1d.rolling(60, min_periods=40).std() * 2).astype(float)
    dn_extreme = (ret_1d < -ret_1d.rolling(60, min_periods=40).std() * 2).astype(float)
    up_freq = up_extreme.rolling(60, min_periods=40).mean()
    dn_freq = dn_extreme.rolling(60, min_periods=40).mean()
    for label, freq in (("positive_tail_frequency", up_freq), ("negative_tail_frequency", dn_freq)):
        corr = cs_spearman(sig, freq).dropna()
        rows.append(
            {
                "vs": label,
                "mean_cs_spearman": float(corr.mean()),
                "median_cs_spearman": float(corr.median()),
                "n_days": int(len(corr)),
            }
        )
    return pd.DataFrame(rows)


def double_sort_tgd(
    skew_alpha: pd.DataFrame,
    tgd: pd.DataFrame,
    ret: pd.DataFrame,
) -> pd.DataFrame:
    """2×5 conditional sorts: TGD tercile × SKEW quintile (by alpha)."""
    s = align_signal(skew_alpha, 1)
    g = tgd.reindex_like(s)
    r = ret.reindex_like(s)
    rows = []
    for dt_ in s.index:
        frame = pd.DataFrame(
            {"skew": s.loc[dt_], "tgd": g.loc[dt_], "ret": r.loc[dt_]}
        ).dropna()
        if len(frame) < 200:
            continue
        frame["tgd_bin"] = pd.qcut(frame["tgd"].rank(method="first"), 2, labels=["TGD_low", "TGD_high"])
        frame["skew_bin"] = pd.qcut(
            frame["skew"].rank(method="first"), 5, labels=[f"S{k}" for k in range(1, 6)]
        )
        for (tb, sb), part in frame.groupby(["tgd_bin", "skew_bin"]):
            rows.append(
                {
                    "date": dt_,
                    "tgd_bin": tb,
                    "skew_bin": sb,
                    "ret": float(part["ret"].mean()),
                    "n": int(len(part)),
                }
            )
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    summary = (
        long.groupby(["tgd_bin", "skew_bin"])["ret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ann_return"] = summary["mean"] * ANNUAL_DAYS
    summary["sharpe"] = summary["mean"] / summary["std"] * math.sqrt(ANNUAL_DAYS)
    # Spread within each TGD half: high-alpha (S5) - low-alpha (S1)
    spreads = []
    for tb, part in long.groupby("tgd_bin"):
        pivot = part.pivot_table(index="date", columns="skew_bin", values="ret")
        if not {"S1", "S5"}.issubset(pivot.columns):
            continue
        spr = (pivot["S5"] - pivot["S1"]).dropna()
        spreads.append(
            {
                "tgd_bin": tb,
                "skew_alpha_spread_ann": float(spr.mean() * ANNUAL_DAYS),
                "skew_alpha_spread_sharpe": float(
                    spr.mean() / spr.std(ddof=1) * math.sqrt(ANNUAL_DAYS)
                )
                if spr.std(ddof=1) > 0
                else np.nan,
                "n_days": int(len(spr)),
            }
        )
    spread_df = pd.DataFrame(spreads)
    summary["metric_type"] = "cell"
    spread_df["metric_type"] = "spread"
    return pd.concat([summary, spread_df], ignore_index=True, sort=False)


def neutralization_ladder(
    alpha: pd.DataFrame,
    ret: pd.DataFrame,
    industry: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    session=None,
    df_not_limit=None,
    df_not_st=None,
    df_trade_status=None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Neutralization ladder using Factor_Dev_Lib.groupTest (tradability-masked)."""
    from factor_runner import compute_group_stats, prepare_signal

    base = alpha.loc[start:end]
    variants = {
        "raw": base,
        "size": neutralize_size_only(base, float_mktcap.loc[start:end]),
        "industry": panel_industry_demean(base, industry.loc[start:end]),
        "size_industry": neutralize_size_industry(
            base, industry.loc[start:end], float_mktcap.loc[start:end]
        ),
    }
    rows = []
    ic_map = {}
    for name, panel in variants.items():
        if session is not None and df_not_limit is not None:
            signal = prepare_signal(
                cs_zscore(panel),
                None,
                df_not_limit,
                df_not_st,
                df_trade_status,
                session,
                start.to_pydatetime() if hasattr(start, "to_pydatetime") else start,
                end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
            )
            _, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(
                signal, ret.loc[start:end], n=10, info="silent"
            )
            plt.close("all")
            stats = compute_group_stats(signal, ret.loc[start:end], group_pnl_df, group_to_df)
            daily = signal.corrwith(ret.loc[start:end], axis=1, method="spearman")
            ic_map[name] = daily
            row = summarize_ic(daily, name)
            means = group_pnl_df[[i for i in range(1, 11)]].mean().to_numpy()
            row.update(
                {
                    "hl_ann_return": stats["hl_annu_ret"],
                    "hl_sharpe": stats["hl_sharpe"],
                    "hl_max_drawdown": stats["hl_mdd"],
                    "avg_turnover": stats["hl_avg_turnover"],
                    "direction": stats["direction"],
                    "decile_mono_corr": float(np.corrcoef(np.arange(1, 11), means)[0, 1])
                    if len(means) == 10
                    else np.nan,
                }
            )
        else:
            # Fallback without masks (research only)
            daily = rank_ic(align_signal(cs_zscore(panel), 1), ret.loc[start:end]).dropna()
            ic_map[name] = daily
            gdf, ls, to = decile_long_short(cs_zscore(panel), ret.loc[start:end])
            perf = series_performance(ls.dropna()) if len(ls.dropna()) else {}
            row = summarize_ic(daily, name)
            row.update(
                {
                    "hl_ann_return": perf.get("annu_ret", np.nan),
                    "hl_sharpe": perf.get("sharpe", np.nan),
                    "hl_max_drawdown": perf.get("max_drawdown", np.nan),
                    "avg_turnover": float(to.mean()) if len(to.dropna()) else np.nan,
                    "decile_mono_corr": float(
                        np.corrcoef(np.arange(1, 11), gdf.mean().to_numpy())[0, 1]
                    )
                    if gdf.shape[1] == 10
                    else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), ic_map


def write_pack_docs(summary: dict) -> None:
    (PACK / "factor_definition.md").write_text(
        f"""# SKEW — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `SKEW` |
| **headline** | `{summary.get("headline_factor")}` |
| **family** | higher_moment · behavioral_lottery |
| **data_level** | daily EOD |
| **status** | research_candidate |
| **formula** | frozen (windows pre-registered) |

## Economic intuition

Investors overweight lottery-like (positively skewed) stocks → overpricing →
lower future returns. Negatively skewed names earn crash-risk compensation.

TGD20 asks *when* returns arrive within the day; SKEW asks *how asymmetric*
the return distribution is. Both are return-distribution information, but different moments.

## Delivery alpha

Raw research quantity has expected **negative** RankIC.
Delivery signal: `Alpha = -SKEW` (long low / negative skew).
""",
        encoding="utf-8",
    )
    (PACK / "formula.md").write_text(
        """# SKEW — Formula

## Baseline total skew

\\[
r_{i,t}=P_{i,t}/P_{i,t-1}-1,\\quad
\\mathrm{SKEW}_{i,t}^{(L)}=
\\frac{\\frac{1}{L}\\sum (r-\\bar r)^3}
{(\\frac{1}{L}\\sum (r-\\bar r)^2)^{3/2}}
\\]

Windows (pre-registered): L ∈ {20, 60, 120}.

## Idiosyncratic skew (headline)

\\[
r_{i,s}=\\alpha_{i,t}+\\beta_{i,t} r_{m,s}+\\varepsilon_{i,s},\\quad
\\mathrm{IdioSKEW}_{i,t}^{(L)}=\\mathrm{Skew}(\\varepsilon)
\\]

Market: CSI300 daily c2c. Windows: L ∈ {60, 120}.

Implementation: `core/factors/skew/` (rolling market-model residual + rolling skew).

## Alpha

\\[
\\mathrm{Alpha}=-\\mathrm{SKEW}
\\]

`signal_shift = 1` applied in evaluation.
""",
        encoding="utf-8",
    )
    (PACK / "data_source.md").write_text(
        """# SKEW — Data Source

| Item | Source |
|------|--------|
| Stock EOD | Wind `ASHAREEODPRICES` via `load_eod_enriched_tables` |
| Returns | close-to-close |
| Market | CSI300 `000300.SH` via `AINDEXEODPRICES` |
| Industry | CITICS industry panel |
| Size | float mktcap |
| TGD20 (interaction) | `research/cache/tgd_panels/TGD20_*_w20.parquet` |

P0 does **not** use minute bars. RSKEW20 is deferred (P1).
""",
        encoding="utf-8",
    )
    (PACK / "implementation.md").write_text(
        """# SKEW — Implementation

| Item | Path |
|------|------|
| Total skew | `core/factors/skew/skew.py` |
| Idio skew | `core/factors/skew/idio_skew.py` |
| Realized (P1 stub) | `core/factors/skew/realized_skew.py` |
| Engine wrappers | `factor_formulas_eod_engine.py::{skew_20d,skew_60d,skew_120d}` |
| Spec | `factor_specs/SKEW.yaml` |
| Runner | `run_skew_validation_v1.py` |

Engine wrappers expose **Alpha = -SKEW** for harness compatibility.
""",
        encoding="utf-8",
    )


def _df_md(df: pd.DataFrame) -> str:
    """Markdown table without optional tabulate dependency."""
    if df is None or df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if np.isfinite(v) else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_validation_md(ic_summary: pd.DataFrame, neut: pd.DataFrame) -> None:
    lines = [
        "# SKEW — Validation",
        "",
        "## IC summary (Alpha signals, ALL, signal_shift=1)",
        "",
        _df_md(ic_summary),
        "",
        "## Neutralization ladder (headline)",
        "",
        _df_md(neut),
        "",
    ]
    (PACK / "validation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SKEW / IdioSKEW validation v1")
    parser.add_argument("--skip-tgd", action="store_true")
    parser.add_argument("--skip-alpha-report", action="store_true")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Reuse research/cache/skew_panels parquet; still reloads EOD for returns/neut",
    )
    args = parser.parse_args()

    ensure_dirs()
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    log(f"Loading EOD ({preheat.date()} → {end.date()})...")

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    market_ret = load_csi300_index_return(preheat, end, session=session)
    # CITICS industry preheat cache starts ~2020-01-02; do not request earlier.
    industry = load_citics_industry_panel(max(start, dt.datetime(2020, 1, 2)), end)
    ret_1d = enriched.close / enriched.close.shift(1) - 1.0
    fwd_ret = ret_1d.copy()

    panels: Dict[str, pd.DataFrame] = {}
    if args.from_cache:
        log("Loading SKEW panels from cache...")
        for path in sorted(CACHE_DIR.glob("*.parquet")):
            name = path.name.split("_20")[0]  # AlphaIdioSKEW60_20200101_...
            # More robust: strip trailing _YYYYMMDD_YYYYMMDD.parquet
            stem = path.stem
            parts = stem.rsplit("_", 2)
            name = parts[0] if len(parts) == 3 and parts[1].isdigit() else stem
            panels[name] = pd.read_parquet(path)
            log(f"  loaded {name}")
        if not panels:
            raise FileNotFoundError(f"No parquet caches in {CACHE_DIR}")
    else:
        log("Building SKEW / IdioSKEW panels...")
        panels = build_factor_panels(ret_1d, market_ret)
        for name, panel in panels.items():
            path = CACHE_DIR / f"{name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
            panel.loc[start:end].to_parquet(path)
            log(f"  cached {name} -> {path.name}")

    # --- IC on Alpha and Raw ---
    ic_rows = []
    ic_daily_map = {}
    for name in list(P0_ALPHA) + list(P0_RAW) + ["AlphaSKEW60", "AlphaIdioSKEW120"]:
        if name not in panels:
            continue
        daily = rank_ic(align_signal(panels[name].loc[start:end], 1), fwd_ret.loc[start:end])
        ic_daily_map[name] = daily
        ic_rows.append(summarize_ic(daily, name))
    ic_summary = pd.DataFrame(ic_rows)
    ic_summary.to_csv(PACK / "tables/ic_summary.csv", index=False)
    ic_summary.to_csv(PACK / "ic_analysis/factor_summary.csv", index=False)
    ic_summary.to_csv(EXP / "ic_summary.csv", index=False)

    # --- Neutralization ladder on headline (tradability-masked groupTest) ---
    log(f"Neutralization ladder: {HEADLINE}")
    df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start, end)
    df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(start, end)
    df_trade_status = Factor_Dev_Lib.get_TradeStatus(start, end)
    ret_all = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c", base_index=None)
    neut, neut_ic = neutralization_ladder(
        panels[HEADLINE],
        ret_all,
        industry,
        enriched.float_mktcap,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        session=session,
        df_not_limit=df_not_limit,
        df_not_st=df_not_st,
        df_trade_status=df_trade_status,
    )
    neut.to_csv(PACK / "tables/neutralization_summary.csv", index=False)
    neut.to_csv(EXP / "neutralization_summary.csv", index=False)

    # --- Quantile / LS for headline size+industry (reuse ladder artifacts) ---
    from factor_runner import compute_group_stats, format_group_stats_title, prepare_signal

    si_panel = neutralize_size_industry(
        panels[HEADLINE].loc[start:end],
        industry.loc[start:end],
        enriched.float_mktcap.loc[start:end],
    )
    si_signal = prepare_signal(
        cs_zscore(si_panel),
        None,
        df_not_limit,
        df_not_st,
        df_trade_status,
        session,
        start,
        end,
    )
    _, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(
        si_signal, ret_all, n=10, info="silent"
    )
    plt.close("all")
    si_stats = compute_group_stats(si_signal, ret_all, group_pnl_df, group_to_df)
    gdf = group_pnl_df[[i for i in range(1, 11)]].copy()
    gdf.columns = [f"G{i}" for i in range(1, 11)]
    ls = group_pnl_df["H-L"] * si_stats["direction"]
    to = group_to_df["H-L"]
    gdf.to_csv(PACK / "quantile_analysis/decile_return.csv")
    gdf.mean().rename("mean_daily_return").reset_index().rename(
        columns={"index": "group"}
    ).to_csv(PACK / "tables/group_return.csv", index=False)
    ls.to_csv(PACK / "quantile_analysis/long_short_daily.csv", header=["ls"])
    to.to_csv(PACK / "execution/turnover_daily.csv", header=["turnover"])
    perf = series_performance(ls.dropna())
    port = pd.DataFrame(
        [
            {
                "factor": HEADLINE,
                "variant": "size_industry",
                "annual_return": perf.get("annu_ret", np.nan),
                "sharpe": perf.get("sharpe", np.nan),
                "max_drawdown": perf.get("max_drawdown", np.nan),
                "avg_turnover": float(to.mean()),
                "n_days": perf.get("n_days", np.nan),
                "direction": int(si_stats["direction"]),
            }
        ]
    )
    port.to_csv(PACK / "tables/portfolio_summary.csv", index=False)

    plot_ic(
        neut_ic["size_industry"],
        PACK / "figures/ic_curve.png",
        f"{HEADLINE} RankIC (size+industry)",
    )
    plot_ic(
        neut_ic["size_industry"],
        PACK / "ic_analysis/ic_curve.png",
        f"{HEADLINE} RankIC (size+industry)",
    )
    plot_decile(gdf, PACK / "figures/quantile_return.png", f"{HEADLINE} decile means (SI)")
    plot_decile(
        gdf, PACK / "quantile_analysis/decile_return.png", f"{HEADLINE} decile means (SI)"
    )
    si_title = format_group_stats_title(si_stats)
    save_grouptest_cum_figure(
        group_pnl_df,
        factor_name=f"{HEADLINE} size_industry",
        stats_title=si_title,
        out_path=PACK / "figures/cumulative_return.png",
    )
    save_grouptest_cum_figure(
        group_pnl_df,
        factor_name=f"{HEADLINE} size_industry",
        stats_title=si_title,
        out_path=PACK / "quantile_analysis/cumulative_long_short.png",
    )
    group_pnl_df.to_csv(PACK / "quantile_analysis/group_daily_pnl.csv")
    group_pnl_df.cumsum().to_csv(PACK / "quantile_analysis/group_cum_pnl.csv")

    # --- Mechanism ---
    log("Lottery / tail mechanism...")
    mech = mechanism_lottery(
        panels["SKEW20"].loc[start:end],
        ret_1d.loc[start:end],
        enriched.turnover.loc[start:end] if enriched.turnover is not None else None,
    )
    mech.to_csv(PACK / "mechanism/lottery_correlation.csv", index=False)
    mech.to_csv(PACK / "tables/lottery_correlation.csv", index=False)

    # --- TGD interaction ---
    interaction = pd.DataFrame()
    corr_tgd = np.nan
    if not args.skip_tgd and TGD_PANEL.exists():
        log("TGD20 interaction...")
        tgd = pd.read_parquet(TGD_PANEL)
        if {"date", "symbol", "TGD20"}.issubset(set(tgd.columns)):
            tgd_wide = tgd.pivot(index="date", columns="symbol", values="TGD20")
        else:
            # Canonical TGD cache is already wide (dates × symbols).
            tgd_wide = tgd.copy()
        tgd_wide.index = pd.to_datetime(tgd_wide.index)
        corr_series = cs_spearman(
            panels["AlphaIdioSKEW60"].loc[start:end], tgd_wide.loc[start:end]
        ).dropna()
        corr_tgd = float(corr_series.mean()) if len(corr_series) else np.nan
        interaction = double_sort_tgd(
            panels["AlphaIdioSKEW60"].loc[start:end],
            tgd_wide.loc[start:end],
            fwd_ret.loc[start:end],
        )
        interaction.to_csv(PACK / "tables/tgd_skew_interaction.csv", index=False)
        interaction.to_csv(EXP / "tgd_skew_interaction.csv", index=False)
        pd.DataFrame(
            [{"pair": "AlphaIdioSKEW60_vs_TGD20", "mean_cs_spearman": corr_tgd}]
        ).to_csv(PACK / "tables/tgd_skew_corr.csv", index=False)
    else:
        log("TGD panel missing or skipped — interaction deferred")

    # --- Optional library alpha-report on AlphaSKEW20 / AlphaIdioSKEW60 ---
    if not args.skip_alpha_report:
        log("Publishing alpha_research_report for P0 alpha factors...")

        def get_ret_matrix(s, e, idx):
            return Factor_Dev_Lib.get_Ret_Matrix(s, e, method="c2c", base_index=idx)

        summary_rows = []
        for name in P0_ALPHA:
            report = build_factor_report(
                name,
                panels[name],
                enriched.close.loc[start:end],
                start_day=start,
                end_day=end,
                session=session,
                df_not_limit=df_not_limit,
                df_not_st=df_not_st,
                df_trade_status=df_trade_status,
                universes=cfg.UNIVERSE_LIST,
                get_ret_matrix=get_ret_matrix,
            )
            publish_factor_report(report, EXP / "alpha_reports")
            summary_rows.append(report_summary_row(report))
        pd.DataFrame(summary_rows).to_csv(EXP / "alpha_report_summary.csv", index=False)

    # --- Yearly IC stability ---
    yearly_rows = []
    for name, daily in ic_daily_map.items():
        if name not in P0_ALPHA:
            continue
        s = daily.dropna()
        for year, part in s.groupby(s.index.year):
            yearly_rows.append(summarize_ic(part, f"{name}_{year}"))
            yearly_rows[-1]["year"] = int(year)
            yearly_rows[-1]["factor"] = name
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(PACK / "stability/yearly_ic.csv", index=False)

    # --- Pack docs / summary.yaml / delivery ---
    headline_ic = neut[neut["label"] == "size_industry"].iloc[0].to_dict() if len(neut) else {}
    summary = {
        "factor_id": "SKEW",
        "headline_factor": HEADLINE,
        "status": "research_candidate",
        "period": f"{start.date()}_{end.date()}",
        "rank_ic_size_industry": headline_ic.get("mean_rank_ic"),
        "icir_size_industry": headline_ic.get("icir_annualized"),
        "hl_sharpe_size_industry": headline_ic.get("hl_sharpe"),
        "hl_ann_return_size_industry": headline_ic.get("hl_ann_return"),
        "mean_corr_vs_tgd20": corr_tgd,
        "p0_factors": list(P0_ALPHA),
    }
    (PACK / "summary.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )
    (PACK / "scout_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_pack_docs(summary)
    write_validation_md(ic_summary, neut)

    (PACK / "README.md").write_text(
        f"""# SKEW Pack

Research-grade skewness anomaly replication (daily + idiosyncratic).

- Headline: **{HEADLINE}** (`Alpha = -IdioSKEW60`)
- Spec: `factor_specs/SKEW.yaml`
- Runner: `run_skew_validation_v1.py`
- Delivery: `research_delivery/factors/SKEW/`

## Headline (size + industry)

| Metric | Value |
|--------|------:|
| RankIC | {headline_ic.get("mean_rank_ic", float("nan")):.4f} |
| ICIR | {headline_ic.get("icir_annualized", float("nan")):.2f} |
| HL Sharpe | {headline_ic.get("hl_sharpe", float("nan")):.2f} |
| Corr vs TGD20 | {corr_tgd if corr_tgd == corr_tgd else float("nan"):.3f} |

See `validation.md`, `tables/`, `figures/`.
""",
        encoding="utf-8",
    )

    # Delivery card
    for src, dst in (
        (PACK / "figures/ic_curve.png", DELIVERY / "plots/ic_curve.png"),
        (PACK / "figures/quantile_return.png", DELIVERY / "plots/decile_return.png"),
        (PACK / "figures/cumulative_return.png", DELIVERY / "plots/cumulative_long_short.png"),
    ):
        if src.exists():
            shutil.copy2(src, dst)
    port.to_csv(DELIVERY / "metrics.csv", index=False)
    (DELIVERY / "formula.md").write_text((PACK / "formula.md").read_text(encoding="utf-8"), encoding="utf-8")

    log("Writing delivery report...")
    _write_delivery_report(
        ic_summary=ic_summary,
        neut=neut,
        mech=mech,
        port=port,
        corr_tgd=corr_tgd,
        interaction=interaction,
        headline_ic=headline_ic,
        start=start,
        end=end,
    )

    session.close()
    log("DONE")
    log(ic_summary.to_string(index=False))
    log(neut.to_string(index=False))


def _write_delivery_report(
    *,
    ic_summary: pd.DataFrame,
    neut: pd.DataFrame,
    mech: pd.DataFrame,
    port: pd.DataFrame,
    corr_tgd: float,
    interaction: pd.DataFrame,
    headline_ic: dict,
    start,
    end,
) -> None:
    def _fmt_table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_n/a_"
        return _df_md(df)

    raw_row = ic_summary[ic_summary["label"] == "IdioSKEW60"]
    raw_ic = float(raw_row["mean_rank_ic"].iloc[0]) if len(raw_row) else np.nan

    md = f"""# SKEW / IdioSKEW — Factor Research Report

**Status:** `research_candidate` · **Tier:** A- · **Family:** higher_moment / behavioral_lottery  
**Delivery card:** `research_delivery/factors/SKEW/`  
**Canonical pack:** `research/reports/factors/SKEW/`  
**Period:** {start.date()} → {end.date()}  
**Headline:** `{HEADLINE}` (Alpha = −IdioSKEW60, size+industry)

---

## 1. Research Question

Does equity-return skewness contain **independent cross-sectional alpha** in A-shares,
and is that information distinct from TGD20’s intraday timing residual?

---

## 2. Economic Hypothesis

Investors prefer lottery-like (positively skewed) payoffs → overpricing → lower future
returns. Negatively skewed names require crash-risk compensation → higher future returns.

This is a **distribution-shape** channel, complementary to TGD20’s **arrival-time** channel.

---

## 3. Formula (frozen)

See [`formula.md`](formula.md).

- Baseline: `SKEW20/60/120` = rolling skew of daily c2c returns  
- Headline: `IdioSKEW60` = rolling skew of CSI300 market-model residuals  
- Delivery: `Alpha = -raw` (long low / negative skew)  
- `signal_shift = 1`

Windows were **pre-registered**; no post-hoc window mining.

---

## 4. IC Statistics

### Alpha signals (expected positive IC)

{_fmt_table(ic_summary[ic_summary["label"].astype(str).str.startswith("Alpha")])}

### Raw research quantities (expected negative IC)

{_fmt_table(ic_summary[~ic_summary["label"].astype(str).str.startswith("Alpha")])}

Raw IdioSKEW60 mean RankIC ≈ **{raw_ic:.4f}** (sign check vs lottery theory).

### IC curve (headline, size+industry)

![IC](plots/ic_curve.png)

---

## 5. Quantile / Long-Short

Size+industry neutralized `{HEADLINE}`:

{_fmt_table(port)}

![Decile](plots/decile_return.png)

![Long-short](plots/cumulative_long_short.png)

Expected: G1 (lowest alpha / highest skew) underperforms G10 (highest alpha / lowest skew).

---

## 6. Neutralization Robustness

{_fmt_table(neut)}

---

## 7. Mechanism — Lottery Link

Cross-sectional Spearman of raw `SKEW20` vs lottery proxies:

{_fmt_table(mech)}

Expected: positive correlation with MAX / turnover / volatility / upside-tail frequency.

---

## 8. Relation with TGD20

Mean daily CS Spearman(`AlphaIdioSKEW60`, `TGD20`) ≈ **{corr_tgd if corr_tgd == corr_tgd else float("nan"):.3f}**.

Broker notes suggest ~0.1–0.2; low overlap would support an independent distribution layer.

### Double sort (2×5)

{_fmt_table(interaction.head(40) if interaction is not None else pd.DataFrame())}

Full table: `research/reports/factors/SKEW/tables/tgd_skew_interaction.csv`.

---

## 9. Findings

1. **Formula:** daily total skew + CICC-style idiosyncratic skew (CSI300 residual).  
2. **Intuition:** lottery / crash-risk compensation in the third moment.  
3. **IC:** headline size+industry RankIC={headline_ic.get("mean_rank_ic", float("nan")):.4f}, ICIR={headline_ic.get("icir_annualized", float("nan")):.2f}.  
4. **Portfolio:** HL Sharpe={headline_ic.get("hl_sharpe", float("nan")):.2f}.  
5. **Neutralization:** see ladder — residual alpha after size+industry is the investability gate.  
6. **vs TGD20:** average rank correlation {corr_tgd if corr_tgd == corr_tgd else float("nan"):.3f}; combination tests are P2.

---

## 10. Do not

- Retune 20/60/120 after looking at the full sample  
- Promote minute RSKEW under the same id  
- Equal-weight with TGD20 without incremental-IC / double-sort evidence  

---

## Artifacts

| Path | Content |
|------|---------|
| `research/reports/factors/SKEW/` | Pack v1 |
| `research/reports/skew_v1/` | Experiment root |
| `research/cache/skew_panels/` | Factor parquet caches |
| `core/factors/skew/` | Canonical formulas |
"""
    (DELIVERY / "report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
