#!/usr/bin/env python3
"""SKEW v2 optimization: purify daily skewness anomaly (no minute RSKEW).

P0: VolAdj / VolResid / TailSKEW
P1: MaxResid / TGDResid / VolMaxResid
P2: MAD sensitivity on baseline

Updates artifacts under research/reports/factors/SKEW/v2/ and the original
buyside report package.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from alpha_research_report import save_cumulative_decile_figure
from core.factors.skew.skew_v2 import (
    build_skew_v2_panels,
    mad_winsorize_cs,
    max_return,
)
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from factor_runner import compute_group_stats, format_group_stats_title, prepare_signal
from industry_neutral import load_citics_industry_panel
from research.extreme_return_study.src.data_loader import load_csi300_index_return

ANNUAL = 250
PACK = ROOT / "research/reports/factors/SKEW"
V2 = PACK / "v2"
CACHE = ROOT / "research/cache/skew_panels"
TGD_PATH = ROOT / "research/cache/tgd_panels/TGD20_20200101_20251231_w20.parquet"
PKG = ROOT / "research_delivery/SKEW_research_package"
DELIV = ROOT / "research_delivery/factors/SKEW"
BASELINE = "AlphaIdioSKEW60"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for p in (V2, V2 / "figures", V2 / "tables", PKG / "data/analysis", PKG / "figures", DELIV / "plots"):
        p.mkdir(parents=True, exist_ok=True)


def summarize_ic(daily: pd.Series, label: str) -> dict:
    s = daily.dropna()
    mean = float(s.mean()) if len(s) else np.nan
    std = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return {
        "label": label,
        "n_days": int(len(s)),
        "mean_rank_ic": mean,
        "ic_std": std,
        "icir_annualized": mean / std * math.sqrt(ANNUAL) if std and std > 0 else np.nan,
        "positive_ic_ratio": float((s > 0).mean()) if len(s) else np.nan,
        "t_stat": mean / std * math.sqrt(len(s)) if std and std > 0 else np.nan,
    }


def eval_panel(
    name: str,
    panel: pd.DataFrame,
    *,
    ret: pd.DataFrame,
    industry: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    session,
    df_not_limit,
    df_not_st,
    df_trade_status,
    start,
    end,
    do_si: bool = True,
) -> Tuple[dict, dict, pd.DataFrame, pd.Series]:
    """Raw + optional size/industry groupTest metrics."""
    rows = {}
    si_stats = {}
    si_group = pd.DataFrame()
    si_ic = pd.Series(dtype=float)

    for mode, signal_panel in (
        ("raw", panel.loc[start:end]),
        (
            "size_industry",
            neutralize_size_industry(
                panel.loc[start:end],
                industry.loc[start:end],
                float_mktcap.loc[start:end],
            )
            if do_si
            else None,
        ),
    ):
        if signal_panel is None:
            continue
        signal = prepare_signal(
            cs_zscore(signal_panel),
            None,
            df_not_limit,
            df_not_st,
            df_trade_status,
            session,
            start,
            end,
        )
        if signal.empty or signal.dropna(how="all").empty:
            log(f"  SKIP {name}|{mode}: empty after prepare_signal")
            continue
        # drop all-NaN rows that break np.apply_along_axis
        signal = signal.dropna(how="all")
        if signal.shape[0] == 0 or signal.shape[1] == 0:
            log(f"  SKIP {name}|{mode}: zero-shaped signal")
            continue
        _, group_pnl, group_to = Factor_Dev_Lib.groupTest(signal, ret, n=10, info="silent")
        plt.close("all")
        stats = compute_group_stats(signal, ret, group_pnl, group_to)
        daily_ic = signal.corrwith(ret, axis=1, method="spearman")
        row = summarize_ic(daily_ic, f"{name}|{mode}")
        row.update(
            {
                "factor": name,
                "mode": mode,
                "hl_ann_return": stats["hl_annu_ret"],
                "hl_sharpe": stats["hl_sharpe"],
                "hl_max_drawdown": stats["hl_mdd"],
                "avg_turnover": stats["hl_avg_turnover"],
                "direction": stats["direction"],
                "mono_corr": float(
                    np.corrcoef(
                        np.arange(1, 11),
                        group_pnl[[i for i in range(1, 11)]].mean().to_numpy(),
                    )[0, 1]
                ),
            }
        )
        rows[mode] = row
        if mode == "size_industry":
            si_stats = stats
            si_group = group_pnl
            si_ic = daily_ic
    if "raw" not in rows:
        raise RuntimeError(f"{name}: raw evaluation failed (empty signal?)")
    return rows["raw"], rows.get("size_industry", {}), si_group, si_ic


def cs_mean_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a, b = a.align(b, join="inner")
    s = a.rank(axis=1, pct=True).corrwith(b.rank(axis=1, pct=True), axis=1).dropna()
    return float(s.mean()) if len(s) else np.nan


def main() -> None:
    ensure_dirs()
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    log(f"SKEW v2 load EOD {preheat.date()} → {end.date()}")

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(max(start, dt.datetime(2020, 1, 2)), end)
    ret_1d = enriched.close / enriched.close.shift(1) - 1.0
    market = load_csi300_index_return(preheat, end, session=session)

    # baseline panels from cache
    idio = pd.read_parquet(CACHE / f"IdioSKEW60_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet")
    alpha_base = pd.read_parquet(
        CACHE / f"AlphaIdioSKEW60_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
    )
    tgd = pd.read_parquet(TGD_PATH)
    tgd.index = pd.to_datetime(tgd.index)

    parser_from_cache = False
    # resume: rebuild only TailSKEW if needed, reuse other caches
    log("Building / refreshing v2 panels...")
    v2 = build_skew_v2_panels(
        ret_1d,
        idio_skew=idio.reindex_like(ret_1d),
        tgd=tgd.reindex(index=ret_1d.index, columns=ret_1d.columns),
        as_alpha=True,
    )
    panels: Dict[str, pd.DataFrame] = {BASELINE: alpha_base.reindex_like(ret_1d)}
    panels.update(v2)
    idio_mad = mad_winsorize_cs(idio.reindex_like(ret_1d))
    panels["AlphaIdioSKEW60_MAD"] = -idio_mad

    for name, panel in panels.items():
        path = CACHE / f"{name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        panel.loc[start:end].to_parquet(path)
        log(f"  cached {name}")

    # Skip variants already fully evaluated if comparison file exists and --resume
    done_factors = set()
    prev_path = V2 / "tables/variant_comparison.csv"
    # always re-run full eval for consistency after TailSKEW fix
    _ = done_factors, prev_path, parser_from_cache

    df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start, end)
    df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(start, end)
    df_trade_status = Factor_Dev_Lib.get_TradeStatus(start, end)
    ret_all = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c", base_index=None)

    # Evaluate all (SI for P0/P1/baseline; raw-only skip none)
    summary_rows: List[dict] = []
    best_name = BASELINE
    best_sharpe = -np.inf
    best_group = None
    best_ic = None
    best_stats = None

    # cheaper: SI only for main candidates
    evaluate = [
        BASELINE,
        "AlphaVolAdj_IdioSKEW60",
        "AlphaVolResid_IdioSKEW60",
        "AlphaTailSKEW60",
        "AlphaMaxResid_IdioSKEW60",
        "AlphaVolMaxResid_IdioSKEW60",
        "AlphaTGDResid_IdioSKEW60",
        "AlphaVolMaxTGDResid_IdioSKEW60",
        "AlphaIdioSKEW60_MAD",
    ]
    for name in evaluate:
        if name not in panels:
            continue
        log(f"eval {name}")
        raw_row, si_row, si_group, si_ic = eval_panel(
            name,
            panels[name],
            ret=ret_all,
            industry=industry,
            float_mktcap=enriched.float_mktcap,
            session=session,
            df_not_limit=df_not_limit,
            df_not_st=df_not_st,
            df_trade_status=df_trade_status,
            start=start,
            end=end,
            do_si=True,
        )
        summary_rows.append(raw_row)
        if si_row:
            summary_rows.append(si_row)
            sharpe = si_row.get("hl_sharpe", np.nan)
            if sharpe == sharpe and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = name
                best_group = si_group
                best_ic = si_ic
                best_stats = si_row

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(V2 / "tables/variant_comparison.csv", index=False)
    summary.to_csv(PKG / "data/analysis/v2_variant_comparison.csv", index=False)

    # correlation matrix of alpha panels (mean CS spearman)
    names = [n for n in evaluate if n in panels]
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for b in names[i:]:
            c = cs_mean_corr(panels[a].loc[start:end], panels[b].loc[start:end])
            corr.loc[a, b] = c
            corr.loc[b, a] = c
    corr.to_csv(V2 / "tables/correlation_matrix.csv")
    corr.to_csv(PKG / "data/analysis/v2_correlation_matrix.csv")

    # vs MAX / VOL / TGD attribution for baseline and best
    max20 = max_return(ret_1d, 20)
    vol20 = ret_1d.rolling(20, min_periods=10).std()
    attr_rows = []
    for name in [BASELINE, best_name]:
        if name not in panels:
            continue
        p = panels[name].loc[start:end]
        attr_rows.append(
            {
                "factor": name,
                "corr_MAX20": cs_mean_corr(p, max20.loc[start:end]),
                "corr_VOL20": cs_mean_corr(p, vol20.loc[start:end]),
                "corr_TGD20": cs_mean_corr(p, tgd.loc[start:end]),
            }
        )
    attr = pd.DataFrame(attr_rows)
    attr.to_csv(V2 / "tables/style_contamination.csv", index=False)
    attr.to_csv(PKG / "data/analysis/v2_style_contamination.csv", index=False)

    # IC improvement table (SI)
    si = summary[summary["mode"] == "size_industry"].copy()
    base_si = si[si["factor"] == BASELINE]
    base_sharpe = float(base_si["hl_sharpe"].iloc[0]) if len(base_si) else np.nan
    base_icir = float(base_si["icir_annualized"].iloc[0]) if len(base_si) else np.nan
    si["delta_hl_sharpe_vs_baseline"] = si["hl_sharpe"] - base_sharpe
    si["delta_icir_vs_baseline"] = si["icir_annualized"] - base_icir
    si = si.sort_values("hl_sharpe", ascending=False)
    si.to_csv(V2 / "tables/ic_improvement_attribution.csv", index=False)
    si.to_csv(PKG / "data/analysis/v2_ic_improvement_attribution.csv", index=False)

    # figures: comparison bars + best cum plot + corr heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df = si.copy()
    axes[0].barh(plot_df["factor"], plot_df["hl_sharpe"], color="#2f80ed")
    axes[0].axvline(base_sharpe, color="red", ls="--", label=f"baseline {base_sharpe:.2f}")
    axes[0].set_xlabel("HL Sharpe (size+industry)")
    axes[0].set_title("SKEW v2 HL Sharpe comparison")
    axes[0].legend()
    axes[1].barh(plot_df["factor"], plot_df["icir_annualized"], color="#27ae60")
    axes[1].axvline(base_icir, color="red", ls="--")
    axes[1].set_xlabel("ICIR (size+industry)")
    axes[1].set_title("SKEW v2 ICIR comparison")
    fig.tight_layout()
    for out in (
        V2 / "figures/variant_comparison.png",
        PKG / "figures/v2_variant_comparison.png",
        DELIV / "plots/v2_variant_comparison.png",
    ):
        fig.savefig(out, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Mean CS Spearman among Alpha panels")
    fig.tight_layout()
    for out in (
        V2 / "figures/correlation_heatmap.png",
        PKG / "figures/v2_correlation_heatmap.png",
        DELIV / "plots/v2_correlation_heatmap.png",
    ):
        fig.savefig(out, dpi=160)
    plt.close(fig)

    if best_group is not None and best_stats is not None:
        title = (
            f"H-L, Direction: {int(best_stats['direction'])}, "
            f"AnnuRet: {best_stats['hl_ann_return']:.2%},"
            f"Sharpe_Ratio: {best_stats['hl_sharpe']:.2f}, "
            f"MDD: {best_stats['hl_max_drawdown']:.2%}, "
            f"Daily Turnover: {best_stats['avg_turnover']:.2f},\n "
            f"Daily IC: {best_stats['mean_rank_ic']:.4f}, "
            f"Annu ICIR: {best_stats['icir_annualized']:.2f}"
        )
        for out in (
            V2 / "figures/best_cumulative_long_short.png",
            PKG / "figures/v2_best_cumulative_long_short.png",
            DELIV / "plots/v2_best_cumulative_long_short.png",
            # overwrite headline delivery plot only if best beats baseline materially
        ):
            save_cumulative_decile_figure(
                best_group.cumsum(),
                factor_name=f"{best_name} size_industry",
                stats_title=title,
                out_path=out,
            )
        # If best improves Sharpe, also refresh main cumulative plot
        if best_name != BASELINE and best_sharpe > base_sharpe + 0.05:
            save_cumulative_decile_figure(
                best_group.cumsum(),
                factor_name=f"{best_name} size_industry (v2 headline)",
                stats_title=title,
                out_path=PKG / "figures/cumulative_long_short.png",
            )
            save_cumulative_decile_figure(
                best_group.cumsum(),
                factor_name=f"{best_name} size_industry (v2 headline)",
                stats_title=title,
                out_path=DELIV / "plots/cumulative_long_short.png",
            )

    meta = {
        "baseline": BASELINE,
        "baseline_si_hl_sharpe": base_sharpe,
        "baseline_si_icir": base_icir,
        "best_variant": best_name,
        "best_si_hl_sharpe": float(best_sharpe) if best_sharpe == best_sharpe else None,
        "period": f"{start.date()}_{end.date()}",
        "note": "Minute RSKEW deferred; v2 focuses on vol/MAX/TGD purification + TailSKEW",
    }
    (V2 / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (PKG / "data/analysis/v2_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    session.close()
    log("=== SI ranking ===")
    log(si[["factor", "mean_rank_ic", "icir_annualized", "hl_sharpe", "hl_ann_return", "mono_corr", "delta_hl_sharpe_vs_baseline"]].to_string(index=False))
    log(f"BEST: {best_name} HL Sharpe={best_sharpe:.3f} (baseline {base_sharpe:.3f})")
    log("DONE v2")


if __name__ == "__main__":
    main()
