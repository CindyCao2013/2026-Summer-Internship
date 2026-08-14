#!/usr/bin/env python3
"""SKEW single-factor v4: formation-window scan + Raw vs Idio + MAD n scan.

No multi-factor synthesis, no minute data, T+1 only.
Writes research/reports/factors/SKEW/v4/ and package analysis/figures.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from alpha_research_report import save_cumulative_decile_figure
from core.factors.skew.idio_skew import build_idio_skew
from core.factors.skew.skew import alpha_from_skew, total_return_skew
from core.factors.skew.skew_v2 import mad_winsorize_cs
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from factor_runner import prepare_signal
from industry_neutral import load_citics_industry_panel
from research.extreme_return_study.src.data_loader import load_csi300_index_return

ANNUAL = 250
PACK = ROOT / "research/reports/factors/SKEW"
V4 = PACK / "v4"
CACHE = ROOT / "research/cache/skew_panels"
PKG = ROOT / "research_delivery/SKEW_research_package"
DELIV = ROOT / "research_delivery/factors/SKEW"
WINDOWS = (40, 50, 60, 75, 90)
MAD_NS = (3.5, 4.0, 5.0, 6.0, 7.0)
BASELINE_LABEL = "AlphaIdioSKEW60_MAD_n5"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for p in (
        V4,
        V4 / "figures",
        V4 / "tables",
        PKG / "data/analysis",
        PKG / "figures",
        DELIV / "plots",
        CACHE,
    ):
        p.mkdir(parents=True, exist_ok=True)


def min_periods_for(window: int) -> int:
    return max(20, int(round(window * 2 / 3)))


def summarize_ic(daily: pd.Series) -> dict:
    s = daily.dropna()
    mean = float(s.mean()) if len(s) else np.nan
    std = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return {
        "n_days": int(len(s)),
        "mean_rank_ic": mean,
        "ic_std": std,
        "icir_annualized": mean / std * math.sqrt(ANNUAL) if std and std > 0 else np.nan,
        "positive_ic_ratio": float((s > 0).mean()) if len(s) else np.nan,
        "t_stat": mean / std * math.sqrt(len(s)) if std and std > 0 else np.nan,
    }


def eval_si(
    name: str,
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    industry: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    session,
    df_not_limit,
    df_not_st,
    df_trade_status,
    start,
    end,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    neut = neutralize_size_industry(
        panel.loc[start:end],
        industry.loc[start:end],
        float_mktcap.loc[start:end],
    )
    signal = prepare_signal(
        cs_zscore(neut),
        None,
        df_not_limit,
        df_not_st,
        df_trade_status,
        session,
        start,
        end,
    ).dropna(how="all")
    ret_use = ret.loc[start:end].reindex(index=signal.index, columns=signal.columns)
    if signal.empty or signal.shape[0] == 0:
        raise RuntimeError(f"{name}: empty signal")

    _, group_pnl, group_to = Factor_Dev_Lib.groupTest(signal, ret_use, n=10, info="silent")
    plt.close("all")

    daily_pnl = group_pnl["H-L"]
    direction = 1 if daily_pnl.mean() > 0 else -1
    pnl_adj = daily_pnl * direction
    ic = signal.corrwith(ret_use, axis=1, method="spearman")
    row = summarize_ic(ic)
    row.update(
        {
            "factor": name,
            "mode": "size_industry",
            "hl_ann_return": float(pnl_adj.mean() * ANNUAL) if len(pnl_adj) else np.nan,
            "hl_sharpe": (
                float(pnl_adj.mean() / pnl_adj.std(ddof=1) * math.sqrt(ANNUAL))
                if len(pnl_adj) > 1 and pnl_adj.std(ddof=1) > 0
                else np.nan
            ),
            "hl_max_drawdown": float(Factor_Dev_Lib.calMDD(pnl_adj)[0]),
            "avg_turnover": float(group_to["H-L"].mean()),
            "direction": int(direction),
            "mono_corr": float(
                np.corrcoef(
                    np.arange(1, 11),
                    group_pnl[[i for i in range(1, 11)]].mean().to_numpy(),
                )[0, 1]
            ),
        }
    )
    return row, group_pnl


def build_alpha_panels(
    ret_1d: pd.DataFrame,
    market: pd.Series,
    windows: tuple,
) -> Dict[str, pd.DataFrame]:
    """Raw / Idio raw-skew panels keyed for Alpha = -skew after MAD."""
    panels: Dict[str, pd.DataFrame] = {}
    for w in windows:
        mp = min_periods_for(w)
        raw = total_return_skew(ret_1d, w, min_periods=mp)
        panels[f"RawSKEW{w}"] = raw
        idio = build_idio_skew(
            ret_1d, market, windows=(w,), as_alpha=False, min_periods=mp
        )[f"IdioSKEW{w}"]
        panels[f"IdioSKEW{w}"] = idio
        log(f"  built Raw/Idio window={w} min_periods={mp}")
    return panels


def main() -> None:
    ensure_dirs()
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    log(f"SKEW v4 load EOD {preheat.date()} → {end.date()}")

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(max(start, dt.datetime(2020, 1, 2)), end)
    ret_1d = enriched.close / enriched.close.shift(1) - 1.0
    market = load_csi300_index_return(preheat, end, session=session)

    df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start, end)
    df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(start, end)
    df_trade_status = Factor_Dev_Lib.get_TradeStatus(start, end)
    ret_daily = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c", base_index=None)

    log("=== Direction 1+2: window × Raw/Idio (MAD n=5) ===")
    skew_panels = build_alpha_panels(ret_1d, market, WINDOWS)

    window_rows: List[dict] = []
    best_row = None
    best_group = None
    best_name = None
    baseline_row = None

    for w in WINDOWS:
        for kind in ("Idio", "Raw"):
            key = f"{kind}SKEW{w}"
            raw_skew = skew_panels[key]
            alpha_mad = -mad_winsorize_cs(raw_skew, n_mad=5.0)
            name = f"Alpha{kind}SKEW{w}_MAD"
            cache_path = CACHE / f"{name}_{tag}.parquet"
            alpha_mad.loc[start:end].to_parquet(cache_path)
            log(f"  eval {name}")
            row, group = eval_si(
                name,
                alpha_mad,
                ret_daily,
                industry=industry,
                float_mktcap=enriched.float_mktcap,
                session=session,
                df_not_limit=df_not_limit,
                df_not_st=df_not_st,
                df_trade_status=df_trade_status,
                start=start,
                end=end,
            )
            row.update({"kind": kind, "window": int(w), "n_mad": 5.0})
            window_rows.append(row)
            if kind == "Idio" and w == 60:
                baseline_row = row
            if best_row is None or row["hl_sharpe"] > best_row["hl_sharpe"]:
                best_row = row
                best_group = group
                best_name = name

    window_df = pd.DataFrame(window_rows).sort_values("hl_sharpe", ascending=False)
    window_df.to_csv(V4 / "tables/window_raw_idio_scan.csv", index=False)
    window_df.to_csv(PKG / "data/analysis/v4_window_raw_idio_scan.csv", index=False)
    log(window_df[["kind", "window", "mean_rank_ic", "icir_annualized", "hl_sharpe", "avg_turnover"]].to_string(index=False))

    # ---------- Direction 3: MAD n on Idio60 (+ on best window if different) ----------
    log("=== Direction 3: MAD n scan ===")
    mad_targets = [("Idio", 60)]
    if best_row is not None and not (best_row["kind"] == "Idio" and best_row["window"] == 60):
        mad_targets.append((best_row["kind"], int(best_row["window"])))

    mad_rows: List[dict] = []
    best_mad_row = None
    best_mad_group = None
    best_mad_name = None

    for kind, w in mad_targets:
        raw_skew = skew_panels[f"{kind}SKEW{w}"]
        for n in MAD_NS:
            alpha = -mad_winsorize_cs(raw_skew, n_mad=float(n))
            name = f"Alpha{kind}SKEW{w}_MAD_n{n:g}"
            log(f"  eval {name}")
            row, group = eval_si(
                name,
                alpha,
                ret_daily,
                industry=industry,
                float_mktcap=enriched.float_mktcap,
                session=session,
                df_not_limit=df_not_limit,
                df_not_st=df_not_st,
                df_trade_status=df_trade_status,
                start=start,
                end=end,
            )
            row.update({"kind": kind, "window": int(w), "n_mad": float(n)})
            mad_rows.append(row)
            if best_mad_row is None or row["hl_sharpe"] > best_mad_row["hl_sharpe"]:
                best_mad_row = row
                best_mad_group = group
                best_mad_name = name

    mad_df = pd.DataFrame(mad_rows).sort_values("hl_sharpe", ascending=False)
    mad_df.to_csv(V4 / "tables/mad_n_scan.csv", index=False)
    mad_df.to_csv(PKG / "data/analysis/v4_mad_n_scan.csv", index=False)
    log(mad_df[["factor", "n_mad", "mean_rank_ic", "icir_annualized", "hl_sharpe"]].to_string(index=False))

    # ---------- figures ----------
    pivot_s = (
        window_df.pivot_table(index="window", columns="kind", values="hl_sharpe")
        .reindex(WINDOWS)
    )
    pivot_i = (
        window_df.pivot_table(index="window", columns="kind", values="icir_annualized")
        .reindex(WINDOWS)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for kind, color in (("Idio", "#2f80ed"), ("Raw", "#e67e22")):
        if kind in pivot_s.columns:
            axes[0].plot(pivot_s.index, pivot_s[kind], "o-", color=color, label=kind)
            axes[1].plot(pivot_i.index, pivot_i[kind], "s--", color=color, label=kind)
    axes[0].axvline(60, color="gray", ls=":", alpha=0.7)
    axes[1].axvline(60, color="gray", ls=":", alpha=0.7)
    axes[0].set_title("HL Sharpe vs formation window (MAD n=5, SI)")
    axes[1].set_title("ICIR vs formation window (MAD n=5, SI)")
    axes[0].set_xlabel("window")
    axes[1].set_xlabel("window")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    for out in (
        V4 / "figures/v4_window_raw_idio.png",
        PKG / "figures/v4_window_raw_idio.png",
        DELIV / "plots/v4_window_raw_idio.png",
    ):
        fig.savefig(out, dpi=160)
    plt.close(fig)

    # MAD n plot for Idio60
    mad60 = mad_df[(mad_df["kind"] == "Idio") & (mad_df["window"] == 60)].sort_values("n_mad")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(mad60["n_mad"], mad60["hl_sharpe"], "o-", color="#2f80ed", label="HL Sharpe")
    ax2 = ax.twinx()
    ax2.plot(mad60["n_mad"], mad60["icir_annualized"], "s--", color="#27ae60", label="ICIR")
    ax.axvline(5.0, color="red", ls="--", alpha=0.6, label="n=5")
    ax.set_xlabel("n_mad")
    ax.set_ylabel("HL Sharpe")
    ax2.set_ylabel("ICIR")
    ax.set_title("AlphaIdioSKEW60 MAD threshold scan")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for out in (
        V4 / "figures/v4_mad_n_scan.png",
        PKG / "figures/v4_mad_n_scan.png",
        DELIV / "plots/v4_mad_n_scan.png",
    ):
        fig.savefig(out, dpi=160)
    plt.close(fig)

    # best cumulative among window scan
    plot_row, plot_group, plot_name = best_row, best_group, best_name
    if best_mad_row is not None and best_mad_row["hl_sharpe"] > (best_row["hl_sharpe"] if best_row else -np.inf):
        plot_row, plot_group, plot_name = best_mad_row, best_mad_group, best_mad_name

    if plot_group is not None and plot_row is not None:
        title = (
            f"H-L, Direction: {int(plot_row['direction'])}, "
            f"AnnuRet: {plot_row['hl_ann_return']:.2%},"
            f"Sharpe_Ratio: {plot_row['hl_sharpe']:.2f}, "
            f"MDD: {plot_row['hl_max_drawdown']:.2%}, "
            f"Daily Turnover: {plot_row['avg_turnover']:.2f},\n "
            f"Daily IC: {plot_row['mean_rank_ic']:.4f}, "
            f"Annu ICIR: {plot_row['icir_annualized']:.2f}"
        )
        for out in (
            V4 / "figures/v4_best_cumulative_long_short.png",
            PKG / "figures/v4_best_cumulative_long_short.png",
            DELIV / "plots/v4_best_cumulative_long_short.png",
        ):
            save_cumulative_decile_figure(
                plot_group.cumsum(),
                factor_name=f"{plot_name} size_industry",
                stats_title=title,
                out_path=out,
            )

    base_sharpe = float(baseline_row["hl_sharpe"]) if baseline_row else np.nan
    base_icir = float(baseline_row["icir_annualized"]) if baseline_row else np.nan
    meta = {
        "baseline": BASELINE_LABEL,
        "baseline_si_hl_sharpe": base_sharpe,
        "baseline_si_icir": base_icir,
        "window_best": {
            "factor": best_name,
            "kind": best_row["kind"] if best_row else None,
            "window": int(best_row["window"]) if best_row else None,
            "hl_sharpe": float(best_row["hl_sharpe"]) if best_row else None,
            "icir": float(best_row["icir_annualized"]) if best_row else None,
        },
        "mad_best": {
            "factor": best_mad_name,
            "n_mad": float(best_mad_row["n_mad"]) if best_mad_row else None,
            "hl_sharpe": float(best_mad_row["hl_sharpe"]) if best_mad_row else None,
            "icir": float(best_mad_row["icir_annualized"]) if best_mad_row else None,
        },
        "period": f"{start.date()}_{end.date()}",
        "note": "Sensitivity scan; frozen formula remains IdioSKEW60 unless clearly dominated",
    }
    (V4 / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (PKG / "data/analysis/v4_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    session.close()
    log("=== DONE v4 ===")
    log(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
