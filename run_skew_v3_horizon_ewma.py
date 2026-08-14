#!/usr/bin/env python3
"""SKEW single-factor optimization v3: horizon + EWMA (no multi-factor, no minute).

Direction 1: forward horizons T+1/2/3/5/10 on AlphaIdioSKEW60_MAD (SI)
Direction 2: EWMA-weighted IdioSKEW60 (half-life 10/15/20) ± MAD

Writes research/reports/factors/SKEW/v3/ and package analysis/figures.
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
from core.factors.skew.ewma_skew import build_forward_return, build_idio_skew_ewma
from core.factors.skew.skew_v2 import mad_winsorize_cs
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from factor_runner import prepare_signal
from industry_neutral import load_citics_industry_panel
from research.extreme_return_study.src.data_loader import load_csi300_index_return

ANNUAL = 250
PACK = ROOT / "research/reports/factors/SKEW"
V3 = PACK / "v3"
CACHE = ROOT / "research/cache/skew_panels"
PKG = ROOT / "research_delivery/SKEW_research_package"
DELIV = ROOT / "research_delivery/factors/SKEW"
HEADLINE = "AlphaIdioSKEW60_MAD"
HORIZONS = (1, 2, 3, 5, 10)
HALFLIVES = (10, 15, 20)


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for p in (
        V3,
        V3 / "figures",
        V3 / "tables",
        PKG / "data/analysis",
        PKG / "figures",
        DELIV / "plots",
        CACHE,
    ):
        p.mkdir(parents=True, exist_ok=True)


def summarize_ic(daily: pd.Series, *, periods_per_year: float) -> dict:
    s = daily.dropna()
    mean = float(s.mean()) if len(s) else np.nan
    std = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return {
        "n_days": int(len(s)),
        "mean_rank_ic": mean,
        "ic_std": std,
        "icir_annualized": mean / std * math.sqrt(periods_per_year) if std and std > 0 else np.nan,
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
    horizon: int = 1,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """size+industry groupTest; Sharpe/ICIR annualized with 250/horizon."""
    ppy = ANNUAL / float(horizon)
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
    )
    signal = signal.dropna(how="all")
    # Align return: drop last horizon-1 days (incomplete forward window)
    ret_use = ret.loc[start:end].reindex(index=signal.index, columns=signal.columns)
    if horizon > 1:
        # last h-1 rows of forward return are NaN by construction; drop all-NaN
        ret_use = ret_use.dropna(how="all")
        common = signal.index.intersection(ret_use.index)
        signal = signal.loc[common]
        ret_use = ret_use.loc[common]

    if signal.empty or signal.shape[0] == 0:
        raise RuntimeError(f"{name}: empty signal")

    _, group_pnl, group_to = Factor_Dev_Lib.groupTest(signal, ret_use, n=10, info="silent")
    plt.close("all")

    daily_pnl = group_pnl["H-L"]
    direction = 1 if daily_pnl.mean() > 0 else -1
    pnl_adj = daily_pnl * direction
    ic = signal.corrwith(ret_use, axis=1, method="spearman")
    row = summarize_ic(ic, periods_per_year=ppy)
    row.update(
        {
            "factor": name,
            "horizon": int(horizon),
            "mode": "size_industry",
            "periods_per_year": ppy,
            "hl_ann_return": float(pnl_adj.mean() * ppy) if len(pnl_adj) else np.nan,
            "hl_sharpe": (
                float(pnl_adj.mean() / pnl_adj.std(ddof=1) * math.sqrt(ppy))
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


def main() -> None:
    ensure_dirs()
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    log(f"SKEW v3 load EOD {preheat.date()} → {end.date()}")

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(max(start, dt.datetime(2020, 1, 2)), end)
    ret_1d = enriched.close / enriched.close.shift(1) - 1.0
    market = load_csi300_index_return(preheat, end, session=session)

    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    mad_path = CACHE / f"{HEADLINE}_{tag}.parquet"
    if mad_path.exists():
        mad_panel = pd.read_parquet(mad_path)
        log(f"loaded cache {mad_path.name}")
    else:
        idio = pd.read_parquet(CACHE / f"IdioSKEW60_{tag}.parquet")
        mad_panel = -mad_winsorize_cs(idio.reindex_like(ret_1d))
        mad_panel.loc[start:end].to_parquet(mad_path)

    df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start, end)
    df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(start, end)
    df_trade_status = Factor_Dev_Lib.get_TradeStatus(start, end)
    ret_daily = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c", base_index=None)

    # ---------- Direction 1: horizons ----------
    log("=== Direction 1: prediction horizons ===")
    horizon_rows: List[dict] = []
    best_h_row = None
    best_h_group = None
    for h in HORIZONS:
        log(f"  horizon H={h}")
        ret_h = build_forward_return(ret_daily, h)
        row, group = eval_si(
            HEADLINE,
            mad_panel,
            ret_h,
            industry=industry,
            float_mktcap=enriched.float_mktcap,
            session=session,
            df_not_limit=df_not_limit,
            df_not_st=df_not_st,
            df_trade_status=df_trade_status,
            start=start,
            end=end,
            horizon=h,
        )
        horizon_rows.append(row)
        if best_h_row is None or row["hl_sharpe"] > best_h_row["hl_sharpe"]:
            best_h_row = row
            best_h_group = group

    horizon_df = pd.DataFrame(horizon_rows).sort_values("horizon")
    horizon_df.to_csv(V3 / "tables/horizon_sweep.csv", index=False)
    horizon_df.to_csv(PKG / "data/analysis/v3_horizon_sweep.csv", index=False)
    log(horizon_df[["horizon", "mean_rank_ic", "icir_annualized", "hl_sharpe", "hl_ann_return", "avg_turnover"]].to_string(index=False))

    # ---------- Direction 2: EWMA ----------
    log("=== Direction 2: EWMA-weighted IdioSKEW60 ===")
    ewma_panels = build_idio_skew_ewma(
        ret_1d,
        market,
        window=60,
        half_lives=HALFLIVES,
        as_alpha=True,
        n_jobs=8,
    )
    # also MAD on each
    panels: Dict[str, pd.DataFrame] = {}
    for name, panel in ewma_panels.items():
        path = CACHE / f"{name}_{tag}.parquet"
        panel.loc[start:end].to_parquet(path)
        panels[name] = panel
        mad_name = f"{name}_MAD"
        # panel is already Alpha = -skew; MAD on raw = -MAD(-alpha) = -MAD(skew)
        raw_skew = -panel
        panels[mad_name] = -mad_winsorize_cs(raw_skew)
        panels[mad_name].loc[start:end].to_parquet(CACHE / f"{mad_name}_{tag}.parquet")
        log(f"  cached {name} / {mad_name}")

    # baseline MAD H=1 for comparison
    ewma_rows: List[dict] = []
    base_row, base_group = eval_si(
        HEADLINE,
        mad_panel,
        ret_daily,
        industry=industry,
        float_mktcap=enriched.float_mktcap,
        session=session,
        df_not_limit=df_not_limit,
        df_not_st=df_not_st,
        df_trade_status=df_trade_status,
        start=start,
        end=end,
        horizon=1,
    )
    base_row["variant_family"] = "equal_weight_mad"
    ewma_rows.append(base_row)

    best_e_row = base_row
    best_e_group = base_group
    best_e_name = HEADLINE

    for name, panel in panels.items():
        log(f"  eval {name}")
        row, group = eval_si(
            name,
            panel,
            ret_daily,
            industry=industry,
            float_mktcap=enriched.float_mktcap,
            session=session,
            df_not_limit=df_not_limit,
            df_not_st=df_not_st,
            df_trade_status=df_trade_status,
            start=start,
            end=end,
            horizon=1,
        )
        row["variant_family"] = "ewma"
        ewma_rows.append(row)
        if row["hl_sharpe"] > best_e_row["hl_sharpe"]:
            best_e_row = row
            best_e_group = group
            best_e_name = name

    ewma_df = pd.DataFrame(ewma_rows).sort_values("hl_sharpe", ascending=False)
    ewma_df.to_csv(V3 / "tables/ewma_sweep.csv", index=False)
    ewma_df.to_csv(PKG / "data/analysis/v3_ewma_sweep.csv", index=False)
    log(ewma_df[["factor", "mean_rank_ic", "icir_annualized", "hl_sharpe", "hl_ann_return", "avg_turnover"]].to_string(index=False))

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(horizon_df["horizon"], horizon_df["hl_sharpe"], "o-", color="#2f80ed", label="HL Sharpe")
    axes[0].plot(horizon_df["horizon"], horizon_df["icir_annualized"], "s--", color="#27ae60", label="ICIR")
    axes[0].set_xlabel("Forward horizon H (days)")
    axes[0].set_title(f"{HEADLINE}: horizon sweep (ann. with 250/H)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].barh(ewma_df["factor"], ewma_df["hl_sharpe"], color="#2f80ed")
    axes[1].axvline(base_row["hl_sharpe"], color="red", ls="--", label=f"MAD baseline {base_row['hl_sharpe']:.2f}")
    axes[1].set_xlabel("HL Sharpe (H=1, SI)")
    axes[1].set_title("EWMA IdioSKEW60 vs MAD baseline")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for out in (
        V3 / "figures/v3_horizon_ewma_comparison.png",
        PKG / "figures/v3_horizon_ewma_comparison.png",
        DELIV / "plots/v3_horizon_ewma_comparison.png",
    ):
        fig.savefig(out, dpi=160)
    plt.close(fig)

    # best cumulative among EWMA/MAD (H=1 investability)
    if best_e_group is not None:
        title = (
            f"H-L, Direction: {int(best_e_row['direction'])}, "
            f"AnnuRet: {best_e_row['hl_ann_return']:.2%},"
            f"Sharpe_Ratio: {best_e_row['hl_sharpe']:.2f}, "
            f"MDD: {best_e_row['hl_max_drawdown']:.2%}, "
            f"Daily Turnover: {best_e_row['avg_turnover']:.2f},\n "
            f"Daily IC: {best_e_row['mean_rank_ic']:.4f}, "
            f"Annu ICIR: {best_e_row['icir_annualized']:.2f}"
        )
        for out in (
            V3 / "figures/v3_best_cumulative_long_short.png",
            PKG / "figures/v3_best_cumulative_long_short.png",
            DELIV / "plots/v3_best_cumulative_long_short.png",
        ):
            save_cumulative_decile_figure(
                best_e_group.cumsum(),
                factor_name=f"{best_e_name} size_industry",
                stats_title=title,
                out_path=out,
            )

    meta = {
        "headline_input": HEADLINE,
        "horizon_best": {
            "horizon": int(best_h_row["horizon"]) if best_h_row else None,
            "hl_sharpe": float(best_h_row["hl_sharpe"]) if best_h_row else None,
            "icir": float(best_h_row["icir_annualized"]) if best_h_row else None,
            "note": "Sharpe/ICIR annualized with periods_per_year=250/H",
        },
        "horizon_h1": {
            "hl_sharpe": float(horizon_df.loc[horizon_df["horizon"] == 1, "hl_sharpe"].iloc[0]),
            "icir": float(horizon_df.loc[horizon_df["horizon"] == 1, "icir_annualized"].iloc[0]),
        },
        "ewma_best": {
            "factor": best_e_name,
            "hl_sharpe": float(best_e_row["hl_sharpe"]),
            "icir": float(best_e_row["icir_annualized"]),
        },
        "mad_baseline_h1": {
            "hl_sharpe": float(base_row["hl_sharpe"]),
            "icir": float(base_row["icir_annualized"]),
        },
        "period": f"{start.date()}_{end.date()}",
    }
    (V3 / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (PKG / "data/analysis/v3_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    session.close()
    log("=== DONE v3 ===")
    log(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
