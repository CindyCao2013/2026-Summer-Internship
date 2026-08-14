#!/usr/bin/env python
"""Mentor single-factor protocol for TGD20 (lean tables).

Convention: G1=short, G10=long, H-L=G10-G1.
Headline: G10 Excess Sharpe vs exact universe EW.
Gates: Excess > 3.5 AND Excess > all G1..G10 / H-L Sharpes.

Usage:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  /opt/conda/anaconda3/envs/base_93/bin/python run_mentor_single_factor_protocol.py

  # MA + neutralization + MAD grid only
  ... run_mentor_single_factor_protocol.py --grid-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib as FDL
from alpha_research_report import save_cumulative_decile_figure
from core.l2_features.tgd import smooth_tgd, tgd20_to_wide
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel, panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_delivery" / "TGD20_research_package" / "data" / "analysis" / "mentor_protocol"
FIG = ROOT / "research_delivery" / "TGD20_research_package" / "figures" / "mentor_protocol"
PANEL_CACHE = ROOT / "research" / "cache" / "tgd_panels"
LONG_PATH = PANEL_CACHE / "TGD20_long_20200101_20251231_w20.parquet"
WIDE_PATH = PANEL_CACHE / "TGD20_20200101_20251231_w20.parquet"

CONFIRM_START = dt.datetime(2022, 1, 28)
CONFIRM_END = dt.datetime(2025, 12, 31)
SIGNAL_SHIFT = 1
N_GROUPS = 10
FEE = 0.0
MA_WINDOWS = (10, 20, 30, 60)


def log(msg: str) -> None:
    print(msg, flush=True)


def monotonicity_spearman(decile_means: pd.Series) -> float:
    try:
        order = pd.to_numeric(decile_means.index, errors="coerce").to_numpy(dtype=float)
    except Exception:
        order = np.arange(1, len(decile_means) + 1, dtype=float)
    y = decile_means.to_numpy(dtype=float)
    mask = np.isfinite(order) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(pd.Series(order[mask]).corr(pd.Series(y[mask]), method="spearman"))


def build_neut_ladder(raw: pd.DataFrame, industry: pd.DataFrame, float_mkt: pd.DataFrame) -> dict:
    log_size = np.log(float_mkt.replace(0, np.nan)).reindex_like(raw)
    ind = industry.reindex_like(raw)
    return {
        "raw": cs_zscore(raw),
        "cap": cs_zscore(panel_cross_sectional_residual(raw, [log_size])),
        "ind": cs_zscore(panel_industry_demean(raw, ind)),
        "ind_cap": cs_zscore(neutralize_size_industry(raw, ind, float_mkt)),
    }


def apply_mad_panel(panel: pd.DataFrame, tanh: bool = True) -> pd.DataFrame:
    return FDL.mad(panel, threshold=3, tanh=tanh)


def eval_one(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    label: str,
    universe: str = "ALL",
    horizon: int = 1,
    save_figs: bool = False,
) -> dict:
    """signal is unshifted factor; ret is already the forward return panel for this horizon."""
    sig = align_signal(signal.reindex_like(ret), SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    # Drop empty days / names
    valid_days = sig.notna().any(axis=1) & r.notna().any(axis=1)
    sig = sig.loc[valid_days]
    r = r.loc[valid_days]

    _, pnl, to = FDL.groupTest(sig, r, n=N_GROUPS, fee=FEE, info="silent")
    group_sharpes = FDL.summarize_group_sharpes(pnl)
    # Direction for H-L display (same as groupTest title): flip if H-L mean < 0
    hl = pnl["H-L"]
    direction = 1 if hl.mean() > 0 else -1
    hl_disp = hl * direction
    hl_sharpe = FDL.calSharpe(hl_disp)
    hl_annu = FDL.calAnnuRet(hl_disp)
    hl_mdd, _ = FDL.calMDD(hl_disp)
    avg_to = float(to["H-L"].mean())
    implied_fee = FDL.implied_annu_fee(avg_to)

    rank_ic_daily = sig.corrwith(r, axis=1, method="spearman")
    rank_ic = float(np.nanmean(rank_ic_daily))
    rank_ic_std = float(np.nanstd(rank_ic_daily))
    icir = rank_ic / rank_ic_std * (250 ** 0.5) if rank_ic_std > 0 else np.nan
    pos_ratio = float((rank_ic_daily > 0).mean())

    cols = [c for c in pnl.columns if c != "H-L"]
    decile_means = pnl[cols].mean()
    mono = monotonicity_spearman(decile_means)

    # G10 excess: pass unshifted signal; helper applies signal_shift
    excess = FDL.g10_excess_vs_universe_ew(signal.reindex_like(ret), ret, signal_shift=SIGNAL_SHIFT)
    gates = FDL.check_excess_gates(excess["excess_sharpe"], group_sharpes)

    row = {
        "label": label,
        "universe": universe,
        "horizon": horizon,
        "rank_ic": rank_ic,
        "icir": icir,
        "ic_positive_ratio": pos_ratio,
        "mono": mono,
        "hl_annu_ret": hl_annu,
        "hl_sharpe": hl_sharpe,
        "hl_mdd": hl_mdd,
        "hl_turnover": avg_to,
        "implied_annu_fee": implied_fee,
        "direction": direction,
        "g10_excess_annu": excess["excess_annu_ret"],
        "g10_excess_sharpe": excess["excess_sharpe"],
        "g10_excess_mdd": excess["excess_max_drawdown"],
        "max_group_or_hl_sharpe": gates["max_group_or_hl_sharpe"],
        "pass_gt_gate": gates["pass_gt_gate"],
        "pass_gt_all_group_sharpes": gates["pass_gt_all_group_sharpes"],
        "pass_all": gates["pass_all"],
        "n_days": excess["n_days"],
        "selected_count_mean": excess["selected_count_mean"],
        "universe_count_mean": excess["universe_count_mean"],
    }
    for g, sh in group_sharpes.items():
        row[f"sharpe_g{g}"] = sh

    if save_figs:
        FIG.mkdir(parents=True, exist_ok=True)
        fig_dec = FIG / f"{label}_decile_return.png"
        fig_cum = FIG / f"{label}_cumulative_long_short.png"
        title = FDL.format_group_test_stats_title(
            direction=direction,
            annu_ret=hl_annu,
            sharpe=hl_sharpe,
            mdd=hl_mdd,
            avg_turnover=avg_to,
            rank_ic=rank_ic,
            icir=icir,
            implied_fee=implied_fee,
        )
        ax = decile_means.plot(kind="bar", title=f"{label} decile means (G1=short, G10=long)")
        ax.set_xlabel(title, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dec, dpi=150)
        plt.close()
        save_cumulative_decile_figure(
            pnl.cumsum(),
            factor_name=label,
            stats_title=title,
            out_path=fig_cum,
        )
        row["fig_decile"] = str(fig_dec.relative_to(ROOT))
        row["fig_cum"] = str(fig_cum.relative_to(ROOT))

        decile_means.to_csv(OUT / f"{label}_decile_means.csv")
        pnl.to_csv(OUT / f"{label}_group_pnl.csv")
        group_sharpes.to_csv(OUT / f"{label}_group_sharpes.csv")

    return row


def load_tgd_wide(window: int = 20) -> pd.DataFrame:
    if window == 20 and WIDE_PATH.exists():
        wide = pd.read_parquet(WIDE_PATH)
        wide.index = pd.to_datetime(wide.index)
        return wide
    if not LONG_PATH.exists():
        raise FileNotFoundError(f"Missing TGD long cache: {LONG_PATH}")
    long = pd.read_parquet(LONG_PATH)
    # Re-smooth innovation with requested MA window
    innov = long[["date", "symbol", "tgd_eps"]].copy()
    smoothed = smooth_tgd(innov, window=window, eps_col="tgd_eps", out_col="TGD")
    wide = tgd20_to_wide(smoothed, value_col="TGD")
    return wide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--skip-universes", action="store_true")
    parser.add_argument("--skip-decay", action="store_true")
    parser.add_argument("--start", default=CONFIRM_START.strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=CONFIRM_END.strftime("%Y-%m-%d"))
    args = parser.parse_args()

    start = dt.datetime.strptime(args.start, "%Y-%m-%d")
    end = dt.datetime.strptime(args.end, "%Y-%m-%d")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    log("Loading EOD / industry / ret ...")
    enriched, _session = load_eod_enriched_tables(start - dt.timedelta(days=40), end)
    float_mkt = enriched.float_mktcap
    industry = load_citics_industry_panel(start - dt.timedelta(days=40), end)
    ret = FDL.get_Ret_Matrix(start, end, method="c2c")
    ret.index = pd.to_datetime(ret.index)

    log("Tradability masks ...")
    not_limit = FDL.get_EOD_Not_Limit(start, end)
    not_st = FDL.get_EOD_Not_ST(start, end)

    rows = []

    def prepare(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        p = panel.loc[start:end].reindex(columns=ret.columns)
        r = ret.reindex_like(p)
        p2, r2 = FDL.apply_tradability_mask(p, r, not_limit=not_limit, not_st=not_st)
        return p2, r2

    # ---- Neutralization ladder on frozen MA20 ----
    if not args.grid_only:
        log("MA20 neutralization ladder ...")
        raw = load_tgd_wide(20)
        raw_c, ret_c = prepare(raw)
        ladder = build_neut_ladder(raw_c, industry, float_mkt)
        for mode, panel in ladder.items():
            log(f"  eval ALL / {mode}")
            rows.append(
                eval_one(
                    panel,
                    ret_c,
                    label=f"MA20_{mode}_ALL",
                    universe="ALL",
                    horizon=1,
                    save_figs=(mode == "ind_cap"),
                )
            )

        headline = ladder["ind_cap"]

        if not args.skip_universes:
            log("Universe masks ...")
            for name, code in FDL.INDEX_CODES.items():
                mask = FDL.get_index_member_mask(code, start, end)
                sig_u, ret_u = FDL.restrict_to_index(headline, ret_c, mask)
                log(f"  eval {name}")
                rows.append(
                    eval_one(
                        sig_u,
                        ret_u,
                        label=f"MA20_ind_cap_{name}",
                        universe=name,
                        horizon=1,
                        save_figs=False,
                    )
                )

        if not args.skip_decay:
            log("Decay horizons ...")
            fwd = FDL.calc_forward_returns(ret_c, periods=(1, 5, 10, 20))
            for h, r_h in fwd.items():
                # Align to confirmation window
                r_h = r_h.loc[start:end].reindex_like(headline)
                log(f"  eval T+{h}")
                rows.append(
                    eval_one(
                        headline,
                        r_h,
                        label=f"MA20_ind_cap_T{h}",
                        universe="ALL",
                        horizon=h,
                        save_figs=False,
                    )
                )

    # ---- Optimization grid: MA × neut × MAD ----
    log("Optimization grid (MA / neut / MAD) ...")
    grid_rows = []
    for window in MA_WINDOWS:
        log(f"  load MA{window}")
        wide = load_tgd_wide(window)
        raw_c, ret_c = prepare(wide)
        for mad_mode in ("none", "mad_tanh", "mad_clip"):
            if mad_mode == "none":
                base = raw_c
            elif mad_mode == "mad_tanh":
                base = apply_mad_panel(raw_c, tanh=True)
            else:
                base = apply_mad_panel(raw_c, tanh=False)
            ladder = build_neut_ladder(base, industry, float_mkt)
            for mode, panel in ladder.items():
                # Focus grid on ind_cap + raw for speed; still cover all nt_types on MA20
                if window != 20 and mode not in ("raw", "ind_cap"):
                    continue
                label = f"MA{window}_{mad_mode}_{mode}"
                log(f"    {label}")
                row = eval_one(panel, ret_c, label=label, universe="ALL", horizon=1)
                grid_rows.append(row)
                rows.append(row)

    summary = pd.DataFrame(rows)
    grid = pd.DataFrame(grid_rows)
    summary.to_csv(OUT / "mentor_protocol_summary.csv", index=False)
    grid.to_csv(OUT / "mentor_optimization_grid.csv", index=False)

    # Best by G10 excess sharpe
    if len(grid):
        best = grid.sort_values("g10_excess_sharpe", ascending=False).iloc[0]
        best_path = OUT / "mentor_best_variant.json"
        best_path.write_text(json.dumps(best.to_dict(), indent=2, default=str), encoding="utf-8")
        log(
            f"BEST: {best['label']} ExcessSharpe={best['g10_excess_sharpe']:.3f} "
            f"HL={best['hl_sharpe']:.3f} pass_all={best['pass_all']}"
        )

    meta = {
        "confirm_start": start.isoformat(),
        "confirm_end": end.isoformat(),
        "signal_shift": SIGNAL_SHIFT,
        "convention": "G1=short, G10=long, H-L=G10-G1",
        "excess_gate": FDL.EXCESS_SHARPE_GATE,
        "n_rows": len(summary),
        "out_dir": str(OUT),
    }
    (OUT / "mentor_protocol_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
