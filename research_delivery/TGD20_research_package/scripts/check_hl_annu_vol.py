#!/usr/bin/env python
"""Check H-L annualized volatility: MA20_raw_ALL vs MA20_ind_cap_ALL.

Computes vol the same way Sharpe's denominator does:
    hl_annu_vol = std(hl_disp) * sqrt(250)

Also reports implied vol = hl_annu_ret / hl_sharpe for cross-check.

Usage:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /opt/conda/anaconda3/envs/base_93/bin/python \\
    research_delivery/TGD20_research_package/scripts/check_hl_annu_vol.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import Factor_Dev_Lib as FDL
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from run_mentor_single_factor_protocol import (
    CONFIRM_END,
    CONFIRM_START,
    FEE,
    N_GROUPS,
    SIGNAL_SHIFT,
    WIDE_PATH,
    build_neut_ladder,
)

OUT = (
    ROOT
    / "research_delivery"
    / "TGD20_research_package"
    / "data"
    / "analysis"
    / "mentor_protocol"
    / "check_hl_annu_vol.csv"
)
LABELS = ("MA20_raw_ALL", "MA20_ind_cap_ALL")
N_ANN = 250


def hl_annu_vol(hl_disp: pd.Series, n: int = N_ANN) -> float:
    """Annualized vol matching FDL.calSharpe denominator (sample std, ddof=1)."""
    s = pd.Series(hl_disp).dropna()
    if s.empty or s.std() == 0 or pd.isna(s.std()):
        return float("nan")
    return float(s.std() * (n ** 0.5))


def eval_hl_vol(signal: pd.DataFrame, ret: pd.DataFrame, label: str) -> dict:
    sig = align_signal(signal.reindex_like(ret), SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    valid = sig.notna().any(axis=1) & r.notna().any(axis=1)
    sig, r = sig.loc[valid], r.loc[valid]

    _, pnl, _ = FDL.groupTest(sig, r, n=N_GROUPS, fee=FEE, info="silent")
    hl = pnl["H-L"]
    direction = 1 if hl.mean() > 0 else -1
    hl_disp = hl * direction

    annu = FDL.calAnnuRet(hl_disp)
    sharpe = FDL.calSharpe(hl_disp)
    vol = hl_annu_vol(hl_disp)
    implied = float(annu / sharpe) if sharpe and np.isfinite(sharpe) and sharpe != 0 else np.nan

    return {
        "label": label,
        "n_days": int(hl_disp.dropna().shape[0]),
        "direction": direction,
        "hl_annu_ret": annu,
        "hl_sharpe": sharpe,
        "hl_annu_vol": vol,
        "implied_vol_annu_over_sharpe": implied,
        "vol_minus_implied": vol - implied if np.isfinite(vol) and np.isfinite(implied) else np.nan,
        "daily_std": float(hl_disp.dropna().std()),
    }


def main() -> None:
    start, end = CONFIRM_START, CONFIRM_END
    print(f"Window: {start.date()} -> {end.date()}")
    print(f"Loading TGD20 wide: {WIDE_PATH}")

    wide = pd.read_parquet(WIDE_PATH)
    wide.index = pd.to_datetime(wide.index)

    enriched, _ = load_eod_enriched_tables(start - dt.timedelta(days=40), end)
    industry = load_citics_industry_panel(start - dt.timedelta(days=40), end)
    ret = FDL.get_Ret_Matrix(start, end, method="c2c")
    ret.index = pd.to_datetime(ret.index)
    not_limit = FDL.get_EOD_Not_Limit(start, end)
    not_st = FDL.get_EOD_Not_ST(start, end)

    raw = wide.loc[start:end].reindex(columns=ret.columns)
    r = ret.reindex_like(raw)
    raw_m, ret_m = FDL.apply_tradability_mask(raw, r, not_limit=not_limit, not_st=not_st)
    ladder = build_neut_ladder(raw_m, industry, enriched.float_mktcap)

    mode_map = {"MA20_raw_ALL": "raw", "MA20_ind_cap_ALL": "ind_cap"}
    rows = []
    for label in LABELS:
        mode = mode_map[label]
        print(f"Evaluating {label} ({mode}) ...")
        rows.append(eval_hl_vol(ladder[mode], ret_m, label))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n=== H-L annualized volatility check ===")
    print(
        out[
            [
                "label",
                "hl_annu_ret",
                "hl_sharpe",
                "hl_annu_vol",
                "implied_vol_annu_over_sharpe",
                "vol_minus_implied",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    raw_row = out.loc[out["label"] == "MA20_raw_ALL"].iloc[0]
    si_row = out.loc[out["label"] == "MA20_ind_cap_ALL"].iloc[0]
    print(
        f"\nVol drop (raw -> ind_cap): "
        f"{raw_row['hl_annu_vol']:.2%} -> {si_row['hl_annu_vol']:.2%} "
        f"(Δ={si_row['hl_annu_vol'] - raw_row['hl_annu_vol']:+.2%})"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
