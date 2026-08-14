#!/usr/bin/env python3
"""Recompute Group10 / long-book excess Sharpe from group_cum_pnl artifacts.

See research_delivery/METRICS_G10_EXCESS.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import Factor_Dev_Lib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_delivery" / "g10_excess_sharpe_proxy.csv"

FACTOR_CUM = {
    "TGD20": ROOT / "research/reports/tgd_v1/portfolio/group_cum_pnl.csv",
    "D1_LiquidityQuality60d": ROOT
    / "research/reports/d1_liquidity_density_v1/confirmation_1455d/low_vol_liquidity_quality_60d/report/group_cum_pnl.csv",
    "FlowDensity20": ROOT
    / "research/reports/l2_flow_density_v1/protocol_charts_1d7/FlowDensity20/report/group_cum_pnl.csv",
    "IdealReversal": ROOT
    / "research/reports/factor_cutting_v1/ideal_reversal/portfolio/group_cum_pnl.csv",
    "IdealAmplitude": ROOT
    / "research/reports/factor_cutting_v1/ideal_amplitude/portfolio/group_cum_pnl.csv",
    "AmihudShockReversal5d": ROOT
    / "research/reports/d1_liquidity_density_v1/confirmation_1455d/amihud_shock_reversal_5d/report/group_cum_pnl.csv",
    "APM_SessionResidual": None,
}


def load_daily_groups(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    cols = []
    for i in range(1, 11):
        if i in df.columns:
            cols.append(i)
        elif str(i) in df.columns:
            cols.append(str(i))
        else:
            raise KeyError(f"missing group {i} in {path}")
    g = df[cols].astype(float)
    g.columns = list(range(1, 11))
    daily = g.diff()
    return daily.dropna(how="all")


def excess_stats(daily_g: pd.DataFrame, group: int) -> dict:
    ew = daily_g.loc[:, 1:10].mean(axis=1)
    x = (daily_g[group] - ew).dropna()
    if len(x) < 50:
        return {"n_days": len(x), "excess_sharpe": np.nan, "excess_annu_ret": np.nan, "excess_mdd": np.nan, "win_rate": np.nan}
    mdd, _ = Factor_Dev_Lib.calMDD(x)
    return {
        "n_days": int(len(x)),
        "excess_sharpe": float(Factor_Dev_Lib.calSharpe(x)),
        "excess_annu_ret": float(Factor_Dev_Lib.calAnnuRet(x)),
        "excess_mdd": float(mdd),
        "win_rate": float((x > 0).mean()),
    }


def main() -> None:
    rows = []
    for fid, path in FACTOR_CUM.items():
        if path is None or not path.exists():
            rows.append({"factor_id": fid, "status": "MISSING_group_cum_pnl"})
            continue
        daily = load_daily_groups(path)
        daily = daily.loc[~(daily.abs().sum(axis=1) < 1e-12)]
        g10 = excess_stats(daily, 10)
        hl = (daily[10] - daily[1]).dropna()
        direction = 1 if hl.mean() >= 0 else -1
        long_g = 10 if direction == 1 else 1
        alpha = excess_stats(daily, long_g)
        hl_sh = float(Factor_Dev_Lib.calSharpe(hl * direction)) if len(hl) >= 50 else np.nan
        rows.append(
            {
                "factor_id": fid,
                "status": "ok",
                "artifact": str(path.relative_to(ROOT)),
                "g10_excess_sharpe": g10["excess_sharpe"],
                "g10_excess_annu_ret": g10["excess_annu_ret"],
                "g10_excess_mdd": g10["excess_mdd"],
                "g10_win_rate": g10["win_rate"],
                "n_days": g10["n_days"],
                "alpha_long_group": long_g,
                "alpha_long_excess_sharpe": alpha["excess_sharpe"],
                "hl_signed_sharpe": hl_sh,
                "benchmark": "decile_EW",
            }
        )
        print(f"{fid}: long_book_xs={alpha['excess_sharpe']:.3f} g10_xs={g10['excess_sharpe']:.3f}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
