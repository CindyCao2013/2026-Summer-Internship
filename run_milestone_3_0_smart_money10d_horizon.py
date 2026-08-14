#!/usr/bin/env python
"""III-A4.3 / Phase 2A.1 — SmartMoney10d holding horizon + turnover diagnosis.

Answers: what is the alpha half-life? Can buffer/rebalance clear 15bp cost?

Uses frozen CSI1000 Phase2A panel. No formula change · No Registry · No Composite.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_horizon.py
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

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_investability import (
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from core.l2_features.smart_money_panel_builder import FORMULA_VERSION
from execution_layer import evaluate_execution
from factor_data_loaders import connect_ddb
from factor_runner import get_universe_mask

REPO = Path(__file__).resolve().parent
PANEL = (
    REPO
    / "research/cache/smart_money/factor_panel"
    / "SmartMoney10d_CSI1000_20230101_20251231.parquet"
)
OUT = REPO / "research/reports/smart_money_v1/phase2a1_horizon"
SCOUT_START = dt.datetime(2023, 1, 1)
SCOUT_END = dt.datetime(2025, 12, 31)
COST_RT = 0.0015
TOP_FRAC = 0.10
DECAY_HS = [1, 3, 5, 10, 20]
HOLD_HS = [1, 5, 10, 20]


def log(msg: str) -> None:
    print(msg, flush=True)


def apply_mask(panel: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m.notna())


def forward_cumret(ret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Sum of daily simple returns over days t+1 .. t+H (investable from signal at t)."""
    parts = [ret.shift(-k) for k in range(1, horizon + 1)]
    out = parts[0].copy()
    for p in parts[1:]:
        out = out.add(p, fill_value=np.nan)
    # require all H legs finite → NaN if any missing
    valid = parts[0].notna()
    for p in parts[1:]:
        valid = valid & p.notna()
    return out.where(valid)


def ic_at_horizon(signal: pd.DataFrame, ret: pd.DataFrame, horizon: int) -> dict:
    """RankIC(Q_t, r_{t+1:t+H}) — no extra shift; fwd already starts at t+1."""
    fwd = forward_cumret(ret, horizon)
    # align: use signal_shift=0 because fwd is already next-day-based
    ic = daily_rank_ic_series(signal, fwd, signal_shift=0)
    ic = ic.dropna()
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    icir = float(icir_from_daily(ic)) if len(ic) >= 20 else float("nan")
    return {
        "horizon": horizon,
        "rank_ic": mean_ic,
        "abs_rank_ic": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
        "icir": icir,
        "abs_icir": abs(icir) if np.isfinite(icir) else np.nan,
        "ic_pos_frac": float((ic > 0).mean()) if len(ic) else np.nan,
        "n_days": int(len(ic)),
    }


def hl_at_horizon(q: pd.DataFrame, ret_u: pd.DataFrame, horizon: int) -> dict:
    """H-L proxies for horizon H (overlapping daily proxy + non-overlap)."""
    fwd = forward_cumret(ret_u, horizon).reindex(index=q.index, columns=q.columns)
    sig = -q
    gross, to = daily_hl_pnl_and_turnover(
        sig, fwd, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=0
    )
    g_proxy = gross / float(horizon)
    to_proxy = to / float(horizon)
    net_proxy = net_pnl_series(g_proxy, to_proxy, COST_RT)
    gp = series_performance(g_proxy.dropna())
    np_ = series_performance(net_proxy.dropna())
    g_nl = gross.iloc[::horizon]
    to_nl = to.iloc[::horizon]
    net_nl = net_pnl_series(g_nl, to_nl, COST_RT).dropna()
    if len(net_nl) >= 5 and net_nl.std() > 0:
        periods = 250.0 / horizon
        nl_sharpe = float(net_nl.mean() / net_nl.std() * np.sqrt(periods))
        nl_ann = float(net_nl.mean() * periods)
    else:
        nl_sharpe, nl_ann = np.nan, np.nan
    return {
        "horizon": horizon,
        "gross_sharpe_proxy": gp["sharpe"],
        "net_sharpe_proxy": np_["sharpe"],
        "daily_turnover_proxy": float(to_proxy.mean()),
        "net_sharpe_nonoverlap": nl_sharpe,
        "net_annu_ret_nonoverlap": nl_ann,
        "n_rebalances": int(len(net_nl)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-exec-grid", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== III-A4.3 Phase 2A.1 SmartMoney horizon + TO diagnosis ===")
    log(f"formula={FORMULA_VERSION} | cost={COST_RT} | no Registry / no formula change")

    if not PANEL.exists():
        raise FileNotFoundError(f"Missing Phase2A panel: {PANEL}")

    q = pd.read_parquet(PANEL)
    q.index = pd.to_datetime(q.index)
    session = connect_ddb()
    try:
        session.run(intraday_lib.ddb_functions)
        mask = get_universe_mask(session, SCOUT_START, SCOUT_END, cfg.UNIVERSE_LIST["CSI1000"])
    finally:
        session.close()
    q = apply_mask(q, mask).dropna(how="all", axis=0).dropna(how="all", axis=1)
    ret = Factor_Dev_Lib.get_Ret_Matrix(SCOUT_START, SCOUT_END + dt.timedelta(days=25), method="c2c")
    ret = ret.reindex(index=q.index.union(ret.index)).reindex(columns=q.columns)
    ret = ret.loc[q.index[0] : q.index[-1] + pd.Timedelta(days=30)]
    # extend ret index for forward windows
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(
        SCOUT_START, SCOUT_END + dt.timedelta(days=40), method="c2c"
    )
    ret_full = ret_full.reindex(columns=q.columns)
    # use union index for forward calc then clip IC to q dates
    all_idx = q.index.union(ret_full.index).sort_values()
    ret_u = ret_full.reindex(index=all_idx, columns=q.columns)
    q_u = q.reindex(index=all_idx, columns=q.columns)

    log(f"Panel {q.shape[0]}d × {q.shape[1]} names (CSI1000 masked)")

    # --- A/C: IC decay ---
    log("IC decay curve ...")
    decay_rows = []
    for h in DECAY_HS:
        row = ic_at_horizon(q_u.loc[q.index], ret_u, h)
        # recompute IC only on scout dates
        fwd = forward_cumret(ret_u, h).reindex(index=q.index, columns=q.columns)
        ic = q.corrwith(fwd, axis=1, method="spearman").dropna()
        row["rank_ic"] = float(ic.mean()) if len(ic) else np.nan
        row["abs_rank_ic"] = abs(row["rank_ic"]) if np.isfinite(row["rank_ic"]) else np.nan
        row["icir"] = float(icir_from_daily(ic)) if len(ic) >= 20 else np.nan
        row["abs_icir"] = abs(row["icir"]) if np.isfinite(row["icir"]) else np.nan
        row["ic_pos_frac"] = float((ic > 0).mean()) if len(ic) else np.nan
        row["n_days"] = int(len(ic))
        decay_rows.append(row)
        log(f"  H={h}: RankIC={row['rank_ic']:.4f} |ICIR|={row['abs_icir']:.2f}")
    decay_df = pd.DataFrame(decay_rows)
    decay_df.to_csv(OUT / "ic_decay.csv", index=False)

    # --- A: holding Sharpe proxies ---
    log("Holding-horizon H-L proxies ...")
    hold_rows = []
    for h in HOLD_HS:
        row = hl_at_horizon(q, ret_u, h)
        ic_match = decay_df.loc[decay_df["horizon"] == h, "rank_ic"]
        row["rank_ic"] = float(ic_match.iloc[0]) if len(ic_match) else np.nan
        hold_rows.append(row)
        log(
            f"  H={h}: gross~{row['gross_sharpe_proxy']:.2f} "
            f"net_proxy~{row['net_sharpe_proxy']:.2f} "
            f"net_nl~{row['net_sharpe_nonoverlap']:.2f} "
            f"TO_proxy={row['daily_turnover_proxy']:.3f}"
        )
    hold_df = pd.DataFrame(hold_rows)
    hold_df.to_csv(OUT / "holding_horizon.csv", index=False)

    # --- B: execution grid (rebalance + buffer) on -Q book ---
    exec_df = pd.DataFrame()
    if not args.skip_exec_grid:
        log("Execution grid (rebalance + buffer) on long-lowQ book ...")
        sig_book = -q  # positive alpha direction for evaluate_execution
        ret_e = ret_u.reindex(index=q.index, columns=q.columns)
        rows = []
        e1 = [
            ("daily", dict(rebalance_freq=1, friday_only=False)),
            ("every_5d", dict(rebalance_freq=5, friday_only=False)),
            ("every_10d", dict(rebalance_freq=10, friday_only=False)),
            ("every_20d", dict(rebalance_freq=20, friday_only=False)),
            ("weekly_friday", dict(rebalance_freq=1, friday_only=True)),
        ]
        for label, kw in e1:
            row = evaluate_execution(
                sig_book,
                ret_e,
                label=f"lowQ|{label}",
                stage="E1_rebalance",
                top_frac=TOP_FRAC,
                weight_method="ew",
                round_trip_cost=COST_RT,
                **kw,
            )
            rows.append(row)
            log(f"  {label}: gross={row['gross_sharpe']:.2f} net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}")

        for blabel, entry, exit_ in [
            ("buffer_5_15", 0.05, 0.15),
            ("buffer_10_20", 0.10, 0.20),
            ("buffer_10_30", 0.10, 0.30),
        ]:
            for rb_name, rb_kw in [
                ("daily", dict(rebalance_freq=1, friday_only=False)),
                ("every_5d", dict(rebalance_freq=5, friday_only=False)),
                ("every_10d", dict(rebalance_freq=10, friday_only=False)),
            ]:
                row = evaluate_execution(
                    sig_book,
                    ret_e,
                    label=f"lowQ|{rb_name}|{blabel}",
                    stage="E2_buffer",
                    entry_frac=entry,
                    exit_frac=exit_,
                    weight_method="ew",
                    round_trip_cost=COST_RT,
                    **rb_kw,
                )
                rows.append(row)
                log(
                    f"  {rb_name}|{blabel}: net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}"
                )

        exec_df = pd.DataFrame(rows)
        exec_df.to_csv(OUT / "execution_grid.csv", index=False)
        ranked = exec_df.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False)
        ranked.to_csv(OUT / "execution_ranked.csv", index=False)
        best = ranked.iloc[0].to_dict() if len(ranked) else {}
        log(f"Best net: {best.get('label')} net={best.get('net_sharpe')} TO={best.get('daily_turnover')}")
    else:
        best = {}

    # --- Classification ---
    # Half-life proxy: smallest H where |IC| drops below half of |IC_H1|
    ic1 = abs(float(decay_df.loc[decay_df["horizon"] == 1, "rank_ic"].iloc[0]))
    half_life = None
    for h in DECAY_HS:
        aic = abs(float(decay_df.loc[decay_df["horizon"] == h, "rank_ic"].iloc[0]))
        if aic <= 0.5 * ic1:
            half_life = h
            break

    best_net = float(best["net_sharpe"]) if best.get("net_sharpe") is not None else np.nan
    investable = bool(np.isfinite(best_net) and best_net > 0.5)

    if half_life is None and ic1 > 0:
        # still strong at H=20
        horizon_class = "medium_or_longer_persistent"
    elif half_life is not None and half_life <= 3:
        horizon_class = "ultra_short_daily"
    elif half_life is not None and half_life <= 10:
        horizon_class = "short_weekly"
    else:
        horizon_class = "medium_horizon"

    status = "research_candidate"
    if investable:
        next_step = "optional testing pack with documented execution recipe"
    else:
        next_step = "keep research_candidate; do not Registry; optional park vs APM/SUE"

    report = {
        "phase": "2A.1_horizon_turnover",
        "formula_version": FORMULA_VERSION,
        "universe": "CSI1000",
        "period": f"{SCOUT_START.date()}_{SCOUT_END.date()}",
        "cost_rt": COST_RT,
        "book": "long_lowQ_short_highQ",
        "ic_decay": decay_rows if False else decay_df.to_dict(orient="records"),
        "holding_horizon": hold_df.to_dict(orient="records"),
        "best_execution": {
            "label": best.get("label"),
            "net_sharpe": best.get("net_sharpe"),
            "gross_sharpe": best.get("gross_sharpe"),
            "daily_turnover": best.get("daily_turnover"),
            "stage": best.get("stage"),
        }
        if best
        else {},
        "half_life_proxy_h": half_life,
        "horizon_class": horizon_class,
        "investable_net_gt_0p5": investable,
        "recommended_status": status,
        "next": next_step,
        "interpretation": (
            "Valid microstructure alpha; diagnose horizon/TO before pack. "
            "Do not modify formula; do not Composite/Registry."
        ),
        "forbidden": ["formula_change", "Registry", "Composite", "Active_*"],
    }
    (OUT / "horizon_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    log(f"Wrote {OUT / 'horizon_report.json'}")
    log(f"horizon_class={horizon_class} half_life~{half_life} investable={investable}")
    log(f"status → {status}")


if __name__ == "__main__":
    main()
