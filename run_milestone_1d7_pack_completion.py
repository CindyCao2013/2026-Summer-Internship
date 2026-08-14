#!/usr/bin/env python
"""Milestone 1D.7 — Pre-Registry pack gap completion (evaluation layer only).

1) FlowDensity20: protocol charts from frozen net_active_flow_mktcap_20d
2) D1_LiquidityQuality60d: execution_layer grid → execution_summary.csv

Does NOT change formulas, mechanism classification, or create Registry.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_1d7_pack_completion.py
  OMP_NUM_THREADS=1 python run_milestone_1d7_pack_completion.py --skip-flow
  OMP_NUM_THREADS=1 python run_milestone_1d7_pack_completion.py --skip-d1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import DEFAULT_ROUND_TRIP_COST
from alpha_research_report import build_factor_report, publish_factor_report
from execution_layer import evaluate_execution
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache

REPO = Path(__file__).resolve().parent
FLOW_PACK = REPO / "research/reports/factors/FlowDensity20"
FLOW_CHART_SRC = REPO / "research/reports/l2_flow_density_v1/protocol_charts_1d7"
D1_PACK = REPO / "research/reports/factors/D1_LiquidityQuality60d"
D1_EXEC_OUT = REPO / "research/reports/d1_liquidity_density_v1/execution"
D1_FACTOR = "low_vol_liquidity_quality_60d"
FLOW_COL = "net_active_flow_mktcap_20d"
TOP_FRAC = 0.10
SIGNAL_SHIFT = 1


def log(msg: str) -> None:
    print(msg, flush=True)


def install_flow_charts(fig_dir: Path) -> None:
    mapping = {
        "ic_timeseries.png": "ic_curve.png",
        "quantile_return.png": "decile_return.png",
        "cumulative_long_short.png": "cumulative_long_short.png",
    }
    charts = FLOW_PACK / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    (FLOW_PACK / "figures").mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in mapping.items():
        src = fig_dir / src_name
        if not src.exists():
            log(f"  WARN missing figure {src}")
            continue
        shutil.copy2(src, charts / dst_name)
        shutil.copy2(src, FLOW_PACK / dst_name)
        shutil.copy2(src, FLOW_PACK / "figures" / dst_name)
        log(f"  installed {dst_name}")


def complete_flow_charts(*, discovery_days: int) -> None:
    log("=== 1D.7 FlowDensity20 protocol charts ===")
    FLOW_CHART_SRC.mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

    float_mkt = enriched.float_mktcap.loc[start:end]
    raw = build_net_active_flow_mktcap(l2_cache, float_mkt, window=20).loc[start:end]
    signal_full = cs_zscore(neutralize_size_industry(raw, industry.reindex_like(raw), float_mkt))
    close_full = enriched.close.loc[start:end]

    _, signal = split_discovery_confirmation(signal_full, discovery_days)
    _, close = split_discovery_confirmation(close_full, discovery_days)
    if signal.empty:
        raise RuntimeError("empty confirmation window for Flow charts")

    log(
        f"  confirmation rows={len(signal)} cols={signal.shape[1]} "
        f"period={signal.index[0].date()}→{signal.index[-1].date()}"
    )
    report = build_factor_report(
        "FlowDensity20",
        signal,
        close,
        start_day=signal.index[0],
        end_day=signal.index[-1],
        session=session,
        df_not_limit=Factor_Dev_Lib.get_EOD_Not_Limit(signal.index[0], signal.index[-1]),
        df_not_st=Factor_Dev_Lib.get_EOD_Not_ST(signal.index[0], signal.index[-1]),
        df_trade_status=Factor_Dev_Lib.get_TradeStatus(signal.index[0], signal.index[-1]),
        universes=cfg.UNIVERSE_LIST,
        get_ret_matrix=lambda s, e, idx: Factor_Dev_Lib.get_Ret_Matrix(
            s, e, method="c2c", base_index=idx
        ),
    )
    md_path = publish_factor_report(report, FLOW_CHART_SRC)
    fig_dir = FLOW_CHART_SRC / "FlowDensity20" / "figures"
    install_flow_charts(fig_dir)
    _ = md_path
    meta = {
        "factor": "FlowDensity20",
        "source_column": FLOW_COL,
        "signal": "size_industry",
        "window": "confirmation_after_discovery",
        "discovery_days": discovery_days,
        "n_days": int(len(signal)),
        "start": str(signal.index[0].date()),
        "end": str(signal.index[-1].date()),
        "note": "Protocol charts only; formula/mechanism unchanged (Milestone 1D.7)",
    }
    (FLOW_CHART_SRC / "chart_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (FLOW_PACK / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(FLOW_CHART_SRC / "chart_meta.json", FLOW_PACK / "artifacts" / "protocol_charts_1d7_meta.json")
    log(f"  Flow charts done → {FLOW_PACK / 'charts'}")


def run_d1_execution_grid(signal: pd.DataFrame, ret: pd.DataFrame, *, signal_mode: str) -> pd.DataFrame:
    rows = []
    e1_specs = [
        ("daily", dict(rebalance_freq=1, friday_only=False)),
        ("weekly_friday", dict(rebalance_freq=1, friday_only=True)),
        ("every_5d", dict(rebalance_freq=5, friday_only=False)),
        ("every_10d", dict(rebalance_freq=10, friday_only=False)),
        ("every_20d", dict(rebalance_freq=20, friday_only=False)),
    ]
    log("\n--- D1 E1 rebalance ---")
    for label, kw in e1_specs:
        row = evaluate_execution(
            signal,
            ret,
            label=f"{signal_mode}|{label}",
            stage="E1",
            top_frac=TOP_FRAC,
            weight_method="ew",
            **kw,
        )
        rows.append(row)
        log(
            f"  {label}: gross={row['gross_sharpe']:.2f} net={row['net_sharpe']:.2f} "
            f"TO={row['daily_turnover']:.3f}"
        )

    best_e1 = max(rows, key=lambda r: (r["net_sharpe"] if pd.notna(r["net_sharpe"]) else -1e9))
    if best_e1["label"].endswith("weekly_friday"):
        base_rb = dict(rebalance_freq=1, friday_only=True)
    else:
        freq = int(best_e1["rebalance_freq"]) if best_e1["rebalance_freq"] != "friday" else 1
        base_rb = dict(rebalance_freq=freq, friday_only=False)

    log("\n--- D1 E2 buffer ---")
    for rb_name, rb_kw in [("daily", dict(rebalance_freq=1, friday_only=False)), ("best_e1", base_rb)]:
        for blabel, entry, exit_ in [
            ("buffer_5_15", 0.05, 0.15),
            ("buffer_10_20", 0.10, 0.20),
            ("buffer_10_30", 0.10, 0.30),
        ]:
            row = evaluate_execution(
                signal,
                ret,
                label=f"{signal_mode}|{rb_name}|{blabel}",
                stage="E2_buffer",
                entry_frac=entry,
                exit_frac=exit_,
                weight_method="ew",
                **rb_kw,
            )
            rows.append(row)
            log(f"  {rb_name}|{blabel}: net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}")

    log("\n--- D1 E2 hold ---")
    for hold in [1, 5, 10]:
        row = evaluate_execution(
            signal,
            ret,
            label=f"{signal_mode}|best_e1|hold_{hold}d",
            stage="E2_hold",
            top_frac=TOP_FRAC,
            min_hold=hold,
            weight_method="ew",
            **base_rb,
        )
        rows.append(row)

    log("\n--- D1 E4 combos ---")
    for clabel, kw in [
        ("best_e1_plain", dict(top_frac=TOP_FRAC, **base_rb)),
        ("best_e1_buffer_5_15", dict(entry_frac=0.05, exit_frac=0.15, **base_rb)),
        ("daily_buffer_5_15", dict(entry_frac=0.05, exit_frac=0.15, rebalance_freq=1, friday_only=False)),
        ("daily_buffer_10_20", dict(entry_frac=0.10, exit_frac=0.20, rebalance_freq=1, friday_only=False)),
    ]:
        row = evaluate_execution(
            signal, ret, label=f"{signal_mode}|{clabel}", stage="E4_combo", weight_method="ew", **kw
        )
        rows.append(row)
        log(f"  {clabel}: net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}")

    return pd.DataFrame(rows)


def complete_d1_execution() -> None:
    log("=== 1D.7 D1_LiquidityQuality60d execution ===")
    D1_EXEC_OUT.mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    raw = build_eod_engine_factor(D1_FACTOR, pv_cache).loc[start:end]
    signal = cs_zscore(raw)
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret = ret.reindex(index=signal.index, columns=signal.columns)

    log(f"  D1 panel rows={len(signal)} period={signal.index[0].date()}→{signal.index[-1].date()}")
    all_rows = run_d1_execution_grid(signal, ret, signal_mode="raw")
    ranked = all_rows.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False)
    ranked.to_csv(D1_EXEC_OUT / "all_experiments.csv", index=False)
    ranked.to_csv(D1_EXEC_OUT / "execution_summary.csv", index=False)

    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    baseline = {
        "factor": "D1_LiquidityQuality60d",
        "source": D1_FACTOR,
        "frozen_formula": True,
        "signal_mode": "raw",
        "signal_shift": SIGNAL_SHIFT,
        "cost_round_trip": DEFAULT_ROUND_TRIP_COST,
        "best_label": best.get("label"),
        "best_net_sharpe": best.get("net_sharpe"),
        "best_gross_sharpe": best.get("gross_sharpe"),
        "best_daily_turnover": best.get("daily_turnover"),
        "note": "Milestone 1D.7 execution only — formula not retuned",
    }
    (D1_EXEC_OUT / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, default=str) + "\n")

    (D1_PACK / "execution").mkdir(parents=True, exist_ok=True)
    (D1_PACK / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(D1_EXEC_OUT / "execution_summary.csv", D1_PACK / "execution_summary.csv")
    shutil.copy2(D1_EXEC_OUT / "execution_summary.csv", D1_PACK / "execution" / "execution_summary.csv")
    shutil.copy2(D1_EXEC_OUT / "baseline_metrics.json", D1_PACK / "artifacts" / "execution_baseline_1d7.json")

    summary_path = D1_PACK / "factor_summary.csv"
    if summary_path.exists() and len(ranked):
        df = pd.read_csv(summary_path)
        df = df[df["mode"] != "execution_best"]
        b = ranked.iloc[0]
        row = {
            "factor": "D1_LiquidityQuality60d",
            "period": "confirmation_1455d",
            "universe": "ALL",
            "mode": "execution_best",
            "rank_ic": b.get("rank_ic"),
            "annu_ic": (float(b["rank_ic"]) * np.sqrt(250) if pd.notna(b.get("rank_ic")) else np.nan),
            "icir": b.get("icir"),
            "hl_annu_ret": b.get("gross_annu_ret"),
            "hl_sharpe": b.get("gross_sharpe"),
            "hl_mdd": b.get("mdd_net"),
            "daily_turnover": b.get("daily_turnover"),
            "implied_annu_fee": b.get("implied_annu_fee"),
            "net_sharpe": b.get("net_sharpe"),
            "monotonicity": np.nan,
            "direction": 1,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(summary_path, index=False)

    log(f"  D1 execution best: {best.get('label')} net={best.get('net_sharpe')} TO={best.get('daily_turnover')}")
    log(f"  installed → {D1_PACK / 'execution_summary.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-flow", action="store_true")
    p.add_argument("--skip-d1", action="store_true")
    p.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = p.parse_args()

    if not args.skip_flow:
        complete_flow_charts(discovery_days=args.discovery_days)
    if not args.skip_d1:
        complete_d1_execution()
    log("=== Milestone 1D.7 evaluation artifacts complete ===")


if __name__ == "__main__":
    main()
