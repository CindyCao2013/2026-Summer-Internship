#!/usr/bin/env python
"""D5 Tail confirmation reports — full 1455d Alpha Report v1."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import pandas as pd

import Factor_Dev_Lib
from alpha_research_report import build_factor_report, publish_factor_report, report_summary_row
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_cn_broker import build_cn_broker_factor
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_tail_d5 import D5_CONFIRMATION_FACTORS, D5_TAIL_DENSITY_CANDIDATES
from run_l2_validation import build_any_eod

OUT = Path("research/reports/d5_tail_density_v1/confirmation_1455d")


def log(msg: str) -> None:
    print(msg, flush=True)


def _source_map() -> dict:
    return {name: source for name, source, _, _ in D5_TAIL_DENSITY_CANDIDATES}


def build_factor_panel(name: str, ctx: dict) -> pd.DataFrame:
    start, end = ctx["start"], ctx["end"]
    source = _source_map().get(name, "eod_engine")
    if source == "cn_broker":
        wide = build_cn_broker_factor(name, ctx["pv_cache"])
    elif source == "pv":
        wide = build_factor(name, ctx["pv_cache"])
    elif source == "eod_engine":
        wide = build_eod_engine_factor(name, ctx["pv_cache"])
    else:
        wide = build_any_eod(name, ctx["pv_cache"], ctx["norm_cache"])
    return wide.reindex(index=ctx["close"].index, columns=ctx["close"].columns).loc[start:end]


def load_report_context() -> dict:
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log(f"Loading data ({start.date()} -> {end.date()})...")
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    from factor_formulas_liquidity_norm import build_liquidity_norm_cache

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )
    close = enriched.close.loc[start:end]

    def get_ret_matrix(s, e, idx):
        return Factor_Dev_Lib.get_Ret_Matrix(s, e, method="c2c", base_index=idx)

    return {
        "start": start,
        "end": end,
        "session": session,
        "close": close,
        "pv_cache": pv_cache,
        "norm_cache": norm_cache,
        "universes": cfg.UNIVERSE_LIST,
        "get_ret_matrix": get_ret_matrix,
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D5 tail confirmation reports (1455d)")
    parser.add_argument("--factors", nargs="*", default=D5_CONFIRMATION_FACTORS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_report_context()
    summary_rows = []

    for name in args.factors:
        log(f"\n=== D5 Confirmation: {name} ===")
        try:
            panel = build_factor_panel(name, ctx)
            report = build_factor_report(
                name,
                panel,
                ctx["close"],
                start_day=ctx["start"],
                end_day=ctx["end"],
                session=ctx["session"],
                df_not_limit=ctx["df_not_limit"],
                df_not_st=ctx["df_not_st"],
                df_trade_status=ctx["df_trade_status"],
                universes=ctx["universes"],
                get_ret_matrix=ctx["get_ret_matrix"],
            )
            path = publish_factor_report(report, OUT)
            summary_rows.append(report_summary_row(report))
            log(f"  Saved -> {path}")
        except Exception as exc:
            log(f"  FAILED {name}: {exc}")
        gc.collect()

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(OUT / "d5_confirmation_summary.csv", index=False)
        log(f"\nSummary -> {OUT / 'd5_confirmation_summary.csv'}")


if __name__ == "__main__":
    main()
