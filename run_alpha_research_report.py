#!/usr/bin/env python
"""Generate Alpha Research Report v1 for Tier-A candidate factors.

Usage:
  OMP_NUM_THREADS=1 python run_alpha_research_report.py
  OMP_NUM_THREADS=1 python run_alpha_research_report.py --factors winner_sentiment_reversal_5d cn_cancel_shock
"""

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

import Factor_Dev_Lib
from alpha_research_report import (
    DEFAULT_TIER_A,
    build_factor_report,
    publish_factor_report,
    report_summary_row,
)
from factor_data_loaders import load_derivative_wide_tables, load_eod_enriched_tables, load_financial_ttmhis_long
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_fundamental import build_fundamental_cache, build_fundamental_factor
from factor_formulas_l2_v2 import build_l2_v2_factor
from factor_formulas_liquidity_norm import build_liquidity_norm_cache, build_liquidity_norm_factor
from factor_formulas_value import build_value_factor
from l2_data_loaders import build_l2_daily_cache
from run_l2_validation import build_any_eod

OUT = Path("research/reports")
REPORT_ROOT = OUT / "tier_a_v1"


def log(msg: str) -> None:
    print(msg, flush=True)


def build_factor_panel(name: str, ctx: dict) -> "pd.DataFrame":
    import pandas as pd

    start, end = ctx["start"], ctx["end"]
    if name == "cn_cancel_shock":
        wide = build_l2_v2_factor(name, ctx["l2_cache"])
    elif name == "quality_composite":
        wide = build_fundamental_factor(name, ctx["fund_cache"])
    elif name == "value_composite":
        wide = build_value_factor(name, ctx["fund_cache"])
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
    der, _ = load_derivative_wide_tables(preheat, end, session=session)
    finance_long, _ = load_financial_ttmhis_long(preheat, end, session=session)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

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
    fund_cache = build_fundamental_cache(
        der, close=enriched.close, finance_long=finance_long
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
        "fund_cache": fund_cache,
        "l2_cache": l2_cache,
        "universes": cfg.UNIVERSE_LIST,
        "get_ret_matrix": get_ret_matrix,
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Research Report v1")
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Regenerate report.md + figures from existing CSV artifacts",
    )
    parser.add_argument(
        "--factors",
        nargs="*",
        default=DEFAULT_TIER_A,
        help="Factor names (default: Tier-A list)",
    )
    args = parser.parse_args()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.markdown_only:
        from alpha_research_report import regenerate_markdown_from_csv

        summary = regenerate_markdown_from_csv(REPORT_ROOT)
        if len(summary):
            summary.to_csv(REPORT_ROOT / "tier_a_summary.csv", index=False)
            log(f"Summary -> {REPORT_ROOT / 'tier_a_summary.csv'}")
            log(summary.to_string(index=False))
        return

    ctx = load_report_context()
    summary_rows = []

    for name in args.factors:
        log(f"\n=== Report: {name} ===")
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
            path = publish_factor_report(report, REPORT_ROOT)
            summary_rows.append(report_summary_row(report))
            log(f"  Saved -> {path}")
        except Exception as exc:
            log(f"  FAILED {name}: {exc}")
        gc.collect()

    if summary_rows:
        import pandas as pd

        summary = pd.DataFrame(summary_rows)
        summary_path = REPORT_ROOT / "tier_a_summary.csv"
        summary.to_csv(summary_path, index=False)
        log(f"\nSummary -> {summary_path}")
        log(summary.to_string(index=False))


if __name__ == "__main__":
    main()
