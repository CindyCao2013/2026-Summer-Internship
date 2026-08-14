#!/usr/bin/env python
"""Compare liquidity factor redundancy before/after size normalization."""

import datetime as dt

import factor_config as cfg
import intraday_lib
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_CORE_LIST,
    build_liquidity_norm_cache,
    build_liquidity_norm_factor,
)
from liquidity_normalization import factor_correlation_matrix


def main():
    start_day = cfg.START_DAY
    end_day = cfg.END_DAY
    start_preheat = start_day - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)

    cache = build_liquidity_norm_cache(
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

    factors = {
        name: build_liquidity_norm_factor(name, cache).loc[start_day:end_day]
        for name in LIQUIDITY_NORM_CORE_LIST
    }

    corr = factor_correlation_matrix(factors)
    out_dir = cfg.RESEARCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "liquidity_factor_correlation.csv"
    corr.to_csv(out_path)

    print("Liquidity factor correlation matrix (stacked obs):")
    print(corr.round(3).to_string())
    print(f"\nSaved -> {out_path}")

    baseline = corr.loc["amount_stability_20d"].drop("amount_stability_20d")
    print("\nCorrelation vs amount_stability_20d (baseline):")
    for name, val in baseline.sort_values(key=abs, ascending=False).items():
        print(f"  {name:40s} {val:+.3f}")


if __name__ == "__main__":
    main()
