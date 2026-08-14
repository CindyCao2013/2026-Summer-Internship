#!/usr/bin/env python
"""Validate a short Order Book daily primitive interval and raw-row parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_order_book import (  # noqa: E402
    COVERAGE_THRESHOLD,
    EXPECTED_MINUTE_COUNT,
    fetch_order_book_daily,
    fetch_raw_row_daily_reference,
)


DEFAULT_ROOT = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/order_book_daily/validation"
)


def _daily_spearman(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return frame.groupby("TradeDate", observed=True).apply(
        lambda group: group[left].corr(group[right], method="spearman"),
        include_groups=False,
    )


def _audit(frame: pd.DataFrame) -> Dict[str, object]:
    numeric = frame.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    obi_columns = (
        "obi_1_mean",
        "obi_5_mean",
        "obi_10_mean",
        "weighted_obi_mean",
        "near_far_imbalance_mean",
    )
    obi_outside = sum(
        int((frame[column].abs() > 1.0 + 1e-12).sum())
        for column in obi_columns
    )
    hhi_outside = sum(
        int(
            (
                frame[column].notna()
                & ~frame[column].between(0.1 - 1e-10, 1.0 + 1e-10)
            ).sum()
        )
        for column in ("bid_depth_hhi_mean", "ask_depth_hhi_mean")
    )
    microprice_bound = int(
        (
            frame["microprice_deviation_mean"].abs()
            > frame["relative_spread_mean"] / 2.0 + 1e-12
        ).sum()
    )
    return {
        "rows": int(len(frame)),
        "dates": int(frame["TradeDate"].nunique()),
        "date_min": str(frame["TradeDate"].min().date()),
        "date_max": str(frame["TradeDate"].max().date()),
        "symbols": int(frame["symbol"].nunique()),
        "sse_rows": int((frame["source_exchange"] == "SSE").sum()),
        "szse_rows": int((frame["source_exchange"] == "SZSE").sum()),
        "duplicate_keys": int(
            frame.duplicated(["symbol", "TradeDate"]).sum()
        ),
        "low_coverage_rows": int(
            (frame["coverage_ratio"] < COVERAGE_THRESHOLD).sum()
        ),
        "coverage_above_one": int((frame["coverage_ratio"] > 1.0).sum()),
        "coverage_q01": float(frame["coverage_ratio"].quantile(0.01)),
        "coverage_q10": float(frame["coverage_ratio"].quantile(0.10)),
        "coverage_median": float(frame["coverage_ratio"].median()),
        "coverage_mean": float(frame["coverage_ratio"].mean()),
        "coverage_q90": float(frame["coverage_ratio"].quantile(0.90)),
        "obi_outside_bounds": int(obi_outside),
        "negative_spread_rows": int(
            (frame["relative_spread_mean"] < -1e-12).sum()
        ),
        "microprice_outside_top_book_rows": microprice_bound,
        "hhi_outside_bounds": int(hhi_outside),
        "inf_values": inf_count,
        "missing_slope_rows": int(
            frame[
                ["bid_depth_slope_mean", "ask_depth_slope_mean"]
            ].isna().any(axis=1).sum()
        ),
        "missing_book_vwap_gap_rows": int(
            frame["book_vwap_gap_mean"].isna().sum()
        ),
        "close_auction_valid_rows": int(frame["close_auction_valid"].sum()),
        "expected_minute_count": EXPECTED_MINUTE_COUNT,
    }


def _quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "valid_minute_count",
        "coverage_ratio",
        "obi_1_mean",
        "obi_5_mean",
        "obi_10_mean",
        "weighted_obi_mean",
        "relative_spread_mean",
        "microprice_deviation_mean",
        "bid_depth_hhi_mean",
        "ask_depth_hhi_mean",
        "bid_depth_slope_mean",
        "ask_depth_slope_mean",
        "book_vwap_gap_mean",
        "log_total_depth_mean",
    ]
    rows: List[Dict[str, object]] = []
    for column in columns:
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        row: Dict[str, object] = {
            "metric": column,
            "n": int(len(values)),
        }
        for quantile in (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0):
            row[f"q{int(quantile * 100):02d}"] = (
                float(values.quantile(quantile)) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _parity(fixed: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    merged = fixed.merge(
        raw,
        on=["symbol", "TradeDate"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    pairs = (
        ("obi_5_mean", "raw_obi_5_mean"),
        ("relative_spread_mean", "raw_relative_spread_mean"),
    )
    for fixed_column, raw_column in pairs:
        pair = merged[
            [
                "TradeDate",
                "raw_snapshot_count",
                fixed_column,
                raw_column,
            ]
        ].dropna()
        differences = pair[fixed_column] - pair[raw_column]
        daily_corr = _daily_spearman(pair, fixed_column, raw_column)
        rows.append(
            {
                "fixed_metric": fixed_column,
                "raw_metric": raw_column,
                "n_symbol_days": int(len(pair)),
                "pearson": float(pair[fixed_column].corr(pair[raw_column])),
                "mean_daily_cross_sectional_spearman": float(
                    daily_corr.mean()
                ),
                "mean_abs_difference": float(differences.abs().mean()),
                "median_abs_difference": float(differences.abs().median()),
                "p90_abs_difference": float(
                    differences.abs().quantile(0.90)
                ),
                "max_abs_difference": float(differences.abs().max()),
                "snapshot_count_vs_abs_difference_spearman": float(
                    pair["raw_snapshot_count"].corr(
                        differences.abs(), method="spearman"
                    )
                ),
                "identical_within_1e_12_fraction": float(
                    differences.abs().le(1e-12).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _assert_hard_checks(audit: Dict[str, object]) -> None:
    hard = (
        "duplicate_keys",
        "coverage_above_one",
        "obi_outside_bounds",
        "negative_spread_rows",
        "microprice_outside_top_book_rows",
        "hhi_outside_bounds",
        "inf_values",
    )
    failures = {name: audit[name] for name in hard if audit[name] != 0}
    if failures:
        raise ValueError(f"Order Book validation failed: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-06-28")
    parser.add_argument("--end", default="2024-06-28")
    parser.add_argument("--label", default="smoke_2024-06-28")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    output = args.output_root / args.label
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[validate] fixed-minute daily {args.start}..{args.end}",
        flush=True,
    )
    fixed = fetch_order_book_daily(args.start, args.end)
    if fixed.empty:
        raise ValueError("Order Book daily query returned no rows")
    print("[validate] raw-row daily reference", flush=True)
    raw = fetch_raw_row_daily_reference(args.start, args.end)

    audit = _audit(fixed)
    quantiles = _quantiles(fixed)
    parity = _parity(fixed, raw)
    fixed.to_parquet(
        output / f"order_book_daily_{args.start}_{args.end}.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame([audit]).to_csv(output / "quality_audit.csv", index=False)
    quantiles.to_csv(output / "metric_quantiles.csv", index=False)
    parity.to_csv(output / "raw_row_parity_summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                **audit,
                "start": args.start,
                "end": args.end,
                "coverage_threshold": COVERAGE_THRESHOLD,
                "raw_row_reference_rows": int(len(raw)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _assert_hard_checks(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
