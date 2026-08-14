#!/usr/bin/env python
"""Read-only cross-family correlation reference for Order Book candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402


ROOT = Path(RESULT_ROOT)
POOL_ROOT = ROOT / "candidate_pool_v1"
ORDER_BOOK_DIR = POOL_ROOT / "order_book_family"
FAMILY_DIRS = {
    "trade_flow": POOL_ROOT / "trade_flow_family",
    "order_size": POOL_ROOT / "order_size_family",
}


def _representatives(
    summary: pd.DataFrame,
    *,
    include_top: int,
) -> Tuple[List[str], Dict[str, str]]:
    score = np.maximum.reduce(
        [
            summary["icir_raw"].abs().fillna(0).to_numpy(),
            summary.get(
                "g10_excess_sharpe",
                pd.Series(0.0, index=summary.index),
            ).abs().fillna(0).to_numpy(),
            summary["hl_sharpe"].abs().fillna(0).to_numpy(),
        ]
    )
    ranked = summary.assign(_score=score).sort_values(
        "_score", ascending=False
    )
    reasons: Dict[str, str] = {
        name: f"performance_top_{include_top}"
        for name in ranked.head(include_top)["factor"]
    }
    cluster_column = (
        "redundancy_cluster_080"
        if "redundancy_cluster_080" in summary.columns
        else None
    )
    if cluster_column:
        for cluster, block in summary.groupby(cluster_column, dropna=True):
            representative = block.loc[
                block["icir_raw"].abs().idxmax(), "factor"
            ]
            previous = reasons.get(representative)
            cluster_reason = f"cluster_representative:{cluster}"
            reasons[representative] = (
                f"{previous}|{cluster_reason}"
                if previous
                else cluster_reason
            )
    ordered = [
        name for name in ranked["factor"] if name in reasons
    ]
    return ordered, reasons


def _external_representatives(
    family: str,
) -> Tuple[List[str], Dict[str, str]]:
    summary = pd.read_csv(FAMILY_DIRS[family] / "candidate_summary.csv")
    names, reasons = _representatives(summary, include_top=0)
    return names, reasons


def _read_factor_year(
    path: Path,
    year: int,
    name: str,
) -> pd.Series:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    frame = pd.read_parquet(
        path,
        columns=["symbol", "tradetime", "value"],
        filters=[
            ("tradetime", ">=", start),
            ("tradetime", "<", end),
        ],
    )
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    frame["tradetime"] = pd.to_datetime(frame["tradetime"]).dt.normalize()
    if frame.duplicated(["tradetime", "symbol"]).any():
        raise ValueError(f"Duplicate narrow key in {path}")
    series = frame.set_index(["tradetime", "symbol"])["value"]
    series.name = name
    return series.sort_index()


def _order_book_path(name: str) -> Path:
    return ORDER_BOOK_DIR / "factors" / name / "factor_narrow.parquet"


def _external_path(name: str) -> Path:
    return ROOT / name / "factor_narrow.parquet"


def _calculate_family(
    *,
    family: str,
    order_book_names: List[str],
    order_book_reasons: Dict[str, str],
    external_names: List[str],
    external_reasons: Dict[str, str],
    start_year: int,
    end_year: int,
    min_names: int,
) -> pd.DataFrame:
    accumulators = {
        (order_book, external): {
            "sum": 0.0,
            "count": 0,
            "date_min": None,
            "date_max": None,
        }
        for order_book in order_book_names
        for external in external_names
    }
    for year in range(start_year, end_year + 1):
        print(f"[cross] {family} year={year}", flush=True)
        order_book_series = [
            _read_factor_year(_order_book_path(name), year, name)
            for name in order_book_names
        ]
        order_book_series = [
            series for series in order_book_series if not series.empty
        ]
        if not order_book_series:
            continue
        panel = pd.concat(order_book_series, axis=1, join="outer")
        available_order_book = list(panel.columns)
        for external in external_names:
            reference = _read_factor_year(
                _external_path(external), year, external
            )
            if reference.empty:
                continue
            panel[external] = reference.reindex(panel.index)
            for trade_date, block in panel.groupby(
                level="tradetime", sort=True
            ):
                correlations = block[
                    [*available_order_book, external]
                ].corr(method="spearman", min_periods=min_names)
                for order_book in available_order_book:
                    rho = correlations.loc[order_book, external]
                    if pd.isna(rho):
                        continue
                    item = accumulators[(order_book, external)]
                    item["sum"] += float(rho)
                    item["count"] += 1
                    timestamp = pd.Timestamp(trade_date)
                    item["date_min"] = (
                        timestamp
                        if item["date_min"] is None
                        else min(item["date_min"], timestamp)
                    )
                    item["date_max"] = (
                        timestamp
                        if item["date_max"] is None
                        else max(item["date_max"], timestamp)
                    )
            panel.drop(columns=external, inplace=True)
        del panel, order_book_series

    rows = []
    for (order_book, external), item in accumulators.items():
        rho = (
            item["sum"] / item["count"]
            if item["count"] > 0
            else np.nan
        )
        rows.append(
            {
                "order_book_factor": order_book,
                "reference_factor": external,
                "reference_family": family,
                "mean_daily_spearman": rho,
                "abs_mean_daily_spearman": abs(rho),
                "n_dates": item["count"],
                "date_min": (
                    str(item["date_min"].date())
                    if item["date_min"] is not None
                    else None
                ),
                "date_max": (
                    str(item["date_max"].date())
                    if item["date_max"] is not None
                    else None
                ),
                "order_book_selection_reason": order_book_reasons[
                    order_book
                ],
                "reference_selection_reason": external_reasons[external],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_mean_daily_spearman", ascending=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--top-order-book", type=int, default=10)
    parser.add_argument("--min-names", type=int, default=100)
    args = parser.parse_args()

    order_book_summary = pd.read_csv(
        ORDER_BOOK_DIR / "candidate_summary.csv"
    )
    order_book_names, order_book_reasons = _representatives(
        order_book_summary,
        include_top=args.top_order_book,
    )
    selection_rows = [
        {
            "family": "order_book",
            "factor": name,
            "selection_reason": order_book_reasons[name],
        }
        for name in order_book_names
    ]
    for family in FAMILY_DIRS:
        external_names, external_reasons = _external_representatives(family)
        selection_rows.extend(
            {
                "family": family,
                "factor": name,
                "selection_reason": external_reasons[name],
            }
            for name in external_names
        )
        result = _calculate_family(
            family=family,
            order_book_names=order_book_names,
            order_book_reasons=order_book_reasons,
            external_names=external_names,
            external_reasons=external_reasons,
            start_year=args.start_year,
            end_year=args.end_year,
            min_names=args.min_names,
        )
        result.to_csv(
            ORDER_BOOK_DIR / f"order_book_vs_{family}_corr.csv",
            index=False,
        )
    pd.DataFrame(selection_rows).drop_duplicates(
        ["family", "factor"]
    ).to_csv(ORDER_BOOK_DIR / "cross_family_selection.csv", index=False)
    print(
        f"[done] cross-family references for "
        f"{len(order_book_names)} Order Book representatives",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
