#!/usr/bin/env python
"""Read-only Price Formation correlation reference versus three frozen families."""

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
PRICE_DIR = POOL_ROOT / "price_formation_family"
FAMILY_DIRS = {
    "trade_flow": POOL_ROOT / "trade_flow_family",
    "order_size": POOL_ROOT / "order_size_family",
    "order_book": POOL_ROOT / "order_book_family",
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
    if "redundancy_cluster_080" in summary.columns:
        for cluster, block in summary.groupby(
            "redundancy_cluster_080", dropna=True
        ):
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
    ordered = [name for name in ranked["factor"] if name in reasons]
    return ordered, reasons


def _external_representatives(
    family: str,
) -> Tuple[List[str], Dict[str, str]]:
    summary = pd.read_csv(FAMILY_DIRS[family] / "candidate_summary.csv")
    return _representatives(summary, include_top=0)


def _factor_path(family: str, name: str) -> Path:
    if family == "price_formation":
        return PRICE_DIR / "factors" / name / "factor_narrow.parquet"
    if family == "order_book":
        return (
            POOL_ROOT
            / "order_book_family"
            / "factors"
            / name
            / "factor_narrow.parquet"
        )
    return ROOT / name / "factor_narrow.parquet"


def _read_factor_full(
    family: str,
    name: str,
    output_name: str,
) -> pd.Series:
    """Read the full narrow history once; year slices are taken in memory."""
    path = _factor_path(family, name)
    frame = pd.read_parquet(
        path,
        columns=["symbol", "tradetime", "value"],
    )
    if frame.empty:
        return pd.Series(dtype=float, name=output_name)
    frame["tradetime"] = pd.to_datetime(frame["tradetime"]).dt.normalize()
    if frame.duplicated(["tradetime", "symbol"]).any():
        raise ValueError(f"Duplicate narrow key in {path}")
    result = frame.set_index(["tradetime", "symbol"])["value"]
    result.name = output_name
    return result.sort_index()


def _slice_year(series: pd.Series, year: int) -> pd.Series:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    sliced = series.loc[
        (series.index.get_level_values("tradetime") >= start)
        & (series.index.get_level_values("tradetime") < end)
    ]
    return sliced


def _calculate_family(
    *,
    family: str,
    price_names: List[str],
    price_reasons: Dict[str, str],
    reference_names: List[str],
    reference_reasons: Dict[str, str],
    start_year: int,
    end_year: int,
    min_names: int,
) -> pd.DataFrame:
    accumulators = {
        (price, reference): {
            "sum": 0.0,
            "count": 0,
            "date_min": None,
            "date_max": None,
        }
        for price in price_names
        for reference in reference_names
    }
    price_full = {
        f"price::{name}": _read_factor_full(
            "price_formation", name, f"price::{name}"
        )
        for name in price_names
    }
    price_full = {
        key: series for key, series in price_full.items() if not series.empty
    }
    reference_full = {
        f"reference::{name}": _read_factor_full(
            family, name, f"reference::{name}"
        )
        for name in reference_names
    }
    reference_full = {
        key: series
        for key, series in reference_full.items()
        if not series.empty
    }
    for year in range(start_year, end_year + 1):
        print(f"[cross] {family} year={year}", flush=True)
        price_series = [
            _slice_year(series, year) for series in price_full.values()
        ]
        price_series = [
            series for series in price_series if not series.empty
        ]
        if not price_series:
            continue
        reference_series = [
            _slice_year(series, year) for series in reference_full.values()
        ]
        reference_series = [
            series for series in reference_series if not series.empty
        ]
        if not reference_series:
            continue
        panel = pd.concat(
            [*price_series, *reference_series], axis=1, join="outer"
        )
        available_price = [
            column.split("::", 1)[1]
            for column in panel.columns
            if column.startswith("price::")
        ]
        available_reference = [
            column.split("::", 1)[1]
            for column in panel.columns
            if column.startswith("reference::")
        ]
        price_columns = [f"price::{name}" for name in available_price]
        reference_columns = [
            f"reference::{name}" for name in available_reference
        ]
        for trade_date, block in panel.groupby(
            level="tradetime", sort=True
        ):
            # Spearman = Pearson over per-column ranks with pairwise
            # complete observations; rank once per day for the full panel.
            ranked = block.rank()
            correlations = ranked.corr(min_periods=min_names)
            block_corr = correlations.loc[price_columns, reference_columns]
            timestamp = pd.Timestamp(trade_date)
            for price in available_price:
                for reference in available_reference:
                    rho = block_corr.loc[
                        f"price::{price}", f"reference::{reference}"
                    ]
                    if pd.isna(rho):
                        continue
                    item = accumulators[(price, reference)]
                    item["sum"] += float(rho)
                    item["count"] += 1
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
        del panel, price_series, reference_series

    rows = []
    for (price, reference), item in accumulators.items():
        rho = (
            item["sum"] / item["count"]
            if item["count"] > 0
            else np.nan
        )
        rows.append(
            {
                "price_formation_factor": price,
                "reference_factor": reference,
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
                "price_formation_selection_reason": price_reasons[price],
                "reference_selection_reason": reference_reasons[reference],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_mean_daily_spearman", ascending=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--top-price-formation", type=int, default=10)
    parser.add_argument("--min-names", type=int, default=100)
    args = parser.parse_args()

    summary = pd.read_csv(PRICE_DIR / "candidate_summary.csv")
    price_names, price_reasons = _representatives(
        summary, include_top=args.top_price_formation
    )
    selection_rows = [
        {
            "family": "price_formation",
            "factor": name,
            "selection_reason": price_reasons[name],
        }
        for name in price_names
    ]
    for family in FAMILY_DIRS:
        reference_names, reference_reasons = _external_representatives(
            family
        )
        selection_rows.extend(
            {
                "family": family,
                "factor": name,
                "selection_reason": reference_reasons[name],
            }
            for name in reference_names
        )
        result = _calculate_family(
            family=family,
            price_names=price_names,
            price_reasons=price_reasons,
            reference_names=reference_names,
            reference_reasons=reference_reasons,
            start_year=args.start_year,
            end_year=args.end_year,
            min_names=args.min_names,
        )
        result.to_csv(
            PRICE_DIR / f"price_formation_vs_{family}_corr.csv",
            index=False,
        )
    pd.DataFrame(selection_rows).drop_duplicates(
        ["family", "factor"]
    ).to_csv(PRICE_DIR / "cross_family_selection.csv", index=False)
    print(
        f"[done] cross-family references for "
        f"{len(price_names)} Price Formation representatives",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
