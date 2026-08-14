#!/usr/bin/env python
"""Build the reusable daily order-size distribution primitive.

The ClickHouse query performs one server-side Tick aggregation per exchange and
quarter. Each chunk is cached, audited, and resumable. No factor or backtest is
computed here.
"""

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

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.ch_tick import (  # noqa: E402
    ORDER_SIZE_BOUNDARIES,
    fetch_order_size_distribution_daily,
)


DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
OUT_DIR = Path(RESULT_ROOT) / "primitives" / "order_size_distribution_daily"


def _required_columns(boundaries: List[int]) -> List[str]:
    columns = [
        "symbol",
        "TradeDate",
        "total_amt",
        "trade_cnt",
        "active_buy_amt",
        "active_sell_amt",
    ]
    for boundary in boundaries:
        columns.extend(
            [
                f"cum_amt_{boundary}",
                f"cum_cnt_{boundary}",
                f"buy_cum_amt_{boundary}",
                f"sell_cum_amt_{boundary}",
            ]
        )
    return columns


def _normalize(part: pd.DataFrame, boundaries: List[int]) -> pd.DataFrame:
    required = _required_columns(boundaries)
    missing = sorted(set(required).difference(part.columns))
    if missing:
        raise ValueError(f"order-size primitive chunk missing columns: {missing}")
    out = part.loc[:, required].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["TradeDate"] = pd.to_datetime(out["TradeDate"]).dt.normalize()
    for column in required[2:]:
        out[column] = pd.to_numeric(
            out[column], errors="coerce"
        ).astype("float64")
    return out


def _audit(part: pd.DataFrame, boundaries: List[int]) -> Dict[str, object]:
    total = part["total_amt"].replace(0, np.nan)
    amount_columns = [f"cum_amt_{boundary}" for boundary in boundaries]
    count_columns = [f"cum_cnt_{boundary}" for boundary in boundaries]
    buy_columns = [
        f"buy_cum_amt_{boundary}" for boundary in boundaries
    ]
    sell_columns = [
        f"sell_cum_amt_{boundary}" for boundary in boundaries
    ]
    amount_values = part[amount_columns].to_numpy(dtype=float)
    count_values = part[count_columns].to_numpy(dtype=float)
    buy_values = part[buy_columns].to_numpy(dtype=float)
    sell_values = part[sell_columns].to_numpy(dtype=float)
    amount_tolerance = np.maximum(
        1e-4, np.abs(amount_values) * 1e-10
    )
    total_values = part["total_amt"].to_numpy(dtype=float)
    total_tolerance = np.maximum(1e-4, np.abs(total_values) * 1e-10)

    amount_monotonic_violations = int(
        (
            np.diff(amount_values, axis=1)
            < -amount_tolerance[:, 1:]
        ).any(axis=1).sum()
    )
    count_monotonic_violations = int(
        (np.diff(count_values, axis=1) < 0).any(axis=1).sum()
    )
    buy_monotonic_violations = int(
        (
            np.diff(buy_values, axis=1)
            < -amount_tolerance[:, 1:]
        ).any(axis=1).sum()
    )
    sell_monotonic_violations = int(
        (
            np.diff(sell_values, axis=1)
            < -amount_tolerance[:, 1:]
        ).any(axis=1).sum()
    )
    amount_above_total = int(
        (
            amount_values[:, -1]
            > total_values + total_tolerance
        ).sum()
    )
    count_above_total = int(
        (
            count_values[:, -1]
            > part["trade_cnt"].to_numpy(dtype=float)
        ).sum()
    )
    side_above_bucket = int(
        (
            (buy_values + sell_values)
            > amount_values + amount_tolerance
        )
        .any(axis=1)
        .sum()
    )
    buy_above_active_total = int(
        (
            buy_values[:, -1]
            > (
                part["active_buy_amt"].to_numpy(dtype=float)
                + total_tolerance
            )
        ).sum()
    )
    sell_above_active_total = int(
        (
            sell_values[:, -1]
            > (
                part["active_sell_amt"].to_numpy(dtype=float)
                + total_tolerance
            )
        ).sum()
    )
    classified_above_total = int(
        (
            part["active_buy_amt"] + part["active_sell_amt"]
            > part["total_amt"] + total_tolerance
        ).sum()
    )

    b1, _, b5, b20, b100 = boundaries
    canonical = pd.DataFrame(
        {
            "le_1w": part[f"cum_amt_{b1}"],
            "1w_5w": part[f"cum_amt_{b5}"] - part[f"cum_amt_{b1}"],
            "5w_20w": part[f"cum_amt_{b20}"] - part[f"cum_amt_{b5}"],
            "20w_100w": part[f"cum_amt_{b100}"] - part[f"cum_amt_{b20}"],
            "gt_100w": part["total_amt"] - part[f"cum_amt_{b100}"],
        }
    )
    canonical_sum_error = (
        canonical.sum(axis=1) - part["total_amt"]
    ).abs()
    negative_bucket_rows = int(
        canonical.lt(-total_tolerance, axis=0).any(axis=1).sum()
    )

    audit: Dict[str, object] = {
        "rows": int(len(part)),
        "n_symbols": int(part["symbol"].nunique()),
        "date_min": (
            str(part["TradeDate"].min().date()) if not part.empty else None
        ),
        "date_max": (
            str(part["TradeDate"].max().date()) if not part.empty else None
        ),
        "amount_monotonic_violations": amount_monotonic_violations,
        "count_monotonic_violations": count_monotonic_violations,
        "buy_monotonic_violations": buy_monotonic_violations,
        "sell_monotonic_violations": sell_monotonic_violations,
        "amount_above_total": amount_above_total,
        "count_above_total": count_above_total,
        "side_above_bucket": side_above_bucket,
        "buy_above_active_total": buy_above_active_total,
        "sell_above_active_total": sell_above_active_total,
        "classified_above_total": classified_above_total,
        "negative_bucket_rows": negative_bucket_rows,
        "max_bucket_sum_error": float(canonical_sum_error.max()),
        "active_buy_share": float(
            part["active_buy_amt"].sum() / part["total_amt"].sum()
        ),
        "active_sell_share": float(
            part["active_sell_amt"].sum() / part["total_amt"].sum()
        ),
    }
    for name, values in canonical.items():
        audit[f"{name}_amount_share"] = float(
            values.sum() / part["total_amt"].sum()
        )
    return audit


def _assert_audit(audit: Dict[str, object], chunk: str) -> None:
    hard_checks = (
        "amount_monotonic_violations",
        "count_monotonic_violations",
        "buy_monotonic_violations",
        "sell_monotonic_violations",
        "amount_above_total",
        "count_above_total",
        "side_above_bucket",
        "buy_above_active_total",
        "sell_above_active_total",
        "classified_above_total",
        "negative_bucket_rows",
    )
    failures = {
        key: audit[key] for key in hard_checks if int(audit[key]) != 0
    }
    if failures:
        raise ValueError(f"primitive audit failed for {chunk}: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument(
        "--force",
        action="store_true",
        help="requery and overwrite existing quarterly chunks",
    )
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    boundaries = [int(value) for value in ORDER_SIZE_BOUNDARIES]

    chunk_dir = OUT_DIR / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    quarter_starts = pd.date_range(start, end, freq="QS")
    edges = [start] + [
        value for value in quarter_starts if start < value <= end
    ]
    edges += [end + pd.Timedelta(days=1)]

    parts = []
    coverage = []
    for position, (chunk_start, chunk_end_exclusive) in enumerate(
        zip(edges[:-1], edges[1:]),
        1,
    ):
        chunk_end = chunk_end_exclusive - pd.Timedelta(days=1)
        label = f"{chunk_start.date()}~{chunk_end.date()}"
        chunk_path = (
            chunk_dir
            / f"chunk_{chunk_start.date()}_{chunk_end.date()}.parquet"
        )
        if chunk_path.exists() and not args.force:
            print(
                f"[{position}/{len(edges) - 1}] reuse {label}",
                flush=True,
            )
            part = pd.read_parquet(chunk_path)
        else:
            print(
                f"[{position}/{len(edges) - 1}] CH query {label}",
                flush=True,
            )
            part = fetch_order_size_distribution_daily(
                chunk_start,
                chunk_end,
                boundaries=boundaries,
            )
            part.to_parquet(chunk_path, index=False)
        part = _normalize(part, boundaries)
        audit = {"chunk": label, **_audit(part, boundaries)}
        _assert_audit(audit, label)
        coverage.append(audit)
        parts.append(part)
        print(
            f"  rows={len(part):,} symbols={part['symbol'].nunique():,} "
            f"buy={audit['active_buy_share']:.2%} "
            f"sell={audit['active_sell_share']:.2%}",
            flush=True,
        )

    combined = pd.concat(parts, ignore_index=True)
    duplicates = int(
        combined.duplicated(["symbol", "TradeDate"], keep=False).sum()
    )
    if duplicates:
        raise ValueError(
            f"combined order-size primitive has {duplicates} duplicate rows"
        )
    combined = combined.sort_values(
        ["TradeDate", "symbol"], kind="stable"
    ).reset_index(drop=True)

    output = (
        OUT_DIR
        / f"order_size_distribution_daily_{start.date()}_{end.date()}.parquet"
    )
    combined.to_parquet(output, index=False)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(OUT_DIR / "coverage_report.csv", index=False)
    full_audit = _audit(combined, boundaries)
    _assert_audit(full_audit, "full_sample")

    manifest = {
        "version": "order_size_distribution_daily_v1",
        "requested_start": str(start.date()),
        "requested_end": str(end.date()),
        "observed_start": str(combined["TradeDate"].min().date()),
        "observed_end": str(combined["TradeDate"].max().date()),
        "rows": int(len(combined)),
        "symbols": int(combined["symbol"].nunique()),
        "boundaries_rmb": boundaries,
        "canonical_buckets": [
            "<=10000",
            "(10000,50000]",
            "(50000,200000]",
            "(200000,1000000]",
            ">1000000",
        ],
        "asset_scope": (
            "A-share code prefixes only: SH 6*; "
            "SZ 000/001/002/003/300/301/302"
        ),
        "frozen_mid_bucket": "(40000,200000]",
        "session": "09:30:00 <= ExchTime < 15:00:01 on every TradeDate",
        "sse_filter": "Type='T'; amount=ifNull(Amount,Price*Volume)",
        "szse_filter": (
            "Type='011' AND BidOrderNo>0 AND AskOrderNo>0; "
            "amount=Price*Volume"
        ),
        "full_sample_audit": full_audit,
        "output": str(output),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] {output}", flush=True)
    print(
        f"[coverage] rows={len(combined):,} "
        f"symbols={combined['symbol'].nunique():,} "
        f"dates={combined['TradeDate'].min().date()}~"
        f"{combined['TradeDate'].max().date()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
