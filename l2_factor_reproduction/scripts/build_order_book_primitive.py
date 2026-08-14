#!/usr/bin/env python
"""Build the partitioned l2_primitive_order_book_daily dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_order_book import (  # noqa: E402
    COVERAGE_THRESHOLD,
    EXPECTED_MINUTE_COUNT,
    FORMULA_VERSION,
    ORDER_BOOK_TABLES,
    SCHEMA_VERSION,
    fetch_order_book_daily,
    order_book_daily_sql,
)


OUT_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/order_book_daily"
)
DATASET_DIR = OUT_DIR / "dataset"
DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start.normalize()
    end = end.normalize()
    while cursor <= end:
        if frequency == "quarter":
            boundary = cursor.to_period("Q").end_time.normalize()
        elif frequency == "month":
            boundary = cursor.to_period("M").end_time.normalize()
        else:
            raise ValueError(f"Unsupported chunk frequency: {frequency}")
        chunk_end = min(boundary, end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _chunk_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    year_dir = DATASET_DIR / f"year={start.year}"
    return year_dir / (
        f"order_book_daily_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
    )


def _disk_snapshot() -> Dict[str, object]:
    usage = shutil.disk_usage("/home")
    return {
        "path": "/home",
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_fraction": usage.used / usage.total,
    }


def _assert_disk(minimum_free_gb: float) -> Dict[str, object]:
    snapshot = _disk_snapshot()
    minimum = int(minimum_free_gb * 1024**3)
    if snapshot["free_bytes"] < minimum:
        raise RuntimeError(
            f"Insufficient /home space: {snapshot['free_bytes'] / 1024**3:.2f} "
            f"GiB free, requires {minimum_free_gb:.2f} GiB"
        )
    return snapshot


def _audit_chunk(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, object]:
    numeric = frame.select_dtypes(include=[np.number])
    obi_columns = (
        "obi_1_mean",
        "obi_5_mean",
        "obi_10_mean",
        "weighted_obi_mean",
        "near_far_imbalance_mean",
    )
    hhi_columns = ("bid_depth_hhi_mean", "ask_depth_hhi_mean")
    return {
        "chunk_start": start.strftime("%Y-%m-%d"),
        "chunk_end": end.strftime("%Y-%m-%d"),
        "actual_date_min": str(frame["TradeDate"].min().date()),
        "actual_date_max": str(frame["TradeDate"].max().date()),
        "rows": int(len(frame)),
        "dates": int(frame["TradeDate"].nunique()),
        "symbols": int(frame["symbol"].nunique()),
        "duplicate_keys": int(
            frame.duplicated(["symbol", "TradeDate"]).sum()
        ),
        "low_coverage_rows": int(
            (frame["coverage_ratio"] < COVERAGE_THRESHOLD).sum()
        ),
        "coverage_above_one": int((frame["coverage_ratio"] > 1.0).sum()),
        "coverage_mean": float(frame["coverage_ratio"].mean()),
        "coverage_q01": float(frame["coverage_ratio"].quantile(0.01)),
        "coverage_q10": float(frame["coverage_ratio"].quantile(0.10)),
        "coverage_median": float(frame["coverage_ratio"].median()),
        "obi_outside_bounds": int(
            sum(
                (frame[column].abs() > 1.0 + 1e-12).sum()
                for column in obi_columns
            )
        ),
        "negative_spread_rows": int(
            (frame["relative_spread_mean"] < -1e-12).sum()
        ),
        "microprice_outside_top_book_rows": int(
            (
                frame["microprice_deviation_mean"].abs()
                > frame["relative_spread_mean"] / 2.0 + 1e-12
            ).sum()
        ),
        "hhi_outside_bounds": int(
            sum(
                (
                    frame[column].notna()
                    & ~frame[column].between(0.1 - 1e-10, 1.0 + 1e-10)
                ).sum()
                for column in hhi_columns
            )
        ),
        "inf_values": int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        ),
        "missing_slope_rows": int(
            frame[
                ["bid_depth_slope_mean", "ask_depth_slope_mean"]
            ].isna().any(axis=1).sum()
        ),
        "missing_book_vwap_gap_rows": int(
            frame["book_vwap_gap_mean"].isna().sum()
        ),
        "close_auction_valid_rows": int(frame["close_auction_valid"].sum()),
    }


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
        raise ValueError(
            f"Order Book primitive chunk failed hard checks: {failures}"
        )


def _coverage_rows(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Dict[str, object]]:
    rows = []
    for exchange, group in frame.groupby("source_exchange", observed=True):
        rows.append(
            {
                "chunk_start": start.strftime("%Y-%m-%d"),
                "chunk_end": end.strftime("%Y-%m-%d"),
                "source_exchange": exchange,
                "rows": int(len(group)),
                "dates": int(group["TradeDate"].nunique()),
                "symbols": int(group["symbol"].nunique()),
                "eligible_rows": int(
                    (group["coverage_ratio"] >= COVERAGE_THRESHOLD).sum()
                ),
                "coverage_mean": float(group["coverage_ratio"].mean()),
                "coverage_q01": float(
                    group["coverage_ratio"].quantile(0.01)
                ),
                "coverage_q10": float(
                    group["coverage_ratio"].quantile(0.10)
                ),
                "coverage_median": float(group["coverage_ratio"].median()),
                "valid_minutes_mean": float(
                    group["valid_minute_count"].mean()
                ),
            }
        )
    return rows


def _query_hashes(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, str]:
    hashes = {}
    for table, suffix, exchange in ORDER_BOOK_TABLES:
        query = order_book_daily_sql(
            table=table,
            exchange_suffix=suffix,
            exchange=exchange,
            start=start,
            end=end,
        )
        hashes[exchange] = _text_sha256(query)
    return hashes


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    total = int(histogram.sum())
    if total == 0:
        return float("nan")
    target = quantile * (total - 1)
    index = int(np.searchsorted(np.cumsum(histogram), target + 1, side="left"))
    return index / EXPECTED_MINUTE_COUNT


def _write_metadata(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    chunks: List[Dict[str, object]],
    audits: List[Dict[str, object]],
    coverage_rows: List[Dict[str, object]],
    initial_disk: Dict[str, object],
    symbols: set[str],
    coverage_histogram: np.ndarray,
) -> None:
    quality = pd.DataFrame(audits)
    coverage = pd.DataFrame(coverage_rows)
    quality.to_csv(OUT_DIR / "quality_audit.csv", index=False)
    coverage.to_csv(OUT_DIR / "coverage_report.csv", index=False)
    code_path = (
        PROJ_ROOT
        / "l2_factor_reproduction/python/ch_order_book.py"
    )
    final_disk = _disk_snapshot()
    manifest = {
        "primitive_name": "l2_primitive_order_book_daily",
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "source_tables": [
            f"cmds.{table}" for table, _, _ in ORDER_BOOK_TABLES
        ],
        "extraction_utc": datetime.now(timezone.utc).isoformat(),
        "date_coverage": {
            "requested_start": start.strftime("%Y-%m-%d"),
            "requested_end": end.strftime("%Y-%m-%d"),
            "actual_min": (
                min(item["actual_date_min"] for item in audits)
                if audits
                else None
            ),
            "actual_max": (
                max(item["actual_date_max"] for item in audits)
                if audits
                else None
            ),
        },
        "row_count": int(sum(item["rows"] for item in audits)),
        "symbol_count": len(symbols),
        "chunk_count": len(chunks),
        "chunk_frequency": frequency,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "expected_minute_count": EXPECTED_MINUTE_COUNT,
        "eligible_row_count": int(
            coverage["eligible_rows"].sum() if len(coverage) else 0
        ),
        "coverage_statistics": {
            "mean": float(
                np.dot(
                    np.arange(len(coverage_histogram)),
                    coverage_histogram,
                )
                / (
                    EXPECTED_MINUTE_COUNT
                    * max(int(coverage_histogram.sum()), 1)
                )
            ),
            "q01": _histogram_quantile(coverage_histogram, 0.01),
            "q10": _histogram_quantile(coverage_histogram, 0.10),
            "median": _histogram_quantile(coverage_histogram, 0.50),
            "q90": _histogram_quantile(coverage_histogram, 0.90),
        },
        "invalid_row_counts": {
            "low_coverage_symbol_days": int(
                quality["low_coverage_rows"].sum() if len(quality) else 0
            ),
            "hard_check_failures": int(
                quality[
                    [
                        "duplicate_keys",
                        "coverage_above_one",
                        "obi_outside_bounds",
                        "negative_spread_rows",
                        "microprice_outside_top_book_rows",
                        "hhi_outside_bounds",
                        "inf_values",
                    ]
                ].to_numpy().sum()
                if len(quality)
                else 0
            ),
            "missing_slope_symbol_days": int(
                quality["missing_slope_rows"].sum() if len(quality) else 0
            ),
            "missing_book_vwap_gap_symbol_days": int(
                quality["missing_book_vwap_gap_rows"].sum()
                if len(quality)
                else 0
            ),
            "raw_source_scope": (
                "Filtered server-side; Phase-0 sample source invalid counts "
                "are in sample_quality_audit.csv"
            ),
        },
        "code": {
            "git_head": _git_head(),
            "module": str(code_path.relative_to(PROJ_ROOT)),
            "module_sha256": _sha256(code_path),
        },
        "storage": {
            "format": "partitioned parquet",
            "compression": "zstd",
            "dataset_path": str(DATASET_DIR.relative_to(PROJ_ROOT)),
            "combined_parquet_written": False,
        },
        "disk_before": initial_disk,
        "disk_after": final_disk,
        "chunks": chunks,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    query_metadata = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "query_builder": (
            "l2_factor_reproduction.python.ch_order_book."
            "order_book_daily_sql"
        ),
        "minute_sampling": "argMax(metric, ExchTime)",
        "sessions": ["[09:30,11:30)", "[13:00,15:00)"],
        "close_auction": "[15:00,15:01) separate fields",
        "chunk_query_hashes": {
            f"{item['start']}_{item['end']}": item["query_sha256"]
            for item in chunks
        },
    }
    (OUT_DIR / "query_metadata.json").write_text(
        json.dumps(query_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument(
        "--chunk-frequency",
        choices=("quarter", "month"),
        default="quarter",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float, default=8.0)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if start > end:
        raise ValueError("start must not exceed end")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    initial_disk = _assert_disk(args.minimum_free_gb)
    print(
        "[disk] /home "
        f"used={initial_disk['used_fraction']:.1%} "
        f"free={initial_disk['free_bytes'] / 1024**3:.2f} GiB",
        flush=True,
    )

    chunk_metadata: List[Dict[str, object]] = []
    audits: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    all_symbols: set[str] = set()
    coverage_histogram = np.zeros(EXPECTED_MINUTE_COUNT + 1, dtype=np.int64)
    planned = list(_chunks(start, end, args.chunk_frequency))
    if args.max_chunks > 0:
        planned = planned[: args.max_chunks]
    for index, (chunk_start, chunk_end) in enumerate(planned, start=1):
        _assert_disk(args.minimum_free_gb)
        path = _chunk_path(chunk_start, chunk_end)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            print(
                f"[resume {index}/{len(planned)}] {path.name}",
                flush=True,
            )
            frame = pd.read_parquet(path)
        else:
            print(
                f"[query {index}/{len(planned)}] "
                f"{chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d}",
                flush=True,
            )
            frame = fetch_order_book_daily(chunk_start, chunk_end)
            if frame.empty:
                raise ValueError(
                    f"No primitive rows for {chunk_start.date()}.."
                    f"{chunk_end.date()}"
                )
            temporary = path.with_suffix(".parquet.tmp")
            frame.to_parquet(
                temporary,
                index=False,
                compression="zstd",
            )
            temporary.replace(path)
        audit = _audit_chunk(frame, chunk_start, chunk_end)
        _assert_hard_checks(audit)
        audits.append(audit)
        all_symbols.update(frame["symbol"].unique())
        minute_counts = frame["valid_minute_count"].clip(
            0, EXPECTED_MINUTE_COUNT
        ).astype(int)
        coverage_histogram += np.bincount(
            minute_counts,
            minlength=EXPECTED_MINUTE_COUNT + 1,
        )
        coverage_rows.extend(
            _coverage_rows(frame, chunk_start, chunk_end)
        )
        query_hash = _query_hashes(chunk_start, chunk_end)
        chunk_metadata.append(
            {
                "start": chunk_start.strftime("%Y-%m-%d"),
                "end": chunk_end.strftime("%Y-%m-%d"),
                "path": str(path.relative_to(OUT_DIR)),
                "rows": int(len(frame)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "query_sha256": query_hash,
            }
        )
        _write_metadata(
            start=start,
            end=end,
            frequency=args.chunk_frequency,
            chunks=chunk_metadata,
            audits=audits,
            coverage_rows=coverage_rows,
            initial_disk=initial_disk,
            symbols=all_symbols,
            coverage_histogram=coverage_histogram,
        )
        print(
            f"[ok] rows={len(frame):,} "
            f"coverage_mean={audit['coverage_mean']:.4f} "
            f"low_coverage={audit['low_coverage_rows']:,}",
            flush=True,
        )
        del frame

    print(
        f"[done] chunks={len(chunk_metadata)} "
        f"rows={sum(item['rows'] for item in audits):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
