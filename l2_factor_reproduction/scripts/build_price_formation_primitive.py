#!/usr/bin/env python
"""Build partitioned l2_primitive_price_formation_daily from DolphinDB."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import UNIVERSE  # noqa: E402
from l2_factor_reproduction.python.price_formation_daily import (  # noqa: E402
    CANONICAL_SOURCE,
    COVERAGE_THRESHOLD,
    EXPECTED_CONTINUOUS_MINUTES,
    FORMULA_VERSION,
    MAX_CONSECUTIVE_PRICE_GAP,
    PRICE_FORMATION_DAILY_COLUMNS,
    SCHEMA_VERSION,
    close_auction_daily_sql,
    fetch_price_formation_daily,
    price_formation_daily_sql,
)
from l2_factor_reproduction.python.price_formation_factors import (  # noqa: E402
    registry_frame,
)


OUT_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/price_formation_daily"
)
DATASET_DIR = OUT_DIR / "dataset"
DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "pyarrow": _package_version("pyarrow"),
        "dolphindb": _package_version("dolphindb"),
    }


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
        elif frequency == "year":
            boundary = cursor.to_period("Y").end_time.normalize()
        elif frequency == "month":
            boundary = cursor.to_period("M").end_time.normalize()
        else:
            raise ValueError(f"unsupported chunk frequency: {frequency}")
        chunk_end = min(boundary, end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _chunk_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return DATASET_DIR / f"year={start.year}" / (
        f"price_formation_daily_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
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
            f"Insufficient /home space: "
            f"{snapshot['free_bytes'] / 1024**3:.2f} GiB free; "
            f"requires {minimum_free_gb:.2f} GiB"
        )
    return snapshot


def _outside(frame: pd.DataFrame, column: str, lower: float, upper: float) -> int:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int((~values.between(lower - 1e-10, upper + 1e-10)).sum())


def _audit_chunk(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, object]:
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "chunk_start": str(start.date()),
        "chunk_end": str(end.date()),
        "actual_date_min": str(frame["TradeDate"].min().date()),
        "actual_date_max": str(frame["TradeDate"].max().date()),
        "rows": int(len(frame)),
        "dates": int(frame["TradeDate"].nunique()),
        "symbols": int(frame["symbol"].nunique()),
        "duplicate_keys": int(
            frame.duplicated(["symbol", "TradeDate"]).sum()
        ),
        "inf_values": int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        ),
        "coverage_above_one": int((frame["coverage_ratio"] > 1).sum()),
        "coverage_below_zero": int((frame["coverage_ratio"] < 0).sum()),
        "low_coverage_rows": int(
            (frame["coverage_ratio"] < COVERAGE_THRESHOLD).sum()
        ),
        "zero_amount_rows": int((frame["daily_amount"] <= 0).sum()),
        "coverage_mean": float(frame["coverage_ratio"].mean()),
        "coverage_q01": float(frame["coverage_ratio"].quantile(0.01)),
        "coverage_q10": float(frame["coverage_ratio"].quantile(0.10)),
        "coverage_median": float(frame["coverage_ratio"].median()),
        "valid_minutes_mean": float(frame["valid_minute_count"].mean()),
        "imputed_minutes_mean": float(
            frame["imputed_price_minute_count"].mean()
        ),
        "imputed_minutes_max": int(
            frame["imputed_price_minute_count"].max()
        ),
        "path_efficiency_outside_bounds": _outside(
            frame, "path_efficiency", 0, 1
        ),
        "clv_outside_bounds": _outside(
            frame, "close_location_value", -1, 1
        ),
        "hhi_outside_bounds": _outside(
            frame, "volume_concentration_hhi", 0, 1
        ),
        "amount_time_center_outside_bounds": _outside(
            frame, "amount_time_center", 0, 1
        ),
        "negative_realized_variance": int(
            (frame["realized_variance"].dropna() < -1e-12).sum()
        ),
        "negative_daily_amount": int(
            (frame["daily_amount"].dropna() < -1e-12).sum()
        ),
        "close_auction_valid_rows": int(
            frame["close_auction_price"].notna().sum()
        ),
    }


def _assert_hard_checks(audit: Dict[str, object]) -> None:
    hard = (
        "duplicate_keys",
        "inf_values",
        "coverage_above_one",
        "coverage_below_zero",
        "path_efficiency_outside_bounds",
        "clv_outside_bounds",
        "hhi_outside_bounds",
        "amount_time_center_outside_bounds",
        "negative_realized_variance",
        "negative_daily_amount",
    )
    failures = {name: audit[name] for name in hard if audit[name] != 0}
    if failures:
        raise ValueError(
            f"Price Formation primitive failed hard checks: {failures}"
        )


def _coverage_rows(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Dict[str, object]]:
    rows = []
    for exchange, block in frame.groupby("source_exchange", observed=True):
        rows.append(
            {
                "chunk_start": str(start.date()),
                "chunk_end": str(end.date()),
                "source_exchange": exchange,
                "rows": int(len(block)),
                "dates": int(block["TradeDate"].nunique()),
                "symbols": int(block["symbol"].nunique()),
                "eligible_rows": int(
                    (
                        block["coverage_ratio"].ge(COVERAGE_THRESHOLD)
                        & block["daily_amount"].gt(0)
                    ).sum()
                ),
                "coverage_mean": float(block["coverage_ratio"].mean()),
                "coverage_q01": float(
                    block["coverage_ratio"].quantile(0.01)
                ),
                "coverage_q10": float(
                    block["coverage_ratio"].quantile(0.10)
                ),
                "coverage_median": float(
                    block["coverage_ratio"].median()
                ),
                "valid_minutes_mean": float(
                    block["valid_minute_count"].mean()
                ),
                "imputed_minutes_mean": float(
                    block["imputed_price_minute_count"].mean()
                ),
            }
        )
    return rows


def _update_previous_close(
    frame: pd.DataFrame,
    state: Dict[str, float],
) -> Dict[str, float]:
    updated = dict(state)
    ordered = frame.sort_values(["symbol", "TradeDate"], kind="stable")
    last_rows = ordered.groupby("symbol", sort=False).tail(1)
    for row in last_rows.itertuples(index=False):
        close = float(row.continuous_close)
        if np.isfinite(close) and close > 0:
            updated[str(row.symbol)] = close
    return updated


def _query_hashes(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, str]:
    return {
        "continuous": _text_sha256(price_formation_daily_sql(start, end)),
        "close_auction": _text_sha256(close_auction_daily_sql(start, end)),
    }


def _source_schema() -> List[Dict[str, object]]:
    path = OUT_DIR / "schema_ddb.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run audit_price_formation_sources.py first"
        )
    return pd.read_csv(path).to_dict("records")


def _lineage_hashes() -> Dict[str, object]:
    paths = {
        "build_script": Path(__file__).resolve(),
        "primitive_module": (
            PROJ_ROOT
            / "l2_factor_reproduction/python/price_formation_daily.py"
        ),
        "factor_formula_module": (
            PROJ_ROOT
            / "l2_factor_reproduction/python/price_formation_factors.py"
        ),
        "evaluation_module": (
            PROJ_ROOT / "l2_factor_reproduction/python/backtest.py"
        ),
        "source_audit_manifest": OUT_DIR / "source_audit_manifest.json",
    }
    output: Dict[str, object] = {}
    for name, path in paths.items():
        output[name] = {
            "path": str(path.relative_to(PROJ_ROOT)),
            "sha256": _sha256(path) if path.exists() else "unavailable",
        }
    registry_csv = registry_frame().to_csv(index=False)
    output["registry"] = {
        "factor_count": int(len(registry_frame())),
        "sha256": _text_sha256(registry_csv),
    }
    return output


def _write_metadata(
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    frequency: str,
    chunks: List[Dict[str, object]],
    audits: List[Dict[str, object]],
    coverage_rows: List[Dict[str, object]],
    initial_disk: Dict[str, object],
    symbols: Set[str],
) -> None:
    quality = pd.DataFrame(audits)
    coverage = pd.DataFrame(coverage_rows)
    quality.to_csv(OUT_DIR / "quality_audit.csv", index=False)
    coverage.to_csv(OUT_DIR / "primitive_coverage.csv", index=False)
    final_disk = _disk_snapshot()
    eligible = (
        int(coverage["eligible_rows"].sum()) if len(coverage) else 0
    )
    row_count = int(quality["rows"].sum()) if len(quality) else 0
    manifest = {
        "primitive_name": "l2_primitive_price_formation_daily",
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "canonical_source": CANONICAL_SOURCE,
        "source_schema": _source_schema(),
        "date_coverage": {
            "requested_start": str(requested_start.date()),
            "requested_end": str(requested_end.date()),
            "actual_min": (
                quality["actual_date_min"].min() if len(quality) else None
            ),
            "actual_max": (
                quality["actual_date_max"].max() if len(quality) else None
            ),
        },
        "row_count": row_count,
        "eligible_row_count": eligible,
        "symbol_count": len(symbols),
        "chunk_count": len(chunks),
        "chunk_frequency": frequency,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "expected_continuous_minutes": EXPECTED_CONTINUOUS_MINUTES,
        "max_consecutive_price_fill": MAX_CONSECUTIVE_PRICE_GAP,
        "coverage_statistics": {
            "mean": (
                float(
                    np.average(
                        quality["coverage_mean"],
                        weights=quality["rows"],
                    )
                )
                if len(quality)
                else np.nan
            ),
            "q01_by_chunk_min": (
                float(quality["coverage_q01"].min())
                if len(quality)
                else np.nan
            ),
            "median_by_chunk_median": (
                float(quality["coverage_median"].median())
                if len(quality)
                else np.nan
            ),
        },
        "invalid_row_counts": {
            "low_coverage_symbol_days": (
                int(quality["low_coverage_rows"].sum())
                if len(quality)
                else 0
            ),
            "zero_amount_symbol_days": (
                int(quality["zero_amount_rows"].sum())
                if len(quality)
                else 0
            ),
            "hard_check_failures": 0,
        },
        "lineage": {
            "git_head": _git_head(),
            **_lineage_hashes(),
        },
        "environment": _environment(),
        "storage": {
            "format": "partitioned parquet",
            "compression": "zstd",
            "dataset_path": str(DATASET_DIR.relative_to(PROJ_ROOT)),
            "combined_parquet_written": False,
            "raw_minute_panel_written": False,
        },
        "benchmark_definition": {
            "benchmark": UNIVERSE,
            "return": "benchmark-relative daily close-to-close",
            "investability_mask": "not_limit * not_ST * trade_status",
            "cost_bps": 7.5,
            "signal_shift": 1,
        },
        "direction_policy": {
            "raw_ic": "frozen formula direction",
            "effective_direction": "display grouping only",
            "production_direction": "not decided in Sprint 6",
        },
        "disk_before": initial_disk,
        "disk_after": final_disk,
        "partition_checksums": chunks,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    query_metadata = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "query_builder": (
            "l2_factor_reproduction.python.price_formation_daily."
            "price_formation_daily_sql"
        ),
        "canonical_source": CANONICAL_SOURCE,
        "sessions": ["[09:30,11:30)", "[13:00,15:00)"],
        "close_auction": "15:00 separate",
        "price_fill": (
            "price state only, in-session, max 3; amount/volume never filled"
        ),
        "realized_return_policy": "observed-to-observed transitions only",
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
        choices=("quarter", "year", "month"),
        default="quarter",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float, default=15.0)
    parser.add_argument("--max-chunks", type=int, default=0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if start > end:
        raise ValueError("start must not exceed end")
    if not (OUT_DIR / "source_audit_manifest.json").exists():
        raise FileNotFoundError(
            "Phase-0 source audit missing; run "
            "audit_price_formation_sources.py first"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    initial_disk = _assert_disk(args.minimum_free_gb)
    print(
        f"[disk] /home used={initial_disk['used_fraction']:.1%} "
        f"free={initial_disk['free_bytes'] / 1024**3:.2f} GiB",
        flush=True,
    )

    planned = list(_chunks(start, end, args.chunk_frequency))
    if args.max_chunks > 0:
        planned = planned[: args.max_chunks]
    chunks: List[Dict[str, object]] = []
    audits: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    symbols: Set[str] = set()
    previous_close: Dict[str, float] = {}

    for position, (chunk_start, chunk_end) in enumerate(planned, start=1):
        _assert_disk(args.minimum_free_gb)
        path = _chunk_path(chunk_start, chunk_end)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            print(
                f"[resume {position}/{len(planned)}] {path.name}",
                flush=True,
            )
            frame = pd.read_parquet(
                path, columns=list(PRICE_FORMATION_DAILY_COLUMNS)
            )
            previous_close = _update_previous_close(frame, previous_close)
        else:
            print(
                f"[query {position}/{len(planned)}] "
                f"{chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d}",
                flush=True,
            )
            frame, previous_close = fetch_price_formation_daily(
                chunk_start,
                chunk_end,
                previous_close=previous_close,
                return_state=True,
            )
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
        coverage_rows.extend(
            _coverage_rows(frame, chunk_start, chunk_end)
        )
        symbols.update(frame["symbol"].astype(str).unique())
        chunks.append(
            {
                "start": str(chunk_start.date()),
                "end": str(chunk_end.date()),
                "path": str(path.relative_to(OUT_DIR)),
                "rows": int(len(frame)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "query_sha256": _query_hashes(chunk_start, chunk_end),
            }
        )
        _write_metadata(
            requested_start=start,
            requested_end=end,
            frequency=args.chunk_frequency,
            chunks=chunks,
            audits=audits,
            coverage_rows=coverage_rows,
            initial_disk=initial_disk,
            symbols=symbols,
        )
        print(
            f"[ok] rows={len(frame):,} "
            f"coverage_mean={audit['coverage_mean']:.4f} "
            f"low_coverage={audit['low_coverage_rows']:,}",
            flush=True,
        )
        del frame

    print(
        f"[done] chunks={len(chunks)} "
        f"rows={sum(item['rows'] for item in audits):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
