#!/usr/bin/env python
"""Build the partitioned cancel_lifecycle_daily primitive (Sprint 6B).

Frozen formula/contract: primitives/cancel_lifecycle_daily/implementation_contract.json
Uses ch_cancel_lifecycle.fetch_cancel_daily day-by-day (CH timeout bound).

Chunks are quarter-partitioned parquet under dataset/year=YYYY/.
Resume-safe: existing quarterly chunks are skipped unless --overwrite.

Usage:
    python build_cancel_lifecycle_primitive.py \\
        --start 2019-01-01 --end 2026-07-31 --chunk-frequency quarter

Worker-friendly: pass disjoint --start/--end ranges to parallel workers
(contract: 4 workers; on KILLED drop to 2).
"""

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

from l2_factor_reproduction.python.ch_cancel_lifecycle import (  # noqa: E402
    PRIMITIVE_COLUMNS,
    PRIMITIVE_VERSION,
    fetch_cancel_daily,
)

OUT_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/cancel_lifecycle_daily"
)
DATASET_DIR = OUT_DIR / "dataset"
DAILY_CACHE = OUT_DIR / "daily"
DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")

INTEGER_COLS = [
    "buy_cancel_event_count", "sell_cancel_event_count",
    "buy_cancelled_unique_order_count",
    "sell_cancelled_unique_order_count",
    "total_trade_count", "zero_price_cancel_count",
    "market_order_price_fill_count", "invalid_cancel_count",
]


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _chunks(
    start: pd.Timestamp, end: pd.Timestamp, frequency: str,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start.normalize()
    end = end.normalize()
    while cursor <= end:
        if frequency == "quarter":
            boundary = cursor.to_period("Q").end_time.normalize()
        elif frequency == "month":
            boundary = cursor.to_period("M").end_time.normalize()
        elif frequency == "day":
            boundary = cursor
        else:
            raise ValueError(f"Unsupported chunk frequency: {frequency}")
        chunk_end = min(boundary, end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _chunk_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return (
        DATASET_DIR / f"year={start.year}"
        / f"cancel_daily_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
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
            f"{snapshot['free_bytes'] / 1024**3:.2f} GiB free, "
            f"requires {minimum_free_gb:.2f} GiB"
        )
    return snapshot


def _load_or_fetch_day(client, day: pd.Timestamp) -> pd.DataFrame:
    cache = DAILY_CACHE / f"cancel_daily_{day:%Y-%m-%d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frame = fetch_cancel_daily(client, day)
    if len(frame) == 0:
        return frame
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False, compression="zstd")
    return frame


def _audit_chunk(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
) -> Dict[str, object]:
    for column in INTEGER_COLS:
        if column in frame and frame[column].isna().any():
            raise ValueError(
                f"NA in integer column {column} for "
                f"{start.date()}..{end.date()}"
            )
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "chunk_start": start.strftime("%Y-%m-%d"),
        "chunk_end": end.strftime("%Y-%m-%d"),
        "actual_date_min": str(pd.to_datetime(frame["TradeDate"]).min().date()),
        "actual_date_max": str(pd.to_datetime(frame["TradeDate"]).max().date()),
        "rows": int(len(frame)),
        "dates": int(pd.to_datetime(frame["TradeDate"]).nunique()),
        "symbols": int(frame["symbol"].nunique()),
        "duplicate_keys": int(
            frame.duplicated(["symbol", "TradeDate"]).sum()
        ),
        "inf_values": int(
            np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()
        ),
        "join_coverage_median": float(frame["join_coverage"].median()),
        "invalid_cancel_total": int(frame["invalid_cancel_count"].sum()),
        "zero_price_cancel_total": int(
            frame["zero_price_cancel_count"].sum()
        ),
    }


def _assert_hard_checks(audit: Dict[str, object]) -> None:
    failures = {}
    if audit["duplicate_keys"] != 0:
        failures["duplicate_keys"] = audit["duplicate_keys"]
    if audit["inf_values"] != 0:
        failures["inf_values"] = audit["inf_values"]
    if audit["join_coverage_median"] < 0.99:
        failures["join_coverage_median"] = audit["join_coverage_median"]
    if failures:
        raise ValueError(
            f"cancel primitive chunk failed hard checks: {failures}"
        )


def _write_manifest(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    chunks: List[Dict[str, object]],
    audits: List[Dict[str, object]],
    initial_disk: Dict[str, object],
    symbols: set,
) -> None:
    code_path = (
        PROJ_ROOT / "l2_factor_reproduction/python/ch_cancel_lifecycle.py"
    )
    manifest = {
        "primitive_name": "l2_primitive_cancel_lifecycle_daily",
        "schema_version": PRIMITIVE_VERSION,
        "formula_version": PRIMITIVE_VERSION,
        "source_tables": [
            "cmds.SSE_AL_TICK_EXG", "cmds.SZSE_AL_TICK_EXG",
        ],
        "extraction_utc": datetime.now(timezone.utc).isoformat(),
        "date_coverage": {
            "requested_start": start.strftime("%Y-%m-%d"),
            "requested_end": end.strftime("%Y-%m-%d"),
            "actual_min": (
                min(item["actual_date_min"] for item in audits)
                if audits else None
            ),
            "actual_max": (
                max(item["actual_date_max"] for item in audits)
                if audits else None
            ),
        },
        "row_count": int(sum(item["rows"] for item in audits)),
        "symbol_count": len(symbols),
        "chunk_count": len(chunks),
        "chunk_frequency": frequency,
        "primitive_columns": PRIMITIVE_COLUMNS,
        "code": {
            "git_head": _git_head(),
            "module": str(code_path.relative_to(PROJ_ROOT)),
            "module_sha256": _sha256(code_path),
        },
        "storage": {
            "format": "partitioned parquet",
            "compression": "zstd",
            "dataset_path": str(DATASET_DIR.relative_to(PROJ_ROOT)),
        },
        "disk_before": initial_disk,
        "disk_after": _disk_snapshot(),
        "chunks": chunks,
    }
    # worker-scoped manifest (avoid races); final merge done by audit script
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    (OUT_DIR / f"manifest_worker_{tag}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(audits).to_csv(
        OUT_DIR / f"quality_audit_worker_{tag}.csv", index=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument(
        "--chunk-frequency", choices=("quarter", "month"), default="quarter",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float, default=8.0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    initial_disk = _assert_disk(args.minimum_free_gb)
    print(
        f"[disk] free={initial_disk['free_bytes'] / 1024**3:.1f} GiB",
        flush=True,
    )

    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client
    client = connect_hf_client()

    chunk_metadata: List[Dict[str, object]] = []
    audits: List[Dict[str, object]] = []
    all_symbols: set = set()
    planned = list(_chunks(start, end, args.chunk_frequency))

    for index, (chunk_start, chunk_end) in enumerate(planned, start=1):
        _assert_disk(args.minimum_free_gb)
        path = _chunk_path(chunk_start, chunk_end)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            print(f"[resume {index}/{len(planned)}] {path.name}", flush=True)
            frame = pd.read_parquet(path)
        else:
            print(
                f"[query {index}/{len(planned)}] "
                f"{chunk_start:%Y-%m-%d}..{chunk_end:%Y-%m-%d}",
                flush=True,
            )
            day_frames: List[pd.DataFrame] = []
            for d_start, _ in _chunks(chunk_start, chunk_end, "day"):
                day_frame = _load_or_fetch_day(client, d_start)
                if len(day_frame) == 0:
                    print(f"  [{d_start:%Y-%m-%d}] empty", flush=True)
                    continue
                day_frames.append(day_frame)
                print(
                    f"  [{d_start:%Y-%m-%d}] rows={len(day_frame):,}",
                    flush=True,
                )
            if not day_frames:
                print(
                    f"[skip] no trading days in "
                    f"{chunk_start.date()}..{chunk_end.date()}",
                    flush=True,
                )
                continue
            frame = pd.concat(day_frames, ignore_index=True)
            temporary = path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)

        audit = _audit_chunk(frame, chunk_start, chunk_end)
        _assert_hard_checks(audit)
        audits.append(audit)
        all_symbols.update(frame["symbol"].unique())
        chunk_metadata.append({
            "start": chunk_start.strftime("%Y-%m-%d"),
            "end": chunk_end.strftime("%Y-%m-%d"),
            "path": str(path.relative_to(OUT_DIR)),
            "rows": int(len(frame)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
        _write_manifest(
            start=start, end=end, frequency=args.chunk_frequency,
            chunks=chunk_metadata, audits=audits,
            initial_disk=initial_disk, symbols=all_symbols,
        )
        print(
            f"[ok] rows={len(frame):,} "
            f"join_cov={audit['join_coverage_median']:.4f} "
            f"invalid={audit['invalid_cancel_total']:,}",
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
