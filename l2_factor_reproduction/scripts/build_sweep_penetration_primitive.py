#!/usr/bin/env python
"""Build sweep_penetration_daily from ClickHouse Tick + SSL2 (Sprint 14B).

Conservative estimated sweep only. No alpha / discovery / FV.

Usage:
  /opt/conda/anaconda3/bin/python -m l2_factor_reproduction.scripts.build_sweep_penetration_primitive \\
      --start 2019-01-01 --end 2026-08-01
  /opt/conda/anaconda3/bin/python -m l2_factor_reproduction.scripts.build_sweep_penetration_primitive \\
      --start 2024-06-28 --end 2024-06-29 --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python import sweep_penetration_daily as spd  # noqa: E402
from l2_factor_reproduction.scripts.build_liquidity_impact_primitive import (  # noqa: E402
    quarter_ranges,
)

OUT_DIR = Path(RESULT_ROOT) / "primitives" / "sweep_penetration_daily"
DATASET_DIR = OUT_DIR / "dataset"


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _module_sha256() -> str:
    import inspect

    return hashlib.sha256(inspect.getsource(spd).encode()).hexdigest()


def _run_period(client, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    hashes: Dict[str, str] = {}
    for exchange in spd.EXCHANGES:
        sql = spd.daily_sql(exchange, start, end)
        hashes[exchange] = spd.query_sha256(sql)
        frame = client.query_df(sql)
        frames.append(frame)
        print(
            f"[build] {exchange} {start}..{end} rows={len(frame)}",
            flush=True,
        )
    daily = spd.finalize_daily(frames)
    return spd.prepare_sweep_penetration_daily(daily), hashes


def _quality_row(frame: pd.DataFrame, tag: str) -> Dict[str, object]:
    return {
        "partition": tag,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "actual_date_min": str(frame["TradeDate"].min().date()),
        "actual_date_max": str(frame["TradeDate"].max().date()),
        "mean_usable_events": float(frame["usable_event_count"].mean()),
        "mean_ambiguous_share": float(frame["ambiguous_event_share"].mean()),
        "mean_sweep_2plus_share": float(
            frame["sweep_2plus_share"].dropna().mean()
        )
        if frame["sweep_2plus_share"].notna().any()
        else float("nan"),
        "mean_median_lag_ms": float(
            frame["median_alignment_lag_ms"].dropna().mean()
        )
        if frame["median_alignment_lag_ms"].notna().any()
        else float("nan"),
        "zero_usable_share": float((frame["usable_event_count"] <= 0).mean()),
    }


def _write_partition(frame: pd.DataFrame, quarter: str) -> Dict[str, object]:
    directory = DATASET_DIR / f"quarter={quarter}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sweep_penetration_daily_{quarter}.parquet"
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return {
        "quarter": quarter,
        "path": str(path.relative_to(PROJ_ROOT)),
        "rows": int(len(frame)),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = connect_hf_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        daily, hashes = _run_period(client, args.start, args.end)
        path = OUT_DIR / f"smoke_{args.start}_{args.end}.parquet"
        daily.to_parquet(path, compression="zstd", index=False)
        print(daily.describe(include="all").to_string(), flush=True)
        print(
            json.dumps(
                {"rows": len(daily), "hashes": hashes, "quality": _quality_row(daily, "smoke")},
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0

    quality_rows: List[Dict[str, object]] = []
    partitions: List[Dict[str, object]] = []
    query_hashes: Dict[str, Dict[str, str]] = {}

    for quarter, start, end in quarter_ranges(args.start, args.end):
        if end > args.end:
            end = args.end
        part_file = (
            DATASET_DIR
            / f"quarter={quarter}"
            / f"sweep_penetration_daily_{quarter}.parquet"
        )
        if part_file.exists() and not args.force:
            print(f"[skip] {quarter}", flush=True)
            existing = pd.read_parquet(part_file)
            quality_rows.append(_quality_row(existing, quarter))
            partitions.append(
                {
                    "quarter": quarter,
                    "path": str(part_file.relative_to(PROJ_ROOT)),
                    "rows": int(len(existing)),
                    "sha256": _sha256(part_file),
                    "reused": True,
                }
            )
            continue
        daily, hashes = _run_period(client, start, end)
        query_hashes[quarter] = hashes
        quality_rows.append(_quality_row(daily, quarter))
        info = _write_partition(daily, quarter)
        info["query_sha256"] = hashes
        partitions.append(info)
        print(f"[done] {quarter} rows={len(daily)}", flush=True)

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT_DIR / "primitive_quality.csv", index=False)
    manifest = {
        "primitive_name": "l2_primitive_sweep_penetration_daily",
        "schema_version": spd.SCHEMA_VERSION,
        "formula_version": spd.FORMULA_VERSION,
        "canonical_source": spd.CANONICAL_SOURCE,
        "formulas": spd.PRIMITIVE_FORMULAS,
        "stale_lag_ms_threshold": spd.STALE_LAG_MS,
        "analysis_unit": "trade_print_vs_strictly_before_stale_snapshot",
        "date_coverage": {
            "requested_start": args.start,
            "requested_end": args.end,
            "actual_min": quality["actual_date_min"].min() if len(quality) else None,
            "actual_max": quality["actual_date_max"].max() if len(quality) else None,
        },
        "row_count": int(quality["rows"].sum()) if len(quality) else 0,
        "partition_checksums": partitions,
        "query_hashes": query_hashes,
        "module_sha256": _module_sha256(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "storage": {
            "format": "quarterly partitioned parquet",
            "compression": "zstd",
            "dataset_path": str(DATASET_DIR.relative_to(PROJ_ROOT)),
            "event_level_written": False,
        },
        "side_direction_contract": {
            "buy": "ASK ladder",
            "sell": "BID ladder",
            "inversion_forbidden": True,
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"[manifest] {OUT_DIR / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
