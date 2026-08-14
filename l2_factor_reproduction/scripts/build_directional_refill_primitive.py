#!/usr/bin/env python
"""Build directional_refill_daily primitive from ClickHouse Tick + SSL2.

Does not rewrite frozen liquidity_impact_daily. Same sources / grid / direction
rules; packs bid/ask depth separately and exports side-conditioned refill.

Usage:
  /opt/conda/anaconda3/bin/python -m l2_factor_reproduction.scripts.build_directional_refill_primitive \\
      --start 2019-01-01 --end 2026-08-01
  # smoke one day:
  ... --start 2024-06-28 --end 2024-06-29 --smoke
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
from l2_factor_reproduction.python import directional_refill_daily as drd  # noqa: E402
from l2_factor_reproduction.scripts.build_liquidity_impact_primitive import (  # noqa: E402
    quarter_ranges,
)

OUT_DIR = Path(RESULT_ROOT) / "primitives" / "directional_refill_daily"
DATASET_DIR = OUT_DIR / "dataset"
LIQ_DS = (
    Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily" / "dataset"
)


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


def _module_sha256() -> str:
    import inspect

    return _text_sha256(inspect.getsource(drd))


def _environment() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "platform": platform.platform(),
    }


def _run_period(client, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    hashes: Dict[str, str] = {}
    for exchange in drd.EXCHANGES:
        sql = drd.daily_sql(exchange, start, end)
        hashes[exchange] = drd.query_sha256(sql)
        frame = client.query_df(sql)
        frames.append(frame)
        print(f"[build] {exchange} {start}..{end} rows={len(frame)}", flush=True)
    daily = drd.finalize_daily(frames)
    return drd.prepare_directional_refill_daily(daily), hashes


def _parity_vs_liquidity(new: pd.DataFrame, quarter: str) -> Dict[str, float]:
    """Compare depth_recovery_5m against frozen liquidity_impact cache."""
    path = (
        LIQ_DS
        / f"quarter={quarter}"
        / f"liquidity_impact_daily_{quarter}.parquet"
    )
    if not path.exists():
        return {"checked": 0.0}
    old = pd.read_parquet(
        path, columns=["symbol", "TradeDate", "depth_recovery_5m"]
    )
    old["TradeDate"] = pd.to_datetime(old["TradeDate"])
    merged = new.merge(
        old,
        on=["symbol", "TradeDate"],
        how="inner",
        suffixes=("_new", "_old"),
    )
    if merged.empty:
        return {"checked": 0.0}
    a = merged["depth_recovery_5m_new"].astype(float)
    b = merged["depth_recovery_5m_old"].astype(float)
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return {"checked": 0.0}
    diff = (a[mask] - b[mask]).abs()
    return {
        "checked": float(mask.sum()),
        "median_abs_diff": float(diff.median()),
        "p95_abs_diff": float(diff.quantile(0.95)),
        "max_abs_diff": float(diff.max()),
        "corr": float(a[mask].corr(b[mask])),
    }


def _quality_row(frame: pd.DataFrame, tag: str) -> Dict[str, object]:
    return {
        "partition": tag,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "actual_date_min": str(frame["TradeDate"].min().date()),
        "actual_date_max": str(frame["TradeDate"].max().date()),
        "coverage_mean": float(frame["coverage_ratio"].mean()),
        "coverage_median": float(frame["coverage_ratio"].median()),
        "bid_recovery_na_share": float(frame["bid_recovery_5m"].isna().mean()),
        "ask_recovery_na_share": float(frame["ask_recovery_5m"].isna().mean()),
        "asym_na_share": float(
            frame["directional_refill_asymmetry"].isna().mean()
        ),
        "mean_sell_shock_events": float(frame["sell_shock_event_count"].mean()),
        "mean_buy_shock_events": float(frame["buy_shock_event_count"].mean()),
    }


def _write_partition(frame: pd.DataFrame, quarter: str) -> Dict[str, object]:
    directory = DATASET_DIR / f"quarter={quarter}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"directional_refill_daily_{quarter}.parquet"
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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="write a smoke parquet instead of quarterly partitions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if quarter parquet exists",
    )
    args = parser.parse_args()

    client = connect_hf_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        daily, hashes = _run_period(client, args.start, args.end)
        smoke_path = OUT_DIR / f"smoke_{args.start}_{args.end}.parquet"
        daily.to_parquet(smoke_path, compression="zstd", index=False)
        print(daily.describe(include="all").to_string(), flush=True)
        print(
            json.dumps(
                {
                    "rows": len(daily),
                    "hashes": hashes,
                    "quality": _quality_row(daily, "smoke"),
                    "path": str(smoke_path),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0

    quality_rows: List[Dict[str, object]] = []
    partitions: List[Dict[str, object]] = []
    parity_rows: List[Dict[str, object]] = []
    query_hashes: Dict[str, Dict[str, str]] = {}

    for quarter, start, end in quarter_ranges(args.start, args.end):
        # Clamp end of last quarter to args.end
        end = min(end, args.end) if end > args.end else end
        part_dir = DATASET_DIR / f"quarter={quarter}"
        part_file = part_dir / f"directional_refill_daily_{quarter}.parquet"
        if part_file.exists() and not args.force:
            print(f"[skip] {quarter} exists", flush=True)
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
        parity = _parity_vs_liquidity(daily, quarter)
        parity["quarter"] = quarter
        parity_rows.append(parity)
        info = _write_partition(daily, quarter)
        info["query_sha256"] = hashes
        partitions.append(info)
        print(
            f"[done] {quarter} rows={len(daily)} parity={parity}",
            flush=True,
        )

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT_DIR / "primitive_quality.csv", index=False)
    if parity_rows:
        pd.DataFrame(parity_rows).to_csv(
            OUT_DIR / "depth_recovery_parity.csv", index=False
        )

    actual_min = quality["actual_date_min"].min() if len(quality) else None
    actual_max = quality["actual_date_max"].max() if len(quality) else None
    manifest = {
        "primitive_name": "l2_primitive_directional_refill_daily",
        "schema_version": drd.SCHEMA_VERSION,
        "formula_version": drd.FORMULA_VERSION,
        "canonical_source": drd.CANONICAL_SOURCE,
        "formulas": drd.PRIMITIVE_FORMULAS,
        "date_coverage": {
            "requested_start": args.start,
            "requested_end": args.end,
            "actual_min": actual_min,
            "actual_max": actual_max,
        },
        "row_count": int(quality["rows"].sum()) if len(quality) else 0,
        "partition_checksums": partitions,
        "query_hashes": query_hashes,
        "module_sha256": _module_sha256(),
        "environment": _environment(),
        "storage": {
            "format": "quarterly partitioned parquet",
            "compression": "zstd",
            "dataset_path": str(DATASET_DIR.relative_to(PROJ_ROOT)),
        },
        "side_direction_contract": {
            "sell_shock": "measures BID recovery",
            "buy_shock": "measures ASK recovery",
            "asymmetry": "bid_recovery_5m - ask_recovery_5m",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"[manifest] wrote {OUT_DIR / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
