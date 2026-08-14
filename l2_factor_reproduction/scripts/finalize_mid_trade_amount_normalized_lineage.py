#!/usr/bin/env python3
"""Enrich normalized dynamic-cache metadata with exact SQL and runtime lineage."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2_factor_reproduction.python import (
    ch_mid_trade_amount_normalization as clickhouse_module,
)
from l2_factor_reproduction.scripts import (
    build_mid_trade_amount_normalized_cache as cache_builder,
)

DEFAULT_CACHE_ROOT = (
    ROOT / "research/results/l2_reproduction/mid_order_ratio/normalized_v1"
)
DEFAULT_REPORT_ROOT = (
    ROOT / "research/reports/factors/mid_order_ratio/normalized_v1"
)

STRICT_FILTERS = {
    "regular_session": "09:30:00 <= ExchTime < 15:00:01",
    "SSE": (
        "Type='T'; amount=ifNull(Amount, Price*Volume); "
        "positive RMB amount; startsWith(Symbol,'6')"
    ),
    "SZSE": (
        "Type='011' AND BidOrderNo>0 AND AskOrderNo>0; "
        "amount=Price*Volume; positive RMB amount; "
        "A-share prefixes 000/001/002/003/300/301/302"
    ),
}


class LineageError(RuntimeError):
    """Raised when persisted SQL lineage cannot be verified."""


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _materialize_frozen_sources(
    cache_root: Path, report_root: Path
) -> Dict[str, Any]:
    """Restore the scale and calibration files referenced by frozen_config."""
    scales_path = cache_root / "scales.parquet"
    dynamic_path = cache_root / "dynamic_aggregates.parquet"
    if not scales_path.is_file() or not dynamic_path.is_file():
        raise LineageError("scales or dynamic aggregates are missing")
    scale_columns = [
        "symbol",
        "TradeDate",
        "total_amount",
        "ADV20_lag1",
        "ADV20_median_lag1",
        "ADV20_history_count",
        "ADV20_source_max_date",
    ]
    scales = pd.read_parquet(scales_path, columns=scale_columns)
    scales["TradeDate"] = pd.to_datetime(scales["TradeDate"]).dt.normalize()
    adv_path = report_root / "artifacts/adv20_lag1.parquet"
    _atomic_parquet(scales, adv_path)
    valid_adv = (
        pd.to_numeric(scales["ADV20_lag1"], errors="coerce").notna()
        & pd.to_numeric(scales["ADV20_lag1"], errors="coerce").gt(0)
    )
    valid_source = pd.to_datetime(
        scales.loc[valid_adv, "ADV20_source_max_date"], errors="coerce"
    )
    valid_dates = scales.loc[valid_adv, "TradeDate"]
    if not valid_source.lt(valid_dates).all():
        raise LineageError("restored ADV20 source-date gate failed")
    adv_metadata = {
        "role": "strict_ADV20_lag1_scale",
        "artifact": str(adv_path.resolve()),
        "artifact_sha256": cache_builder._sha256_file(adv_path),
        "source": str(scales_path.resolve()),
        "source_sha256": cache_builder._sha256_file(scales_path),
        "window": 20,
        "lag": 1,
        "calendar": "complete observed SSE/SZSE market trading dates",
        "missing_policy": "require all 20 prior market dates; no backfill",
        "rows": int(len(scales)),
        "valid_rows": int(valid_adv.sum()),
        "symbols": int(scales["symbol"].nunique()),
        "observed_start": str(scales["TradeDate"].min().date()),
        "observed_end": str(scales["TradeDate"].max().date()),
        "max_source_date_le_t_minus_1": True,
    }
    adv_metadata_path = report_root / "artifacts/adv20_lag1.metadata.json"
    adv_metadata_path.write_text(
        json.dumps(adv_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    calibration_start = pd.Timestamp("2023-01-03")
    calibration_end = pd.Timestamp("2023-06-30")
    a1_columns = [
        clickhouse_module.a1_selected_amount_column(lower, upper)
        for lower, upper in clickhouse_module.DEFAULT_A1_GRID
    ]
    dynamic_columns = [
        "symbol",
        "TradeDate",
        "total_amount",
        "a0_abs_4w20w_selected_amount",
        *a1_columns,
    ]
    calibration = pd.read_parquet(
        dynamic_path,
        columns=dynamic_columns,
        filters=[
            ("TradeDate", ">=", calibration_start),
            ("TradeDate", "<=", calibration_end),
        ],
    )
    calibration["TradeDate"] = pd.to_datetime(
        calibration["TradeDate"]
    ).dt.normalize()
    valid_keys = scales.loc[
        valid_adv
        & scales["TradeDate"].between(calibration_start, calibration_end),
        ["symbol", "TradeDate"],
    ]
    calibration = calibration.merge(
        valid_keys,
        on=["symbol", "TradeDate"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["TradeDate", "symbol"], kind="stable")
    calibration_path = (
        cache_root
        / "calibration_adv_grid/calibration_adv_grid_daily.parquet"
    )
    calibration_manifest_path = cache_root / "calibration_adv_grid/manifest.json"
    existing_manifest: Dict[str, Any] = {}
    if calibration_manifest_path.is_file():
        existing_manifest = json.loads(
            calibration_manifest_path.read_text(encoding="utf-8")
        )
    expected_rows = existing_manifest.get("rows")
    if expected_rows is not None and int(expected_rows) != len(calibration):
        raise LineageError(
            "restored calibration row count differs from frozen manifest: "
            f"{len(calibration)} != {expected_rows}"
        )
    _atomic_parquet(calibration, calibration_path)
    existing_manifest.update(
        {
            "output": str(calibration_path.resolve()),
            "artifact_sha256": cache_builder._sha256_file(calibration_path),
            "rows": int(len(calibration)),
            "symbols": int(calibration["symbol"].nunique()),
            "restored_from_dynamic_aggregate": str(dynamic_path.resolve()),
            "restored_from_dynamic_sha256": cache_builder._sha256_file(
                dynamic_path
            ),
            "valid_ADV20_scale_source": str(adv_path.resolve()),
            "valid_ADV20_scale_sha256": cache_builder._sha256_file(adv_path),
        }
    )
    calibration_manifest_path.write_text(
        json.dumps(existing_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "adv_scale": str(adv_path.resolve()),
        "adv_scale_sha256": adv_metadata["artifact_sha256"],
        "adv_scale_rows": int(len(scales)),
        "adv_valid_rows": int(valid_adv.sum()),
        "calibration_grid": str(calibration_path.resolve()),
        "calibration_grid_sha256": existing_manifest["artifact_sha256"],
        "calibration_grid_rows": int(len(calibration)),
    }


def _clickhouse_runtime() -> Dict[str, Any]:
    try:
        client_version = importlib.metadata.version("clickhouse-connect")
    except importlib.metadata.PackageNotFoundError:
        client_version = "unknown"
    client = clickhouse_module._get_ch_client()
    try:
        server_version = getattr(client, "server_version", None)
        if not server_version:
            result = client.query("SELECT version()")
            server_version = result.result_rows[0][0]
    finally:
        client.close()
    return {
        "client": "clickhouse-connect",
        "client_version": str(client_version),
        "server_version": str(server_version),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    payload = cache_builder._load_verified_metadata(path)
    if not isinstance(payload, dict):
        raise LineageError(f"metadata is not an object: {path}")
    return payload


def _render_example(
    report_root: Path,
    runtime: Dict[str, Any],
    first_record: Dict[str, Any],
) -> Path:
    output = report_root / "appendix/dynamic_factor_sql_example.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    sql = first_record["sql_text_by_exchange"]
    text = f"""# Dynamic Factor SQL Evidence

This is the exact first monthly query pair from the persisted dynamic cache.
Every monthly chunk metadata file stores its own SSE and SZSE SQL text and
SHA256. The execution used `clickhouse-connect`
`{runtime['client_version']}` against ClickHouse
`{runtime['server_version']}` with `CSVWithNames` ExternalData.

Chunk: `{first_record['chunk_start']}` to `{first_record['chunk_end']}`.

## SSE

```sql
{sql['SSE'].strip()}
```

## SZSE

```sql
{sql['SZSE'].strip()}
```
"""
    output.write_text(text, encoding="utf-8")
    return output


def finalize_lineage(
    cache_root: Path = DEFAULT_CACHE_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> Dict[str, Any]:
    runtime = _clickhouse_runtime()
    frozen_sources = _materialize_frozen_sources(cache_root, report_root)
    chunk_root = cache_root / "dynamic_aggregates/chunks"
    metadata_paths = sorted(chunk_root.glob("chunk_*.metadata.json"))
    if not metadata_paths:
        raise LineageError(f"no dynamic chunk metadata found under {chunk_root}")

    records: List[Dict[str, Any]] = []
    for metadata_path in metadata_paths:
        metadata = _load_json(metadata_path)
        artifact_path = Path(str(metadata["artifact"]))
        if not artifact_path.is_file():
            raise LineageError(f"chunk artifact is missing: {artifact_path}")
        if cache_builder._sha256_file(artifact_path) != metadata["artifact_sha256"]:
            raise LineageError(f"chunk artifact hash mismatch: {artifact_path}")
        start = pd.Timestamp(metadata["chunk_start"])
        end = pd.Timestamp(metadata["chunk_end"])
        queries = clickhouse_module.build_dynamic_factor_queries(
            start,
            end,
            a1_grid=clickhouse_module.DEFAULT_A1_GRID,
            a2_grid=clickhouse_module.DEFAULT_A2_GRID,
        )
        sql_sha256 = cache_builder._sql_sha256(queries)
        if sql_sha256 != metadata.get("sql_sha256"):
            raise LineageError(
                f"SQL hash mismatch for {metadata_path}: "
                f"{sql_sha256} != {metadata.get('sql_sha256')}"
            )
        metadata.update(
            {
                "sql_text_by_exchange": queries,
                "strict_filters": STRICT_FILTERS,
                "symbol_scope": "SSE/SZSE listed A-shares; BSE excluded",
                "amount_definition": "RMB traded amount, not volume",
                "external_data": {
                    "table": clickhouse_module.SCALE_ROWS_EXTERNAL_TABLE,
                    "format": "CSVWithNames",
                    "join_keys": ["Symbol=symbol", "toDate(ExchTime)=TradeDate"],
                    "raw_ticks_transferred_to_python": False,
                },
                "clickhouse_runtime": runtime,
            }
        )
        written = cache_builder._write_metadata(metadata_path, metadata)
        records.append(
            {
                "metadata": str(metadata_path.resolve()),
                "metadata_sha256": written["metadata_sha256"],
                "artifact": str(artifact_path.resolve()),
                "artifact_sha256": written["artifact_sha256"],
                "chunk_start": written["chunk_start"],
                "chunk_end": written["chunk_end"],
                "rows": int(written["rows"]),
                "sql_sha256": written["sql_sha256"],
                "sql_text_by_exchange": queries,
            }
        )

    aggregate_metadata_path = cache_root / "dynamic_aggregates.metadata.json"
    aggregate_metadata = _load_json(aggregate_metadata_path)
    aggregate_metadata.update(
        {
            "strict_filters": STRICT_FILTERS,
            "symbol_scope": "SSE/SZSE listed A-shares; BSE excluded",
            "amount_definition": "RMB traded amount, not volume",
            "clickhouse_runtime": runtime,
            "external_data_format": "CSVWithNames",
            "raw_ticks_transferred_to_python": False,
            "sql_lineage": [
                {
                    key: record[key]
                    for key in (
                        "metadata",
                        "metadata_sha256",
                        "artifact",
                        "artifact_sha256",
                        "chunk_start",
                        "chunk_end",
                        "rows",
                        "sql_sha256",
                    )
                }
                for record in records
            ],
        }
    )
    aggregate_metadata = cache_builder._write_metadata(
        aggregate_metadata_path, aggregate_metadata
    )

    lineage_path = report_root / "appendix/dynamic_sql_lineage.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_payload = {
        "version": "mid_trade_amount_dynamic_sql_lineage_v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "cache_root": str(cache_root.resolve()),
        "runtime": runtime,
        "frozen_sources": frozen_sources,
        "strict_filters": STRICT_FILTERS,
        "chunk_count": len(records),
        "chunks": records,
        "aggregate_metadata": str(aggregate_metadata_path.resolve()),
        "aggregate_metadata_sha256": aggregate_metadata["metadata_sha256"],
    }
    lineage_path.write_text(
        json.dumps(lineage_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    example_path = _render_example(report_root, runtime, records[0])
    return {
        "chunk_count": len(records),
        "runtime": runtime,
        "frozen_sources": frozen_sources,
        "aggregate_metadata_sha256": aggregate_metadata["metadata_sha256"],
        "lineage_path": str(lineage_path.resolve()),
        "sql_example_path": str(example_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize_lineage(args.cache_root.resolve(), args.report_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LineageError as exc:
        raise SystemExit(f"ERROR: {exc}")
