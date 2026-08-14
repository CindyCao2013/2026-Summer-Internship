#!/usr/bin/env python3
"""Append a new complete month to the frozen normalized factor cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2_factor_reproduction.python import (
    ch_mid_trade_amount_normalization as ch_normalization,
)
from l2_factor_reproduction.python.mid_trade_amount_normalization import (
    A0_FACTOR_ID,
    A1_FACTOR_ID,
    A2_FACTOR_ID,
    A3_FACTOR_ID,
    build_lagged_trade_size_scales,
    validate_frozen_config,
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
FROZEN_CONFIG_SHA_FILE = "frozen_config.sha256"


class ExtensionError(RuntimeError):
    """Raised when an append-only cache extension fails a hard gate."""


def _json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExtensionError(f"expected JSON object: {path}")
    return payload


def _scale_median_column(scales: pd.DataFrame) -> str:
    candidates = [
        column
        for column in (
            "daily_median_trade_amount_y",
            "daily_median_trade_amount",
            "q50",
            "daily_median_trade_amount_x",
        )
        if column in scales.columns
    ]
    if not candidates:
        raise ExtensionError("existing scales lack a daily median source")
    if (
        "daily_median_trade_amount_x" in scales.columns
        and "daily_median_trade_amount_y" in scales.columns
    ):
        left = pd.to_numeric(
            scales["daily_median_trade_amount_x"], errors="coerce"
        )
        right = pd.to_numeric(
            scales["daily_median_trade_amount_y"], errors="coerce"
        )
        paired = left.notna() & right.notna()
        if paired.any() and not np.allclose(
            left.loc[paired],
            right.loc[paired],
            rtol=0.0,
            atol=1e-10,
        ):
            raise ExtensionError("existing daily median evidence is inconsistent")
    return candidates[0]


def _compare_overlap(
    existing: pd.DataFrame,
    rebuilt: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, float]:
    columns = ["ADV20_lag1", "ADV20_median_lag1", "ATS20_lag1"]
    left = existing.loc[
        existing["TradeDate"].between(start, end),
        ["symbol", "TradeDate", *columns],
    ]
    right = rebuilt.loc[
        rebuilt["TradeDate"].between(start, end),
        ["symbol", "TradeDate", *columns],
    ]
    merged = left.merge(
        right,
        on=["symbol", "TradeDate"],
        how="inner",
        validate="one_to_one",
        suffixes=("_old", "_new"),
    )
    if merged.empty:
        raise ExtensionError("scale overlap check has no rows")
    audit: Dict[str, float] = {"overlap_rows": float(len(merged))}
    for column in columns:
        old = merged[f"{column}_old"]
        new = merged[f"{column}_new"]
        if not old.isna().equals(new.isna()):
            raise ExtensionError(f"{column} overlap NaN pattern changed")
        valid = old.notna()
        difference = (old.loc[valid] - new.loc[valid]).abs()
        relative = difference / old.loc[valid].abs().clip(lower=1.0)
        maximum = float(relative.max()) if len(relative) else 0.0
        audit[f"{column}_max_relative_error"] = maximum
        if maximum > 1e-12:
            raise ExtensionError(
                f"{column} overlap changed: max relative error={maximum}"
            )
    return audit


def _build_extended_scales(
    existing: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple:
    existing = existing.copy()
    existing["TradeDate"] = pd.to_datetime(
        existing["TradeDate"]
    ).dt.normalize()
    existing_end = pd.Timestamp(existing["TradeDate"].max())
    if existing_end >= end:
        extension = existing.loc[
            existing["TradeDate"].between(start, end)
        ].copy()
        return existing, extension, {"scale_rows_reused": float(len(extension))}

    warmup_start = start - pd.Timedelta(days=75)
    primitive = ch_normalization.fetch_daily_scale_primitive(
        warmup_start, end
    )
    if primitive.empty:
        raise ExtensionError("ClickHouse extension primitive is empty")
    primitive["TradeDate"] = pd.to_datetime(
        primitive["TradeDate"]
    ).dt.normalize()
    calendar = pd.DatetimeIndex(
        sorted(
            set(
                existing.loc[
                    existing["TradeDate"].ge(warmup_start), "TradeDate"
                ].tolist()
            )
            | set(primitive["TradeDate"].tolist())
        ),
        name="TradeDate",
    )
    symbols = sorted(
        set(existing["symbol"].astype(str))
        | set(primitive["symbol"].astype(str))
    )
    source = primitive[
        ["symbol", "TradeDate", "total_amount", "q50"]
    ].rename(columns={"q50": "daily_median_trade_amount"})
    missing_symbols = sorted(set(symbols) - set(source["symbol"].astype(str)))
    if missing_symbols:
        placeholders = pd.DataFrame(
            {
                "symbol": missing_symbols,
                "TradeDate": calendar[0],
                "total_amount": np.nan,
                "daily_median_trade_amount": np.nan,
            }
        )
        source = pd.concat([source, placeholders], ignore_index=True)
    rebuilt = build_lagged_trade_size_scales(
        source,
        calendar,
        symbol_col="symbol",
        date_col="TradeDate",
        total_amount_col="total_amount",
        daily_median_col="daily_median_trade_amount",
    )
    overlap_start = max(
        warmup_start + pd.Timedelta(days=45),
        start - pd.Timedelta(days=10),
    )
    audit = _compare_overlap(
        existing,
        rebuilt,
        overlap_start,
        min(existing_end, end),
    )

    extension_dates = calendar[
        (calendar > existing_end) & (calendar <= end)
    ]
    if len(extension_dates) == 0:
        raise ExtensionError("no new market dates were found")
    lag_extension = rebuilt.loc[
        rebuilt["TradeDate"].isin(extension_dates)
    ].rename(
        columns={
            "daily_median_trade_amount": "daily_median_trade_amount_x"
        }
    )
    primitive_extra = primitive.drop(columns=["total_amount"]).rename(
        columns={"q50": "daily_median_trade_amount_y"}
    )
    extension = lag_extension.merge(
        primitive_extra,
        on=["symbol", "TradeDate"],
        how="left",
        validate="one_to_one",
    )
    expected_rows = len(symbols) * len(extension_dates)
    if len(extension) != expected_rows:
        raise ExtensionError(
            f"extension scale grid is incomplete: {len(extension)} != {expected_rows}"
        )
    for column in existing.columns:
        if column not in extension.columns:
            extension[column] = np.nan
    extension = extension.loc[:, existing.columns]
    full = pd.concat([existing, extension], ignore_index=True)
    if full.duplicated(["symbol", "TradeDate"]).any():
        raise ExtensionError("extended scales contain duplicate keys")
    audit.update(
        {
            "primitive_rows": float(len(primitive)),
            "extension_calendar_days": float(len(extension_dates)),
            "extension_rows": float(len(extension)),
            "full_rows": float(len(full)),
        }
    )
    return full, extension, audit


def _backup_once(path: Path, end: pd.Timestamp) -> Path:
    backup = path.with_name(
        f"{path.stem}_through_{end.strftime('%Y-%m-%d')}{path.suffix}"
    )
    if not backup.exists():
        os.link(str(path), str(backup))
    return backup


def _replace_hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    os.link(str(source), str(target))


def _restore_stage_a_artifacts(
    full_scales: pd.DataFrame,
    scales_path: Path,
    cache_root: Path,
    report_root: Path,
) -> Dict[str, Any]:
    """Restore the deleted warmup primitive and report scale hardlink."""
    median_column = _scale_median_column(full_scales)
    primitive_columns = [
        "symbol",
        "TradeDate",
        "total_amount",
        "positive_trade_count",
        "daily_mean_trade_amount",
        "q20",
        "q30",
        median_column,
        "q70",
        "q80",
    ]
    research_primitive = full_scales.loc[
        full_scales["total_amount"].notna(), primitive_columns
    ].rename(columns={median_column: "q50"})
    warmup_primitive = ch_normalization.fetch_daily_scale_primitive(
        pd.Timestamp("2022-11-01"),
        pd.Timestamp("2022-12-31"),
    )
    primitive = ch_normalization.audit_daily_scale_result(
        pd.concat(
            [warmup_primitive, research_primitive],
            ignore_index=True,
        )
    )
    stage_a_manifest_path = cache_root / "daily_trade_size_scale/manifest.json"
    stage_a_manifest = _json(stage_a_manifest_path)
    expected_rows = int(stage_a_manifest["rows"])
    if len(primitive) != expected_rows:
        raise ExtensionError(
            "restored Stage-A primitive row count differs from manifest: "
            f"{len(primitive)} != {expected_rows}"
        )
    if (
        pd.Timestamp(primitive["TradeDate"].min())
        != pd.Timestamp(stage_a_manifest["observed_start"])
        or pd.Timestamp(primitive["TradeDate"].max())
        != pd.Timestamp(stage_a_manifest["observed_end"])
    ):
        raise ExtensionError("restored Stage-A primitive date range changed")

    primitive_path = cache_root / cache_builder.PRIMITIVE_FILE
    primitive_metadata = cache_builder._write_dataframe_artifact(
        primitive,
        primitive_path,
        cache_root / cache_builder.PRIMITIVE_METADATA_FILE,
        role="strict_daily_trade_size_primitive_restored",
        details={
            "requested_warmup_start": "2022-11-01",
            "requested_end": str(primitive["TradeDate"].max().date()),
            "rows": int(len(primitive)),
            "symbols": int(primitive["symbol"].nunique()),
            "restoration": (
                "2022-11/12 re-fetched from ClickHouse; research-period "
                "primitive recovered losslessly from persisted Stage-A scales"
            ),
        },
    )
    nested_primitive = (
        cache_root / "daily_trade_size_scale/daily_trade_size_primitive.parquet"
    )
    _replace_hardlink(primitive_path, nested_primitive)
    report_scale = report_root / "artifacts/daily_trade_size_scale.parquet"
    _replace_hardlink(scales_path, report_scale)
    stage_a_manifest.update(
        {
            "primitive": str(nested_primitive.resolve()),
            "primitive_sha256": primitive_metadata["artifact_sha256"],
            "scale_output": str(report_scale.resolve()),
            "scale_output_sha256": cache_builder._sha256_file(report_scale),
            "rows": int(len(primitive)),
            "scale_rows": int(len(full_scales)),
            "observed_scale_rows": int(full_scales["total_amount"].notna().sum()),
            "max_source_date_le_t_minus_1": True,
        }
    )
    stage_a_manifest_path.write_text(
        json.dumps(stage_a_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    scale_manifest_path = cache_root / "scale_manifest.json"
    scale_manifest = _json(scale_manifest_path)
    scale_manifest.update(
        {
            "primitive_rows": int(len(primitive)),
            "scale_rows": int(len(full_scales)),
            "observed_scale_rows": int(
                full_scales["total_amount"].notna().sum()
            ),
            "primitive_sha256": primitive_metadata["artifact_sha256"],
            "scales_sha256": cache_builder._sha256_file(scales_path),
            "scale_start": str(full_scales["TradeDate"].min().date()),
            "scale_end": str(full_scales["TradeDate"].max().date()),
            "source_max_date_strict": True,
        }
    )
    scale_manifest_path.write_text(
        json.dumps(scale_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_scale_manifest = report_root / "artifacts/scale_manifest.json"
    report_scale_manifest.write_text(
        json.dumps(scale_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "primitive": str(primitive_path.resolve()),
        "primitive_sha256": primitive_metadata["artifact_sha256"],
        "primitive_rows": int(len(primitive)),
        "report_scale": str(report_scale.resolve()),
        "report_scale_sha256": cache_builder._sha256_file(report_scale),
    }


def extend_cache(
    cache_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    workers: int = 2,
) -> Dict[str, Any]:
    if start > end:
        raise ExtensionError("extension start is after end")
    config_path = cache_root / cache_builder.FROZEN_CONFIG_FILE
    config = _json(config_path)
    pin = (cache_root / FROZEN_CONFIG_SHA_FILE).read_text(
        encoding="utf-8"
    ).strip()
    config_sha = validate_frozen_config(config, expected_sha256=pin)

    scales_path = cache_root / cache_builder.SCALES_FILE
    scales_metadata_path = cache_root / cache_builder.SCALES_METADATA_FILE
    existing_scales, _ = cache_builder._read_verified_parquet(
        scales_path, scales_metadata_path
    )
    existing_scales = cache_builder._canonicalize_scales(existing_scales)
    previous_end = pd.Timestamp(existing_scales["TradeDate"].max())
    backup = _backup_once(scales_path, previous_end)
    full_scales, extension_scales, scale_audit = _build_extended_scales(
        existing_scales, start, end
    )
    full_scales = cache_builder._canonicalize_scales(full_scales)
    frozen_scale_manifest_path = cache_root / "scale_manifest.json"
    if frozen_scale_manifest_path.is_file():
        frozen_scale_manifest = _json(frozen_scale_manifest_path)
        if pd.Timestamp(frozen_scale_manifest.get("scale_end")) == end:
            expected_rows = int(frozen_scale_manifest["scale_rows"])
            if len(full_scales) != expected_rows:
                raise ExtensionError(
                    "extended scale row count differs from frozen Stage-A "
                    f"manifest: {len(full_scales)} != {expected_rows}"
                )
            expected_observed = int(
                frozen_scale_manifest["observed_scale_rows"]
            )
            observed = int(full_scales["total_amount"].notna().sum())
            if observed != expected_observed:
                raise ExtensionError(
                    "extended observed scale rows differ from frozen Stage-A "
                    f"manifest: {observed} != {expected_observed}"
                )
    full_scale_metadata = cache_builder._write_dataframe_artifact(
        full_scales,
        scales_path,
        scales_metadata_path,
        role="factor_scales_extended",
        details={
            "requested_start": str(full_scales["TradeDate"].min().date()),
            "requested_end": str(end.date()),
            "previous_scale_artifact": str(backup.resolve()),
            "previous_scale_sha256": cache_builder._sha256_file(backup),
            "config_sha256": config_sha,
            **scale_audit,
        },
    )
    restored_stage_a = _restore_stage_a_artifacts(
        full_scales,
        scales_path,
        cache_root,
        DEFAULT_REPORT_ROOT,
    )

    extension_aggregates, extension_chunks = cache_builder._run_dynamic_chunks(
        stage="dynamic_aggregates",
        start=start,
        end=end,
        scales=extension_scales,
        scale_sha256=full_scale_metadata["artifact_sha256"],
        output_root=cache_root,
        force=False,
        workers=workers,
        fetcher=ch_normalization.fetch_dynamic_factor_aggregates,
        config_sha256=config_sha,
    )
    if extension_aggregates.empty:
        raise ExtensionError("extension dynamic aggregates are empty")

    dynamic_path = cache_root / cache_builder.DYNAMIC_AGGREGATES_FILE
    dynamic_metadata_path = cache_root / cache_builder.DYNAMIC_METADATA_FILE
    existing_dynamic, old_dynamic_metadata = (
        cache_builder._read_verified_parquet(
            dynamic_path, dynamic_metadata_path
        )
    )
    existing_dynamic = cache_builder._canonicalize_dynamic(existing_dynamic)
    existing_dynamic = existing_dynamic.loc[
        ~existing_dynamic["TradeDate"].between(start, end)
    ]
    combined_dynamic = cache_builder._canonicalize_dynamic(
        pd.concat(
            [existing_dynamic, extension_aggregates],
            ignore_index=True,
        )
    )
    dynamic_metadata = cache_builder._write_dataframe_artifact(
        combined_dynamic,
        dynamic_path,
        dynamic_metadata_path,
        role="strict_dynamic_factor_aggregates_extended",
        details={
            "requested_start": str(combined_dynamic["TradeDate"].min().date()),
            "requested_end": str(combined_dynamic["TradeDate"].max().date()),
            "config_sha256": config_sha,
            "source_scales_sha256": full_scale_metadata["artifact_sha256"],
            "previous_dynamic_sha256": old_dynamic_metadata["artifact_sha256"],
            "extension_chunks": extension_chunks,
            "rows": int(len(combined_dynamic)),
            "symbols": int(combined_dynamic["symbol"].nunique()),
            "workers_requested": int(workers),
            "workers_effective": int(workers),
            "max_workers": cache_builder.MAX_WORKERS,
        },
    )

    extension_panels = cache_builder.build_factor_panels(
        extension_aggregates,
        config,
        scale_rows=extension_scales,
    )
    panel_path = cache_root / cache_builder.FACTOR_PANELS_FILE
    panel_metadata_path = cache_root / cache_builder.FACTOR_METADATA_FILE
    existing_panels, old_panel_metadata = cache_builder._read_verified_parquet(
        panel_path, panel_metadata_path
    )
    existing_panels["TradeDate"] = pd.to_datetime(
        existing_panels["TradeDate"]
    ).dt.normalize()
    existing_panels = existing_panels.loc[
        ~existing_panels["TradeDate"].between(start, end)
    ]
    combined_panels = pd.concat(
        [existing_panels, extension_panels], ignore_index=True
    )
    if combined_panels.duplicated(
        ["factor_id", "TradeDate", "symbol"]
    ).any():
        raise ExtensionError("extended factor panels contain duplicate keys")
    headline_ids = [A0_FACTOR_ID, A1_FACTOR_ID, A2_FACTOR_ID, A3_FACTOR_ID]
    panel_metadata = cache_builder._write_dataframe_artifact(
        combined_panels,
        panel_path,
        panel_metadata_path,
        role="normalized_factor_panels_long_extended",
        details={
            "requested_start": str(
                pd.to_datetime(combined_panels["TradeDate"]).min().date()
            ),
            "requested_end": str(end.date()),
            "config_sha256": config_sha,
            "source_dynamic_sha256": dynamic_metadata["artifact_sha256"],
            "source_scales_sha256": full_scale_metadata["artifact_sha256"],
            "previous_factor_panels_sha256": old_panel_metadata[
                "artifact_sha256"
            ],
            "rows": int(len(combined_panels)),
            "factor_ids": sorted(
                combined_panels["factor_id"].unique().tolist()
            ),
            "headline_factor_ids": headline_ids,
            "grid_factor_ids": sorted(
                set(combined_panels["factor_id"].unique()) - set(headline_ids)
            ),
            "a0_factor_id": A0_FACTOR_ID,
            "legacy_aliases": {"mid_order_ratio": A0_FACTOR_ID},
            "legacy_alias_materialized": False,
            "columns": list(cache_builder.FACTOR_PANEL_COLUMNS),
            "sort_order": "factor_specification_then_TradeDate_symbol_by_stage",
        },
    )

    build_path = cache_root / cache_builder.BUILD_METADATA_FILE
    build_metadata = (
        cache_builder._load_verified_metadata(build_path)
        if build_path.is_file()
        else {}
    )
    build_metadata.update(
        {
            "pipeline_version": cache_builder.PIPELINE_VERSION,
            "stage": "factors_extended",
            "extended_at": pd.Timestamp.now().isoformat(),
            "extension_start": str(start.date()),
            "extension_end": str(end.date()),
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha,
            "scales": {
                "path": str(scales_path.resolve()),
                "sha256": full_scale_metadata["artifact_sha256"],
            },
            "dynamic_aggregates": {
                "path": str(dynamic_path.resolve()),
                "sha256": dynamic_metadata["artifact_sha256"],
            },
            "factor_panels": {
                "path": str(panel_path.resolve()),
                "sha256": panel_metadata["artifact_sha256"],
            },
            "workers_requested": int(workers),
            "workers_effective": int(workers),
            "max_workers": cache_builder.MAX_WORKERS,
            "stage_b_refroze_parameters": False,
        }
    )
    build_metadata = cache_builder._write_metadata(build_path, build_metadata)
    return {
        "config_sha256": config_sha,
        "extension_start": str(start.date()),
        "extension_end": str(end.date()),
        "scales_rows": int(len(full_scales)),
        "dynamic_rows": int(len(combined_dynamic)),
        "factor_panel_rows": int(len(combined_panels)),
        "scale_sha256": full_scale_metadata["artifact_sha256"],
        "dynamic_sha256": dynamic_metadata["artifact_sha256"],
        "factor_panel_sha256": panel_metadata["artifact_sha256"],
        "build_metadata_sha256": build_metadata["metadata_sha256"],
        "scale_audit": scale_audit,
        "restored_stage_a": restored_stage_a,
        "extension_chunks": extension_chunks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= cache_builder.MAX_WORKERS:
        raise ExtensionError(
            f"workers must be in [1, {cache_builder.MAX_WORKERS}]"
        )
    result = extend_cache(
        args.cache_root.resolve(),
        pd.Timestamp(args.start).normalize(),
        pd.Timestamp(args.end).normalize(),
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExtensionError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
