#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the resumable normalized mid-trade-amount cache.

The workflow has an explicit stage boundary:

``primitives``
    Fetch strict daily Tick aggregates in monthly chunks and build lagged
    ADV20/ATS20 scales on the complete market calendar.
``calibrate``
    Use only CSI1000 point-in-time membership, Wind market cap, and trade
    amount coverage to freeze A1.  No return panel is imported or loaded.
``factors``
    Verify the frozen configuration (and an optional SHA256 pin) before the
    first query, then fetch A0/A1-grid/A2-grid/A3 aggregates in one dynamic
    query per monthly chunk and materialize long factor panels.
``all``
    Run the three stages in order.

Existing chunks are reused only after their request hash, metadata hash, and
artifact SHA256 pass validation.  ``--force`` is therefore an explicit
replacement operation, not a way to silently trust a stale chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_mid_trade_amount_normalization import (  # noqa: E402
    DEFAULT_A1_GRID,
    DEFAULT_A2_GRID,
    a1_selected_amount_column,
    a2_selected_amount_column,
    audit_dynamic_factor_result,
    build_daily_scale_queries,
    build_dynamic_factor_queries,
    dynamic_result_columns,
    fetch_daily_scale_primitive,
    fetch_dynamic_factor_aggregates,
)
from l2_factor_reproduction.python.mid_trade_amount_normalization import (  # noqa: E402
    A0_FACTOR_ID,
    A0_LOWER_RMB,
    A0_UPPER_RMB,
    A1_FACTOR_ID,
    A2_FACTOR_ID,
    A3_FACTOR_ID,
    FROZEN_A2_LOWER_MULTIPLE,
    FROZEN_A2_UPPER_MULTIPLE,
    amount_share_from_aggregates,
    assert_unique_symbol_trade_date,
    build_lagged_trade_size_scales,
    candidate_grid_name,
    freeze_a1_distribution_candidate,
    freeze_config,
    validate_frozen_config,
)


DEFAULT_WARMUP_START = "2022-11-01"
DEFAULT_FACTOR_START = "2023-01-03"
DEFAULT_END = "2026-07-31"
DEFAULT_CALIBRATION_START = "2023-01-03"
DEFAULT_CALIBRATION_END = "2023-06-30"
DEFAULT_WORKERS = 2
MAX_WORKERS = 10
CSI1000_INDEX_CODE = "000852.SH"

DEFAULT_OUTPUT_ROOT = (
    PROJ_ROOT
    / "research/results/l2_reproduction/mid_order_ratio/normalized_v1"
)

PRIMITIVE_FILE = "daily_trade_size_primitive.parquet"
PRIMITIVE_METADATA_FILE = "daily_trade_size_primitive.metadata.json"
SCALES_FILE = "scales.parquet"
SCALES_METADATA_FILE = "scales.metadata.json"
CALIBRATION_AGGREGATES_FILE = "calibration_dynamic_aggregates.parquet"
CALIBRATION_METADATA_FILE = "calibration_dynamic_aggregates.metadata.json"
FROZEN_CONFIG_FILE = "frozen_config.json"
FROZEN_CONFIG_METADATA_FILE = "frozen_config.metadata.json"
DYNAMIC_AGGREGATES_FILE = "dynamic_aggregates.parquet"
DYNAMIC_METADATA_FILE = "dynamic_aggregates.metadata.json"
FACTOR_PANELS_FILE = "factor_panels.parquet"
FACTOR_METADATA_FILE = "factor_panels.metadata.json"
BUILD_METADATA_FILE = "build_metadata.json"

PIPELINE_VERSION = "mid_trade_amount_normalized_cache_v1"
FACTOR_PANEL_COLUMNS = ("TradeDate", "symbol", "value", "factor_id")


class CacheBuildError(RuntimeError):
    """Raised when a cache stage violates a hard data or lineage gate."""


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{label} cannot be NaT")
    if result.tzinfo is not None:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    return result.normalize()


def _date_text(value: Any) -> str:
    return _timestamp(value, "date").strftime("%Y-%m-%d")


def _validate_workers(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("workers must be an integer between 1 and 10")
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("workers must be an integer between 1 and 10") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(workers)
    if not math.isfinite(numeric) or numeric != workers:
        raise ValueError("workers must be an integer between 1 and 10")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAX_WORKERS}; got {workers}"
        )
    return workers


validate_workers = _validate_workers


def monthly_chunks(start: Any, end: Any) -> Tuple[Tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Return inclusive calendar-month chunks clipped to ``start``/``end``."""
    first = _timestamp(start, "start")
    last = _timestamp(end, "end")
    if first > last:
        raise ValueError(f"start must not be after end: {first.date()}>{last.date()}")
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first
    while cursor <= last:
        month_end = cursor + pd.offsets.MonthEnd(0)
        chunk_end = min(last, pd.Timestamp(month_end).normalize())
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return tuple(chunks)


def _validate_periods(
    warmup_start: Any,
    factor_start: Any,
    end: Any,
    calibration_start: Any,
    calibration_end: Any,
) -> Dict[str, pd.Timestamp]:
    dates = {
        "warmup_start": _timestamp(warmup_start, "warmup_start"),
        "factor_start": _timestamp(factor_start, "factor_start"),
        "end": _timestamp(end, "end"),
        "calibration_start": _timestamp(
            calibration_start, "calibration_start"
        ),
        "calibration_end": _timestamp(calibration_end, "calibration_end"),
    }
    if not dates["warmup_start"] <= dates["factor_start"] <= dates["end"]:
        raise ValueError("require warmup_start <= factor_start <= end")
    if not (
        dates["factor_start"]
        <= dates["calibration_start"]
        <= dates["calibration_end"]
        <= dates["end"]
    ):
        raise ValueError(
            "require factor_start <= calibration_start <= "
            "calibration_end <= end"
        )
    return dates


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _date_text(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.tmp.parquet"
    )
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_with_hash(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(_json_safe(payload))
    result.pop("metadata_sha256", None)
    result["metadata_sha256"] = _canonical_sha256(result)
    return result


def _write_metadata(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = _metadata_with_hash(payload)
    _atomic_write_json(path, result)
    return result


def _load_verified_metadata(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise CacheBuildError(f"required metadata is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheBuildError(f"unable to read metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CacheBuildError(f"metadata must contain a JSON object: {path}")
    stored = payload.get("metadata_sha256")
    if not isinstance(stored, str):
        raise CacheBuildError(f"metadata_sha256 is missing: {path}")
    check = dict(payload)
    check.pop("metadata_sha256", None)
    actual = _canonical_sha256(check)
    if stored != actual:
        raise CacheBuildError(
            f"metadata SHA256 mismatch for {path}: {stored} != {actual}"
        )
    return payload


def _artifact_metadata(
    artifact_path: Path,
    *,
    role: str,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not artifact_path.is_file():
        raise CacheBuildError(f"artifact was not written: {artifact_path}")
    artifact_hash = _sha256_file(artifact_path)
    payload: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "role": role,
        "artifact": str(artifact_path.resolve()),
        "sha256": artifact_hash,
        "artifact_sha256": artifact_hash,
        "bytes": int(artifact_path.stat().st_size),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    if details:
        payload.update(dict(details))
    return payload


def _write_dataframe_artifact(
    frame: pd.DataFrame,
    artifact_path: Path,
    metadata_path: Path,
    *,
    role: str,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _atomic_write_parquet(frame, artifact_path)
    payload = _artifact_metadata(artifact_path, role=role, details=details)
    return _write_metadata(metadata_path, payload)


def _refresh_dataframe_metadata(
    artifact_path: Path,
    metadata_path: Path,
    *,
    role: str,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _artifact_metadata(artifact_path, role=role, details=details)
    return _write_metadata(metadata_path, payload)


def _read_verified_parquet(
    artifact_path: Path,
    metadata_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not artifact_path.is_file():
        raise CacheBuildError(f"required parquet is missing: {artifact_path}")
    metadata = _load_verified_metadata(metadata_path)
    stored = metadata.get("artifact_sha256", metadata.get("sha256"))
    actual = _sha256_file(artifact_path)
    if stored != actual:
        raise CacheBuildError(
            f"artifact SHA256 mismatch for {artifact_path}: {stored} != {actual}"
        )
    try:
        return pd.read_parquet(artifact_path), metadata
    except Exception as exc:
        raise CacheBuildError(
            f"unable to read parquet {artifact_path}: {exc}"
        ) from exc


def _column_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _resolve_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    description: str,
    required: bool = True,
) -> Optional[str]:
    token_map: Dict[str, List[str]] = {}
    for column in frame.columns:
        token_map.setdefault(_column_token(column), []).append(str(column))
    matches: List[str] = []
    for alias in aliases:
        for column in token_map.get(_column_token(alias), []):
            if column not in matches:
                matches.append(column)
    if not matches:
        if required:
            raise CacheBuildError(
                f"{description} is missing; accepted aliases={list(aliases)}, "
                f"columns={list(frame.columns)}"
            )
        return None
    if len(matches) > 1:
        raise CacheBuildError(
            f"{description} is ambiguous; matching columns={matches}"
        )
    return matches[0]


def _rename_alias(
    frame: pd.DataFrame,
    canonical: str,
    aliases: Sequence[str],
    *,
    required: bool = True,
) -> pd.DataFrame:
    source = _resolve_column(
        frame,
        (canonical, *aliases),
        description=canonical,
        required=required,
    )
    if source is None or source == canonical:
        return frame
    if canonical in frame.columns:
        raise CacheBuildError(
            f"cannot rename {source!r} to existing column {canonical!r}"
        )
    return frame.rename(columns={source: canonical})


def _normalize_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if re.fullmatch(r"\d{6}", text):
        text += ".SH" if text.startswith("6") else ".SZ"
    return text


def _normalize_key_columns(
    frame: pd.DataFrame,
    *,
    description: str,
) -> pd.DataFrame:
    out = frame.copy()
    out = _rename_alias(
        out,
        "symbol",
        ("Symbol", "S_INFO_WINDCODE", "S_CON_WINDCODE", "windcode"),
    )
    out = _rename_alias(
        out,
        "TradeDate",
        ("trade_date", "TRADE_DT", "TradingDay", "date", "Date"),
    )
    out["symbol"] = out["symbol"].map(_normalize_symbol)
    out["TradeDate"] = pd.to_datetime(
        out["TradeDate"], errors="raise"
    ).dt.normalize()
    if out[["symbol", "TradeDate"]].isna().any().any():
        raise CacheBuildError(f"{description} has null symbol/TradeDate keys")
    if out["symbol"].eq("").any():
        raise CacheBuildError(f"{description} has empty symbols")
    assert_unique_symbol_trade_date(
        out,
        symbol_col="symbol",
        frame_name=description,
    )
    return out


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    allow_null: bool,
    nonnegative: bool = True,
) -> None:
    frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[column].dropna()
    if not allow_null and frame[column].isna().any():
        raise CacheBuildError(f"{column} contains null values")
    if (~values.map(math.isfinite)).any():
        raise CacheBuildError(f"{column} contains non-finite values")
    if nonnegative and (values < -1e-10).any():
        raise CacheBuildError(f"{column} contains negative values")


def _canonicalize_primitive(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    required_columns = [
        "symbol",
        "TradeDate",
        "total_amount",
        "daily_median_trade_amount",
        "q20",
        "q80",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=required_columns)
    out = frame.copy()
    out = _rename_alias(
        out,
        "symbol",
        ("Symbol", "S_INFO_WINDCODE", "windcode"),
    )
    out = _rename_alias(
        out,
        "TradeDate",
        ("trade_date", "TRADE_DT", "TradingDay", "date"),
    )
    out = _rename_alias(
        out,
        "total_amount",
        ("TotalAmount", "daily_total_amount", "sum_amount"),
    )
    out = _rename_alias(
        out,
        "q20",
        ("daily_q20", "amount_q20", "p20"),
    )
    out = _rename_alias(
        out,
        "q80",
        ("daily_q80", "amount_q80", "p80"),
    )

    median = _resolve_column(
        out,
        ("daily_median_trade_amount", "median_trade_amount"),
        description="daily_median_trade_amount",
        required=False,
    )
    q50 = _resolve_column(
        out,
        ("q50", "daily_q50", "amount_q50", "p50"),
        description="q50",
        required=False,
    )
    if median is None and q50 is None:
        raise CacheBuildError(
            "daily primitive must contain q50 or daily_median_trade_amount"
        )
    if median is not None and q50 is not None and median != q50:
        left = pd.to_numeric(out[median], errors="raise")
        right = pd.to_numeric(out[q50], errors="raise")
        same = np.isclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        )
        if not bool(np.all(same)):
            raise CacheBuildError(
                "q50 and daily_median_trade_amount contain different values"
            )
        out = out.drop(columns=[q50])
        if median != "daily_median_trade_amount":
            out = out.rename(columns={median: "daily_median_trade_amount"})
    elif median is not None:
        if median != "daily_median_trade_amount":
            out = out.rename(columns={median: "daily_median_trade_amount"})
    else:
        out = out.rename(columns={q50: "daily_median_trade_amount"})

    out = _normalize_key_columns(out, description="daily trade-size primitive")
    for column in out.columns:
        if column not in ("symbol", "TradeDate"):
            _numeric_column(out, column, allow_null=True)
    if out["total_amount"].isna().any():
        raise CacheBuildError("daily primitive total_amount contains nulls")
    return out.sort_values(
        ["TradeDate", "symbol"], kind="stable"
    ).reset_index(drop=True)


def _canonicalize_scales(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        raise CacheBuildError("scales cannot be None")
    out = frame.copy()
    if out.empty and not len(out.columns):
        raise CacheBuildError("scales have no schema")
    out = _rename_alias(
        out,
        "symbol",
        ("Symbol", "S_INFO_WINDCODE", "windcode"),
    )
    out = _rename_alias(
        out,
        "TradeDate",
        ("trade_date", "TRADE_DT", "TradingDay", "date"),
    )
    out = _rename_alias(
        out,
        "ADV20_lag1",
        ("adv20_lag1", "adv20_mean_lag1", "adv20"),
    )
    out = _rename_alias(
        out,
        "ATS20_lag1",
        ("ats20_lag1", "ats20_median_lag1", "ats20"),
    )
    out = _rename_alias(
        out,
        "q20",
        ("daily_q20", "amount_q20", "p20"),
    )
    out = _rename_alias(
        out,
        "q80",
        ("daily_q80", "amount_q80", "p80"),
    )
    out = _normalize_key_columns(out, description="lagged trade-size scales")
    for column in out.columns:
        if column in ("symbol", "TradeDate"):
            continue
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
        else:
            _numeric_column(out, column, allow_null=True)
    return out.sort_values(
        ["TradeDate", "symbol"], kind="stable"
    ).reset_index(drop=True)


def _number_token(value: float) -> str:
    return (
        format(float(value), ".12g")
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def _a1_column_aliases(lower: float, upper: float) -> Tuple[str, ...]:
    expected = a1_selected_amount_column(lower, upper)
    candidate = candidate_grid_name("A1", lower, upper)
    low = _number_token(lower)
    high = _number_token(upper)
    return (
        expected,
        f"{candidate}_selected_amount",
        candidate,
        f"a1_adv20_l{low}_h{high}_selected_amount",
        f"a1_l{low}_h{high}_selected_amount",
    )


def _a2_column_aliases(lower: float, upper: float) -> Tuple[str, ...]:
    expected = a2_selected_amount_column(lower, upper)
    candidate = candidate_grid_name("A2", lower, upper)
    low = _number_token(lower)
    high = _number_token(upper)
    return (
        expected,
        f"{candidate}_selected_amount",
        candidate,
        f"a2_ats20_l{low}_h{high}_x_selected_amount",
        f"a2_l{low}_h{high}_selected_amount",
    )


def _canonicalize_dynamic(
    frame: Optional[pd.DataFrame],
    *,
    a1_grid: Sequence[Tuple[float, float]] = DEFAULT_A1_GRID,
    a2_grid: Sequence[Tuple[float, float]] = DEFAULT_A2_GRID,
) -> pd.DataFrame:
    expected = list(dynamic_result_columns(a1_grid, a2_grid))
    if frame is None or frame.empty:
        return pd.DataFrame(columns=expected)
    out = frame.copy()
    out = _rename_alias(
        out,
        "symbol",
        ("Symbol", "S_INFO_WINDCODE", "windcode"),
    )
    out = _rename_alias(
        out,
        "TradeDate",
        ("trade_date", "TRADE_DT", "TradingDay", "date"),
    )
    out = _rename_alias(
        out,
        "total_amount",
        ("TotalAmount", "daily_total_amount", "sum_amount"),
    )
    out = _rename_alias(
        out,
        "a0_abs_4w20w_selected_amount",
        (
            "a0_selected_amount",
            "a0_abs_selected_amount",
            "mid_order_selected_amount",
            "mid_trade_amount_selected_amount",
        ),
    )
    for lower, upper in a1_grid:
        canonical = a1_selected_amount_column(lower, upper)
        out = _rename_alias(
            out,
            canonical,
            tuple(
                alias
                for alias in _a1_column_aliases(lower, upper)
                if alias != canonical
            ),
        )
    for lower, upper in a2_grid:
        canonical = a2_selected_amount_column(lower, upper)
        out = _rename_alias(
            out,
            canonical,
            tuple(
                alias
                for alias in _a2_column_aliases(lower, upper)
                if alias != canonical
            ),
        )
    out = _rename_alias(
        out,
        "a3_q20_q80_selected_amount",
        (
            "a3_selected_amount",
            "rollq_selected_amount",
            "q20_q80_selected_amount",
        ),
    )
    missing = sorted(set(expected).difference(out.columns))
    if missing:
        raise CacheBuildError(f"dynamic aggregates missing columns: {missing}")
    try:
        return audit_dynamic_factor_result(
            out.loc[:, expected],
            a1_grid=a1_grid,
            a2_grid=a2_grid,
        )
    except (TypeError, ValueError) as exc:
        raise CacheBuildError(f"invalid dynamic aggregates: {exc}") from exc


def _validate_chunk_dates(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    description: str,
) -> None:
    if frame.empty:
        return
    outside = ~frame["TradeDate"].between(start, end)
    if outside.any():
        examples = (
            frame.loc[outside, "TradeDate"]
            .drop_duplicates()
            .sort_values()
            .head(5)
            .astype(str)
            .tolist()
        )
        raise CacheBuildError(
            f"{description} contains dates outside its chunk; examples={examples}"
        )


def _validate_dynamic_join(
    aggregate: pd.DataFrame,
    scale_rows: pd.DataFrame,
) -> Dict[str, int]:
    scales = _canonicalize_scales(scale_rows)
    expected = scales
    if "total_amount" in expected.columns:
        total = pd.to_numeric(expected["total_amount"], errors="coerce")
        expected = expected.loc[
            total.notna() & np.isfinite(total) & total.gt(0)
        ]
    left = aggregate[["symbol", "TradeDate"]]
    right = expected[["symbol", "TradeDate"]]
    joined = left.merge(
        right,
        on=["symbol", "TradeDate"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    matched = int(joined["_merge"].eq("both").sum())
    if (
        len(aggregate) != len(expected)
        or len(joined) != len(aggregate)
        or matched != len(aggregate)
    ):
        missing = joined.loc[
            joined["_merge"].ne("both"), ["symbol", "TradeDate"]
        ].head(5)
        raise CacheBuildError(
            "dynamic aggregate/scale join row gate failed; "
            f"aggregate_rows={len(aggregate)}, "
            f"expected_scale_rows={len(expected)}, matched={matched}, "
            f"sample={missing.to_dict('records')}"
        )
    return {
        "scale_input_rows": int(len(scales)),
        "expected_join_rows": int(len(expected)),
        "aggregate_rows": int(len(aggregate)),
        "join_matched_rows": matched,
    }


def _sql_sha256(queries: Mapping[str, str]) -> str:
    return _canonical_sha256(dict(sorted(queries.items())))


def _chunk_paths(chunk_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Path, Path]:
    stem = f"chunk_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return chunk_root / f"{stem}.parquet", chunk_root / f"{stem}.metadata.json"


def _run_monthly_chunks(
    *,
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_root: Path,
    force: bool,
    request_base: Mapping[str, Any],
    request_for_chunk: Optional[
        Callable[[pd.Timestamp, pd.Timestamp], Mapping[str, Any]]
    ] = None,
    build_chunk: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    validate_chunk: Callable[
        [pd.DataFrame, pd.Timestamp, pd.Timestamp],
        Tuple[pd.DataFrame, Mapping[str, Any]],
    ],
    workers: int = 1,
) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    chunk_root.mkdir(parents=True, exist_ok=True)
    worker_count = _validate_workers(workers)

    def process_chunk(
        chunk: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        chunk_start, chunk_end = chunk
        artifact_path, metadata_path = _chunk_paths(
            chunk_root, chunk_start, chunk_end
        )
        chunk_request = (
            dict(request_for_chunk(chunk_start, chunk_end))
            if request_for_chunk is not None
            else {}
        )
        request = {
            **dict(request_base),
            **chunk_request,
            "pipeline_version": PIPELINE_VERSION,
            "stage": stage,
            "chunk_start": _date_text(chunk_start),
            "chunk_end": _date_text(chunk_end),
        }
        request_hash = _canonical_sha256(request)
        reused = artifact_path.is_file() and not force
        if reused:
            metadata = _load_verified_metadata(metadata_path)
            if metadata.get("request_sha256") != request_hash:
                raise CacheBuildError(
                    f"stale {stage} chunk request hash for {artifact_path}; "
                    "rerun with --force"
                )
            actual_hash = _sha256_file(artifact_path)
            stored_hash = metadata.get(
                "artifact_sha256", metadata.get("sha256")
            )
            if stored_hash != actual_hash:
                raise CacheBuildError(
                    f"chunk SHA256 mismatch for {artifact_path}; "
                    "rerun with --force only after investigating the mutation"
                )
            raw = pd.read_parquet(artifact_path)
            validated, audit = validate_chunk(raw, chunk_start, chunk_end)
        else:
            raw = build_chunk(chunk_start, chunk_end)
            validated, audit = validate_chunk(raw, chunk_start, chunk_end)
            _atomic_write_parquet(validated, artifact_path)
            details = {
                **request,
                "request_sha256": request_hash,
                "rows": int(len(validated)),
                **dict(audit),
            }
            metadata = _write_metadata(
                metadata_path,
                _artifact_metadata(
                    artifact_path,
                    role=f"{stage}_monthly_chunk",
                    details=details,
                ),
            )
        record = {
            "path": str(artifact_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "sha256": _sha256_file(artifact_path),
            "request_sha256": request_hash,
            "start": _date_text(chunk_start),
            "end": _date_text(chunk_end),
            "rows": int(len(validated)),
            "reused": bool(reused),
            **dict(audit),
        }
        print(
            f"[{stage}] {record['start']}~{record['end']} "
            f"rows={record['rows']:,} "
            f"{'reused' if reused else 'built'}",
            flush=True,
        )
        return validated, record

    chunks = monthly_chunks(start, end)
    if worker_count == 1 or len(chunks) <= 1:
        completed = [process_chunk(chunk) for chunk in chunks]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            completed = list(pool.map(process_chunk, chunks))
    frames = [item[0] for item in completed]
    records = [item[1] for item in completed]
    return frames, records


def load_market_calendar(start: Any, end: Any) -> Iterable[Any]:
    """Load the SSE trading calendar without loading any return panel."""
    from Factor_Dev_Lib import get_TradingDay

    return get_TradingDay(
        _timestamp(start, "calendar start"),
        _timestamp(end, "calendar end"),
    )


def load_csi1000_membership(start: Any, end: Any) -> pd.DataFrame:
    """Load CSI1000 point-in-time membership without loading returns."""
    from Factor_Dev_Lib import get_index_member_mask

    return get_index_member_mask(
        CSI1000_INDEX_CODE,
        _timestamp(start, "membership start"),
        _timestamp(end, "membership end"),
    )


def load_wind_market_cap(start: Any, end: Any) -> pd.DataFrame:
    """Load the Wind total-market-cap panel lazily."""
    from l2_factor_reproduction.python.mid_trade_amount_research_data import (
        load_market_cap_wide,
    )

    return load_market_cap_wide(start, end)


def _normalize_calendar(values: Iterable[Any]) -> pd.DatetimeIndex:
    if isinstance(values, pd.DataFrame):
        column = next(
            (
                name
                for name in (
                    "TradeDate",
                    "TradingDay",
                    "TRADE_DAYS",
                    "TRADE_DT",
                    "date",
                )
                if name in values.columns
            ),
            None,
        )
        raw: Iterable[Any] = (
            values[column] if column is not None else values.index
        )
    elif isinstance(values, pd.Series):
        raw = values
    else:
        raw = values
    dates = pd.DatetimeIndex(pd.to_datetime(list(raw), errors="raise"))
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Shanghai").tz_localize(None)
    dates = dates.normalize().sort_values()
    if dates.empty:
        raise CacheBuildError("market calendar is empty")
    if dates.hasnans or dates.duplicated().any():
        raise CacheBuildError("market calendar contains null/duplicate dates")
    return dates


def run_primitives(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    warmup_start: Any = DEFAULT_WARMUP_START,
    factor_start: Any = DEFAULT_FACTOR_START,
    end: Any = DEFAULT_END,
    calibration_start: Any = DEFAULT_CALIBRATION_START,
    calibration_end: Any = DEFAULT_CALIBRATION_END,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    fetcher: Optional[Callable[..., pd.DataFrame]] = None,
    calendar_loader: Optional[Callable[[Any, Any], Iterable[Any]]] = None,
) -> Dict[str, Any]:
    """Build resumable daily primitives and complete-calendar lagged scales."""
    worker_count = _validate_workers(workers)
    dates = _validate_periods(
        warmup_start,
        factor_start,
        end,
        calibration_start,
        calibration_end,
    )
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or fetch_daily_scale_primitive
    calendar_fn = calendar_loader or load_market_calendar
    chunk_root = root / "daily_trade_size_primitive" / "chunks"

    def build_chunk(chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.DataFrame:
        return fetch(chunk_start, chunk_end)

    def validate_chunk(
        raw: pd.DataFrame,
        chunk_start: pd.Timestamp,
        chunk_end: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
        part = _canonicalize_primitive(raw)
        _validate_chunk_dates(
            part,
            chunk_start,
            chunk_end,
            description="daily primitive",
        )
        return part, {
            "symbols": int(part["symbol"].nunique()) if len(part) else 0,
        }

    parts, chunks = _run_monthly_chunks(
        stage="primitives",
        start=dates["warmup_start"],
        end=dates["end"],
        chunk_root=chunk_root,
        force=bool(force),
        request_base={
            "query_family": "strict_daily_trade_size_primitive",
        },
        request_for_chunk=lambda chunk_start, chunk_end: {
            "sql_sha256": _sql_sha256(
                build_daily_scale_queries(chunk_start, chunk_end)
            )
        },
        build_chunk=build_chunk,
        validate_chunk=validate_chunk,
        workers=worker_count,
    )
    primitive = _canonicalize_primitive(
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame()
    )
    primitive_path = root / PRIMITIVE_FILE
    primitive_metadata_path = root / PRIMITIVE_METADATA_FILE
    primitive_metadata = _write_dataframe_artifact(
        primitive,
        primitive_path,
        primitive_metadata_path,
        role="daily_trade_size_primitive",
        details={
            "requested_warmup_start": _date_text(dates["warmup_start"]),
            "requested_end": _date_text(dates["end"]),
            "factor_start": _date_text(dates["factor_start"]),
            "rows": int(len(primitive)),
            "symbols": int(primitive["symbol"].nunique()) if len(primitive) else 0,
            "q50_output_column": "daily_median_trade_amount",
            "q50_column_retained": False,
            "workers_requested": worker_count,
            "workers_effective": worker_count,
            "max_workers": MAX_WORKERS,
            "chunks": chunks,
        },
    )

    calendar = _normalize_calendar(
        calendar_fn(dates["warmup_start"], dates["end"])
    )
    if len(primitive) and not primitive["TradeDate"].isin(calendar).all():
        missing = (
            primitive.loc[~primitive["TradeDate"].isin(calendar), "TradeDate"]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .tolist()
        )
        raise CacheBuildError(
            f"primitive dates are absent from market calendar: {missing}"
        )
    try:
        scale_base = build_lagged_trade_size_scales(
            primitive,
            calendar,
            symbol_col="symbol",
            daily_median_col="daily_median_trade_amount",
        )
    except (TypeError, ValueError) as exc:
        raise CacheBuildError(f"unable to build lagged scales: {exc}") from exc

    quantiles = primitive[["symbol", "TradeDate", "q20", "q80"]].copy()
    scale_rows_before = int(len(scale_base))
    scales = scale_base.merge(
        quantiles,
        on=["symbol", "TradeDate"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(scales) != scale_rows_before:
        raise CacheBuildError(
            "scale/primitive quantile join changed row count: "
            f"{scale_rows_before} -> {len(scales)}"
        )
    scales = _canonicalize_scales(scales)
    scales = scales.loc[
        scales["TradeDate"].between(dates["factor_start"], dates["end"])
    ].reset_index(drop=True)
    scale_path = root / SCALES_FILE
    scale_metadata_path = root / SCALES_METADATA_FILE
    scale_metadata = _write_dataframe_artifact(
        scales,
        scale_path,
        scale_metadata_path,
        role="lagged_trade_size_scales",
        details={
            "source_primitive": str(primitive_path.resolve()),
            "source_primitive_sha256": primitive_metadata["artifact_sha256"],
            "requested_start": _date_text(dates["factor_start"]),
            "requested_end": _date_text(dates["end"]),
            "calendar_start": _date_text(calendar.min()),
            "calendar_end": _date_text(calendar.max()),
            "calendar_rows": int(len(calendar)),
            "rows": int(len(scales)),
            "symbols": int(scales["symbol"].nunique()) if len(scales) else 0,
            "rolling_window": 20,
            "lag": 1,
            "quantile_join_input_rows": scale_rows_before,
            "quantile_join_output_rows": int(len(scale_base)),
        },
    )
    return {
        "stage": "primitives",
        "output_root": str(root),
        "primitive_path": str(primitive_path),
        "primitive_sha256": primitive_metadata["artifact_sha256"],
        "scales_path": str(scale_path),
        "scales_sha256": scale_metadata["artifact_sha256"],
        "chunks": chunks,
    }


def _dynamic_sql_hash(
    start: pd.Timestamp,
    end: pd.Timestamp,
    a1_grid: Sequence[Tuple[float, float]],
    a2_grid: Sequence[Tuple[float, float]],
) -> str:
    return _sql_sha256(
        build_dynamic_factor_queries(
            start,
            end,
            a1_grid=a1_grid,
            a2_grid=a2_grid,
        )
    )


def _run_dynamic_chunks(
    *,
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scales: pd.DataFrame,
    scale_sha256: str,
    output_root: Path,
    force: bool,
    workers: int,
    fetcher: Callable[..., pd.DataFrame],
    config_sha256: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    a1_grid = tuple(DEFAULT_A1_GRID)
    a2_grid = tuple(DEFAULT_A2_GRID)
    chunk_root = output_root / stage / "chunks"

    def chunk_scales(
        chunk_start: pd.Timestamp, chunk_end: pd.Timestamp
    ) -> pd.DataFrame:
        return scales.loc[
            scales["TradeDate"].between(chunk_start, chunk_end)
        ].copy()

    def build_chunk(chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.DataFrame:
        local_scales = chunk_scales(chunk_start, chunk_end)
        return fetcher(
            chunk_start,
            chunk_end,
            local_scales,
            a1_grid=a1_grid,
            a2_grid=a2_grid,
        )

    def validate_chunk(
        raw: pd.DataFrame,
        chunk_start: pd.Timestamp,
        chunk_end: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
        part = _canonicalize_dynamic(
            raw,
            a1_grid=a1_grid,
            a2_grid=a2_grid,
        )
        _validate_chunk_dates(
            part,
            chunk_start,
            chunk_end,
            description=f"{stage} dynamic aggregates",
        )
        join_audit = _validate_dynamic_join(
            part, chunk_scales(chunk_start, chunk_end)
        )
        return part, join_audit

    parts, chunks = _run_monthly_chunks(
        stage=stage,
        start=start,
        end=end,
        chunk_root=chunk_root,
        force=force,
        request_base={
            "query_family": "strict_dynamic_trade_amount_grid",
            "scale_sha256": scale_sha256,
            "config_sha256": config_sha256,
            "a1_grid": a1_grid,
            "a2_grid": a2_grid,
        },
        request_for_chunk=lambda chunk_start, chunk_end: {
            "sql_sha256": _dynamic_sql_hash(
                chunk_start, chunk_end, a1_grid, a2_grid
            )
        },
        build_chunk=build_chunk,
        validate_chunk=validate_chunk,
        workers=workers,
    )
    combined = _canonicalize_dynamic(
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(),
        a1_grid=a1_grid,
        a2_grid=a2_grid,
    )
    return combined, chunks


def _wide_or_long_panel(
    frame: pd.DataFrame,
    *,
    description: str,
    value_aliases: Sequence[str],
    implicit_value: Optional[float] = None,
) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise CacheBuildError(f"{description} panel is empty")
    date_column = _resolve_column(
        frame,
        ("TradeDate", "TRADE_DT", "TradingDay", "date", "Date"),
        description=f"{description} date",
        required=False,
    )
    symbol_column = _resolve_column(
        frame,
        ("symbol", "Symbol", "S_INFO_WINDCODE", "S_CON_WINDCODE"),
        description=f"{description} symbol",
        required=False,
    )
    value_column = _resolve_column(
        frame,
        value_aliases,
        description=f"{description} value",
        required=False,
    )
    if date_column is not None and symbol_column is not None:
        columns = [date_column, symbol_column]
        if value_column is not None:
            columns.append(value_column)
        out = frame.loc[:, columns].copy()
        out = out.rename(
            columns={
                date_column: "TradeDate",
                symbol_column: "symbol",
                **({value_column: "panel_value"} if value_column else {}),
            }
        )
        if value_column is None:
            if implicit_value is None:
                raise CacheBuildError(f"{description} value column is missing")
            out["panel_value"] = implicit_value
    else:
        wide = frame.copy()
        wide.index = pd.to_datetime(wide.index, errors="raise").normalize()
        wide.index.name = "TradeDate"
        out = (
            wide.rename_axis(columns="symbol")
            .stack(dropna=False)
            .rename("panel_value")
            .reset_index()
        )
    out["TradeDate"] = pd.to_datetime(
        out["TradeDate"], errors="raise"
    ).dt.normalize()
    out["symbol"] = out["symbol"].map(_normalize_symbol)
    if out[["TradeDate", "symbol"]].isna().any().any():
        raise CacheBuildError(f"{description} contains null keys")
    if out.duplicated(["symbol", "TradeDate"]).any():
        raise CacheBuildError(
            f"{description} contains duplicate symbol/TradeDate keys"
        )
    return out


def _membership_long(frame: pd.DataFrame) -> pd.DataFrame:
    out = _wide_or_long_panel(
        frame,
        description="CSI1000 PIT membership",
        value_aliases=(
            "member",
            "membership",
            "flag",
            "weight",
            "S_CON_WT",
            "I_WEIGHT",
        ),
        implicit_value=1.0,
    )
    numeric = pd.to_numeric(out["panel_value"], errors="coerce")
    text_true = (
        out["panel_value"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "yes", "y", "member", "in"})
    )
    keep = numeric.gt(0) | text_true
    return out.loc[keep, ["TradeDate", "symbol"]].reset_index(drop=True)


def _market_cap_long(frame: pd.DataFrame) -> pd.DataFrame:
    out = _wide_or_long_panel(
        frame,
        description="Wind market cap",
        value_aliases=(
            "market_cap",
            "total_market_cap",
            "total_mktcap",
            "S_VAL_MV",
            "mkt_cap",
            "cap",
        ),
    ).rename(columns={"panel_value": "market_cap"})
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    finite_positive = (
        out["market_cap"].notna()
        & np.isfinite(out["market_cap"])
        & out["market_cap"].gt(0)
    )
    return out.loc[
        finite_positive, ["TradeDate", "symbol", "market_cap"]
    ].reset_index(drop=True)


def _assign_daily_cap_quintile(values: pd.Series) -> pd.Series:
    count = len(values)
    if count < 5:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    ranks = values.rank(method="first")
    quintiles = np.ceil(ranks * 5.0 / count).clip(1, 5)
    return quintiles.astype("Int64")


def build_a1_calibration_distribution(
    aggregates: pd.DataFrame,
    membership: pd.DataFrame,
    market_cap: pd.DataFrame,
    *,
    scale_rows: Optional[pd.DataFrame] = None,
    a1_grid: Sequence[Tuple[float, float]] = DEFAULT_A1_GRID,
    a2_grid: Sequence[Tuple[float, float]] = DEFAULT_A2_GRID,
) -> Tuple[pd.DataFrame, float, Dict[int, float], Dict[str, Any]]:
    """Build distribution-only A1 coverage rows on CSI1000 cap quintiles."""
    dynamic = _canonicalize_dynamic(
        aggregates,
        a1_grid=a1_grid,
        a2_grid=a2_grid,
    )
    if dynamic.empty:
        raise CacheBuildError("calibration dynamic aggregates are empty")
    scale_valid_rows = len(dynamic)
    if scale_rows is not None:
        scales = _canonicalize_scales(scale_rows)
        dynamic = dynamic.merge(
            scales[["symbol", "TradeDate", "ADV20_lag1"]],
            on=["symbol", "TradeDate"],
            how="left",
            validate="one_to_one",
        )
        valid_adv = (
            dynamic["ADV20_lag1"].notna()
            & np.isfinite(dynamic["ADV20_lag1"])
            & dynamic["ADV20_lag1"].gt(0)
        )
        dynamic = dynamic.loc[valid_adv].drop(columns="ADV20_lag1")
        scale_valid_rows = int(len(dynamic))
        if dynamic.empty:
            raise CacheBuildError(
                "no calibration aggregate rows have a valid ADV20_lag1 scale"
            )
    members = _membership_long(membership)
    cap = _market_cap_long(market_cap)
    eligible = dynamic.merge(
        members,
        on=["TradeDate", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    eligible = eligible.merge(
        cap,
        on=["TradeDate", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    if eligible.empty:
        raise CacheBuildError(
            "no calibration rows align across aggregates, CSI1000 PIT, and cap"
        )
    eligible["market_cap_quintile"] = eligible.groupby(
        "TradeDate", sort=False
    )["market_cap"].transform(
        _assign_daily_cap_quintile
    )
    eligible = eligible.loc[
        eligible["market_cap_quintile"].notna()
    ].copy()
    eligible["market_cap_quintile"] = eligible[
        "market_cap_quintile"
    ].astype(int)
    observed_quintiles = sorted(
        eligible["market_cap_quintile"].unique().tolist()
    )
    if observed_quintiles != [1, 2, 3, 4, 5]:
        raise CacheBuildError(
            "calibration requires all five market-cap quintiles; "
            f"found={observed_quintiles}"
        )

    total = float(eligible["total_amount"].sum())
    a0_selected = float(
        eligible["a0_abs_4w20w_selected_amount"].sum()
    )
    if not math.isfinite(total) or total <= 0:
        raise CacheBuildError("calibration total amount must be positive")
    a0_overall = a0_selected / total
    a0_grouped = eligible.groupby(
        "market_cap_quintile", sort=True
    )[["a0_abs_4w20w_selected_amount", "total_amount"]].sum()
    a0_by_quintile = {
        int(quintile): float(
            row["a0_abs_4w20w_selected_amount"] / row["total_amount"]
        )
        for quintile, row in a0_grouped.iterrows()
    }

    rows: List[Dict[str, Any]] = []
    for lower, upper in a1_grid:
        selected_column = a1_selected_amount_column(lower, upper)
        grouped = eligible.groupby(
            "market_cap_quintile", sort=True
        )[[selected_column, "total_amount"]].sum()
        candidate = candidate_grid_name("A1", lower, upper)
        for quintile, row in grouped.iterrows():
            rows.append(
                {
                    "candidate": candidate,
                    "market_cap_quintile": int(quintile),
                    "selected_amount": float(row[selected_column]),
                    "total_amount": float(row["total_amount"]),
                    "lower_bps": float(lower),
                    "upper_bps": float(upper),
                }
            )
    distribution = pd.DataFrame(rows)
    audit = {
        "dynamic_rows": int(len(dynamic)),
        "adv20_scale_valid_rows": scale_valid_rows,
        "pit_cap_eligible_rows": int(len(eligible)),
        "eligible_dates": int(eligible["TradeDate"].nunique()),
        "eligible_symbols": int(eligible["symbol"].nunique()),
        "a0_overall_coverage": float(a0_overall),
        "a0_quintile_coverage": a0_by_quintile,
    }
    return distribution, float(a0_overall), a0_by_quintile, audit


def _frozen_config_path(
    output_root: Path, explicit: Optional[Path]
) -> Path:
    return (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else output_root / FROZEN_CONFIG_FILE
    )


def run_calibrate(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    warmup_start: Any = DEFAULT_WARMUP_START,
    factor_start: Any = DEFAULT_FACTOR_START,
    end: Any = DEFAULT_END,
    calibration_start: Any = DEFAULT_CALIBRATION_START,
    calibration_end: Any = DEFAULT_CALIBRATION_END,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    frozen_config_path: Optional[Path] = None,
    fetcher: Optional[Callable[..., pd.DataFrame]] = None,
    membership_loader: Optional[Callable[[Any, Any], pd.DataFrame]] = None,
    market_cap_loader: Optional[Callable[[Any, Any], pd.DataFrame]] = None,
    freeze_fn: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Freeze A1 from distribution coverage only; never load returns."""
    worker_count = _validate_workers(workers)
    dates = _validate_periods(
        warmup_start,
        factor_start,
        end,
        calibration_start,
        calibration_end,
    )
    root = Path(output_root).resolve()
    scales, scale_metadata = _read_verified_parquet(
        root / SCALES_FILE,
        root / SCALES_METADATA_FILE,
    )
    scales = _canonicalize_scales(scales)
    calibration_scales = scales.loc[
        scales["TradeDate"].between(
            dates["calibration_start"], dates["calibration_end"]
        )
    ].copy()
    if calibration_scales.empty:
        raise CacheBuildError("no scale rows in the calibration period")

    dynamic_fetch = fetcher or fetch_dynamic_factor_aggregates
    aggregates, chunks = _run_dynamic_chunks(
        stage="calibration_dynamic_aggregates",
        start=dates["calibration_start"],
        end=dates["calibration_end"],
        scales=calibration_scales,
        scale_sha256=str(scale_metadata["artifact_sha256"]),
        output_root=root,
        force=bool(force),
        workers=worker_count,
        fetcher=dynamic_fetch,
    )
    aggregate_path = root / CALIBRATION_AGGREGATES_FILE
    aggregate_metadata_path = root / CALIBRATION_METADATA_FILE
    aggregate_metadata = _write_dataframe_artifact(
        aggregates,
        aggregate_path,
        aggregate_metadata_path,
        role="calibration_dynamic_aggregates",
        details={
            "selection_uses_returns": False,
            "requested_start": _date_text(dates["calibration_start"]),
            "requested_end": _date_text(dates["calibration_end"]),
            "rows": int(len(aggregates)),
            "symbols": int(aggregates["symbol"].nunique())
            if len(aggregates)
            else 0,
            "source_scales_sha256": scale_metadata["artifact_sha256"],
            "a1_grid": DEFAULT_A1_GRID,
            "a2_grid": DEFAULT_A2_GRID,
            "workers_requested": worker_count,
            "workers_effective": worker_count,
            "max_workers": MAX_WORKERS,
            "chunks": chunks,
        },
    )

    membership_fn = membership_loader or load_csi1000_membership
    cap_fn = market_cap_loader or load_wind_market_cap
    membership = membership_fn(
        dates["calibration_start"], dates["calibration_end"]
    )
    market_cap = cap_fn(
        dates["calibration_start"], dates["calibration_end"]
    )
    distribution, a0_coverage, a0_quintiles, coverage_audit = (
        build_a1_calibration_distribution(
            aggregates,
            membership,
            market_cap,
            scale_rows=calibration_scales,
        )
    )
    selected = freeze_a1_distribution_candidate(
        distribution,
        a0_overall_coverage=a0_coverage,
        a0_quintile_coverage=a0_quintiles,
    )
    selected_lower = float(selected["lower_bps"])
    selected_upper = float(selected["upper_bps"])
    a1_block = {
        **selected,
        "selected_aggregate_column": a1_selected_amount_column(
            selected_lower, selected_upper
        ),
        "scale": (
            "ADV20_lag1 mean of exactly 20 prior market trading dates"
        ),
        "grid_L_bps": sorted({float(pair[0]) for pair in DEFAULT_A1_GRID}),
        "grid_H_bps": sorted({float(pair[1]) for pair in DEFAULT_A1_GRID}),
        "selection_rule": (
            "CSI1000 PIT and Wind market-cap quintiles; require 10%-80% "
            "amount coverage in every quintile, then minimize the overall "
            "coverage gap versus A0 with deterministic tie-breaks"
        ),
        "effective_direction": -1,
    }
    config_payload: Dict[str, Any] = {
        "version": "mid_trade_amount_normalized_v1",
        "headline_factor_ids": [
            A0_FACTOR_ID,
            A1_FACTOR_ID,
            A2_FACTOR_ID,
            A3_FACTOR_ID,
        ],
        "calibration_period": {
            "start": _date_text(dates["calibration_start"]),
            "end": _date_text(dates["calibration_end"]),
        },
        "selection_uses_returns": False,
        "a0": {
            "factor_id": A0_FACTOR_ID,
            "legacy_alias": "mid_order_ratio",
            "lower_rmb_exclusive": float(A0_LOWER_RMB),
            "upper_rmb_inclusive": float(A0_UPPER_RMB),
            "effective_direction": -1,
        },
        "a1": a1_block,
        "a2": {
            "factor_id": A2_FACTOR_ID,
            "lower_multiple": float(FROZEN_A2_LOWER_MULTIPLE),
            "upper_multiple": float(FROZEN_A2_UPPER_MULTIPLE),
            "selected_aggregate_column": a2_selected_amount_column(
                FROZEN_A2_LOWER_MULTIPLE,
                FROZEN_A2_UPPER_MULTIPLE,
            ),
            "scale": (
                "ATS20_lag1 median of daily median positive Tick amount "
                "over exactly 20 prior market trading dates"
            ),
            "grid_L": sorted({float(pair[0]) for pair in DEFAULT_A2_GRID}),
            "grid_H": sorted({float(pair[1]) for pair in DEFAULT_A2_GRID}),
            "effective_direction": -1,
        },
        "a3": {
            "factor_id": A3_FACTOR_ID,
            "lower_daily_quantile_exclusive": 0.2,
            "upper_daily_quantile_inclusive": 0.8,
            "role": "P1 secondary",
            "effective_direction": -1,
        },
        "effective_direction": {
            "A0": -1,
            "A1": -1,
            "A2": -1,
            "A3": -1,
        },
        "direction_policy": (
            "frozen ex ante; raw RankIC reported; never inferred per window"
        ),
        "periods": {
            "factor": [
                _date_text(dates["factor_start"]),
                _date_text(dates["end"]),
            ],
            "calibration": [
                _date_text(dates["calibration_start"]),
                _date_text(dates["calibration_end"]),
            ],
        },
        "sources": {
            "scales": str((root / SCALES_FILE).resolve()),
            "scales_sha256": scale_metadata["artifact_sha256"],
            "calibration_dynamic": str(aggregate_path.resolve()),
            "calibration_dynamic_sha256": aggregate_metadata[
                "artifact_sha256"
            ],
        },
        "calibration_coverage_audit": coverage_audit,
    }
    freezer = freeze_fn or freeze_config
    frozen = freezer(config_payload)
    verified_sha = validate_frozen_config(
        frozen,
        required_keys=(
            "a0",
            "a1",
            "a2",
            "a3",
            "effective_direction",
        ),
    )
    config_path = _frozen_config_path(root, frozen_config_path)
    _atomic_write_json(config_path, frozen)
    sha_path = config_path.with_suffix(".sha256")
    _atomic_write_text(sha_path, verified_sha + "\n")
    config_metadata_path = (
        config_path.parent / FROZEN_CONFIG_METADATA_FILE
    )
    config_metadata = _write_metadata(
        config_metadata_path,
        _artifact_metadata(
            config_path,
            role="frozen_distribution_only_config",
            details={
                "config_sha256": verified_sha,
                "selection_uses_returns": False,
                "calibration_dynamic_sha256": aggregate_metadata[
                    "artifact_sha256"
                ],
                "coverage_audit": coverage_audit,
                "companion_sha256_file": str(sha_path.resolve()),
            },
        ),
    )
    return {
        "stage": "calibrate",
        "output_root": str(root),
        "calibration_aggregates": str(aggregate_path),
        "calibration_aggregates_sha256": aggregate_metadata[
            "artifact_sha256"
        ],
        "frozen_config": str(config_path),
        "config_sha256": verified_sha,
        "frozen_config_file_sha256": config_metadata["artifact_sha256"],
        "selected_a1": {
            "candidate": selected["candidate"],
            "lower_bps": selected_lower,
            "upper_bps": selected_upper,
        },
        "chunks": chunks,
    }


def _config_block(
    config: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = config.get(name, config.get(name.upper()))
    if not isinstance(value, Mapping):
        raise CacheBuildError(f"frozen config missing {name.upper()} block")
    return value


def _config_number(
    block: Mapping[str, Any],
    aliases: Sequence[str],
    description: str,
) -> float:
    found = [key for key in aliases if key in block]
    if not found:
        raise CacheBuildError(f"frozen config missing {description}")
    if len(found) > 1:
        values = {float(block[key]) for key in found}
        if len(values) != 1:
            raise CacheBuildError(
                f"frozen config has conflicting {description}: {found}"
            )
    value = float(block[found[0]])
    if not math.isfinite(value):
        raise CacheBuildError(f"frozen config {description} is not finite")
    return value


def build_factor_panels(
    aggregates: pd.DataFrame,
    frozen_config: Mapping[str, Any],
    *,
    scale_rows: Optional[pd.DataFrame] = None,
    a1_grid: Sequence[Tuple[float, float]] = DEFAULT_A1_GRID,
    a2_grid: Sequence[Tuple[float, float]] = DEFAULT_A2_GRID,
) -> pd.DataFrame:
    """Convert selected/total aggregates into headline and grid factor rows."""
    dynamic = _canonicalize_dynamic(
        aggregates,
        a1_grid=a1_grid,
        a2_grid=a2_grid,
    )
    a1 = _config_block(frozen_config, "a1")
    a2 = _config_block(frozen_config, "a2")
    selected_a1 = (
        _config_number(
            a1,
            ("lower_bps", "lower", "L", "l"),
            "A1 lower_bps",
        ),
        _config_number(
            a1,
            ("upper_bps", "upper", "H", "h"),
            "A1 upper_bps",
        ),
    )
    selected_a2 = (
        _config_number(
            a2,
            ("lower_multiple", "lower", "L", "l"),
            "A2 lower_multiple",
        ),
        _config_number(
            a2,
            ("upper_multiple", "upper", "H", "h"),
            "A2 upper_multiple",
        ),
    )
    if selected_a1 not in {
        (float(lower), float(upper)) for lower, upper in a1_grid
    }:
        raise CacheBuildError(
            f"frozen A1 pair is outside the supplied grid: {selected_a1}"
        )
    if selected_a2 not in {
        (float(lower), float(upper)) for lower, upper in a2_grid
    }:
        raise CacheBuildError(
            f"frozen A2 pair is outside the supplied grid: {selected_a2}"
        )
    if not (
        math.isclose(
            selected_a2[0],
            FROZEN_A2_LOWER_MULTIPLE,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            selected_a2[1],
            FROZEN_A2_UPPER_MULTIPLE,
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        raise CacheBuildError(
            "A2 headline must remain frozen at (0.5, 2.0]"
        )

    validity: Dict[str, np.ndarray] = {}
    if scale_rows is not None:
        scales = _canonicalize_scales(scale_rows)
        evidence = dynamic[["symbol", "TradeDate"]].merge(
            scales[
                [
                    "symbol",
                    "TradeDate",
                    "ADV20_lag1",
                    "ATS20_lag1",
                    "q20",
                    "q80",
                ]
            ],
            on=["symbol", "TradeDate"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not evidence["_merge"].eq("both").all():
            raise CacheBuildError(
                "factor-panel scale evidence join has unmatched aggregate rows"
            )
        validity = {
            "A1": (
                evidence["ADV20_lag1"].notna()
                & np.isfinite(evidence["ADV20_lag1"])
                & evidence["ADV20_lag1"].gt(0)
            ).to_numpy(),
            "A2": (
                evidence["ATS20_lag1"].notna()
                & np.isfinite(evidence["ATS20_lag1"])
                & evidence["ATS20_lag1"].gt(0)
            ).to_numpy(),
            "A3": (
                evidence["q20"].notna()
                & evidence["q80"].notna()
                & np.isfinite(evidence["q20"])
                & np.isfinite(evidence["q80"])
                & evidence["q20"].ge(0)
                & evidence["q80"].ge(evidence["q20"])
            ).to_numpy(),
        }

    specifications: List[Tuple[str, str, Optional[str]]] = [
        (A0_FACTOR_ID, "a0_abs_4w20w_selected_amount", None),
        (
            A1_FACTOR_ID,
            a1_selected_amount_column(*selected_a1),
            "A1",
        ),
        (
            A2_FACTOR_ID,
            a2_selected_amount_column(*selected_a2),
            "A2",
        ),
        (A3_FACTOR_ID, "a3_q20_q80_selected_amount", "A3"),
    ]
    specifications.extend(
        (
            candidate_grid_name("A1", lower, upper),
            a1_selected_amount_column(lower, upper),
            "A1",
        )
        for lower, upper in a1_grid
    )
    specifications.extend(
        (
            candidate_grid_name("A2", lower, upper),
            a2_selected_amount_column(lower, upper),
            "A2",
        )
        for lower, upper in a2_grid
    )
    factor_ids = [factor_id for factor_id, _, _ in specifications]
    if len(factor_ids) != len(set(factor_ids)):
        raise CacheBuildError(f"duplicate output factor ids: {factor_ids}")

    parts: List[pd.DataFrame] = []
    for factor_id, selected_column, scale_family in specifications:
        values = amount_share_from_aggregates(
            dynamic[selected_column].to_numpy(dtype=float),
            dynamic["total_amount"].to_numpy(dtype=float),
        )
        if scale_family is not None and scale_family in validity:
            values = np.where(validity[scale_family], values, np.nan)
        part = dynamic[["TradeDate", "symbol"]].copy()
        part["value"] = values
        part["factor_id"] = factor_id
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=list(FACTOR_PANEL_COLUMNS))
    panels = pd.concat(parts, ignore_index=True)
    panels = panels.loc[:, FACTOR_PANEL_COLUMNS]
    if panels.duplicated(
        ["factor_id", "TradeDate", "symbol"]
    ).any():
        raise CacheBuildError(
            "factor panels contain duplicate factor_id/symbol/TradeDate keys"
        )
    if (panels["factor_id"] == "mid_order_ratio").any():
        raise CacheBuildError(
            "legacy alias mid_order_ratio must remain metadata-only"
        )
    # ``dynamic`` is already ordered by TradeDate/symbol and specifications are
    # deterministic.  Avoid a redundant global sort of the 22x-expanded panel.
    return panels.reset_index(drop=True)


def _read_frozen_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise CacheBuildError(f"frozen config is missing: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheBuildError(f"unable to read frozen config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CacheBuildError("frozen_config.json must contain a JSON object")
    return config


def run_factors(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    warmup_start: Any = DEFAULT_WARMUP_START,
    factor_start: Any = DEFAULT_FACTOR_START,
    end: Any = DEFAULT_END,
    calibration_start: Any = DEFAULT_CALIBRATION_START,
    calibration_end: Any = DEFAULT_CALIBRATION_END,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    frozen_config_path: Optional[Path] = None,
    expected_config_sha256: Optional[str] = None,
    fetcher: Optional[Callable[..., pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Build factor panels after validating, but never changing, the freeze."""
    root = Path(output_root).resolve()
    config_path = _frozen_config_path(root, frozen_config_path)
    config = _read_frozen_config(config_path)
    if expected_config_sha256 is None:
        pin_path = config_path.with_suffix(".sha256")
        if not pin_path.is_file():
            raise CacheBuildError(
                f"frozen config companion SHA256 is missing: {pin_path}"
            )
        expected_config_sha256 = pin_path.read_text(
            encoding="utf-8"
        ).strip()
    # This is deliberately the first stage-B data gate.  No scale read, fetch,
    # freeze, or output write happens before the config and optional pin pass.
    try:
        config_sha256 = validate_frozen_config(
            config,
            expected_sha256=expected_config_sha256,
            required_keys=(
                "a0",
                "a1",
                "a2",
                "a3",
                "effective_direction",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CacheBuildError(f"frozen config validation failed: {exc}") from exc

    worker_count = _validate_workers(workers)
    dates = _validate_periods(
        warmup_start,
        factor_start,
        end,
        calibration_start,
        calibration_end,
    )
    scales, scale_metadata = _read_verified_parquet(
        root / SCALES_FILE,
        root / SCALES_METADATA_FILE,
    )
    scales = _canonicalize_scales(scales)
    factor_scales = scales.loc[
        scales["TradeDate"].between(dates["factor_start"], dates["end"])
    ].reset_index(drop=True)
    if factor_scales.empty:
        raise CacheBuildError("no scales in the requested factor period")

    scale_path = root / SCALES_FILE
    scale_metadata_path = root / SCALES_METADATA_FILE
    # Stage B consumes a date view but must never truncate or rewrite the
    # Stage-A scale artifact.  Its full hash remains the immutable lineage
    # parent for every monthly request.

    dynamic_fetch = fetcher or fetch_dynamic_factor_aggregates
    aggregates, chunks = _run_dynamic_chunks(
        stage="dynamic_aggregates",
        start=dates["factor_start"],
        end=dates["end"],
        scales=factor_scales,
        scale_sha256=str(scale_metadata["artifact_sha256"]),
        output_root=root,
        force=bool(force),
        workers=worker_count,
        fetcher=dynamic_fetch,
        config_sha256=config_sha256,
    )
    if aggregates.empty:
        raise CacheBuildError(
            "strict dynamic factor aggregates are empty for the factor period"
        )
    dynamic_path = root / DYNAMIC_AGGREGATES_FILE
    dynamic_metadata = _write_dataframe_artifact(
        aggregates,
        dynamic_path,
        root / DYNAMIC_METADATA_FILE,
        role="strict_dynamic_factor_aggregates",
        details={
            "requested_start": _date_text(dates["factor_start"]),
            "requested_end": _date_text(dates["end"]),
            "config_sha256": config_sha256,
            "source_scales_sha256": scale_metadata["artifact_sha256"],
            "rows": int(len(aggregates)),
            "symbols": int(aggregates["symbol"].nunique())
            if len(aggregates)
            else 0,
            "a1_grid": DEFAULT_A1_GRID,
            "a2_grid": DEFAULT_A2_GRID,
            "workers_requested": worker_count,
            "workers_effective": worker_count,
            "max_workers": MAX_WORKERS,
            "chunks": chunks,
        },
    )
    panels = build_factor_panels(
        aggregates,
        config,
        scale_rows=factor_scales,
    )
    panel_path = root / FACTOR_PANELS_FILE
    headline_ids = [A0_FACTOR_ID, A1_FACTOR_ID, A2_FACTOR_ID, A3_FACTOR_ID]
    grid_ids = sorted(
        set(panels["factor_id"].unique()).difference(headline_ids)
    )
    panel_metadata = _write_dataframe_artifact(
        panels,
        panel_path,
        root / FACTOR_METADATA_FILE,
        role="normalized_factor_panels_long",
        details={
            "requested_start": _date_text(dates["factor_start"]),
            "requested_end": _date_text(dates["end"]),
            "config_sha256": config_sha256,
            "source_dynamic_sha256": dynamic_metadata["artifact_sha256"],
            "source_scales_sha256": scale_metadata["artifact_sha256"],
            "rows": int(len(panels)),
            "factor_ids": sorted(panels["factor_id"].unique().tolist()),
            "headline_factor_ids": headline_ids,
            "grid_factor_ids": grid_ids,
            "a0_factor_id": A0_FACTOR_ID,
            "legacy_aliases": {"mid_order_ratio": A0_FACTOR_ID},
            "legacy_alias_materialized": False,
            "columns": list(FACTOR_PANEL_COLUMNS),
            "sort_order": "factor_specification_then_TradeDate_symbol",
        },
    )
    build_metadata = _write_metadata(
        root / BUILD_METADATA_FILE,
        {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "factors",
            "created_at": pd.Timestamp.now().isoformat(),
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
            "scales": {
                "path": str(scale_path.resolve()),
                "sha256": scale_metadata["artifact_sha256"],
            },
            "dynamic_aggregates": {
                "path": str(dynamic_path.resolve()),
                "sha256": dynamic_metadata["artifact_sha256"],
            },
            "factor_panels": {
                "path": str(panel_path.resolve()),
                "sha256": panel_metadata["artifact_sha256"],
            },
            "workers_requested": worker_count,
            "workers_effective": worker_count,
            "max_workers": MAX_WORKERS,
            "stage_b_refroze_parameters": False,
            "legacy_aliases": {"mid_order_ratio": A0_FACTOR_ID},
        },
    )
    return {
        "stage": "factors",
        "output_root": str(root),
        "config_sha256": config_sha256,
        "dynamic_aggregates": str(dynamic_path),
        "dynamic_aggregates_sha256": dynamic_metadata["artifact_sha256"],
        "factor_panels": str(panel_path),
        "factor_panels_sha256": panel_metadata["artifact_sha256"],
        "scales": str(scale_path),
        "scales_sha256": scale_metadata["artifact_sha256"],
        "build_metadata_sha256": build_metadata["metadata_sha256"],
        "chunks": chunks,
    }


# Explicit stage aliases make orchestration and tests discoverable.
stage_primitives = run_primitives
stage_calibrate = run_calibrate
stage_factors = run_factors


def run_pipeline(
    *,
    stage: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    warmup_start: Any = DEFAULT_WARMUP_START,
    factor_start: Any = DEFAULT_FACTOR_START,
    end: Any = DEFAULT_END,
    calibration_start: Any = DEFAULT_CALIBRATION_START,
    calibration_end: Any = DEFAULT_CALIBRATION_END,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    frozen_config_path: Optional[Path] = None,
    expected_config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = str(stage).strip().lower()
    if normalized not in {"primitives", "calibrate", "factors", "all"}:
        raise ValueError(f"unknown stage: {stage!r}")
    common = {
        "output_root": output_root,
        "warmup_start": warmup_start,
        "factor_start": factor_start,
        "end": end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "workers": workers,
        "force": force,
    }
    results: Dict[str, Any] = {}
    if normalized in {"primitives", "all"}:
        results["primitives"] = run_primitives(**common)
    if normalized in {"calibrate", "all"}:
        results["calibrate"] = run_calibrate(
            **common,
            frozen_config_path=frozen_config_path,
        )
    if normalized in {"factors", "all"}:
        results["factors"] = run_factors(
            **common,
            frozen_config_path=frozen_config_path,
            expected_config_sha256=expected_config_sha256,
        )
    return results[normalized] if normalized != "all" else results


run_stage = run_pipeline


def _workers_argument(value: str) -> int:
    try:
        return _validate_workers(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("primitives", "calibrate", "factors", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--warmup-start", default=DEFAULT_WARMUP_START)
    parser.add_argument("--factor-start", default=DEFAULT_FACTOR_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--calibration-start", default=DEFAULT_CALIBRATION_START
    )
    parser.add_argument(
        "--calibration-end", default=DEFAULT_CALIBRATION_END
    )
    parser.add_argument(
        "--workers",
        type=_workers_argument,
        default=DEFAULT_WORKERS,
        help="Requested concurrency (1-10); execution is currently sequential.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace monthly chunks instead of resuming verified chunks.",
    )
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument(
        "--config-sha256",
        "--expected-config-sha256",
        dest="expected_config_sha256",
        help="Optional Stage-B pin for frozen_config canonical SHA256.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pipeline(
        stage=args.stage,
        output_root=args.output_root,
        warmup_start=args.warmup_start,
        factor_start=args.factor_start,
        end=args.end,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
        workers=args.workers,
        force=args.force,
        frozen_config_path=args.frozen_config,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CacheBuildError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
