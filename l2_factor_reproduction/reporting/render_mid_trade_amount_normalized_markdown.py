#!/usr/bin/env python3
"""Render the numeric chapters of the normalized mid-trade-amount report.

The renderer is intentionally a strict, read-only consumer of the persisted
``normalized_v1`` artifacts.  It never substitutes a value, silently drops a
factor, or treats a missing figure as optional.  All inputs are validated
before any Markdown chapter is replaced.
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
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l2_factor_reproduction.python.mid_trade_amount_normalization import (  # noqa: E402
    validate_frozen_config,
)


DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT
    / "research"
    / "reports"
    / "factors"
    / "mid_order_ratio"
    / "normalized_v1"
)

ROLES: Tuple[str, ...] = ("A0", "A1", "A2", "A3")
UNIVERSES: Tuple[str, ...] = ("ALL", "CSI300", "CSI500", "CSI1000")
SEGMENTS: Tuple[str, ...] = ("IS", "validation", "OOS")
OLS_METHODS: Tuple[str, ...] = ("raw", "industry", "cap", "joint")
TURNOVER_TERCILES: Tuple[str, ...] = ("Low", "Mid", "High")
QUINTILES: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5")
SIGNIFICANCE_TSTAT = 1.96
EXPECTED_FEE_BPS = 7.5

CHAPTER_FILES: Tuple[str, ...] = (
    "01_executive_summary.md",
    "05_standalone_validation.md",
    "06_normalization_diagnostics.md",
    "07_exposure_diagnostics.md",
    "08_time_and_state_robustness.md",
    "10_research_decision.md",
)

FIGURE_CLASSES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("01_factor_variant_summary", "Factor variant summary"),
        ("02_universe_variant_summary", "PIT universe comparison"),
        ("03_decile_annualized", "Per-variant decile annualized return"),
        ("04_decile_cumulative", "Per-variant decile cumulative return"),
        ("05_ic_stability", "Daily, monthly, and 63-day RankIC"),
        ("06_cap_adv_quintiles", "Market-cap and ADV quintiles"),
        ("07_parameter_stability", "Frozen-grid parameter stability"),
        ("08_turnover_tercile", "Lagged turnover-state terciles"),
        ("09_ols_diagnostics", "Raw/industry/cap/joint OLS"),
        ("10_segments_and_coverage", "IS/validation/OOS and coverage"),
    ]
)

CSV_COLUMNS: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict(
    [
        (
            "factor_variant_summary.csv",
            (
                "factor_id",
                "factor_role",
                "rank_ic",
                "rank_ic_tstat",
                "icir",
                "hl_annu_ret",
                "hl_sharpe",
                "hl_mdd",
                "hl_turnover",
                "factor_coverage_ratio",
                "implied_annu_fee_7p5bps",
            ),
        ),
        (
            "universe_variant_summary.csv",
            (
                "factor_id",
                "factor_role",
                "universe",
                "rank_ic",
                "rank_ic_tstat",
                "icir",
                "hl_sharpe",
                "hl_mdd",
                "hl_turnover",
            ),
        ),
        (
            "csi1000_decile_summary.csv",
            (
                "factor_id",
                "factor_role",
                "decile_monotonicity_spearman",
                "csi1000_index_excess_hl_annu_ret",
                "csi1000_index_excess_hl_turnover",
                "implied_annu_fee_7p5bps",
            ),
        ),
        (
            "missing_scale_coverage.csv",
            (
                "factor_id",
                "factor_role",
                "required_scale",
                "expected_stock_days",
                "factor_stock_days",
                "factor_coverage_ratio",
                "missing_scale_ratio",
                "factor_coverage_given_scale",
            ),
        ),
        (
            "csi1000_monthly_rank_ic.csv",
            (
                "factor_id",
                "month",
                "rank_ic_mean",
                "icir",
                "negative_ic_day_share",
                "n_days",
            ),
        ),
        (
            "csi1000_rolling_63d_rank_ic.csv",
            (
                "TradeDate",
                "factor_id",
                "rank_ic_raw",
                "rank_ic_63d_mean",
                "rank_ic_63d_count",
            ),
        ),
        (
            "csi1000_cap_quintile_statistics.csv",
            (
                "factor_id",
                "dimension",
                "quantile",
                "n_days",
                "n_names_avg",
                "coverage_rate",
                "rank_ic_mean",
                "icir",
            ),
        ),
        (
            "csi1000_adv_quintile_statistics.csv",
            (
                "factor_id",
                "dimension",
                "quantile",
                "n_days",
                "n_names_avg",
                "coverage_rate",
                "rank_ic_mean",
                "icir",
            ),
        ),
        (
            "parameter_stability.csv",
            (
                "factor_id",
                "factor_role",
                "factor_family",
                "rank_ic",
                "rank_ic_tstat",
                "icir",
                "lower_bound",
                "upper_bound",
                "parameter_unit",
                "is_selected",
            ),
        ),
        (
            "state_turnover_tercile_summary.csv",
            (
                "factor_id",
                "factor_role",
                "turnover_tercile",
                "rank_ic",
                "icir",
                "negative_ic_day_share",
                "n_days",
            ),
        ),
        (
            "ols_diagnostics.csv",
            (
                "factor_id",
                "factor_role",
                "ols_method",
                "rank_ic",
                "rank_ic_tstat",
                "icir",
                "abs_rank_ic_retained_vs_raw",
            ),
        ),
        (
            "sample_segment_results.csv",
            (
                "factor_id",
                "factor_role",
                "segment",
                "status",
                "actual_start",
                "actual_end",
                "rank_ic",
                "rank_ic_tstat",
                "icir",
                "hl_sharpe",
                "hl_mdd",
                "hl_turnover",
            ),
        ),
        (
            "normalization_distribution_summary_adv.csv",
            (
                "scale",
                "unit",
                "group_type",
                "group",
                "quantile",
                "value",
                "calibration_start",
                "calibration_end",
            ),
        ),
        (
            "normalization_by_size_bucket_calibration.csv",
            (
                "variant",
                "bucket_type",
                "bucket",
                "amount_coverage",
                "a0_amount_coverage",
                "calibration_start",
                "calibration_end",
            ),
        ),
        (
            "parameter_stability_adv_distribution.csv",
            (
                "lower_adv_bps_exclusive",
                "upper_adv_bps_inclusive",
                "overall_amount_coverage",
                "a0_overall_amount_coverage",
                "abs_coverage_diff_vs_a0",
                "mean_abs_quintile_coverage_diff_vs_a0",
                "all_quintiles_between_10pct_80pct",
                "minimum_quintile_coverage",
                "maximum_quintile_coverage",
                "frozen_main",
            ),
        ),
    ]
)

MANIFEST_GENERATED_CSVS: Tuple[str, ...] = tuple(
    name
    for name in CSV_COLUMNS
    if name
    not in {
        "normalization_distribution_summary_adv.csv",
        "normalization_by_size_bucket_calibration.csv",
        "parameter_stability_adv_distribution.csv",
    }
)

NUMERIC_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "factor_variant_summary.csv": (
        "rank_ic",
        "rank_ic_tstat",
        "icir",
        "hl_annu_ret",
        "hl_sharpe",
        "hl_mdd",
        "hl_turnover",
        "factor_coverage_ratio",
        "implied_annu_fee_7p5bps",
    ),
    "universe_variant_summary.csv": (
        "rank_ic",
        "rank_ic_tstat",
        "icir",
        "hl_sharpe",
        "hl_mdd",
        "hl_turnover",
    ),
    "csi1000_decile_summary.csv": (
        "decile_monotonicity_spearman",
        "csi1000_index_excess_hl_annu_ret",
        "csi1000_index_excess_hl_turnover",
        "implied_annu_fee_7p5bps",
    ),
    "missing_scale_coverage.csv": (
        "expected_stock_days",
        "factor_stock_days",
        "factor_coverage_ratio",
        "missing_scale_ratio",
        "factor_coverage_given_scale",
    ),
    "csi1000_monthly_rank_ic.csv": (
        "rank_ic_mean",
        "icir",
        "negative_ic_day_share",
        "n_days",
    ),
    "csi1000_rolling_63d_rank_ic.csv": (
        "rank_ic_raw",
        "rank_ic_63d_count",
    ),
    "csi1000_cap_quintile_statistics.csv": (
        "n_days",
        "n_names_avg",
        "coverage_rate",
        "rank_ic_mean",
        "icir",
    ),
    "csi1000_adv_quintile_statistics.csv": (
        "n_days",
        "n_names_avg",
        "coverage_rate",
        "rank_ic_mean",
        "icir",
    ),
    "parameter_stability.csv": (
        "rank_ic",
        "rank_ic_tstat",
        "icir",
        "lower_bound",
        "upper_bound",
    ),
    "state_turnover_tercile_summary.csv": (
        "rank_ic",
        "icir",
        "negative_ic_day_share",
        "n_days",
    ),
    "ols_diagnostics.csv": (
        "rank_ic",
        "rank_ic_tstat",
        "icir",
        "abs_rank_ic_retained_vs_raw",
    ),
    "sample_segment_results.csv": (
        "rank_ic",
        "rank_ic_tstat",
        "icir",
        "hl_sharpe",
        "hl_mdd",
        "hl_turnover",
    ),
    "normalization_distribution_summary_adv.csv": ("quantile", "value"),
    "normalization_by_size_bucket_calibration.csv": (
        "bucket",
        "amount_coverage",
        "a0_amount_coverage",
    ),
    "parameter_stability_adv_distribution.csv": (
        "lower_adv_bps_exclusive",
        "upper_adv_bps_inclusive",
        "overall_amount_coverage",
        "a0_overall_amount_coverage",
        "abs_coverage_diff_vs_a0",
        "mean_abs_quintile_coverage_diff_vs_a0",
        "minimum_quintile_coverage",
        "maximum_quintile_coverage",
    ),
}


class MarkdownRenderError(RuntimeError):
    """Raised when persisted evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class FigureClass:
    description: str
    files: Tuple[str, ...]


@dataclass(frozen=True)
class DecisionEvidence:
    role: str
    oos_rank_ic: float
    oos_tstat: float
    expected_direction: bool
    significant: bool
    stable_pools: int
    total_pools: int
    cross_pool_stable: bool
    retained: bool


@dataclass(frozen=True)
class DecisionResult:
    branch: str
    evidence: Mapping[str, DecisionEvidence]


@dataclass
class ReportInputs:
    report_root: Path
    artifacts_root: Path
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    parity: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]
    figures: Mapping[str, FigureClass]
    role_to_factor: Mapping[str, str]
    direction: int
    fee_bps: float


def _fail(message: str) -> None:
    raise MarkdownRenderError(message)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        _fail("Required {} is missing: {}".format(description, path))
    if path.stat().st_size <= 0:
        _fail("Required {} is empty: {}".format(description, path))
    return path


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    _require_file(path, description)

    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant {!r}".format(value))

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail("Invalid {} at {}: {}".format(description, path, exc))
    if not isinstance(payload, Mapping):
        _fail("{} must contain a JSON object: {}".format(description, path))
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail("{} must be numeric; got {!r}".format(context, value))
    if not math.isfinite(number):
        _fail("{} must be finite; got {!r}".format(context, value))
    return number


def _bool_value(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    _fail("{} must be boolean; got {!r}".format(context, value))
    return False


def _read_csv(
    artifacts_root: Path,
    filename: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    path = _require_file(artifacts_root / filename, filename)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        _fail("Cannot read {}: {}".format(path, exc))
    if frame.empty:
        _fail("{} has no data rows".format(path))
    if frame.columns.duplicated().any():
        _fail("{} contains duplicate column names".format(path))
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        _fail("{} is missing columns: {}".format(path, missing))
    for column in NUMERIC_COLUMNS.get(filename, ()):
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = converted.isna() | ~converted.map(math.isfinite)
        if invalid.any():
            rows = list(frame.index[invalid][:5])
            _fail(
                "{} column {!r} has non-finite values at rows {}".format(
                    path, column, rows
                )
            )
        frame[column] = converted
    if filename == "csi1000_rolling_63d_rank_ic.csv":
        rolling = pd.to_numeric(frame["rank_ic_63d_mean"], errors="coerce")
        finite = rolling.notna() & rolling.map(math.isfinite)
        frame["rank_ic_63d_mean"] = rolling
        if not finite.any():
            _fail("{} has no finite 63-day RankIC values".format(path))
    return frame


def _assert_unique(
    frame: pd.DataFrame,
    columns: Sequence[str],
    description: str,
) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, list(columns)].head(5).to_dict("records")
        _fail("{} has duplicate keys: {}".format(description, sample))


def _assert_role_rows(
    frame: pd.DataFrame,
    description: str,
    role_column: str = "factor_role",
) -> None:
    found = set(frame[role_column].astype(str))
    missing = sorted(set(ROLES) - found)
    if missing:
        _fail("{} is missing factor roles: {}".format(description, missing))


def _attach_roles(
    frame: pd.DataFrame,
    role_to_factor: Mapping[str, str],
    description: str,
) -> pd.DataFrame:
    factor_to_role = {factor_id: role for role, factor_id in role_to_factor.items()}
    out = frame.copy()
    out["factor_id"] = out["factor_id"].astype(str)
    out["factor_role"] = out["factor_id"].map(factor_to_role)
    if out["factor_role"].isna().any():
        unknown = sorted(set(out.loc[out["factor_role"].isna(), "factor_id"]))
        _fail("{} contains unknown factor IDs: {}".format(description, unknown))
    return out


def _resolve_direction(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> int:
    declared = manifest.get("frozen_effective_direction")
    try:
        direction = int(declared)
    except (TypeError, ValueError):
        _fail("artifact_manifest frozen_effective_direction must be -1 or 1")
    if direction not in (-1, 1) or float(declared) != direction:
        _fail("artifact_manifest frozen_effective_direction must be exactly -1 or 1")
    config_directions = config.get("effective_direction")
    values: Iterable[Any]
    if isinstance(config_directions, Mapping):
        values = config_directions.values()
    else:
        values = (config_directions,)
    for value in values:
        if value is None or int(value) != direction or float(value) != direction:
            _fail("frozen_config direction disagrees with artifact_manifest")
    return direction


def _validate_manifest_records(
    report_root: Path,
    manifest: Mapping[str, Any],
    required_relative_paths: Sequence[str],
) -> None:
    records = manifest.get("generated_files")
    if not isinstance(records, list) or not records:
        _fail("artifact_manifest generated_files must be a non-empty list")
    by_path: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail("artifact_manifest generated_files[{}] is not an object".format(index))
        relative = str(record.get("path", "")).replace("\\", "/")
        if not relative or Path(relative).is_absolute():
            _fail("artifact_manifest generated path must be relative: {!r}".format(relative))
        if relative in by_path:
            _fail("artifact_manifest repeats generated path: {}".format(relative))
        by_path[relative] = record

    for relative in required_relative_paths:
        normalized = str(Path(relative)).replace("\\", "/")
        if normalized not in by_path:
            _fail("artifact_manifest does not record required output: {}".format(normalized))
        record = by_path[normalized]
        path = _require_file(report_root / normalized, normalized)
        try:
            path.resolve().relative_to(report_root.resolve())
        except ValueError:
            _fail("Manifest path escapes report root: {}".format(normalized))
        expected_bytes = int(record.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            _fail("Byte-size mismatch for manifest output: {}".format(normalized))
        expected_hash = str(record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            _fail("Invalid SHA256 in manifest for {}".format(normalized))
        if _sha256(path) != expected_hash:
            _fail("SHA256 mismatch for manifest output: {}".format(normalized))


def _validate_figure_classes(
    report_root: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, FigureClass]:
    records = manifest.get("figure_classes")
    if not isinstance(records, list):
        _fail("artifact_manifest figure_classes must be a list")
    figures: Dict[str, FigureClass] = {}
    for record in records:
        if not isinstance(record, Mapping):
            _fail("Every figure class in artifact_manifest must be an object")
        class_id = str(record.get("id", ""))
        description = str(record.get("description", "")).strip()
        files = record.get("files")
        if class_id in figures:
            _fail("Duplicate figure class in artifact_manifest: {}".format(class_id))
        if not description or not isinstance(files, list) or not files:
            _fail("Figure class {} lacks a description or files".format(class_id))
        normalized_files: List[str] = []
        for value in files:
            relative = str(value).replace("\\", "/")
            if Path(relative).is_absolute():
                _fail("Figure path must be relative: {}".format(relative))
            path = _require_file(report_root / relative, "figure {}".format(relative))
            try:
                path.resolve().relative_to(report_root.resolve())
            except ValueError:
                _fail("Figure path escapes report root: {}".format(relative))
            normalized_files.append(relative)
        figures[class_id] = FigureClass(description, tuple(normalized_files))
    found = set(figures)
    expected = set(FIGURE_CLASSES)
    if found != expected:
        _fail(
            "Expected exactly ten figure classes; missing={}, extra={}".format(
                sorted(expected - found), sorted(found - expected)
            )
        )
    return figures


def _validate_inputs(data: ReportInputs) -> None:
    tables = data.tables
    factor = tables["factor_variant_summary.csv"]
    factor["factor_role"] = factor["factor_role"].astype(str)
    _assert_role_rows(factor, "factor_variant_summary.csv")
    _assert_unique(factor, ("factor_role",), "factor_variant_summary.csv")
    if set(factor["factor_role"]) != set(ROLES):
        _fail("factor_variant_summary.csv must contain exactly A0/A1/A2/A3")

    universe = tables["universe_variant_summary.csv"]
    universe["factor_role"] = universe["factor_role"].astype(str)
    universe["universe"] = universe["universe"].astype(str)
    _assert_unique(
        universe,
        ("factor_role", "universe"),
        "universe_variant_summary.csv",
    )
    for role in ROLES:
        found = set(universe.loc[universe["factor_role"] == role, "universe"])
        if found != set(UNIVERSES):
            _fail(
                "universe_variant_summary.csv {} pools differ from {}".format(
                    role, list(UNIVERSES)
                )
            )

    decile = tables["csi1000_decile_summary.csv"]
    _assert_role_rows(decile, "csi1000_decile_summary.csv")
    _assert_unique(decile, ("factor_role",), "csi1000_decile_summary.csv")

    coverage = tables["missing_scale_coverage.csv"]
    _assert_role_rows(coverage, "missing_scale_coverage.csv")
    _assert_unique(coverage, ("factor_role",), "missing_scale_coverage.csv")

    for filename in (
        "csi1000_monthly_rank_ic.csv",
        "csi1000_rolling_63d_rank_ic.csv",
    ):
        _assert_role_rows(tables[filename], filename)
        for role in ROLES:
            if tables[filename].loc[
                tables[filename]["factor_role"] == role
            ].empty:
                _fail("{} has no rows for {}".format(filename, role))

    rolling = tables["csi1000_rolling_63d_rank_ic.csv"]
    rolling["TradeDate"] = pd.to_datetime(rolling["TradeDate"], errors="coerce")
    if rolling["TradeDate"].isna().any():
        _fail("csi1000_rolling_63d_rank_ic.csv contains invalid TradeDate")
    for role in ROLES:
        part = rolling.loc[rolling["factor_role"] == role, "rank_ic_63d_mean"]
        if not part.map(lambda value: pd.notna(value) and math.isfinite(float(value))).any():
            _fail("63-day RankIC has no finite value for {}".format(role))

    for filename in (
        "csi1000_cap_quintile_statistics.csv",
        "csi1000_adv_quintile_statistics.csv",
    ):
        frame = tables[filename]
        _assert_role_rows(frame, filename)
        _assert_unique(frame, ("factor_role", "quantile"), filename)
        for role in ROLES:
            found = set(
                frame.loc[frame["factor_role"] == role, "quantile"].astype(str)
            )
            if not set(QUINTILES).issubset(found):
                _fail("{} {} lacks Q1-Q5".format(filename, role))

    parameters = tables["parameter_stability.csv"]
    _assert_role_rows(parameters, "parameter_stability.csv")
    selected_count: Dict[str, int] = {}
    for role in ROLES:
        part = parameters.loc[parameters["factor_role"].astype(str) == role]
        selected_count[role] = sum(
            _bool_value(value, "parameter_stability.csv is_selected")
            for value in part["is_selected"]
        )
    if any(count != 1 for count in selected_count.values()):
        _fail(
            "parameter_stability.csv needs exactly one selected row per role: {}".format(
                selected_count
            )
        )

    state = tables["state_turnover_tercile_summary.csv"]
    _assert_unique(
        state,
        ("factor_role", "turnover_tercile"),
        "state_turnover_tercile_summary.csv",
    )
    for role in ROLES:
        found = set(
            state.loc[state["factor_role"].astype(str) == role, "turnover_tercile"]
        )
        if found != set(TURNOVER_TERCILES):
            _fail("Turnover-state rows are incomplete for {}".format(role))

    ols = tables["ols_diagnostics.csv"]
    _assert_unique(ols, ("factor_role", "ols_method"), "ols_diagnostics.csv")
    for role in ROLES:
        found = set(ols.loc[ols["factor_role"].astype(str) == role, "ols_method"])
        if found != set(OLS_METHODS):
            _fail("OLS raw/industry/cap/joint rows are incomplete for {}".format(role))

    segments = tables["sample_segment_results.csv"]
    _assert_unique(
        segments,
        ("factor_role", "segment"),
        "sample_segment_results.csv",
    )
    if not segments["status"].astype(str).eq("ok").all():
        bad = segments.loc[
            ~segments["status"].astype(str).eq("ok"),
            ["factor_role", "segment", "status"],
        ].to_dict("records")
        _fail("IS/validation/OOS contains unavailable rows: {}".format(bad))
    for role in ROLES:
        found = set(
            segments.loc[segments["factor_role"].astype(str) == role, "segment"]
        )
        if found != set(SEGMENTS):
            _fail("IS/validation/OOS rows are incomplete for {}".format(role))

    distribution = tables["normalization_distribution_summary_adv.csv"]
    universe_groups = set(
        distribution.loc[distribution["group_type"] == "universe", "group"].astype(str)
    )
    cap_groups = set(
        distribution.loc[
            distribution["group_type"] == "market_cap_quintile", "group"
        ].astype(str)
    )
    if not set(UNIVERSES).issubset(universe_groups) or not set(QUINTILES).issubset(
        cap_groups
    ):
        _fail("ADV calibration distributions lack four pools or cap quintiles")

    adv_selection = tables["parameter_stability_adv_distribution.csv"]
    frozen = [
        _bool_value(value, "parameter_stability_adv_distribution.csv frozen_main")
        for value in adv_selection["frozen_main"]
    ]
    if sum(frozen) != 1:
        _fail("ADV distribution grid needs exactly one frozen_main row")
    if not all(
        _bool_value(
            value,
            "parameter_stability_adv_distribution.csv "
            "all_quintiles_between_10pct_80pct",
        )
        for value in adv_selection["all_quintiles_between_10pct_80pct"]
    ):
        _fail("ADV calibration grid contains a candidate outside its frozen coverage gate")

    parity = data.parity
    if parity.get("gate") != "passed":
        _fail("a0_parity.json gate is not passed")
    for key in (
        "pearson",
        "spearman",
        "max_abs_error",
        "mean_abs_error",
        "within_1e_12_share",
    ):
        _finite_number(parity.get(key), "a0_parity.json {}".format(key))


def load_report_inputs(report_root: Path = DEFAULT_REPORT_ROOT) -> ReportInputs:
    """Load and validate every artifact needed by the six numeric chapters."""
    root = Path(report_root).expanduser().resolve()
    if not root.is_dir():
        _fail("Report root is missing: {}".format(root))
    artifacts = root / "artifacts"
    if not artifacts.is_dir():
        _fail("Artifacts directory is missing: {}".format(artifacts))

    config_path = artifacts / "frozen_config.json"
    config = _load_json(config_path, "frozen_config.json")
    checksum_path = _require_file(
        artifacts / "frozen_config.sha256",
        "frozen_config.sha256",
    )
    expected_checksum = checksum_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checksum):
        _fail("frozen_config.sha256 does not contain one lowercase SHA256")
    try:
        validate_frozen_config(
            config,
            expected_sha256=expected_checksum,
            required_keys=("headline_factor_ids", "a0", "a1", "a2", "a3"),
        )
    except (TypeError, ValueError) as exc:
        _fail("frozen_config integrity check failed: {}".format(exc))

    manifest = _load_json(artifacts / "artifact_manifest.json", "artifact_manifest.json")
    parity = _load_json(artifacts / "a0_parity.json", "a0_parity.json")
    if manifest.get("frozen_config_snapshot") != config:
        _fail("artifact_manifest frozen_config_snapshot differs from frozen_config.json")

    figures = _validate_figure_classes(root, manifest)
    required_manifest_paths = [
        "artifacts/{}".format(filename) for filename in MANIFEST_GENERATED_CSVS
    ]
    required_manifest_paths.extend(
        relative
        for figure in figures.values()
        for relative in figure.files
    )
    _validate_manifest_records(root, manifest, required_manifest_paths)

    tables: Dict[str, pd.DataFrame] = {}
    for filename, columns in CSV_COLUMNS.items():
        tables[filename] = _read_csv(artifacts, filename, columns)

    factor = tables["factor_variant_summary.csv"]
    role_to_factor = {
        str(row.factor_role): str(row.factor_id)
        for row in factor[["factor_role", "factor_id"]].itertuples(index=False)
    }
    if set(role_to_factor) != set(ROLES):
        _fail("factor_variant_summary.csv must map exactly A0/A1/A2/A3")

    configured_ids = config.get("headline_factor_ids")
    if not isinstance(configured_ids, list) or set(map(str, configured_ids)) != set(
        role_to_factor.values()
    ):
        _fail("frozen_config headline_factor_ids disagree with factor summary")

    for filename in (
        "csi1000_monthly_rank_ic.csv",
        "csi1000_rolling_63d_rank_ic.csv",
        "csi1000_cap_quintile_statistics.csv",
        "csi1000_adv_quintile_statistics.csv",
    ):
        tables[filename] = _attach_roles(
            tables[filename],
            role_to_factor,
            filename,
        )

    direction = _resolve_direction(config, manifest)
    fee = manifest.get("fee")
    if not isinstance(fee, Mapping):
        _fail("artifact_manifest fee block is missing")
    fee_bps = _finite_number(fee.get("one_way_bps"), "manifest fee.one_way_bps")
    config_fee = _finite_number(config.get("fee_bps"), "frozen_config fee_bps")
    if fee_bps != EXPECTED_FEE_BPS or config_fee != EXPECTED_FEE_BPS:
        _fail("The normalized_v1 fee label must be exactly 7.5 bps")

    manifest_roles = manifest.get("factor_roles")
    if not isinstance(manifest_roles, Mapping):
        _fail("artifact_manifest factor_roles is missing")
    normalized_manifest_roles = {
        str(role): str(factor_id) for factor_id, role in manifest_roles.items()
    }
    if normalized_manifest_roles != role_to_factor:
        _fail("artifact_manifest factor_roles disagree with factor summary")

    data = ReportInputs(
        report_root=root,
        artifacts_root=artifacts,
        config=config,
        manifest=manifest,
        parity=parity,
        tables=tables,
        figures=figures,
        role_to_factor=role_to_factor,
        direction=direction,
        fee_bps=fee_bps,
    )
    _validate_inputs(data)
    return data


def determine_research_decision(
    sample_segments: pd.DataFrame,
    universe_summary: pd.DataFrame,
    direction: int,
) -> DecisionResult:
    """Choose one frozen five-way branch without consulting Sharpe.

    A role is retained only when its CSI1000 OOS raw RankIC has the frozen
    direction, its direction-adjusted t-stat is at least 1.96, and all four PIT
    universe rows have the same frozen direction.
    """
    if direction not in (-1, 1):
        _fail("Decision direction must be -1 or 1")
    evidence: Dict[str, DecisionEvidence] = {}
    for role in ROLES:
        oos = sample_segments.loc[
            (sample_segments["factor_role"].astype(str) == role)
            & (sample_segments["segment"].astype(str) == "OOS")
        ]
        if len(oos) != 1:
            _fail("Decision requires exactly one OOS row for {}".format(role))
        row = oos.iloc[0]
        rank_ic = _finite_number(row["rank_ic"], "{} OOS RankIC".format(role))
        tstat = _finite_number(row["rank_ic_tstat"], "{} OOS t-stat".format(role))
        pool_rows = universe_summary.loc[
            universe_summary["factor_role"].astype(str) == role
        ]
        if set(pool_rows["universe"].astype(str)) != set(UNIVERSES):
            _fail("Decision requires all four universes for {}".format(role))
        stable_pools = int(
            sum(
                _finite_number(value, "{} pool RankIC".format(role)) * direction > 0
                for value in pool_rows["rank_ic"]
            )
        )
        expected = rank_ic * direction > 0
        significant = tstat * direction >= SIGNIFICANCE_TSTAT
        cross_pool = stable_pools == len(UNIVERSES)
        retained = expected and significant and cross_pool
        evidence[role] = DecisionEvidence(
            role=role,
            oos_rank_ic=rank_ic,
            oos_tstat=tstat,
            expected_direction=expected,
            significant=significant,
            stable_pools=stable_pools,
            total_pools=len(UNIVERSES),
            cross_pool_stable=cross_pool,
            retained=retained,
        )

    if evidence["A1"].retained and evidence["A2"].retained:
        branch = "both ADV20 and ATS20 retain the relation"
    elif evidence["A1"].retained:
        branch = "only ADV20 retains it"
    elif evidence["A2"].retained:
        branch = "only ATS20 retains it"
    elif evidence["A0"].retained:
        branch = "only the fixed-RMB phenomenon remains"
    else:
        branch = "all normalized variants lose the relation"
    return DecisionResult(branch=branch, evidence=evidence)


def _pct(value: Any, digits: int = 2) -> str:
    number = _finite_number(value, "rendered percentage")
    return ("{:.%df}%%" % digits).format(number * 100.0)


def _number(value: Any, digits: int = 3) -> str:
    number = _finite_number(value, "rendered number")
    return ("{:.%df}" % digits).format(number)


def _integer(value: Any) -> str:
    number = _finite_number(value, "rendered integer")
    return "{:,.0f}".format(number)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| {} |".format(" | ".join(_cell(value) for value in headers)),
        "| {} |".format(" | ".join("---" for _ in headers)),
    ]
    for row in rows:
        values = list(row)
        if len(values) != len(headers):
            _fail("Internal Markdown table width mismatch")
        lines.append("| {} |".format(" | ".join(_cell(value) for value in values)))
    return "\n".join(lines)


def _figure_markdown(data: ReportInputs, class_ids: Sequence[str]) -> str:
    blocks: List[str] = []
    for class_id in class_ids:
        figure = data.figures[class_id]
        blocks.append("### {} — {}".format(class_id, figure.description))
        for relative in figure.files:
            caption = "{} — {}".format(figure.description, Path(relative).stem)
            blocks.append("![{}]({})".format(caption, relative))
    return "\n\n".join(blocks)


def _ordered(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column, order in zip(columns, (ROLES, UNIVERSES, SEGMENTS)):
        if column in out:
            rank = {value: index for index, value in enumerate(order)}
            out["_order_{}".format(column)] = out[column].map(rank)
    sort_columns = [
        "_order_{}".format(column)
        for column in columns
        if "_order_{}".format(column) in out
    ]
    if sort_columns:
        out = out.sort_values(sort_columns)
    return out.drop(columns=sort_columns, errors="ignore")


def _factor_definition_rows(data: ReportInputs) -> List[Sequence[Any]]:
    config = data.config
    return [
        (
            "A0",
            data.role_to_factor["A0"],
            "fixed RMB",
            "({:,.0f}, {:,.0f}] RMB".format(
                _finite_number(config["a0"]["lower_rmb_exclusive"], "A0 lower"),
                _finite_number(config["a0"]["upper_rmb_inclusive"], "A0 upper"),
            ),
        ),
        (
            "A1",
            data.role_to_factor["A1"],
            "ADV20 lag-1",
            "({}, {}] bps of ADV".format(
                _number(config["a1"]["lower_bps"], 1),
                _number(config["a1"]["upper_bps"], 1),
            ),
        ),
        (
            "A2",
            data.role_to_factor["A2"],
            "ATS20 lag-1",
            "({}, {}] × ATS20".format(
                _number(config["a2"]["lower_multiple"], 2),
                _number(config["a2"]["upper_multiple"], 2),
            ),
        ),
        (
            "A3",
            data.role_to_factor["A3"],
            "same-day execution quantiles",
            "(Q{}, Q{}] amount share; P1 diagnostic".format(
                int(
                    _finite_number(
                        config["a3"]["lower_daily_quantile_exclusive"],
                        "A3 lower",
                    )
                    * 100
                ),
                int(
                    _finite_number(
                        config["a3"]["upper_daily_quantile_inclusive"],
                        "A3 upper",
                    )
                    * 100
                ),
            ),
        ),
    ]


def _render_executive(data: ReportInputs, decision: DecisionResult) -> str:
    factor = _ordered(data.tables["factor_variant_summary.csv"], ("factor_role",))
    universe = _ordered(
        data.tables["universe_variant_summary.csv"],
        ("factor_role", "universe"),
    )
    factor_rows = [
        (
            row.factor_role,
            _pct(row.rank_ic),
            _number(row.icir, 2),
            _number(row.rank_ic_tstat, 2),
            _number(row.hl_sharpe, 2),
            _pct(row.hl_mdd),
            _number(row.hl_turnover, 2),
            _pct(row.factor_coverage_ratio),
        )
        for row in factor.itertuples(index=False)
    ]
    pool_rows = [
        (
            row.factor_role,
            row.universe,
            _pct(row.rank_ic),
            _number(row.icir, 2),
        )
        for row in universe.itertuples(index=False)
    ]
    parity = data.parity
    return """# 01 — Executive Summary

## Frozen five-way decision

**Decision: `{branch}`.**

The branch is selected from frozen OOS raw RankIC direction, its two-sided
5% t-stat threshold (`|t| >= {threshold:.2f}` in the frozen direction), and
same-direction RankIC in all four PIT pools. Sharpe is shown as a diagnostic
and is not used to select the branch. A3 is a secondary P1 diagnostic and
does not alter this five-way A1/A2/A0 decision tree.

## A0/A1/A2/A3 definitions

{definitions}

All four variants preserve raw-direction RankIC and use frozen effective
direction `{direction}` for H-L displays. No window is allowed to re-infer
the sign.

## CSI1000 full-common-sample metrics

{factor_table}

RankIC and ICIR above are raw-direction statistics. H-L Sharpe, MDD and
turnover use the frozen effective direction. Coverage is persisted stock-day
factor coverage, not an estimate inserted by this renderer.

The one-way cost label is **{fee:.1f} bps**, i.e.
`turnover × {fee:.1f}/10,000 × 250` for the display-only implied annual fee.
Gross H-L returns remain fee-zero sorting diagnostics.

## Four PIT pools

{pool_table}

The pools are ALL SSE/SZSE A-shares, CSI300, CSI500 and CSI1000, with
point-in-time membership where applicable.

## A0 parity gate

- gate: `{gate}`;
- Pearson: `{pearson}`;
- Spearman: `{spearman}`;
- maximum absolute error: `{max_error}`;
- mean absolute error: `{mean_error}`;
- share within `1e-12`: `{within}`.

## Headline figures

{figures}

## Scope boundary

This is standalone single-factor evidence. It does not run factor-library
correlation, factor combination, incremental IC, portfolio optimization or
alpha stacking.
""".format(
        branch=decision.branch,
        threshold=SIGNIFICANCE_TSTAT,
        definitions=_table(
            ("Role", "Factor ID", "Normalization", "Frozen bucket"),
            _factor_definition_rows(data),
        ),
        direction=data.direction,
        factor_table=_table(
            (
                "Role",
                "RankIC",
                "ICIR",
                "IC t-stat",
                "H-L Sharpe",
                "H-L MDD",
                "H-L turnover",
                "Coverage",
            ),
            factor_rows,
        ),
        fee=data.fee_bps,
        pool_table=_table(("Role", "Pool", "RankIC", "ICIR"), pool_rows),
        gate=parity["gate"],
        pearson=_number(parity["pearson"], 12),
        spearman=_number(parity["spearman"], 12),
        max_error="{:.3e}".format(
            _finite_number(parity["max_abs_error"], "parity max error")
        ),
        mean_error="{:.3e}".format(
            _finite_number(parity["mean_abs_error"], "parity mean error")
        ),
        within=_pct(parity["within_1e_12_share"]),
        figures=_figure_markdown(
            data,
            ("01_factor_variant_summary", "02_universe_variant_summary"),
        ),
    )


def _render_standalone(data: ReportInputs) -> str:
    factor = _ordered(data.tables["factor_variant_summary.csv"], ("factor_role",))
    decile = _ordered(data.tables["csi1000_decile_summary.csv"], ("factor_role",))
    universe = _ordered(
        data.tables["universe_variant_summary.csv"],
        ("factor_role", "universe"),
    )
    merged = factor.merge(
        decile,
        on=("factor_id", "factor_role"),
        validate="one_to_one",
        suffixes=("", "_decile"),
    )
    rows = [
        (
            row.factor_role,
            _pct(row.rank_ic),
            _number(row.icir, 2),
            _number(row.rank_ic_tstat, 2),
            _pct(row.hl_annu_ret),
            _number(row.hl_sharpe, 2),
            _pct(row.hl_mdd),
            _number(row.hl_turnover, 2),
            _number(row.decile_monotonicity_spearman, 3),
            _pct(row.csi1000_index_excess_hl_annu_ret),
            _number(row.csi1000_index_excess_hl_turnover, 2),
            _pct(row.implied_annu_fee_7p5bps_decile),
            _pct(row.factor_coverage_ratio),
        )
        for row in merged.itertuples(index=False)
    ]
    universe_rows = [
        (
            row.factor_role,
            row.universe,
            _pct(row.rank_ic),
            _number(row.rank_ic_tstat, 2),
            _number(row.icir, 2),
            _number(row.hl_sharpe, 2),
            _pct(row.hl_mdd),
            _number(row.hl_turnover, 2),
        )
        for row in universe.itertuples(index=False)
    ]
    return """# 05 — Standalone Validation

## CSI1000 standalone and decile diagnostics

{summary}

`Implied fee` is the persisted display-only annual deduction under a one-way
**{fee:.1f} bps** label. It is not a claim that the gross H-L sort is a
deployable strategy. RankIC, ICIR, Sharpe, MDD, turnover and coverage are
reported together so that a high Sharpe cannot hide weak information
coefficient evidence or sparse coverage.

## Four-pool validation

{universes}

The four-pool table uses exact valid-universe equal weighting. Cross-pool
direction is a decision input; the maximum Sharpe across pools is not.

## Per-variant decile figures

{figures}
""".format(
        summary=_table(
            (
                "Role",
                "RankIC",
                "ICIR",
                "t-stat",
                "H-L ann. return",
                "Sharpe",
                "MDD",
                "Turnover",
                "Decile mono.",
                "Index-excess H-L",
                "Decile turnover",
                "Implied fee",
                "Coverage",
            ),
            rows,
        ),
        fee=data.fee_bps,
        universes=_table(
            (
                "Role",
                "Pool",
                "RankIC",
                "t-stat",
                "ICIR",
                "Sharpe",
                "MDD",
                "Turnover",
            ),
            universe_rows,
        ),
        figures=_figure_markdown(
            data,
            ("03_decile_annualized", "04_decile_cumulative"),
        ),
    )


def _render_normalization(data: ReportInputs) -> str:
    distribution = data.tables["normalization_distribution_summary_adv.csv"]
    universe_distribution = distribution.loc[
        distribution["group_type"] == "universe"
    ].sort_values(["group", "quantile"])
    cap_distribution = distribution.loc[
        distribution["group_type"] == "market_cap_quintile"
    ].sort_values(["group", "quantile"])
    adv_grid = data.tables["parameter_stability_adv_distribution.csv"].copy()
    adv_grid["render_frozen"] = adv_grid["frozen_main"].map(
        lambda value: _bool_value(value, "ADV frozen_main")
    )
    adv_grid = adv_grid.sort_values(
        ["render_frozen", "abs_coverage_diff_vs_a0"],
        ascending=[False, True],
    )
    size = data.tables["normalization_by_size_bucket_calibration.csv"]
    frozen_adv = adv_grid.loc[adv_grid["render_frozen"]].iloc[0]
    expected_variant = "A1_L{}_H{}".format(
        _number(frozen_adv["lower_adv_bps_exclusive"], 1).rstrip("0").rstrip("."),
        _number(frozen_adv["upper_adv_bps_inclusive"], 1).rstrip("0").rstrip("."),
    )
    selected_size = size.loc[
        size["variant"].astype(str).str.upper() == expected_variant.upper()
    ].sort_values("bucket")
    if selected_size.empty:
        _fail(
            "normalization_by_size_bucket_calibration.csv lacks frozen {}".format(
                expected_variant
            )
        )

    coverage = _ordered(
        data.tables["missing_scale_coverage.csv"],
        ("factor_role",),
    )
    parameters = data.tables["parameter_stability.csv"].copy()
    parameters["_role_order"] = parameters["factor_role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    parameters = parameters.sort_values(["_role_order", "lower_bound", "upper_bound"])

    return """# 06 — Normalization Diagnostics

## Frozen A1 distribution calibration

The A1 bounds were selected from the 2023-H1 execution-size distribution,
without return data. Values below are persisted bps-of-ADV20 observations.

### Quantiles across four pools

{universe_distribution}

### Median by CSI1000 market-cap quintile

{cap_distribution}

## ADV candidate coverage grid

{adv_grid}

The frozen row is selected by distribution coverage relative to A0, subject
to the persisted 10%–80% cap-quintile gate. Return, RankIC and Sharpe are not
inputs to this calibration.

## Frozen A1 coverage by market-cap bucket

{size_table}

## Missing-scale and factor coverage

{coverage}

ADV20 and ATS20 require exactly 20 lagged market trading dates. Missing
history remains missing; the renderer does not fill or extrapolate it.

## Return-side parameter stability

{parameters}

This table is diagnostic after freezing. The selected flags and all candidate
metrics are read from `parameter_stability.csv`; no candidate is selected by
the best Sharpe or ICIR.

## Parameter-stability figure

{figures}
""".format(
        universe_distribution=_table(
            ("Pool", "Quantile", "bps of ADV"),
            (
                (row.group, _number(row.quantile, 2), _number(row.value, 4))
                for row in universe_distribution.itertuples(index=False)
            ),
        ),
        cap_distribution=_table(
            ("Cap quintile", "Median bps of ADV"),
            (
                (row.group, _number(row.value, 4))
                for row in cap_distribution.loc[
                    (cap_distribution["quantile"] - 0.5).abs() < 1e-12
                ].itertuples(index=False)
            ),
        ),
        adv_grid=_table(
            (
                "L bps",
                "H bps",
                "Frozen",
                "Coverage",
                "A0 coverage",
                "|diff|",
                "Mean cap-Q |diff|",
                "Min cap-Q",
                "Max cap-Q",
            ),
            (
                (
                    _number(row.lower_adv_bps_exclusive, 1),
                    _number(row.upper_adv_bps_inclusive, 1),
                    _yes_no(bool(row.render_frozen)),
                    _pct(row.overall_amount_coverage),
                    _pct(row.a0_overall_amount_coverage),
                    _pct(row.abs_coverage_diff_vs_a0),
                    _pct(row.mean_abs_quintile_coverage_diff_vs_a0),
                    _pct(row.minimum_quintile_coverage),
                    _pct(row.maximum_quintile_coverage),
                )
                for row in adv_grid.itertuples(index=False)
            ),
        ),
        size_table=_table(
            ("Bucket type", "Bucket", "A1 coverage", "A0 coverage"),
            (
                (
                    row.bucket_type,
                    _integer(row.bucket),
                    _pct(row.amount_coverage),
                    _pct(row.a0_amount_coverage),
                )
                for row in selected_size.itertuples(index=False)
            ),
        ),
        coverage=_table(
            (
                "Role",
                "Required scale",
                "Expected stock-days",
                "Factor stock-days",
                "Coverage",
                "Missing scale",
                "Coverage given scale",
            ),
            (
                (
                    row.factor_role,
                    row.required_scale,
                    _integer(row.expected_stock_days),
                    _integer(row.factor_stock_days),
                    _pct(row.factor_coverage_ratio),
                    _pct(row.missing_scale_ratio),
                    _pct(row.factor_coverage_given_scale),
                )
                for row in coverage.itertuples(index=False)
            ),
        ),
        parameters=_table(
            (
                "Role",
                "Factor ID",
                "Bounds",
                "Unit",
                "Selected",
                "RankIC",
                "ICIR",
                "t-stat",
            ),
            (
                (
                    row.factor_role,
                    row.factor_id,
                    "({}, {}]".format(
                        _number(row.lower_bound, 2),
                        _number(row.upper_bound, 2),
                    ),
                    row.parameter_unit,
                    _yes_no(
                        _bool_value(row.is_selected, "parameter is_selected")
                    ),
                    _pct(row.rank_ic),
                    _number(row.icir, 2),
                    _number(row.rank_ic_tstat, 2),
                )
                for row in parameters.itertuples(index=False)
            ),
        ),
        figures=_figure_markdown(data, ("07_parameter_stability",)),
    )


def _quintile_rows(frame: pd.DataFrame) -> Iterable[Sequence[Any]]:
    order = {value: index for index, value in enumerate(QUINTILES)}
    use = frame.loc[frame["quantile"].isin(QUINTILES)].copy()
    use["_role"] = use["factor_role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    use["_quantile"] = use["quantile"].map(order)
    use = use.sort_values(["_role", "_quantile"])
    for row in use.itertuples(index=False):
        yield (
            row.factor_role,
            row.quantile,
            _pct(row.rank_ic_mean),
            _number(row.icir, 2),
            _pct(row.coverage_rate),
            _number(row.n_names_avg, 1),
            _integer(row.n_days),
        )


def _render_exposure(data: ReportInputs) -> str:
    cap = data.tables["csi1000_cap_quintile_statistics.csv"]
    adv = data.tables["csi1000_adv_quintile_statistics.csv"]
    ols = data.tables["ols_diagnostics.csv"].copy()
    ols["_role"] = ols["factor_role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    ols["_method"] = ols["ols_method"].map(
        {method: index for index, method in enumerate(OLS_METHODS)}
    )
    ols = ols.sort_values(["_role", "_method"])
    return """# 07 — Exposure Diagnostics

## CSI1000 market-cap quintiles

{cap}

## CSI1000 lagged-ADV quintiles

{adv}

Both characteristic sorts are daily cross-sectional quintiles. The tables
show actual per-stratum coverage and name counts; a missing stratum is a hard
input failure rather than an omitted row.

## OLS diagnostics: raw / industry / cap / joint

{ols}

OLS rows are standalone residualization diagnostics. Residualization cannot
reclassify Tick executions and does not create alpha. Retention is measured
against each role's persisted raw RankIC.

## Exposure figures

{figures}
""".format(
        cap=_table(
            (
                "Role",
                "Cap quintile",
                "RankIC",
                "ICIR",
                "Coverage",
                "Avg names",
                "Days",
            ),
            _quintile_rows(cap),
        ),
        adv=_table(
            (
                "Role",
                "ADV quintile",
                "RankIC",
                "ICIR",
                "Coverage",
                "Avg names",
                "Days",
            ),
            _quintile_rows(adv),
        ),
        ols=_table(
            (
                "Role",
                "OLS method",
                "RankIC",
                "ICIR",
                "t-stat",
                "|RankIC| retained",
            ),
            (
                (
                    row.factor_role,
                    row.ols_method,
                    _pct(row.rank_ic),
                    _number(row.icir, 2),
                    _number(row.rank_ic_tstat, 2),
                    _pct(row.abs_rank_ic_retained_vs_raw),
                )
                for row in ols.itertuples(index=False)
            ),
        ),
        figures=_figure_markdown(
            data,
            ("06_cap_adv_quintiles", "09_ols_diagnostics"),
        ),
    )


def _rolling_rows(data: ReportInputs) -> Iterable[Sequence[Any]]:
    rolling = data.tables["csi1000_rolling_63d_rank_ic.csv"].copy()
    for role in ROLES:
        part = rolling.loc[rolling["factor_role"] == role].sort_values("TradeDate")
        finite = part.loc[part["rank_ic_63d_mean"].notna()]
        values = finite["rank_ic_63d_mean"].astype(float)
        latest = finite.iloc[-1]
        yield (
            role,
            pd.Timestamp(latest["TradeDate"]).strftime("%Y-%m-%d"),
            _pct(latest["rank_ic_63d_mean"]),
            _pct(values.min()),
            _pct(values.max()),
            _pct(float((values * data.direction > 0).mean())),
            _integer(latest["rank_ic_63d_count"]),
        )


def _render_time_state(data: ReportInputs) -> str:
    monthly = data.tables["csi1000_monthly_rank_ic.csv"].copy()
    monthly["_role"] = monthly["factor_role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    monthly = monthly.sort_values(["_role", "month"])
    state = data.tables["state_turnover_tercile_summary.csv"].copy()
    state["_role"] = state["factor_role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    state["_tercile"] = state["turnover_tercile"].map(
        {value: index for index, value in enumerate(TURNOVER_TERCILES)}
    )
    state = state.sort_values(["_role", "_tercile"])
    segments = data.tables["sample_segment_results.csv"].copy()
    segments["_role"] = segments["factor_role"].map(
        {role: index for index, role in enumerate(ROLES)}
    )
    segments["_segment"] = segments["segment"].map(
        {value: index for index, value in enumerate(SEGMENTS)}
    )
    segments = segments.sort_values(["_role", "_segment"])
    return """# 08 — Time and State Robustness

## Monthly raw RankIC

{monthly}

## Rolling 63-trading-day RankIC

{rolling}

The 63-day table is derived only from persisted rolling observations. Its
direction share is the fraction of finite windows matching frozen direction
`{direction}`.

## Lagged-turnover terciles

{state}

Low/Mid/High states use lagged turnover information. The direction is not
re-fitted within a state.

## Frozen IS / validation / OOS

{segments}

IS, validation and OOS are shown separately. Parameters and direction remain
frozen in every segment; OOS raw RankIC and t-stat, rather than the segment's
highest Sharpe, drive the final decision.

## Time, state, segment and coverage figures

{figures}
""".format(
        monthly=_table(
            (
                "Role",
                "Month",
                "RankIC",
                "ICIR",
                "Negative-IC days",
                "Days",
            ),
            (
                (
                    row.factor_role,
                    row.month,
                    _pct(row.rank_ic_mean),
                    _number(row.icir, 2),
                    _pct(row.negative_ic_day_share),
                    _integer(row.n_days),
                )
                for row in monthly.itertuples(index=False)
            ),
        ),
        rolling=_table(
            (
                "Role",
                "Latest date",
                "Latest 63d",
                "Minimum 63d",
                "Maximum 63d",
                "Frozen-direction windows",
                "Latest count",
            ),
            _rolling_rows(data),
        ),
        direction=data.direction,
        state=_table(
            (
                "Role",
                "Turnover tercile",
                "RankIC",
                "ICIR",
                "Negative-IC days",
                "Days",
            ),
            (
                (
                    row.factor_role,
                    row.turnover_tercile,
                    _pct(row.rank_ic),
                    _number(row.icir, 2),
                    _pct(row.negative_ic_day_share),
                    _integer(row.n_days),
                )
                for row in state.itertuples(index=False)
            ),
        ),
        segments=_table(
            (
                "Role",
                "Segment",
                "Actual dates",
                "RankIC",
                "t-stat",
                "ICIR",
                "Sharpe",
                "MDD",
                "Turnover",
            ),
            (
                (
                    row.factor_role,
                    row.segment,
                    "{} to {}".format(row.actual_start, row.actual_end),
                    _pct(row.rank_ic),
                    _number(row.rank_ic_tstat, 2),
                    _number(row.icir, 2),
                    _number(row.hl_sharpe, 2),
                    _pct(row.hl_mdd),
                    _number(row.hl_turnover, 2),
                )
                for row in segments.itertuples(index=False)
            ),
        ),
        figures=_figure_markdown(
            data,
            (
                "05_ic_stability",
                "08_turnover_tercile",
                "10_segments_and_coverage",
            ),
        ),
    )


def _parameter_direction_rows(data: ReportInputs) -> Iterable[Sequence[Any]]:
    parameters = data.tables["parameter_stability.csv"]
    for role in ROLES:
        part = parameters.loc[parameters["factor_role"].astype(str) == role]
        matching = int((part["rank_ic"].astype(float) * data.direction > 0).sum())
        selected = part.loc[
            part["is_selected"].map(
                lambda value: _bool_value(value, "parameter is_selected")
            )
        ].iloc[0]
        yield (
            role,
            "{}/{}".format(matching, len(part)),
            _pct(matching / float(len(part))),
            _pct(selected["rank_ic"]),
            _number(selected["rank_ic_tstat"], 2),
            _number(selected["icir"], 2),
        )


def _render_decision(data: ReportInputs, decision: DecisionResult) -> str:
    evidence_rows = [
        (
            role,
            _pct(item.oos_rank_ic),
            _number(item.oos_tstat, 2),
            _yes_no(item.expected_direction),
            _yes_no(item.significant),
            "{}/{}".format(item.stable_pools, item.total_pools),
            _yes_no(item.retained),
        )
        for role, item in (
            (role, decision.evidence[role]) for role in ROLES
        )
    ]
    return """# 10 — Research Decision

## Final branch

**`{branch}`**

## Frozen decision evidence

{evidence}

A role is retained only if all three primary gates pass:

1. CSI1000 OOS raw RankIC has frozen direction `{direction}`;
2. its direction-adjusted raw RankIC t-stat is at least `{threshold:.2f}`;
3. raw RankIC has the frozen direction in ALL, CSI300, CSI500 and CSI1000.

The four-pool evidence is the persisted full-common-sample universe table;
the persisted segment table supplies CSI1000 OOS significance. No
segment-by-pool result is invented.

The branch order is fixed: both A1/A2, A1 only, A2 only, A0-only fallback,
then no retained normalized relation. A3 remains a secondary P1 diagnostic
and cannot select a branch.

## Parameter-stability evidence

{parameters}

Parameter stability is supporting evidence after parameters were frozen. It
cannot overturn failed OOS direction/significance gates and cannot select the
highest Sharpe.

## Cost and interpretation limits

- the cost label is one-way **{fee:.1f} bps**, display-only;
- H-L is a gross standalone sorting diagnostic, not a portfolio strategy;
- Tick execution amount does not identify investor type;
- no causal claim follows from OLS residualization.

## Explicitly outside scope

This report does **not** perform factor-library correlation analysis, factor
combination, incremental-IC testing, portfolio optimization, alpha stacking,
or return-based parameter optimization.
""".format(
        branch=decision.branch,
        evidence=_table(
            (
                "Role",
                "OOS raw RankIC",
                "OOS t-stat",
                "Frozen direction",
                "Significant",
                "Pools same direction",
                "Retained",
            ),
            evidence_rows,
        ),
        direction=data.direction,
        threshold=SIGNIFICANCE_TSTAT,
        parameters=_table(
            (
                "Role",
                "Candidates same direction",
                "Share",
                "Selected RankIC",
                "Selected t-stat",
                "Selected ICIR",
            ),
            _parameter_direction_rows(data),
        ),
        fee=data.fee_bps,
    )


def render_chapters(data: ReportInputs) -> Mapping[str, str]:
    """Create all chapter text in memory, after input validation."""
    decision = determine_research_decision(
        data.tables["sample_segment_results.csv"],
        data.tables["universe_variant_summary.csv"],
        data.direction,
    )
    chapters: "OrderedDict[str, str]" = OrderedDict(
        [
            ("01_executive_summary.md", _render_executive(data, decision)),
            ("05_standalone_validation.md", _render_standalone(data)),
            ("06_normalization_diagnostics.md", _render_normalization(data)),
            ("07_exposure_diagnostics.md", _render_exposure(data)),
            ("08_time_and_state_robustness.md", _render_time_state(data)),
            ("10_research_decision.md", _render_decision(data, decision)),
        ]
    )
    if tuple(chapters) != CHAPTER_FILES:
        _fail("Internal chapter list changed unexpectedly")
    combined = "\n".join(chapters.values())
    forbidden = ("to" + "do", "place" + "holder")
    for token in forbidden:
        if re.search(r"\b{}\b".format(token), combined, flags=re.IGNORECASE):
            _fail("Rendered Markdown contains forbidden draft marker")
    if "7.5 bps" not in combined:
        _fail("Rendered Markdown lacks the required 7.5 bps label")
    for class_id in FIGURE_CLASSES:
        expected_files = data.figures[class_id].files
        if not all("]({})".format(path) in combined for path in expected_files):
            _fail("Rendered Markdown omits figure class {}".format(class_id))
    return chapters


def _atomic_write_chapters(
    report_root: Path,
    chapters: Mapping[str, str],
) -> Mapping[str, Path]:
    temporary: List[Tuple[Path, Path]] = []
    try:
        for filename, text in chapters.items():
            target = report_root / filename
            temp = target.with_name(
                ".{}.{}.tmp".format(target.name, uuid.uuid4().hex)
            )
            temp.write_text(text.rstrip() + "\n", encoding="utf-8")
            temporary.append((temp, target))
        for temp, target in temporary:
            os.replace(str(temp), str(target))
        return OrderedDict(
            (filename, (report_root / filename).resolve()) for filename in chapters
        )
    finally:
        for temp, _ in temporary:
            if temp.exists():
                temp.unlink()


def render_report(
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> Mapping[str, Path]:
    """Validate persisted evidence, then replace exactly six Markdown chapters."""
    data = load_report_inputs(report_root)
    chapters = render_chapters(data)
    return _atomic_write_chapters(data.report_root, chapters)


render_normalized_markdown = render_report
render_markdown = render_report
render_mid_trade_amount_normalized_markdown = render_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="normalized_v1 report root containing artifacts/ and figures/",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    written = render_report(args.report_root)
    for path in written.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
