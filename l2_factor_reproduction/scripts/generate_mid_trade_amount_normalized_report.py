#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate normalized mid-trade-amount research artifacts.

The script consumes already persisted ``normalized_v1`` factor panels, daily
scale panels, and a frozen configuration.  It deliberately does not query raw
Tick data and never selects a factor direction or a normalized threshold from
evaluated returns.

Input contracts
---------------
Factor panels are long parquet data with:
``TradeDate, symbol, value, factor_id``.

Scale panels are long parquet data with ``TradeDate, symbol`` plus the scale
columns produced by the normalized cache builder (notably ``adv20_lag1`` and
``ats20_lag1``).  ``Symbol`` is accepted as an alias for ``symbol``.

The command-line entry point loads the same Wind return, tradability, and PIT
membership panels used by ``generate_mid_order_ratio_report_artifacts.py``.
All computational helpers below accept DataFrames directly so unit tests do
not need a database connection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import groupTest, mad, zsc  # noqa: E402
from l2_factor_reproduction.python.mid_trade_amount_normalization import (  # noqa: E402
    A0_FACTOR_ID,
    A1_FACTOR_ID,
    A2_FACTOR_ID,
    A3_FACTOR_ID,
    validate_frozen_config,
)
from l2_factor_reproduction.scripts import (  # noqa: E402
    generate_mid_order_ratio_report_artifacts as official_report,
)


DEFAULT_INPUT_ROOT = (
    PROJ_ROOT
    / "research/results/l2_reproduction/mid_order_ratio/normalized_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJ_ROOT
    / "research/reports/factors/mid_order_ratio/normalized_v1"
)

FROZEN_EFFECTIVE_DIRECTION = official_report.FROZEN_EFFECTIVE_DIRECTION
UNIVERSES = official_report.UNIVERSES
align_core_panels = official_report.align_core_panels
evaluate_prepared = official_report.evaluate_prepared

FEE_BPS = 7.5
ANNUALIZATION = 250
MIN_CROSS_SECTION = 20
STYLE_WARMUP_CALENDAR_DAYS = 60

PANEL_FILE_CANDIDATES = (
    "factor_panels.parquet",
    "factor_panels_long.parquet",
    "normalized_factor_panels.parquet",
    "factor_panel.parquet",
    "factors.parquet",
)
SCALE_FILE_CANDIDATES = (
    "scales.parquet",
    "lagged_scales.parquet",
    "trade_size_scales.parquet",
    "adv20_lag1.parquet",
    "daily_scales.parquet",
    "scale_panels.parquet",
    "scale_panel.parquet",
    "daily_scale_primitives.parquet",
)
CONFIG_FILE_CANDIDATES = ("frozen_config.json",)

DEFAULT_SEGMENTS = OrderedDict(
    [
        ("IS", ("2023-01-04", "2024-06-28")),
        ("validation", ("2023-07-03", "2024-06-28")),
        ("OOS", ("2024-07-01", "2026-07-31")),
    ]
)

# Exactly ten chart classes.  Classes 03 and 04 create one file per factor
# version, as requested; they are never combined into a single ambiguous plot.
FIGURE_CLASSES = OrderedDict(
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

ROLE_DISPLAY = {
    "A0": "A0 fixed RMB (40k, 200k]",
    "A1": "A1 ADV20-normalized (frozen)",
    "A2": "A2 ATS20-normalized (0.5x, 2.0x]",
    "A3": "A3 daily Q20-Q80 (P1)",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
    }
)


class ReportDataError(RuntimeError):
    """Raised when a required persisted or market-data input is unavailable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "factor"


def _normalize_symbol(value: Any) -> str:
    text = str(value).strip().upper()
    if re.fullmatch(r"\d{6}", text):
        suffix = ".SH" if text.startswith("6") else ".SZ"
        return text + suffix
    return text


def _fmt_date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _fmt_date(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _file_record(path: Path, role: str, relative_to: Optional[Path] = None) -> Dict[str, Any]:
    resolved = path.resolve()
    display = str(resolved)
    if relative_to is not None:
        try:
            display = str(resolved.relative_to(relative_to.resolve()))
        except ValueError:
            pass
    return {
        "role": role,
        "path": display,
        "absolute_path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _discover_file(
    root: Path,
    explicit: Optional[Path],
    candidates: Sequence[str],
    description: str,
) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else root / explicit
        if not path.is_file():
            raise ReportDataError(f"Required {description} does not exist: {path}")
        return path.resolve()

    matches: List[Path] = []
    for name in candidates:
        matches.extend(path for path in root.rglob(name) if path.is_file())
    unique = sorted(set(path.resolve() for path in matches))
    if not unique:
        expected = ", ".join(candidates)
        raise ReportDataError(
            f"Required {description} is missing under {root}. "
            f"Expected one of: {expected}. Build normalized_v1 first."
        )
    if len(unique) > 1:
        listed = "\n  - ".join(str(path) for path in unique)
        raise ReportDataError(
            f"Multiple possible {description} files were found; pass an explicit path:\n"
            f"  - {listed}"
        )
    return unique[0]


def resolve_input_paths(
    input_root: Path,
    panel_file: Optional[Path] = None,
    scales_file: Optional[Path] = None,
    frozen_config: Optional[Path] = None,
) -> Dict[str, Path]:
    """Resolve all mandatory persisted inputs before creating report output."""
    root = input_root.resolve()
    if not root.is_dir():
        raise ReportDataError(
            f"normalized_v1 input directory is missing: {root}. "
            "No report was generated."
        )
    return {
        "factor_panels": _discover_file(
            root, panel_file, PANEL_FILE_CANDIDATES, "factor panel parquet"
        ),
        "scales": _discover_file(
            root, scales_file, SCALE_FILE_CANDIDATES, "scale panel parquet"
        ),
        "frozen_config": _discover_file(
            root, frozen_config, CONFIG_FILE_CANDIDATES, "frozen_config JSON"
        ),
    }


def _canonicalize_key_columns(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    aliases = {
        "TradeDate": ("TradeDate", "trade_date", "tradetime", "date", "Date"),
        "symbol": ("symbol", "Symbol", "S_INFO_WINDCODE"),
    }
    rename: Dict[str, str] = {}
    for canonical, choices in aliases.items():
        found = next((name for name in choices if name in frame.columns), None)
        if found is None:
            raise ReportDataError(
                f"{description} must contain {canonical!r}; columns={list(frame.columns)}"
            )
        rename[found] = canonical
    out = frame.rename(columns=rename).copy()
    out["TradeDate"] = pd.to_datetime(out["TradeDate"], errors="coerce").dt.normalize()
    if out["TradeDate"].isna().any():
        count = int(out["TradeDate"].isna().sum())
        raise ReportDataError(f"{description} contains {count} invalid TradeDate values")
    out["symbol"] = out["symbol"].map(_normalize_symbol)
    if out["symbol"].eq("").any():
        raise ReportDataError(f"{description} contains empty symbols")
    return out


def validate_factor_panels(
    frame: pd.DataFrame, *, sort_output: bool = True
) -> pd.DataFrame:
    """Validate and normalize the persisted long factor-panel contract."""
    out = _canonicalize_key_columns(frame, "factor panels")
    required = {"value", "factor_id"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ReportDataError(f"factor panels missing columns: {missing}")
    out["factor_id"] = out["factor_id"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if out["factor_id"].eq("").any():
        raise ReportDataError("factor panels contain empty factor_id values")
    duplicated = out.duplicated(["factor_id", "TradeDate", "symbol"], keep=False)
    if duplicated.any():
        sample = out.loc[
            duplicated, ["factor_id", "TradeDate", "symbol"]
        ].head(5)
        raise ReportDataError(
            "factor panels violate unique factor_id+TradeDate+symbol keys; "
            f"sample={sample.to_dict(orient='records')}"
        )
    if not out["value"].notna().any():
        raise ReportDataError("factor panels contain no finite factor values")
    if sort_output:
        return out.sort_values(
            ["factor_id", "TradeDate", "symbol"]
        ).reset_index(drop=True)
    return out.reset_index(drop=True)


def validate_scales(
    frame: pd.DataFrame, *, sort_output: bool = True
) -> pd.DataFrame:
    """Validate and normalize the persisted long daily-scale contract."""
    out = _canonicalize_key_columns(frame, "scales")
    duplicated = out.duplicated(["TradeDate", "symbol"], keep=False)
    if duplicated.any():
        sample = out.loc[duplicated, ["TradeDate", "symbol"]].head(5)
        raise ReportDataError(
            "scales violate unique TradeDate+symbol keys; "
            f"sample={sample.to_dict(orient='records')}"
        )
    for column in out.columns:
        if column not in ("TradeDate", "symbol"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if sort_output:
        return out.sort_values(["TradeDate", "symbol"]).reset_index(drop=True)
    return out.reset_index(drop=True)


def load_persisted_inputs(
    paths: Mapping[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    try:
        panels = pd.read_parquet(paths["factor_panels"])
    except Exception as exc:
        raise ReportDataError(
            f"Unable to read factor panels {paths['factor_panels']}: {exc}"
        ) from exc
    try:
        scales = pd.read_parquet(paths["scales"])
    except Exception as exc:
        raise ReportDataError(
            f"Unable to read scales {paths['scales']}: {exc}"
        ) from exc
    try:
        config = json.loads(paths["frozen_config"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReportDataError(
            f"Unable to read frozen config {paths['frozen_config']}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ReportDataError("frozen_config.json must contain a JSON object")
    try:
        validate_frozen_config(config)
    except (TypeError, ValueError) as exc:
        raise ReportDataError(f"frozen_config integrity check failed: {exc}") from exc
    return (
        validate_factor_panels(panels, sort_output=False),
        validate_scales(scales, sort_output=False),
        config,
    )


def replace_with_authoritative_a0(
    factor_panels: pd.DataFrame, authoritative_path: Path
) -> pd.DataFrame:
    """Use the parity-gated strict A0 panel as the report headline source."""
    try:
        authoritative = validate_factor_panels(
            pd.read_parquet(authoritative_path)
        )
    except Exception as exc:
        raise ReportDataError(
            f"Unable to read authoritative A0 panel {authoritative_path}: {exc}"
        ) from exc
    factor_ids = set(authoritative["factor_id"].unique())
    if factor_ids != {A0_FACTOR_ID}:
        raise ReportDataError(
            "authoritative A0 panel must contain exactly "
            f"{A0_FACTOR_ID!r}; observed={sorted(factor_ids)}"
        )
    existing = factor_panels.loc[
        factor_panels["factor_id"].eq(A0_FACTOR_ID),
        ["TradeDate", "symbol"],
    ]
    if existing.empty:
        raise ReportDataError("dynamic factor panels do not contain A0")
    key_audit = existing.merge(
        authoritative[["TradeDate", "symbol"]],
        on=["TradeDate", "symbol"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not key_audit["_merge"].eq("both").all():
        counts = key_audit["_merge"].value_counts().to_dict()
        raise ReportDataError(
            "authoritative A0 keys differ from the parity-gated dynamic panel: "
            f"{counts}"
        )
    replacement = authoritative.set_index(["TradeDate", "symbol"])["value"]
    a0_mask = factor_panels["factor_id"].eq(A0_FACTOR_ID)
    existing_index = pd.MultiIndex.from_frame(
        factor_panels.loc[a0_mask, ["TradeDate", "symbol"]]
    )
    factor_panels.loc[a0_mask, "value"] = replacement.reindex(
        existing_index
    ).to_numpy()
    return factor_panels


def factor_panels_to_wide(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Convert validated long panels to one Date×symbol panel per factor_id."""
    panels: Dict[str, pd.DataFrame] = {}
    for factor_id, part in frame.groupby("factor_id", sort=True):
        wide = part.pivot(index="TradeDate", columns="symbol", values="value")
        wide.index = pd.to_datetime(wide.index).normalize()
        panels[str(factor_id)] = wide.sort_index().sort_index(axis=1)
    return panels


def scale_to_wide(scales: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in scales.columns:
        raise ReportDataError(
            f"Required scale column {column!r} is missing; "
            f"available={list(scales.columns)}"
        )
    wide = scales.pivot(index="TradeDate", columns="symbol", values=column)
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.sort_index().sort_index(axis=1)


def find_scale_column(
    scales: pd.DataFrame, aliases: Sequence[str], description: str
) -> str:
    lower = {str(column).lower(): str(column) for column in scales.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise ReportDataError(
        f"Required {description} scale is missing. Accepted aliases={list(aliases)}; "
        f"available={list(scales.columns)}"
    )


def infer_factor_role(factor_id: str) -> Optional[str]:
    text = factor_id.lower()
    if re.search(r"(^|[_-])a0([_-]|$)", text) or (
        ("abs" in text or "absolute" in text) and "adv" not in text and "ats" not in text
    ):
        return "A0"
    if re.search(r"(^|[_-])a1([_-]|$)", text) or "adv" in text:
        return "A1"
    if re.search(r"(^|[_-])a2([_-]|$)", text) or "ats" in text:
        return "A2"
    if re.search(r"(^|[_-])a3([_-]|$)", text) or "rollq" in text or (
        ("q20" in text and "q80" in text) or "quantile" in text
    ):
        return "A3"
    return None


def _extract_factor_id(value: Any, available: Sequence[str]) -> Optional[str]:
    available_set = set(map(str, available))
    if isinstance(value, str) and value in available_set:
        return value
    if isinstance(value, Mapping):
        for key in ("factor_id", "selected_factor_id", "candidate_id", "name"):
            item = value.get(key)
            if isinstance(item, str) and item in available_set:
                return item
    return None


def _find_values_for_keys(obj: Any, keys: Iterable[str]) -> List[Any]:
    wanted = {key.lower() for key in keys}
    found: List[Any] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                found.append(value)
            found.extend(_find_values_for_keys(value, keys))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_find_values_for_keys(value, keys))
    return found


def resolve_headline_factors(
    factor_ids: Sequence[str],
    frozen_config: Mapping[str, Any],
    require_core: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """Resolve report variants without consulting any return statistic.

    Explicit frozen-config ids take precedence.  Ambiguous A1/A2 grids are a
    hard error because choosing among them from report performance would violate
    the calibration freeze.
    """
    available = sorted(set(map(str, factor_ids)))
    if not available:
        raise ReportDataError("No factor_id values are available")

    explicit_lists = _find_values_for_keys(
        frozen_config,
        (
            "headline_factor_ids",
            "main_factor_ids",
            "selected_factor_ids",
            "report_factor_ids",
        ),
    )
    for value in explicit_lists:
        if isinstance(value, list) and value:
            selected = [str(item) for item in value]
            missing = sorted(set(selected) - set(available))
            if missing:
                raise ReportDataError(
                    f"Frozen headline factors are absent from panels: {missing}"
                )
            roles = {
                factor_id: infer_factor_role(factor_id) or f"V{index + 1}"
                for index, factor_id in enumerate(selected)
            }
            return selected, roles

    selected_by_role: Dict[str, str] = {}
    key_map = {
        "A0": ("a0", "a0_factor_id", "baseline_factor_id", "absolute_factor_id"),
        "A1": ("a1", "a1_factor_id", "selected_a1", "selected_a1_factor_id"),
        "A2": ("a2", "a2_factor_id", "selected_a2", "selected_a2_factor_id"),
        "A3": ("a3", "a3_factor_id", "selected_a3", "selected_a3_factor_id"),
    }
    for role, keys in key_map.items():
        for value in _find_values_for_keys(frozen_config, keys):
            factor_id = _extract_factor_id(value, available)
            if factor_id is not None:
                selected_by_role[role] = factor_id
                break

    canonical_ids = {
        "A0": A0_FACTOR_ID,
        "A1": A1_FACTOR_ID,
        "A2": A2_FACTOR_ID,
        "A3": A3_FACTOR_ID,
    }
    for role, factor_id in canonical_ids.items():
        if role not in selected_by_role and factor_id in available:
            selected_by_role[role] = factor_id

    candidates: Dict[str, List[str]] = {role: [] for role in key_map}
    for factor_id in available:
        role = infer_factor_role(factor_id)
        if role is not None:
            candidates[role].append(factor_id)

    for role in ("A0", "A1", "A2", "A3"):
        if role in selected_by_role:
            continue
        role_candidates = candidates[role]
        if len(role_candidates) == 1:
            selected_by_role[role] = role_candidates[0]
        elif role == "A2" and len(role_candidates) > 1:
            fixed = [
                item
                for item in role_candidates
                if re.search(r"(0[._p]?5|05x)", item.lower())
                and re.search(r"(^|[^0-9])2([._p]?0)?x?([^0-9]|$)", item.lower())
            ]
            if len(fixed) == 1:
                selected_by_role[role] = fixed[0]

    core_missing = [role for role in ("A0", "A1", "A2") if role not in selected_by_role]
    if require_core and core_missing:
        details = {
            role: candidates[role]
            for role in core_missing
        }
        raise ReportDataError(
            "Frozen headline factor ids are missing or ambiguous for "
            f"{core_missing}; candidates={details}. Record the selected ids in "
            "frozen_config.json. The report will not select by returns."
        )

    if selected_by_role:
        ordered_roles = [role for role in ("A0", "A1", "A2", "A3") if role in selected_by_role]
        selected = [selected_by_role[role] for role in ordered_roles]
        return selected, {selected_by_role[role]: role for role in ordered_roles}

    if len(available) == 1 or not require_core:
        return available, {
            factor_id: infer_factor_role(factor_id) or f"V{index + 1}"
            for index, factor_id in enumerate(available)
        }
    raise ReportDataError("Unable to resolve frozen headline variants")


def _direction_values(obj: Any) -> List[int]:
    values: List[int] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in ("effective_direction", "frozen_effective_direction"):
                if isinstance(value, Mapping):
                    for direction in value.values():
                        if isinstance(
                            direction,
                            (int, float, np.integer, np.floating),
                        ):
                            values.append(int(direction))
                        else:
                            values.extend(_direction_values(direction))
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(int(value))
            else:
                values.extend(_direction_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_direction_values(value))
    return values


def resolve_frozen_direction(frozen_config: Mapping[str, Any]) -> int:
    """Enforce the official direction; never infer sign from report returns."""
    declared = _direction_values(frozen_config)
    invalid = sorted(set(value for value in declared if value != FROZEN_EFFECTIVE_DIRECTION))
    if invalid:
        raise ReportDataError(
            "frozen_config direction conflicts with the official frozen direction "
            f"{FROZEN_EFFECTIVE_DIRECTION}: {invalid}"
        )
    if FROZEN_EFFECTIVE_DIRECTION not in (-1, 1):
        raise ReportDataError("Official frozen direction must be -1 or 1")
    return int(FROZEN_EFFECTIVE_DIRECTION)


def factor_display_name(factor_id: str, role: Optional[str]) -> str:
    if role in ROLE_DISPLAY:
        return ROLE_DISPLAY[str(role)]
    return str(factor_id)


def implied_annual_fee(avg_daily_turnover: float, fee_bps: float = FEE_BPS) -> float:
    """Display-only annual fee: daily H-L turnover × bps/1e4 × 250."""
    if pd.isna(avg_daily_turnover):
        return float("nan")
    return float(avg_daily_turnover) * float(fee_bps) / 10_000.0 * ANNUALIZATION


def expand_evaluation_summary(
    summary: Mapping[str, Any],
    factor_id: str,
    universe: str,
    rank_ic: Optional[pd.Series] = None,
    role: Optional[str] = None,
    fee_bps: float = FEE_BPS,
) -> Dict[str, Any]:
    """Add normalized-report labels, frozen sign, and 7.5bps diagnostics."""
    out = dict(summary)
    direction = int(out.get("effective_direction", FROZEN_EFFECTIVE_DIRECTION))
    if direction != FROZEN_EFFECTIVE_DIRECTION:
        raise ReportDataError(
            f"{factor_id} was evaluated with direction={direction}; "
            f"expected frozen {FROZEN_EFFECTIVE_DIRECTION}"
        )
    raw_ic = float(out.get("rank_ic", np.nan))
    raw_icir = float(out.get("icir", np.nan))
    turnover = float(out.get("hl_turnover", np.nan))
    gross = float(out.get("hl_annu_ret", np.nan))
    fee = implied_annual_fee(turnover, fee_bps=fee_bps)
    out.update(
        {
            "factor_id": str(factor_id),
            "factor_role": role or infer_factor_role(factor_id) or "other",
            "factor_label": factor_display_name(factor_id, role),
            "universe": str(universe),
            "rank_ic_direction": "raw",
            "effective_direction": direction,
            "effective_rank_ic": raw_ic * direction,
            "effective_icir": raw_icir * direction,
            "fee_bps_one_way": float(fee_bps),
            "implied_annu_fee_7p5bps": fee,
            "hl_net_annu_ret_after_implied_fee": gross - fee,
            "cost_treatment": "display-only; gross group returns are not fee-deducted",
        }
    )
    if rank_ic is not None:
        values = rank_ic.dropna()
        out["raw_ic_negative_day_share"] = (
            float((values < 0).mean()) if len(values) else np.nan
        )
        out["effective_ic_positive_day_share_recomputed"] = (
            float((values * direction > 0).mean()) if len(values) else np.nan
        )
    return out


def evaluate_wide_factor(
    factor: pd.DataFrame,
    ret_raw: pd.DataFrame,
    tradable: pd.DataFrame,
    member: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    effective_direction: int,
) -> Dict[str, Any]:
    signal, ret = align_core_panels(
        factor, ret_raw, tradable, member, pd.Timestamp(start), pd.Timestamp(end)
    )
    if signal.empty or ret.empty:
        raise ReportDataError(
            f"No aligned observations for {start.date()}~{end.date()}"
        )
    valid = (signal.notna() & ret.notna()).sum(axis=1)
    if not valid.ge(MIN_CROSS_SECTION).any():
        raise ReportDataError(
            f"No date has at least {MIN_CROSS_SECTION} aligned names"
        )
    return evaluate_prepared(signal, ret, effective_direction=effective_direction)


def evaluate_prepared_segment(
    prepared: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    effective_direction: int,
) -> Dict[str, Any]:
    """Evaluate a date slice of an already shifted signal.

    Slicing after alignment retains the prior factor date for the first return
    date in each segment and avoids an accidental extra shift.
    """
    signal = prepared["signal_raw"].loc[pd.Timestamp(start) : pd.Timestamp(end)]
    ret = prepared["ret_raw"].reindex_like(signal)
    if signal.empty:
        raise ReportDataError(
            f"No prepared observations in segment {start.date()}~{end.date()}"
        )
    return evaluate_prepared(signal, ret, effective_direction=effective_direction)


def evaluate_index_deciles(
    signal_effective: pd.DataFrame,
    ret_index_excess: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """CSI1000 index-excess deciles using the already frozen effective signal."""
    ret = ret_index_excess.reindex_like(signal_effective)
    _, pnl, turnover = groupTest(signal_effective, ret, n=10, info="silent")
    pnl = pnl.rename(columns=lambda value: str(value))
    turnover = turnover.rename(columns=lambda value: str(value))
    deciles = [str(value) for value in range(1, 11)]
    missing = [value for value in deciles + ["H-L"] if value not in pnl.columns]
    if missing:
        raise ReportDataError(f"groupTest did not return expected deciles: {missing}")
    annualized = pnl[deciles].mean() * ANNUALIZATION
    monotonicity = pd.Series(
        np.arange(1, 11, dtype=float), index=deciles
    ).corr(annualized, method="spearman")
    avg_turnover = float(turnover["H-L"].mean())
    summary = {
        "decile_monotonicity_spearman": float(monotonicity),
        "csi1000_index_excess_g10_minus_g1_annu_ret": float(
            annualized["10"] - annualized["1"]
        ),
        "csi1000_index_excess_hl_annu_ret": float(
            pnl["H-L"].mean() * ANNUALIZATION
        ),
        "csi1000_index_excess_hl_turnover": avg_turnover,
        "implied_annu_fee_7p5bps": implied_annual_fee(avg_turnover),
    }
    return pnl, turnover, summary


def summarize_monthly_ic(rank_ic: pd.Series, factor_id: str) -> pd.DataFrame:
    values = rank_ic.dropna().sort_index()
    if values.empty:
        return pd.DataFrame(
            columns=[
                "factor_id",
                "month",
                "rank_ic_mean",
                "rank_ic_std",
                "n_days",
                "icir",
                "negative_ic_day_share",
            ]
        )
    grouped = values.groupby(values.index.to_period("M"))
    rows = []
    for month, part in grouped:
        std = float(part.std(ddof=1))
        rows.append(
            {
                "factor_id": factor_id,
                "month": str(month),
                "rank_ic_mean": float(part.mean()),
                "rank_ic_std": std,
                "n_days": int(len(part)),
                "icir": (
                    float(part.mean() / std * math.sqrt(ANNUALIZATION))
                    if std > 0
                    else np.nan
                ),
                "negative_ic_day_share": float((part < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def rolling_ic_frame(
    rank_ic: pd.Series,
    factor_id: str,
    window: int = 63,
    min_periods: int = 21,
) -> pd.DataFrame:
    values = rank_ic.sort_index()
    return pd.DataFrame(
        {
            "TradeDate": values.index,
            "factor_id": factor_id,
            "rank_ic_raw": values.to_numpy(),
            f"rank_ic_{window}d_mean": values.rolling(
                window, min_periods=min_periods
            ).mean().to_numpy(),
            f"rank_ic_{window}d_count": values.rolling(
                window, min_periods=1
            ).count().to_numpy(),
        }
    )


def _quantile_daily_rows(
    signal_raw: pd.DataFrame,
    ret_raw: pd.DataFrame,
    characteristic: pd.DataFrame,
    factor_id: str,
    dimension: str,
    n_quantiles: int,
    characteristic_lag: int,
    min_group_names: int,
) -> pd.DataFrame:
    state = characteristic.shift(characteristic_lag).reindex_like(signal_raw)
    ret = ret_raw.reindex_like(signal_raw)
    rows: List[Dict[str, Any]] = []
    for date in signal_raw.index:
        signal_row = signal_raw.loc[date]
        ret_row = ret.loc[date]
        state_row = state.loc[date]
        base = signal_row.notna() & ret_row.notna()
        valid = base & state_row.notna() & np.isfinite(state_row)
        n_base = int(base.sum())
        n_valid = int(valid.sum())
        if n_valid < n_quantiles * min_group_names:
            continue
        state_valid = state_row[valid].rank(method="first", pct=True)
        quantile = np.ceil(state_valid * n_quantiles).clip(1, n_quantiles).astype(int)
        for number in range(1, n_quantiles + 1):
            names = quantile.index[quantile == number]
            if len(names) < min_group_names:
                continue
            values = signal_row.loc[names]
            returns = ret_row.loc[names]
            rows.append(
                {
                    "TradeDate": pd.Timestamp(date),
                    "factor_id": factor_id,
                    "dimension": dimension,
                    "quantile_number": number,
                    "quantile": f"Q{number}",
                    "n_names": int(len(names)),
                    "n_base_names": n_base,
                    "n_characteristic_valid": n_valid,
                    "coverage_rate": float(n_valid / n_base) if n_base else np.nan,
                    "factor_mean": float(values.mean()),
                    "factor_median": float(values.median()),
                    "factor_std": float(values.std(ddof=1)),
                    "rank_ic": float(values.corr(returns, method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def compute_quantile_statistics(
    signal_raw: pd.DataFrame,
    ret_raw: pd.DataFrame,
    characteristic: pd.DataFrame,
    factor_id: str = "factor",
    dimension: str = "characteristic",
    n_quantiles: int = 5,
    characteristic_lag: int = 0,
    min_group_names: int = 2,
    effective_direction: int = FROZEN_EFFECTIVE_DIRECTION,
    include_spread: bool = True,
) -> pd.DataFrame:
    """Daily cross-sectional characteristic strata summarized across time."""
    daily = _quantile_daily_rows(
        signal_raw=signal_raw,
        ret_raw=ret_raw,
        characteristic=characteristic,
        factor_id=factor_id,
        dimension=dimension,
        n_quantiles=n_quantiles,
        characteristic_lag=characteristic_lag,
        min_group_names=min_group_names,
    )
    columns = [
        "factor_id",
        "dimension",
        "quantile",
        "n_days",
        "n_obs",
        "n_names_avg",
        "coverage_rate",
        "factor_mean",
        "factor_median",
        "factor_std",
        "rank_ic_mean",
        "rank_ic_std",
        "icir",
        "effective_rank_ic_mean",
        "negative_ic_day_share",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for number, part in daily.groupby("quantile_number", sort=True):
        rank_ic = part["rank_ic"].dropna()
        std = float(rank_ic.std(ddof=1))
        weighted_mean = float(
            np.average(part["factor_mean"], weights=part["n_names"])
        )
        rows.append(
            {
                "factor_id": factor_id,
                "dimension": dimension,
                "quantile": f"Q{int(number)}",
                "n_days": int(part["TradeDate"].nunique()),
                "n_obs": int(part["n_names"].sum()),
                "n_names_avg": float(part["n_names"].mean()),
                "coverage_rate": float(part["coverage_rate"].mean()),
                "factor_mean": weighted_mean,
                "factor_median": float(part["factor_median"].median()),
                "factor_std": float(part["factor_std"].mean()),
                "rank_ic_mean": float(rank_ic.mean()),
                "rank_ic_std": std,
                "icir": (
                    float(rank_ic.mean() / std * math.sqrt(ANNUALIZATION))
                    if std > 0
                    else np.nan
                ),
                "effective_rank_ic_mean": float(rank_ic.mean())
                * effective_direction,
                "negative_ic_day_share": float((rank_ic < 0).mean()),
            }
        )
    summary = pd.DataFrame(rows, columns=columns)
    if include_spread and len(summary) == n_quantiles:
        low = summary.loc[summary["quantile"] == "Q1"].iloc[0]
        high = summary.loc[
            summary["quantile"] == f"Q{n_quantiles}"
        ].iloc[0]
        spread: Dict[str, Any] = {
            "factor_id": factor_id,
            "dimension": dimension,
            "quantile": f"Q{n_quantiles}-Q1",
        }
        for column in columns:
            if column in spread or column in ("n_days", "n_obs"):
                continue
            spread[column] = (
                float(high[column] - low[column])
                if pd.notna(high[column]) and pd.notna(low[column])
                else np.nan
            )
        spread["n_days"] = int(min(high["n_days"], low["n_days"]))
        spread["n_obs"] = int(high["n_obs"] + low["n_obs"])
        summary = pd.concat([summary, pd.DataFrame([spread])], ignore_index=True)
    return summary


def compute_turnover_tercile_fixture_stats(
    signal_raw: pd.DataFrame,
    ret_raw: pd.DataFrame,
    turnover_state: pd.DataFrame,
    factor_id: str = "factor",
    characteristic_lag: int = 1,
    min_group_names: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pure-DataFrame turnover-tercile helper used by tests and diagnostics."""
    daily = _quantile_daily_rows(
        signal_raw,
        ret_raw,
        turnover_state,
        factor_id,
        "turnover_state",
        3,
        characteristic_lag,
        min_group_names,
    )
    if not daily.empty:
        labels = {1: "Low", 2: "Mid", 3: "High"}
        daily["turnover_tercile"] = daily["quantile_number"].map(labels)
    summary = compute_quantile_statistics(
        signal_raw,
        ret_raw,
        turnover_state,
        factor_id=factor_id,
        dimension="turnover_state",
        n_quantiles=3,
        characteristic_lag=characteristic_lag,
        min_group_names=min_group_names,
        include_spread=False,
    )
    if not summary.empty:
        summary["turnover_tercile"] = summary["quantile"].map(
            {"Q1": "Low", "Q2": "Mid", "Q3": "High"}
        )
    return daily, summary


def infer_scale_requirement(
    factor_id: str,
    role: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    text = factor_id.lower()
    if role == "A1" or "adv" in text:
        return "adv20_lag1"
    if role == "A2" or "ats" in text:
        return "ats20_lag1"
    if config:
        values = _find_values_for_keys(config, ("scale_column", "required_scale"))
        for value in values:
            if isinstance(value, str) and value.lower() in text:
                return value
    return None


def _decode_number_token(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def factor_parameter_fields(
    factor_id: str,
    role: Optional[str],
    frozen_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expose economic bounds in parameter-stability artifacts."""
    resolved_role = role or infer_factor_role(factor_id) or "other"
    if resolved_role == "A0":
        return {
            "lower_bound": 40_000.0,
            "upper_bound": 200_000.0,
            "parameter_unit": "RMB_per_trade",
        }
    if resolved_role == "A3":
        return {
            "lower_bound": 0.20,
            "upper_bound": 0.80,
            "parameter_unit": "same_day_trade_amount_quantile",
        }

    match = re.search(
        r"(?:^|_)l(?P<lower>m?\d+(?:p\d+)?)"
        r"_h(?P<upper>m?\d+(?:p\d+)?)(?:_|$)",
        factor_id.lower(),
    )
    if match is not None:
        return {
            "lower_bound": _decode_number_token(match.group("lower")),
            "upper_bound": _decode_number_token(match.group("upper")),
            "parameter_unit": (
                "bps_of_ADV20_lag1"
                if resolved_role == "A1"
                else "multiple_of_ATS20_lag1"
            ),
        }

    config_keys = ("a1", "A1") if resolved_role == "A1" else ("a2", "A2")
    block: Optional[Mapping[str, Any]] = None
    for key in config_keys:
        value = frozen_config.get(key)
        if isinstance(value, Mapping):
            block = value
            break
    if block is not None and resolved_role == "A1":
        lower = block.get("lower_bps")
        upper = block.get("upper_bps")
        unit = "bps_of_ADV20_lag1"
    elif block is not None and resolved_role == "A2":
        lower = block.get("lower_multiple")
        upper = block.get("upper_multiple")
        unit = "multiple_of_ATS20_lag1"
    else:
        lower, upper, unit = np.nan, np.nan, "unknown"
    return {
        "lower_bound": float(lower) if lower is not None else np.nan,
        "upper_bound": float(upper) if upper is not None else np.nan,
        "parameter_unit": unit,
    }


def compute_missing_scale_coverage(
    factor_panels: pd.DataFrame,
    scales: pd.DataFrame,
    factor_ids: Optional[Sequence[str]] = None,
    roles: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Coverage diagnostics using scale rows as the observable stock-day grid."""
    panels = validate_factor_panels(factor_panels)
    scale_frame = validate_scales(scales)
    ids = list(factor_ids) if factor_ids is not None else sorted(panels["factor_id"].unique())
    base = scale_frame.copy()
    amount_column = next(
        (
            column
            for column in ("total_amount", "TotalAmount", "daily_total_amount")
            if column in base.columns
        ),
        None,
    )
    if amount_column is not None:
        eligible = base[amount_column].notna() & base[amount_column].gt(0)
        base = base.loc[eligible].copy()
    base_keys = base[["TradeDate", "symbol"]]
    expected = int(len(base_keys))
    rows: List[Dict[str, Any]] = []
    for factor_id in ids:
        role = roles.get(factor_id) if roles else infer_factor_role(factor_id)
        requirement = infer_scale_requirement(factor_id, role)
        actual = panels.loc[
            (panels["factor_id"] == factor_id) & panels["value"].notna(),
            ["TradeDate", "symbol"],
        ].drop_duplicates()
        present = int(len(base_keys.merge(actual, on=["TradeDate", "symbol"], how="inner")))
        resolved_requirement: Optional[str] = None
        if requirement is not None:
            aliases = {
                "adv20_lag1": (
                    "adv20_lag1",
                    "ADV20_lag1",
                    "adv20_mean_lag1",
                    "adv20",
                ),
                "ats20_lag1": (
                    "ats20_lag1",
                    "ATS20_lag1",
                    "ats20_median_lag1",
                    "ats20",
                ),
            }.get(requirement, (requirement,))
            lower = {str(column).lower(): str(column) for column in base.columns}
            resolved_requirement = next(
                (lower[alias.lower()] for alias in aliases if alias.lower() in lower),
                None,
            )
        if requirement is None:
            scale_available = expected
        elif resolved_requirement is None:
            scale_available = 0
        else:
            scale_available = int(base[resolved_requirement].notna().sum())
        missing_scale = expected - scale_available
        rows.append(
            {
                "factor_id": factor_id,
                "factor_role": role or "other",
                "required_scale": requirement or "none",
                "resolved_scale_column": resolved_requirement or (
                    "not_applicable" if requirement is None else "missing"
                ),
                "expected_stock_days": expected,
                "factor_stock_days": present,
                "missing_factor_stock_days": expected - present,
                "factor_coverage_ratio": (
                    float(present / expected) if expected else np.nan
                ),
                "scale_available_stock_days": scale_available,
                "missing_scale_stock_days": missing_scale,
                "missing_scale_ratio": (
                    float(missing_scale / expected) if expected else np.nan
                ),
                "factor_coverage_given_scale": (
                    float(present / scale_available) if scale_available else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def daily_ols_residualize(
    factor: pd.DataFrame,
    industry: pd.DataFrame,
    cap_control: pd.DataFrame,
    method: str,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Daily cross-sectional OLS residuals for exposure diagnostics."""
    aliases = {
        "industry": "industry",
        "ind": "industry",
        "cap": "cap",
        "market_cap": "cap",
        "joint": "joint",
        "industry+cap": "joint",
        "ind_cap": "joint",
    }
    normalized = aliases.get(method)
    if normalized is None:
        raise ValueError(f"Unknown OLS method: {method}")
    idx = factor.index.intersection(industry.index).intersection(cap_control.index)
    cols = factor.columns.intersection(industry.columns).intersection(cap_control.columns)
    y_panel = factor.reindex(index=idx, columns=cols)
    ind_panel = industry.reindex(index=idx, columns=cols)
    cap_panel = cap_control.reindex(index=idx, columns=cols)
    out = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    for date in idx:
        y = pd.to_numeric(y_panel.loc[date], errors="coerce")
        ind = ind_panel.loc[date]
        cap = pd.to_numeric(cap_panel.loc[date], errors="coerce")
        if normalized == "industry":
            valid = y.notna() & ind.notna()
        elif normalized == "cap":
            valid = y.notna() & cap.notna()
        else:
            valid = y.notna() & ind.notna() & cap.notna()
        if int(valid.sum()) < min_observations:
            continue
        yv = y[valid].astype(float)
        pieces = [pd.Series(1.0, index=yv.index, name="const")]
        if normalized in ("industry", "joint"):
            dummies = pd.get_dummies(ind[valid].astype(str), dtype=float)
            pieces.append(dummies)
        if normalized in ("cap", "joint"):
            pieces.append(cap[valid].astype(float).rename("cap"))
        x = pd.concat(pieces, axis=1).astype(float)
        finite = np.isfinite(x).all(axis=1) & np.isfinite(yv)
        x = x.loc[finite]
        yv = yv.loc[finite]
        if len(yv) < min_observations:
            continue
        coefficients, _, _, _ = np.linalg.lstsq(
            x.to_numpy(), yv.to_numpy(), rcond=None
        )
        residual = yv.to_numpy() - x.to_numpy().dot(coefficients)
        out.loc[date, yv.index] = residual
    return out


def resolve_segments(config: Mapping[str, Any]) -> "OrderedDict[str, Tuple[pd.Timestamp, pd.Timestamp]]":
    resolved: "OrderedDict[str, Tuple[pd.Timestamp, pd.Timestamp]]" = OrderedDict()
    containers = _find_values_for_keys(
        config, ("segments", "sample_segments", "sample_windows", "evaluation_windows")
    )
    source: Mapping[str, Any] = {}
    for value in containers:
        if isinstance(value, Mapping):
            source = value
            break
    aliases = {
        "IS": ("IS", "is", "in_sample", "formal_is"),
        "validation": ("validation", "independent_validation", "val"),
        "OOS": ("OOS", "oos", "out_of_sample"),
    }
    for name, default in DEFAULT_SEGMENTS.items():
        value: Any = None
        for alias in aliases[name]:
            if alias in source:
                value = source[alias]
                break
        start, end = default
        if isinstance(value, Mapping):
            start = value.get("start", value.get("date_start", start))
            end = value.get("end", value.get("date_end", end))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = value
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if start_ts > end_ts:
            raise ReportDataError(f"Invalid {name} segment: {start_ts}>{end_ts}")
        resolved[name] = (start_ts, end_ts)
    return resolved


def _savefig(fig: plt.Figure, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.01,
        0.005,
        caption,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_decile_files(
    pnl: pd.DataFrame,
    output_dir: Path,
    factor_id: str,
    factor_label: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Write separate annualized-decile and cumulative-decile PNG files."""
    frame = pnl.copy()
    frame.columns = [str(column) for column in frame.columns]
    deciles = [str(value) for value in range(1, 11)]
    required = deciles + ["H-L"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Decile PnL missing columns: {missing}")
    label = factor_label or factor_id
    slug = _slug(factor_id)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 10))

    annualized = frame[deciles].mean() * ANNUALIZATION * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar([f"G{value}" for value in deciles], annualized, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, annualized):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.5,
        )
    ax.set_title(
        f"{label}\nCSI1000 PIT annualized index-excess return by effective decile"
    )
    ax.set_xlabel("Effective signal (G1 low, G10 high); frozen direction = -1")
    ax.set_ylabel("Arithmetic annualized return (%)")
    annualized_path = output_dir / f"03_decile_annualized__{slug}.png"
    _savefig(
        fig,
        annualized_path,
        "T-1 signal; daily c2c stock return minus CSI1000 index return; "
        "gross fee=0; parameters and direction frozen before return evaluation.",
    )

    cumulative = frame[required].cumsum() * 100
    fig, ax = plt.subplots(figsize=(13.5, 7))
    for index, column in enumerate(deciles):
        ax.plot(
            cumulative.index,
            cumulative[column],
            color=colors[index],
            linewidth=1.0,
            label=f"G{column}",
        )
    ax.plot(
        cumulative.index,
        cumulative["H-L"],
        color="black",
        linewidth=2.4,
        label="H-L (effective)",
    )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title(
        f"{label}\nCSI1000 PIT cumulative index-excess return by effective decile"
    )
    ax.set_xlabel("Return date")
    ax.set_ylabel("Cumulative arithmetic return (%)")
    ax.legend(ncol=4, fontsize=8)
    cumulative_path = output_dir / f"04_decile_cumulative__{slug}.png"
    _savefig(
        fig,
        cumulative_path,
        "T-1 signal; daily c2c CSI1000 index-excess return; gross fee=0. "
        "The 7.5bps fee is reported separately as an implied annual estimate.",
    )
    return annualized_path, cumulative_path


def _plot_factor_summary(frame: pd.DataFrame, path: Path) -> None:
    show = frame.sort_values("factor_role")
    labels = show["factor_role"].astype(str)
    x = np.arange(len(show))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x - width, show["effective_icir"], width, label="Effective ICIR")
    ax.bar(x, show["hl_sharpe"], width, label="Gross H-L Sharpe")
    ax.bar(
        x + width,
        show["hl_net_annu_ret_after_implied_fee"],
        width,
        label="Net annual return after implied fee",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("CSI1000 PIT — Frozen normalized factor variants")
    ax.set_xlabel("Factor version (no return-based selection)")
    ax.legend()
    _savefig(
        fig,
        path,
        "Raw RankIC is sign-adjusted only for effective ICIR display. "
        "H-L is gross; net annual return subtracts TO × 7.5bps × 250.",
    )


def _plot_universe_summary(frame: pd.DataFrame, path: Path) -> None:
    pivot = frame.pivot(index="factor_role", columns="universe", values="effective_icir")
    ordered = [name for name in UNIVERSES if name in pivot.columns]
    pivot = pivot.reindex(columns=ordered)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    image = ax.imshow(values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                "NA" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Effective ICIR")
    ax.set_title("PIT universe comparison — frozen direction and exact-universe EW")
    _savefig(
        fig,
        path,
        "ALL is SSE/SZSE A-shares; CSI universes use point-in-time membership. "
        "T-1 factor and the official tradability/ST/limit masks are applied.",
    )


def _plot_ic_stability(
    daily: pd.DataFrame, monthly: pd.DataFrame, rolling: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for factor_id, part in daily.groupby("factor_id"):
        axes[0].plot(
            pd.to_datetime(part["TradeDate"]),
            part["rank_ic_raw"],
            linewidth=0.65,
            label=factor_id,
        )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_title("Daily raw-direction RankIC")
    axes[0].legend(fontsize=7)
    for factor_id, part in rolling.groupby("factor_id"):
        column = next(name for name in part.columns if name.endswith("d_mean"))
        axes[1].plot(
            pd.to_datetime(part["TradeDate"]),
            part[column],
            linewidth=1.6,
            label=factor_id,
        )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_title("63-trading-day mean raw RankIC (minimum 21)")
    for factor_id, part in monthly.groupby("factor_id"):
        axes[2].plot(
            part["month"],
            part["rank_ic_mean"],
            marker="o",
            linewidth=1.2,
            label=factor_id,
        )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_title("Calendar-month mean raw RankIC")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_xlabel("Month")
    _savefig(
        fig,
        path,
        "Daily cross-sectional Spearman correlation of T-1 raw factor with T c2c "
        "stock return. Negative raw IC is consistent with frozen direction=-1.",
    )


def _plot_quintiles(cap: pd.DataFrame, adv: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, frame, title in (
        (axes[0], cap, "Lagged market-cap quintile"),
        (axes[1], adv, "Lagged ADV20 quintile"),
    ):
        use = frame[frame["quantile"].isin([f"Q{i}" for i in range(1, 6)])]
        for factor_id, part in use.groupby("factor_id"):
            part = part.assign(
                order=part["quantile"].str.extract(
                    r"(\d+)", expand=False
                ).astype(int)
            ).sort_values("order")
            ax.plot(
                part["quantile"],
                part["rank_ic_mean"],
                marker="o",
                label=factor_id,
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("Mean raw RankIC within quintile")
    axes[1].legend(fontsize=7)
    _savefig(
        fig,
        path,
        "Quintiles are formed cross-sectionally using characteristics known on "
        "the signal date. Statistics include coverage in the companion CSVs.",
    )


def _plot_parameter_stability(frame: pd.DataFrame, path: Path) -> None:
    use = frame.sort_values(["factor_family", "factor_id"])
    colors = ["#c44e52" if bool(value) else "#4c78a8" for value in use["is_selected"]]
    fig, ax = plt.subplots(figsize=(max(11, len(use) * 0.45), 5.8))
    ax.bar(use["factor_id"], use["effective_icir"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=55)
    ax.set_title("Parameter stability — frozen candidate grids (red = preselected)")
    ax.set_ylabel("Effective ICIR")
    _savefig(
        fig,
        path,
        "Diagnostic only. Candidate performance is not used to refreeze thresholds, "
        "direction, or report variants.",
    )


def _plot_turnover_state(frame: pd.DataFrame, path: Path) -> None:
    use = frame.copy()
    order = {"Low": 0, "Mid": 1, "High": 2}
    use["order"] = use["turnover_tercile"].map(order)
    roles = list(use["factor_role"].drop_duplicates())
    x = np.arange(3)
    width = 0.8 / max(len(roles), 1)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, role in enumerate(roles):
        part = use[use["factor_role"] == role].sort_values("order")
        ax.bar(
            x - 0.4 + width / 2 + index * width,
            part["rank_ic"],
            width,
            label=role,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("CSI1000 PIT — Raw RankIC by lagged turnover-state tercile")
    ax.set_ylabel("Mean raw RankIC")
    ax.legend()
    _savefig(
        fig,
        path,
        "Turnover state is the official lagged 20-day log-mean S_DQ_TURN panel; "
        "terciles are known before the evaluated return.",
    )


def _plot_ols(frame: pd.DataFrame, path: Path) -> None:
    methods = ["raw", "industry", "cap", "joint"]
    pivot = frame.pivot(index="factor_role", columns="ols_method", values="effective_icir")
    pivot = pivot.reindex(columns=methods)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(pivot.index))
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        ax.bar(
            x - 0.4 + width / 2 + index * width,
            pivot[method],
            width,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("CSI1000 PIT — Daily OLS exposure diagnostics")
    ax.set_ylabel("Effective ICIR")
    ax.legend()
    _savefig(
        fig,
        path,
        "Daily cross-sectional OLS: raw, CITICS industry dummies, standardized "
        "log total market cap, and joint controls. Direction remains frozen.",
    )


def _plot_segments_and_coverage(
    segments: pd.DataFrame, coverage: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    available = segments[segments["status"] == "ok"].copy()
    if len(available):
        pivot = available.pivot(
            index="factor_role", columns="segment", values="effective_icir"
        )
        methods = [name for name in DEFAULT_SEGMENTS if name in pivot.columns]
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(methods), 1)
        for index, segment in enumerate(methods):
            axes[0].bar(
                x - 0.4 + width / 2 + index * width,
                pivot[segment],
                width,
                label=segment,
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pivot.index)
        axes[0].legend()
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Frozen performance by sample segment")
    axes[0].set_ylabel("Effective ICIR")

    use = coverage.sort_values("factor_role")
    axes[1].bar(use["factor_role"], use["factor_coverage_ratio"], label="Factor")
    axes[1].plot(
        use["factor_role"],
        1 - use["missing_scale_ratio"],
        color="#c44e52",
        marker="o",
        label="Required scale available",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Persisted stock-day coverage")
    axes[1].set_ylabel("Coverage ratio")
    axes[1].legend()
    _savefig(
        fig,
        path,
        "IS/validation/OOS use the same frozen parameters and direction. Missing "
        "segments are explicitly marked in CSV and are never replaced by estimates.",
    )


def generate_figure_suite(
    output_dir: Path,
    factor_summary: pd.DataFrame,
    universe_summary: pd.DataFrame,
    decile_pnl: Mapping[str, pd.DataFrame],
    roles: Mapping[str, str],
    daily_ic: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    rolling_ic: pd.DataFrame,
    cap_stats: pd.DataFrame,
    adv_stats: pd.DataFrame,
    parameter_stability: pd.DataFrame,
    state_summary: pd.DataFrame,
    ols_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> Dict[str, List[Path]]:
    """Generate exactly the ten declared figure classes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, List[Path]] = {name: [] for name in FIGURE_CLASSES}

    path = output_dir / "01_factor_variant_summary.png"
    _plot_factor_summary(factor_summary, path)
    produced["01_factor_variant_summary"].append(path)

    path = output_dir / "02_universe_variant_summary.png"
    _plot_universe_summary(universe_summary, path)
    produced["02_universe_variant_summary"].append(path)

    for factor_id, pnl in decile_pnl.items():
        annualized, cumulative = plot_decile_files(
            pnl,
            output_dir,
            factor_id,
            factor_display_name(factor_id, roles.get(factor_id)),
        )
        produced["03_decile_annualized"].append(annualized)
        produced["04_decile_cumulative"].append(cumulative)

    path = output_dir / "05_ic_stability.png"
    _plot_ic_stability(daily_ic, monthly_ic, rolling_ic, path)
    produced["05_ic_stability"].append(path)

    path = output_dir / "06_cap_adv_quintiles.png"
    _plot_quintiles(cap_stats, adv_stats, path)
    produced["06_cap_adv_quintiles"].append(path)

    path = output_dir / "07_parameter_stability.png"
    _plot_parameter_stability(parameter_stability, path)
    produced["07_parameter_stability"].append(path)

    path = output_dir / "08_turnover_tercile.png"
    _plot_turnover_state(state_summary, path)
    produced["08_turnover_tercile"].append(path)

    path = output_dir / "09_ols_diagnostics.png"
    _plot_ols(ols_summary, path)
    produced["09_ols_diagnostics"].append(path)

    path = output_dir / "10_segments_and_coverage.png"
    _plot_segments_and_coverage(segment_summary, coverage, path)
    produced["10_segments_and_coverage"].append(path)
    return produced


def _write_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def _config_source_references(
    config: Mapping[str, Any], config_path: Path
) -> List[Dict[str, Any]]:
    """Record path-like lineage references, hashing those available locally."""
    references: List[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and any(
            token in key.lower()
            for token in ("path", "cache", "source", "metadata", "manifest")
        ):
            if "\n" not in value and len(value) < 4096:
                references.append(value)

    visit(config)
    rows: List[Dict[str, Any]] = []
    seen = set()
    for value in references:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            workspace_candidate = (PROJ_ROOT / candidate).resolve()
            config_candidate = (config_path.parent / candidate).resolve()
            candidate = (
                workspace_candidate
                if workspace_candidate.exists()
                else config_candidate
            )
        else:
            candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            exists = candidate.is_file()
        except OSError:
            exists = False
        row: Dict[str, Any] = {
            "declared_value": value,
            "resolved_path": key,
            "exists": exists,
        }
        if exists:
            row.update(
                {
                    "bytes": int(candidate.stat().st_size),
                    "sha256": _sha256(candidate),
                }
            )
        rows.append(row)
    return rows


def generate_report_artifacts(
    factor_panels: pd.DataFrame,
    scales: pd.DataFrame,
    frozen_config: Mapping[str, Any],
    ret_raw: pd.DataFrame,
    ret_csi1000_excess: pd.DataFrame,
    tradable: pd.DataFrame,
    members: Mapping[str, Optional[pd.DataFrame]],
    market_cap: pd.DataFrame,
    industry: pd.DataFrame,
    turnover_state: pd.DataFrame,
    output_root: Path,
    source_paths: Optional[Mapping[str, Path]] = None,
    require_core_variants: bool = True,
    require_all_segments: bool = True,
    inputs_validated: bool = False,
) -> Dict[str, Any]:
    """Build all CSV/PNG/JSON artifacts from in-memory panels."""
    factor_long = (
        factor_panels
        if inputs_validated
        else validate_factor_panels(factor_panels)
    )
    scale_long = scales if inputs_validated else validate_scales(scales)
    try:
        validate_frozen_config(frozen_config)
    except (TypeError, ValueError) as exc:
        raise ReportDataError(f"frozen_config integrity check failed: {exc}") from exc
    panels = factor_panels_to_wide(factor_long)
    headline, roles = resolve_headline_factors(
        list(panels), frozen_config, require_core=require_core_variants
    )
    direction = resolve_frozen_direction(frozen_config)
    missing_members = sorted(set(UNIVERSES) - set(members))
    if missing_members:
        raise ReportDataError(f"Missing PIT membership panels: {missing_members}")
    if market_cap is None or market_cap.empty:
        raise ReportDataError("Market-cap panel is required for quintile/OLS diagnostics")
    if industry is None or industry.empty:
        raise ReportDataError("Industry panel is required for OLS diagnostics")
    if turnover_state is None or turnover_state.empty:
        raise ReportDataError("Turnover state panel is required for tercile diagnostics")

    panel_start = min(panel.index.min() for panel in panels.values())
    panel_end = max(panel.index.max() for panel in panels.values())
    start = max(pd.Timestamp(panel_start), pd.Timestamp(ret_raw.index.min()))
    end = min(pd.Timestamp(panel_end), pd.Timestamp(ret_raw.index.max()))
    if start > end:
        raise ReportDataError(
            f"Factor and return panels have no common date range: {start}>{end}"
        )

    root = output_root.resolve()
    artifacts = root / "artifacts"
    figures = root / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = [
        path
        for path in (
            artifacts / "a0_parity.json",
            artifacts / "a0_parity_top100.csv",
        )
        if path.is_file()
    ]

    universe_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    universe_rows: List[Dict[str, Any]] = []
    for factor_id in headline:
        for universe in UNIVERSES:
            result = evaluate_wide_factor(
                panels[factor_id],
                ret_raw,
                tradable,
                members[universe],
                start,
                end,
                direction,
            )
            universe_results[(factor_id, universe)] = result
            universe_rows.append(
                expand_evaluation_summary(
                    result["summary"],
                    factor_id,
                    universe,
                    rank_ic=result["rank_ic"],
                    role=roles.get(factor_id),
                )
            )
    segments = resolve_segments(frozen_config)
    unavailable_preflight: List[str] = []
    if require_all_segments:
        for factor_id in headline:
            prepared = universe_results[(factor_id, "CSI1000")]
            observed_start = pd.Timestamp(prepared["signal_raw"].index.min())
            observed_end = pd.Timestamp(prepared["signal_raw"].index.max())
            for segment, (requested_start, requested_end) in segments.items():
                if max(requested_start, observed_start) > min(
                    requested_end, observed_end
                ):
                    unavailable_preflight.append(
                        f"{factor_id}:{segment} requested "
                        f"{requested_start.date()}~{requested_end.date()}, observed "
                        f"{observed_start.date()}~{observed_end.date()}"
                    )
        if unavailable_preflight:
            raise ReportDataError(
                "Required IS/validation/OOS data are missing; no report artifacts "
                "were written:\n  - " + "\n  - ".join(unavailable_preflight)
            )
    universe_summary = pd.DataFrame(universe_rows)
    generated.append(
        _write_csv(
            universe_summary,
            artifacts / "universe_variant_summary.csv",
        )
    )

    daily_parts: List[pd.DataFrame] = []
    monthly_parts: List[pd.DataFrame] = []
    rolling_parts: List[pd.DataFrame] = []
    decile_summary_rows: List[Dict[str, Any]] = []
    decile_pnl: Dict[str, pd.DataFrame] = {}
    factor_rows: List[Dict[str, Any]] = []
    for factor_id in headline:
        role = roles.get(factor_id)
        result = universe_results[(factor_id, "CSI1000")]
        daily = result["rank_ic"].rename("rank_ic_raw").to_frame().reset_index()
        daily.columns = ["TradeDate", "rank_ic_raw"]
        daily.insert(0, "factor_id", factor_id)
        daily["effective_rank_ic"] = daily["rank_ic_raw"] * direction
        daily_parts.append(daily)
        generated.append(
            _write_csv(
                daily,
                artifacts
                / f"csi1000_daily_rank_ic__{_slug(factor_id)}.csv",
            )
        )
        monthly_parts.append(summarize_monthly_ic(result["rank_ic"], factor_id))
        rolling_parts.append(rolling_ic_frame(result["rank_ic"], factor_id))

        pnl, turnover, decile_summary = evaluate_index_deciles(
            result["signal_effective"], ret_csi1000_excess
        )
        decile_pnl[factor_id] = pnl
        pnl_out = pnl.copy()
        pnl_out.index.name = "TradeDate"
        turnover_out = turnover.copy()
        turnover_out.index.name = "TradeDate"
        generated.append(
            _write_csv(
                pnl_out,
                artifacts
                / f"csi1000_decile_index_excess_daily__{_slug(factor_id)}.csv",
                index=True,
            )
        )
        generated.append(
            _write_csv(
                turnover_out,
                artifacts
                / f"csi1000_decile_turnover_daily__{_slug(factor_id)}.csv",
                index=True,
            )
        )
        decile_summary_rows.append(
            {
                "factor_id": factor_id,
                "factor_role": role,
                **decile_summary,
            }
        )
        factor_rows.append(
            {
                **expand_evaluation_summary(
                    result["summary"],
                    factor_id,
                    "CSI1000",
                    rank_ic=result["rank_ic"],
                    role=role,
                ),
                **decile_summary,
            }
        )

    daily_ic = pd.concat(daily_parts, ignore_index=True)
    monthly_ic = pd.concat(monthly_parts, ignore_index=True)
    rolling_ic = pd.concat(rolling_parts, ignore_index=True)
    generated.extend(
        [
            _write_csv(daily_ic, artifacts / "csi1000_daily_rank_ic.csv"),
            _write_csv(monthly_ic, artifacts / "csi1000_monthly_rank_ic.csv"),
            _write_csv(rolling_ic, artifacts / "csi1000_rolling_63d_rank_ic.csv"),
            _write_csv(
                pd.DataFrame(decile_summary_rows),
                artifacts / "csi1000_decile_summary.csv",
            ),
        ]
    )

    coverage = compute_missing_scale_coverage(
        factor_long, scale_long, factor_ids=headline, roles=roles
    )
    generated.append(
        _write_csv(coverage, artifacts / "missing_scale_coverage.csv")
    )
    factor_summary = pd.DataFrame(factor_rows).merge(
        coverage[
            [
                "factor_id",
                "expected_stock_days",
                "factor_stock_days",
                "factor_coverage_ratio",
                "required_scale",
                "missing_scale_stock_days",
                "missing_scale_ratio",
                "factor_coverage_given_scale",
            ]
        ],
        on="factor_id",
        how="left",
        validate="one_to_one",
    )
    generated.append(
        _write_csv(factor_summary, artifacts / "factor_variant_summary.csv")
    )

    adv_column = find_scale_column(
        scale_long,
        ("adv20_lag1", "ADV20_lag1", "adv20_mean_lag1", "adv20"),
        "ADV20 lag-1",
    )
    adv_wide = scale_to_wide(scale_long, adv_column)
    cap_parts: List[pd.DataFrame] = []
    adv_parts: List[pd.DataFrame] = []
    for factor_id in headline:
        result = universe_results[(factor_id, "CSI1000")]
        cap_parts.append(
            compute_quantile_statistics(
                result["signal_raw"],
                result["ret_raw"],
                market_cap,
                factor_id=factor_id,
                dimension="market_cap",
                n_quantiles=5,
                characteristic_lag=1,
                min_group_names=20,
                effective_direction=direction,
            )
        )
        adv_parts.append(
            compute_quantile_statistics(
                result["signal_raw"],
                result["ret_raw"],
                adv_wide,
                factor_id=factor_id,
                dimension="adv20_lag1",
                n_quantiles=5,
                characteristic_lag=1,
                min_group_names=20,
                effective_direction=direction,
            )
        )
    cap_stats = pd.concat(cap_parts, ignore_index=True)
    adv_stats = pd.concat(adv_parts, ignore_index=True)
    if cap_stats.empty or adv_stats.empty:
        raise ReportDataError(
            "Cap/ADV quintile diagnostics have no valid daily strata; "
            "check style-panel coverage and symbol conventions"
        )
    generated.extend(
        [
            _write_csv(
                cap_stats, artifacts / "csi1000_cap_quintile_statistics.csv"
            ),
            _write_csv(
                adv_stats, artifacts / "csi1000_adv_quintile_statistics.csv"
            ),
        ]
    )

    parameter_rows: List[Dict[str, Any]] = []
    for factor_id, panel in panels.items():
        result = (
            universe_results[(factor_id, "CSI1000")]
            if factor_id in headline
            else evaluate_wide_factor(
                panel,
                ret_raw,
                tradable,
                members["CSI1000"],
                start,
                end,
                direction,
            )
        )
        role = infer_factor_role(factor_id) or "other"
        parameter_rows.append(
            {
                **expand_evaluation_summary(
                    result["summary"],
                    factor_id,
                    "CSI1000",
                    rank_ic=result["rank_ic"],
                    role=role,
                ),
                "factor_family": role,
                "is_selected": factor_id in headline,
                **factor_parameter_fields(
                    factor_id,
                    role,
                    frozen_config,
                ),
                "selection_policy": (
                    "frozen before returns; diagnostic only"
                ),
            }
        )
    parameter_stability = pd.DataFrame(parameter_rows)
    generated.append(
        _write_csv(
            parameter_stability, artifacts / "parameter_stability.csv"
        )
    )

    state_daily_parts: List[pd.DataFrame] = []
    state_summary_parts: List[pd.DataFrame] = []
    for factor_id in headline:
        result = universe_results[(factor_id, "CSI1000")]
        state_daily, state_summary = official_report.compute_state_dependence(
            result["signal_raw"], result["ret_raw"], turnover_state
        )
        if state_daily.empty:
            raise ReportDataError(
                f"Turnover-state tercile diagnostics are empty for {factor_id}"
            )
        daily_long = (
            state_daily.rename_axis("TradeDate")
            .reset_index()
            .melt(
                id_vars="TradeDate",
                var_name="turnover_tercile",
                value_name="rank_ic",
            )
        )
        daily_long.insert(0, "factor_id", factor_id)
        daily_long.insert(1, "factor_role", roles.get(factor_id))
        summary_part = state_summary.copy()
        summary_part.insert(0, "factor_id", factor_id)
        summary_part.insert(1, "factor_role", roles.get(factor_id))
        state_daily_parts.append(daily_long)
        state_summary_parts.append(summary_part)
    state_daily_all = pd.concat(state_daily_parts, ignore_index=True)
    state_summary_all = pd.concat(state_summary_parts, ignore_index=True)
    generated.extend(
        [
            _write_csv(
                state_daily_all,
                artifacts / "state_turnover_tercile_daily_ic.csv",
            ),
            _write_csv(
                state_summary_all,
                artifacts / "state_turnover_tercile_summary.csv",
            ),
        ]
    )

    log_cap = np.log(market_cap.where(market_cap > 0))
    cap_control = zsc(mad(log_cap))
    ols_rows: List[Dict[str, Any]] = []
    csi_member = members["CSI1000"]
    if csi_member is None:
        raise ReportDataError("CSI1000 PIT member panel is required")
    for factor_id in headline:
        factor = panels[factor_id]
        idx = factor.index.intersection(csi_member.index)
        cols = factor.columns.intersection(csi_member.columns)
        local = factor.reindex(index=idx, columns=cols).where(
            csi_member.reindex(index=idx, columns=cols) == 1
        )
        raw_result = universe_results[(factor_id, "CSI1000")]
        raw_ic = float(raw_result["summary"]["rank_ic"])
        method_results: Dict[str, Dict[str, Any]] = {"raw": raw_result}
        for method in ("industry", "cap", "joint"):
            residual = daily_ols_residualize(
                local, industry, cap_control, method, min_observations=20
            )
            method_results[method] = evaluate_wide_factor(
                residual,
                ret_raw,
                tradable,
                csi_member,
                start,
                end,
                direction,
            )
        for method, result in method_results.items():
            row = expand_evaluation_summary(
                result["summary"],
                factor_id,
                "CSI1000",
                rank_ic=result["rank_ic"],
                role=roles.get(factor_id),
            )
            row["ols_method"] = method
            row["raw_rank_ic_baseline"] = raw_ic
            row["abs_rank_ic_retained_vs_raw"] = (
                abs(float(row["rank_ic"])) / abs(raw_ic)
                if raw_ic != 0
                else np.nan
            )
            ols_rows.append(row)
    ols_summary = pd.DataFrame(ols_rows)
    generated.append(
        _write_csv(ols_summary, artifacts / "ols_diagnostics.csv")
    )

    segment_universe_rows: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    for factor_id in headline:
        for universe in UNIVERSES:
            prepared = universe_results[(factor_id, universe)]
            observed_start = pd.Timestamp(prepared["signal_raw"].index.min())
            observed_end = pd.Timestamp(prepared["signal_raw"].index.max())
            for segment, (requested_start, requested_end) in segments.items():
                actual_start = max(requested_start, observed_start)
                actual_end = min(requested_end, observed_end)
                if actual_start > actual_end:
                    unavailable.append(
                        f"{factor_id}:{universe}:{segment} requested "
                        f"{requested_start.date()}~{requested_end.date()}, observed "
                        f"{observed_start.date()}~{observed_end.date()}"
                    )
                    segment_universe_rows.append(
                        {
                            "factor_id": factor_id,
                            "factor_role": roles.get(factor_id),
                            "universe": universe,
                            "segment": segment,
                            "status": "no_common_data",
                            "requested_start": _fmt_date(requested_start),
                            "requested_end": _fmt_date(requested_end),
                            "actual_start": None,
                            "actual_end": None,
                            "parameters_refrozen": False,
                            "direction_refrozen": False,
                        }
                    )
                    continue
                result = evaluate_prepared_segment(
                    prepared, actual_start, actual_end, direction
                )
                row = expand_evaluation_summary(
                    result["summary"],
                    factor_id,
                    universe,
                    rank_ic=result["rank_ic"],
                    role=roles.get(factor_id),
                )
                row.update(
                    {
                        "segment": segment,
                        "status": "ok",
                        "requested_start": _fmt_date(requested_start),
                        "requested_end": _fmt_date(requested_end),
                        "actual_start": _fmt_date(actual_start),
                        "actual_end": _fmt_date(actual_end),
                        "parameters_refrozen": False,
                        "direction_refrozen": False,
                    }
                )
                segment_universe_rows.append(row)
    segment_by_universe = pd.DataFrame(segment_universe_rows)
    segment_summary = segment_by_universe.loc[
        segment_by_universe["universe"].eq("CSI1000")
    ].reset_index(drop=True)
    generated.extend(
        [
            _write_csv(
                segment_by_universe,
                artifacts / "sample_segment_results_by_universe.csv",
            ),
            _write_csv(
                segment_summary,
                artifacts / "sample_segment_results.csv",
            ),
        ]
    )
    if unavailable and require_all_segments:
        raise ReportDataError(
            "Required IS/validation/OOS data are missing; no substitute results were "
            "created:\n  - " + "\n  - ".join(unavailable)
        )

    figure_map = generate_figure_suite(
        figures,
        factor_summary,
        universe_summary,
        decile_pnl,
        roles,
        daily_ic,
        monthly_ic,
        rolling_ic,
        cap_stats,
        adv_stats,
        parameter_stability,
        state_summary_all,
        ols_summary,
        segment_summary,
        coverage,
    )
    for paths in figure_map.values():
        generated.extend(paths)

    source_records: List[Dict[str, Any]] = []
    declared_refs: List[Dict[str, Any]] = []
    if source_paths:
        for role, path in source_paths.items():
            source_records.append(_file_record(path, role))
        config_path = source_paths.get("frozen_config")
        if config_path is not None:
            declared_refs = _config_source_references(frozen_config, config_path)

    generated_records = [
        _file_record(path, "artifact" if path.parent == artifacts else "figure", root)
        for path in sorted(set(generated))
    ]
    manifest = {
        "version": "mid_trade_amount_normalized_report_v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "input_root": (
            str(next(iter(source_paths.values())).resolve().parent)
            if source_paths
            else "in_memory"
        ),
        "output_root": str(root),
        "observed_common_sample": {
            "start": _fmt_date(start),
            "end": _fmt_date(end),
        },
        "headline_factor_ids": headline,
        "factor_roles": roles,
        "frozen_effective_direction": direction,
        "direction_policy": (
            "official direction frozen before return evaluation; never inferred "
            "from report metrics or sample segments"
        ),
        "fee": {
            "one_way_bps": FEE_BPS,
            "annualization": ANNUALIZATION,
            "formula": "mean_daily_H-L_turnover * 7.5/10000 * 250",
            "treatment": "implied display-only; gross returns are not fee-deducted",
        },
        "evaluation": {
            "signal_lag_days": 1,
            "return": "daily close-to-close",
            "rank_ic": "daily cross-sectional Spearman, raw factor direction",
            "icir": "mean/std(ddof=1)*sqrt(250)",
            "universe_membership": "point-in-time",
            "tradability": "official not-limit * not-ST * trade-status masks",
            "headline_deciles": "CSI1000 index-excess; effective direction",
            "universe_summary_benchmark": "exact valid-universe equal weight",
        },
        "segments": {
            name: {"start": _fmt_date(values[0]), "end": _fmt_date(values[1])}
            for name, values in segments.items()
        },
        "scope_exclusions": [
            "factor-library correlation",
            "factor combination",
            "incremental IC",
            "portfolio optimization",
            "alpha stacking",
            "return-based parameter selection",
        ],
        "figure_classes": [
            {
                "id": name,
                "description": description,
                "files": [
                    str(path.resolve().relative_to(root))
                    for path in figure_map[name]
                ],
            }
            for name, description in FIGURE_CLASSES.items()
        ],
        "source_files": source_records,
        "declared_source_cache_and_config_references": declared_refs,
        "generated_files": generated_records,
        "hash_policy": (
            "SHA256 is recorded for every source file directly read and every "
            "generated CSV/PNG. The manifest excludes its own self-referential hash."
        ),
        "frozen_config_snapshot": _json_safe(frozen_config),
    }
    manifest_path = artifacts / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_market_context(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Any]:
    """Load official returns, masks, PIT memberships, and style diagnostics."""
    try:
        ret_raw = official_report.get_Ret_Matrix(
            start, end, method="c2c", base_index=None
        )
        ret_csi1000_excess = official_report.get_Ret_Matrix(
            start, end, method="c2c", base_index="000852.SH"
        )
        tradable = (
            official_report.get_EOD_Not_Limit(start, end)
            * official_report.get_EOD_Not_ST(start, end)
            * official_report.get_TradeStatus(start, end)
        )
        members = {
            name: (
                None
                if code is None
                else official_report.get_index_member_mask(code, start, end)
            )
            for name, code in UNIVERSES.items()
        }
    except Exception as exc:
        raise ReportDataError(
            "Required Wind/DolphinDB return, tradability, or PIT membership data "
            f"could not be loaded for {start.date()}~{end.date()}: {exc}"
        ) from exc

    try:
        from Factor_Dev_Lib import get_preheat_ind_data_citics
        from factor_data_loaders import load_derivative_wide_tables

        derivative, session = load_derivative_wide_tables(start, end)
        try:
            market_cap = derivative.total_mktcap
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()
        industry_raw = get_preheat_ind_data_citics(start, end)
        if "TradingDay" in industry_raw.columns:
            industry = industry_raw.set_index("TradingDay")
        else:
            industry = industry_raw.copy()
        industry.index = pd.to_datetime(industry.index).normalize()
        market_cap.index = pd.to_datetime(market_cap.index).normalize()
    except Exception as exc:
        raise ReportDataError(
            "Required market-cap/CITICS industry data could not be loaded for "
            f"quintile and OLS diagnostics: {exc}"
        ) from exc

    try:
        from l2_factor_reproduction.scripts.test_double_neutralization import (
            _get_turnover_wide,
        )

        warmup_start = start - pd.Timedelta(days=STYLE_WARMUP_CALENDAR_DAYS)
        turnover_state = _get_turnover_wide(warmup_start, end)
    except Exception as exc:
        raise ReportDataError(
            "Required S_DQ_TURN state data could not be loaded: "
            f"{exc}"
        ) from exc
    return {
        "ret_raw": ret_raw,
        "ret_csi1000_excess": ret_csi1000_excess,
        "tradable": tradable,
        "members": members,
        "market_cap": market_cap,
        "industry": industry,
        "turnover_state": turnover_state,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--panel-file", type=Path)
    parser.add_argument(
        "--authoritative-a0",
        type=Path,
        help=(
            "Parity-gated strict A0 panel. Defaults to "
            "<input-root>/normalized_factor_panel_a0.parquet."
        ),
    )
    parser.add_argument("--scales-file", type=Path)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument(
        "--allow-missing-segments",
        action="store_true",
        help=(
            "Write explicit no_common_data rows instead of failing when a frozen "
            "IS/validation/OOS segment is unavailable."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_input_paths(
        args.input_root,
        panel_file=args.panel_file,
        scales_file=args.scales_file,
        frozen_config=args.frozen_config,
    )
    factor_panels, scales, frozen_config = load_persisted_inputs(paths)
    authoritative_a0 = args.authoritative_a0
    if authoritative_a0 is None:
        authoritative_a0 = (
            args.input_root / "normalized_factor_panel_a0.parquet"
        )
    elif not authoritative_a0.is_absolute():
        authoritative_a0 = args.input_root / authoritative_a0
    authoritative_a0 = authoritative_a0.resolve()
    if not authoritative_a0.is_file():
        raise ReportDataError(
            "Parity-gated authoritative A0 panel is missing: "
            f"{authoritative_a0}. Run the panel parity gate first."
        )
    factor_panels = replace_with_authoritative_a0(
        factor_panels, authoritative_a0
    )
    paths["authoritative_a0"] = authoritative_a0
    observed_start = pd.Timestamp(factor_panels["TradeDate"].min())
    observed_end = pd.Timestamp(factor_panels["TradeDate"].max())
    start = pd.Timestamp(args.start) if args.start else observed_start
    end = pd.Timestamp(args.end) if args.end else observed_end
    if start > end:
        raise ReportDataError(f"Invalid requested date range: {start}>{end}")
    factor_panels = factor_panels[
        factor_panels["TradeDate"].between(start, end)
    ].copy()
    scales = scales[scales["TradeDate"].between(start, end)].copy()
    if factor_panels.empty:
        raise ReportDataError(
            f"No persisted factor rows in requested range {start.date()}~{end.date()}"
        )
    if scales.empty:
        raise ReportDataError(
            f"No persisted scale rows in requested range {start.date()}~{end.date()}"
        )

    context = load_market_context(start, end)
    manifest = generate_report_artifacts(
        factor_panels=factor_panels,
        scales=scales,
        frozen_config=frozen_config,
        output_root=args.output_root,
        source_paths=paths,
        require_core_variants=True,
        require_all_segments=not args.allow_missing_segments,
        inputs_validated=True,
        **context,
    )
    print(
        "Generated normalized mid-trade-amount artifacts: "
        f"{manifest['output_root']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReportDataError as exc:
        raise SystemExit(
            "ERROR: normalized report was not generated because required data are "
            f"missing or invalid.\n{exc}"
        )
