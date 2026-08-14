#!/usr/bin/env python3
"""Run the normalized-study A0 panel and formal-metric parity hard gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = (
    ROOT / "research/results/l2_reproduction/mid_order_ratio/normalized_v1"
)
REPORT_ROOT = (
    ROOT / "research/reports/factors/mid_order_ratio/normalized_v1"
)
CANONICAL_A0 = (
    ROOT
    / "research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity"
    / "tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet"
)
STRICT_PRIMITIVE = (
    ROOT
    / "research/results/l2_reproduction/primitives/order_size_distribution_daily"
    / "order_size_distribution_daily_2019-01-01_2026-07-31.parquet"
)
OFFICIAL_ARTIFACTS = (
    ROOT / "research/reports/factors/mid_order_ratio/artifacts"
)
EXPECTED_CANONICAL_SHA256 = (
    "ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f"
)
A0_FACTOR_ID = "mid_trade_amount_share_abs_4w20w"
MIN_SPEARMAN = 1.0 - 1e-12
MAX_ABS_ERROR = 1e-10


class ParityGateError(RuntimeError):
    """Raised when an A0 hard gate does not pass."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["TradeDate"] = pd.to_datetime(
        out["TradeDate"], errors="raise"
    ).dt.normalize()
    if out.duplicated(["symbol", "TradeDate"]).any():
        raise ParityGateError("parity input contains duplicate symbol/TradeDate keys")
    return out


def _amount_share(
    frame: pd.DataFrame,
    total_column: str,
    selected_column: Optional[str] = None,
    lower_column: Optional[str] = None,
    upper_column: Optional[str] = None,
) -> pd.Series:
    total = pd.to_numeric(frame[total_column], errors="raise").to_numpy(float)
    if selected_column is not None:
        selected = pd.to_numeric(
            frame[selected_column], errors="raise"
        ).to_numpy(float)
    else:
        if lower_column is None or upper_column is None:
            raise ValueError("lower_column and upper_column are required")
        selected = (
            pd.to_numeric(frame[upper_column], errors="raise").to_numpy(float)
            - pd.to_numeric(frame[lower_column], errors="raise").to_numpy(float)
        )
    values = np.full(len(frame), np.nan, dtype=float)
    valid = np.isfinite(total) & np.isfinite(selected) & (total > 0)
    values[valid] = selected[valid] / total[valid]
    return pd.Series(values, index=frame.index)


def _paired_statistics(
    left: pd.Series, right: pd.Series
) -> Tuple[bool, int, float, float, float, float]:
    nan_equal = bool(left.isna().equals(right.isna()))
    finite = (
        left.notna()
        & right.notna()
        & np.isfinite(left.to_numpy(float))
        & np.isfinite(right.to_numpy(float))
    )
    left_valid = left.loc[finite]
    right_valid = right.loc[finite]
    if left_valid.empty:
        return nan_equal, 0, math.nan, math.nan, math.nan, math.nan
    difference = (left_valid - right_valid).abs()
    return (
        nan_equal,
        int(len(difference)),
        float(left_valid.corr(right_valid, method="pearson")),
        float(left_valid.corr(right_valid, method="spearman")),
        float(difference.max()),
        float(difference.mean()),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def run_panel_gate(
    cache_root: Path = CACHE_ROOT,
    report_root: Path = REPORT_ROOT,
    canonical_path: Path = CANONICAL_A0,
    primitive_path: Path = STRICT_PRIMITIVE,
) -> Dict[str, Any]:
    """Validate dynamic A0 and persist the authoritative strict A0 panel."""
    dynamic_path = cache_root / "dynamic_aggregates.parquet"
    if not dynamic_path.is_file():
        raise ParityGateError(f"dynamic aggregates are missing: {dynamic_path}")
    if _sha256(canonical_path) != EXPECTED_CANONICAL_SHA256:
        raise ParityGateError("authoritative A0 cache checksum changed")

    dynamic_raw = pd.read_parquet(
        dynamic_path,
        columns=[
            "symbol",
            "TradeDate",
            "total_amount",
            "a0_abs_4w20w_selected_amount",
        ],
    )
    dynamic = _normalize_keys(dynamic_raw[["symbol", "TradeDate"]])
    dynamic["dynamic_value"] = _amount_share(
        dynamic_raw,
        "total_amount",
        selected_column="a0_abs_4w20w_selected_amount",
    ).to_numpy()

    canonical_raw = pd.read_parquet(
        canonical_path,
        columns=[
            "symbol",
            "TradeDate",
            "TotalAmount",
            "cum_40000",
            "cum_200000",
        ],
    )
    canonical = _normalize_keys(canonical_raw[["symbol", "TradeDate"]])
    canonical["canonical_value"] = _amount_share(
        canonical_raw,
        "TotalAmount",
        lower_column="cum_40000",
        upper_column="cum_200000",
    ).to_numpy()

    canonical_compare = canonical.merge(
        dynamic,
        on=["symbol", "TradeDate"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=True,
    )
    canonical_left_only = int(canonical_compare["_merge"].eq("left_only").sum())
    dynamic_extra = int(canonical_compare["_merge"].eq("right_only").sum())
    matched = canonical_compare.loc[canonical_compare["_merge"].eq("both")].copy()
    (
        nan_equal,
        compared_count,
        pearson,
        spearman,
        max_error,
        mean_error,
    ) = _paired_statistics(
        matched["dynamic_value"], matched["canonical_value"]
    )
    matched["abs_diff"] = (
        matched["dynamic_value"] - matched["canonical_value"]
    ).abs()
    finite_canonical_differences = matched["abs_diff"].dropna()
    within_1e_12_share = (
        float(finite_canonical_differences.le(1e-12).mean())
        if len(finite_canonical_differences)
        else math.nan
    )
    p99_error = (
        float(finite_canonical_differences.quantile(0.99))
        if len(finite_canonical_differences)
        else math.nan
    )

    start = pd.Timestamp(dynamic["TradeDate"].min())
    end = pd.Timestamp(dynamic["TradeDate"].max())
    primitive_raw = pd.read_parquet(
        primitive_path,
        columns=[
            "symbol",
            "TradeDate",
            "total_amt",
            "cum_amt_40000",
            "cum_amt_200000",
        ],
        filters=[
            ("TradeDate", ">=", start),
            ("TradeDate", "<=", end),
        ],
    )
    primitive = _normalize_keys(primitive_raw[["symbol", "TradeDate"]])
    primitive["primitive_value"] = _amount_share(
        primitive_raw,
        "total_amt",
        lower_column="cum_amt_40000",
        upper_column="cum_amt_200000",
    ).to_numpy()

    full_compare = dynamic.merge(
        primitive,
        on=["symbol", "TradeDate"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    dynamic_without_primitive = int(full_compare["_merge"].ne("both").sum())
    (
        full_nan_equal,
        full_compared_count,
        full_pearson,
        full_spearman,
        full_max_error,
        full_mean_error,
    ) = _paired_statistics(
        full_compare["dynamic_value"], full_compare["primitive_value"]
    )

    canonical_evidence = canonical[["symbol", "TradeDate", "canonical_value"]]
    authoritative = full_compare.merge(
        canonical_evidence,
        on=["symbol", "TradeDate"],
        how="left",
        validate="one_to_one",
        indicator="canonical_merge",
    )
    canonical_key = authoritative["canonical_merge"].eq("both")
    canonical_date_range = authoritative["TradeDate"].between(
        canonical["TradeDate"].min(), canonical["TradeDate"].max()
    )
    canonical_grid_exclusion = canonical_date_range & ~canonical_key
    primitive_extension = ~canonical_date_range
    authoritative["value"] = authoritative["primitive_value"]
    authoritative.loc[canonical_key, "value"] = authoritative.loc[
        canonical_key, "canonical_value"
    ]
    authoritative.loc[canonical_grid_exclusion, "value"] = np.nan
    authoritative["factor_id"] = A0_FACTOR_ID
    authoritative_panel = authoritative[
        ["TradeDate", "symbol", "value", "factor_id"]
    ].sort_values(["TradeDate", "symbol"], kind="stable")

    gate_passed = bool(
        canonical_left_only == 0
        and nan_equal
        and np.isfinite(spearman)
        and spearman >= MIN_SPEARMAN
        and np.isfinite(max_error)
        and max_error <= MAX_ABS_ERROR
        and dynamic_without_primitive == 0
        and full_nan_equal
        and np.isfinite(full_spearman)
        and full_spearman >= MIN_SPEARMAN
        and np.isfinite(full_max_error)
        and full_max_error <= MAX_ABS_ERROR
    )

    artifacts = report_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    top100_path = artifacts / "a0_parity_top100.csv"
    matched.nlargest(100, "abs_diff")[
        [
            "symbol",
            "TradeDate",
            "canonical_value",
            "dynamic_value",
            "abs_diff",
        ]
    ].to_csv(top100_path, index=False)

    result: Dict[str, Any] = {
        "gate": "passed" if gate_passed else "failed",
        "canonical_rows": int(len(canonical)),
        "dynamic_rows": int(len(dynamic)),
        "matched_rows": int(len(matched)),
        "canonical_left_only": canonical_left_only,
        "dynamic_extra_rows": dynamic_extra,
        "nan_pattern_equal_on_canonical_grid": nan_equal,
        "compared_value_count": compared_count,
        "pearson": _finite_or_none(pearson),
        "spearman": _finite_or_none(spearman),
        "max_abs_error": _finite_or_none(max_error),
        "mean_abs_error": _finite_or_none(mean_error),
        "p99_abs_error": _finite_or_none(p99_error),
        "within_1e_12_share": _finite_or_none(within_1e_12_share),
        "required_spearman_min": MIN_SPEARMAN,
        "required_max_abs_error": MAX_ABS_ERROR,
        "full_extension": {
            "strict_primitive_rows_in_date_slice": int(len(primitive)),
            "dynamic_without_strict_primitive": dynamic_without_primitive,
            "nan_pattern_equal": full_nan_equal,
            "compared_value_count": full_compared_count,
            "pearson": _finite_or_none(full_pearson),
            "spearman": _finite_or_none(full_spearman),
            "max_abs_error": _finite_or_none(full_max_error),
            "mean_abs_error": _finite_or_none(full_mean_error),
        },
        "authoritative_source_counts": {
            "canonical_overlap": int(canonical_key.sum()),
            "canonical_grid_excluded_dynamic_extras": int(
                canonical_grid_exclusion.sum()
            ),
            "strict_primitive_extension": int(primitive_extension.sum()),
        },
        "authoritative_a0_source": str(canonical_path.resolve()),
        "strict_primitive_source": str(primitive_path.resolve()),
        "dynamic_aggregate_source": str(dynamic_path.resolve()),
        "source_sha256": {
            "canonical": _sha256(canonical_path),
            "strict_primitive": _sha256(primitive_path),
            "dynamic_aggregates": _sha256(dynamic_path),
        },
    }
    parity_path = artifacts / "a0_parity.json"
    _write_json(parity_path, result)
    if not gate_passed:
        raise ParityGateError(f"A0 panel parity gate failed: {result}")

    panel_path = cache_root / "normalized_factor_panel_a0.parquet"
    temporary = panel_path.with_suffix(".tmp.parquet")
    authoritative_panel.to_parquet(temporary, index=False)
    temporary.replace(panel_path)
    panel_metadata = {
        "role": "authoritative_strict_a0_headline_panel",
        "factor_id": A0_FACTOR_ID,
        "rows": int(len(authoritative_panel)),
        "start": str(authoritative_panel["TradeDate"].min().date()),
        "end": str(authoritative_panel["TradeDate"].max().date()),
        "artifact_sha256": _sha256(panel_path),
        "parity_sha256": _sha256(parity_path),
        "canonical_overlap_rows": int(canonical_key.sum()),
        "canonical_grid_excluded_dynamic_extra_rows": int(
            canonical_grid_exclusion.sum()
        ),
        "strict_primitive_extension_rows": int(primitive_extension.sum()),
        "legacy_aliases": {"mid_order_ratio": A0_FACTOR_ID},
        "legacy_alias_materialized": False,
    }
    _write_json(panel_path.with_suffix(".metadata.json"), panel_metadata)
    return result


def _read_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = next(
        (
            column
            for column in ("TradeDate", "date", "Date")
            if column in frame.columns
        ),
        None,
    )
    if date_column is None:
        unnamed = [
            column for column in frame.columns if str(column).startswith("Unnamed:")
        ]
        date_column = unnamed[0] if unnamed else str(frame.columns[0])
    dates = pd.to_datetime(frame.pop(date_column), errors="raise").dt.normalize()
    frame.index = pd.DatetimeIndex(dates)
    frame.index.name = "TradeDate"
    if frame.index.duplicated().any():
        raise ParityGateError(f"duplicate dates in {path}")
    return frame.sort_index()


def _update_artifact_manifest(report_root: Path, paths: Sequence[Path]) -> None:
    manifest_path = report_root / "artifacts/artifact_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("absolute_path")): record
        for record in manifest.get("generated_files", [])
    }
    for path in paths:
        resolved = path.resolve()
        records[str(resolved)] = {
            "role": "artifact",
            "path": str(resolved.relative_to(report_root.resolve())),
            "absolute_path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": _sha256(resolved),
        }
    manifest["generated_files"] = [
        records[key] for key in sorted(records)
    ]
    _write_json(manifest_path, manifest)


def run_formal_metric_gate(
    report_root: Path = REPORT_ROOT,
    official_artifacts: Path = OFFICIAL_ARTIFACTS,
) -> Dict[str, Any]:
    """Compare normalized A0 formal outputs with the frozen strict report."""
    artifacts = report_root / "artifacts"
    slug = A0_FACTOR_ID
    normalized_ic_path = artifacts / f"csi1000_daily_rank_ic__{slug}.csv"
    normalized_decile_path = (
        artifacts / f"csi1000_decile_index_excess_daily__{slug}.csv"
    )
    official_ic_path = official_artifacts / "csi1000_rank_ic_daily.csv"
    official_decile_path = (
        official_artifacts / "csi1000_decile_index_excess_daily.csv"
    )
    for path in (
        normalized_ic_path,
        normalized_decile_path,
        official_ic_path,
        official_decile_path,
    ):
        if not path.is_file():
            raise ParityGateError(f"formal parity input is missing: {path}")

    normalized_ic = _read_daily(normalized_ic_path)
    official_ic = _read_daily(official_ic_path)
    normalized_ic_column = next(
        (
            column
            for column in ("rank_ic_raw", "rank_ic")
            if column in normalized_ic.columns
        ),
        None,
    )
    official_ic_column = next(
        (
            column
            for column in ("rank_ic_raw", "rank_ic")
            if column in official_ic.columns
        ),
        None,
    )
    if normalized_ic_column is None or official_ic_column is None:
        raise ParityGateError("formal RankIC files do not contain a RankIC column")
    ic = official_ic[[official_ic_column]].join(
        normalized_ic[[normalized_ic_column]],
        how="left",
        lsuffix="_official",
        rsuffix="_normalized",
    )
    ic_key_match = bool(ic.notna().all(axis=1).all())
    ic_overlap = ic.dropna()
    ic_difference = (
        ic_overlap.iloc[:, 0] - ic_overlap.iloc[:, 1]
    ).abs()
    rank_ic_abs_error = (
        float(ic_difference.max()) if len(ic_difference) else math.inf
    )
    official_icir = (
        float(ic_overlap.iloc[:, 0].mean())
        / float(ic_overlap.iloc[:, 0].std(ddof=1))
        * math.sqrt(250)
    )
    normalized_icir = (
        float(ic_overlap.iloc[:, 1].mean())
        / float(ic_overlap.iloc[:, 1].std(ddof=1))
        * math.sqrt(250)
    )
    icir_abs_error = abs(official_icir - normalized_icir)

    normalized_decile = _read_daily(normalized_decile_path)
    official_decile = _read_daily(official_decile_path)
    columns = [str(value) for value in range(1, 11)] + ["H-L"]
    missing = [
        column
        for column in columns
        if column not in normalized_decile.columns
        or column not in official_decile.columns
    ]
    if missing:
        raise ParityGateError(f"formal decile files miss columns: {missing}")
    decile = official_decile[columns].join(
        normalized_decile[columns],
        how="left",
        lsuffix="_official",
        rsuffix="_normalized",
    )
    decile_key_match = bool(decile.notna().all(axis=1).all())
    decile_overlap = decile.dropna()
    official_values = decile_overlap[
        [f"{column}_official" for column in columns]
    ].to_numpy(float)
    normalized_values = decile_overlap[
        [f"{column}_normalized" for column in columns]
    ].to_numpy(float)
    decile_max_abs_error = (
        float(np.max(np.abs(official_values - normalized_values)))
        if len(decile_overlap)
        else math.inf
    )
    official_hl = decile_overlap["H-L_official"]
    normalized_hl = decile_overlap["H-L_normalized"]
    hl_annu_ret_abs_error = abs(
        float(official_hl.mean() * 250)
        - float(normalized_hl.mean() * 250)
    )

    passed = bool(
        ic_key_match
        and decile_key_match
        and rank_ic_abs_error <= 1e-12
        and icir_abs_error <= 1e-10
        and decile_max_abs_error <= 1e-10
        and hl_annu_ret_abs_error <= 1e-10
    )
    result = {
        "gate": "passed" if passed else "failed",
        "rank_ic_date_key_match": ic_key_match,
        "decile_date_key_match": decile_key_match,
        "rank_ic_days": int(len(ic_overlap)),
        "decile_days": int(len(decile_overlap)),
        "rank_ic_abs_error": rank_ic_abs_error,
        "icir_abs_error": icir_abs_error,
        "decile_max_abs_error": decile_max_abs_error,
        "hl_annu_ret_abs_error": hl_annu_ret_abs_error,
        "thresholds": {
            "rank_ic_abs_error": 1e-12,
            "icir_abs_error": 1e-10,
            "decile_max_abs_error": 1e-10,
            "hl_annu_ret_abs_error": 1e-10,
        },
        "official_artifacts": str(official_artifacts.resolve()),
        "normalized_artifacts": str(artifacts.resolve()),
    }
    output_path = artifacts / "a0_formal_metric_parity.json"
    _write_json(output_path, result)
    _update_artifact_manifest(
        report_root,
        [
            artifacts / "a0_parity.json",
            artifacts / "a0_parity_top100.csv",
            output_path,
        ],
    )
    if not passed:
        raise ParityGateError(f"A0 formal metric parity gate failed: {result}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("panel", "formal", "all"), default="panel"
    )
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--canonical-a0", type=Path, default=CANONICAL_A0)
    parser.add_argument("--strict-primitive", type=Path, default=STRICT_PRIMITIVE)
    parser.add_argument(
        "--official-artifacts", type=Path, default=OFFICIAL_ARTIFACTS
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in ("panel", "all"):
        result = run_panel_gate(
            cache_root=args.cache_root,
            report_root=args.report_root,
            canonical_path=args.canonical_a0,
            primitive_path=args.strict_primitive,
        )
        print(f"A0 panel parity: {result['gate']}", flush=True)
    if args.stage in ("formal", "all"):
        result = run_formal_metric_gate(
            report_root=args.report_root,
            official_artifacts=args.official_artifacts,
        )
        print(f"A0 formal metric parity: {result['gate']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityGateError as exc:
        raise SystemExit(f"ERROR: {exc}")
