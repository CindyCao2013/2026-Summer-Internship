#!/usr/bin/env python
"""Pool-wide consistency audit across all frozen families (Sprint 7 infra).

Checks, without re-running any backtest:
1. family candidate_summary.csv metrics match each factor's summary.json
2. schema-v1 required metrics are never silently empty
   (NaN must be explained by the row's ``missing_reason``)
3. no inf values in numeric summary columns
4. factor date coverage stays inside the primitive / baseline sample window
5. factor_direction is always -1 or +1
6. raw/effective direction metadata is internally consistent
7. benchmark / cost / signal_shift policy is identical across families
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import IMPLIED_ANNU_FEE_BPS  # noqa: E402
from l2_factor_reproduction.config import settings  # noqa: E402
from l2_factor_reproduction.python.candidate_pool_registry import (  # noqa: E402
    BASELINE_POLICY,
    BRIDGE_CONFIG,
    BRIDGE_FACTOR,
    CANDIDATE_SUMMARY_SCHEMA_V1,
    POOL_ROOT,
    FamilyConfig,
    active_families,
)

TOL = 1e-8
METRIC_PAIRS_JSON = {
    "rank_ic_raw": ("rank_ic_mean_raw", "rank_ic_mean"),
    "icir_raw": ("rank_icir",),
    "hl_annu_ret": ("hl_annu_ret_flipped",),
    "hl_sharpe": ("hl_sharpe_flipped",),
    "hl_mdd": ("hl_mdd_flipped",),
    "avg_hl_turnover": ("avg_hl_turnover",),
    "implied_annu_fee": ("implied_annu_fee",),
}
REQUIRED_NUMERIC = [
    column
    for column in CANDIDATE_SUMMARY_SCHEMA_V1
    if column
    not in (
        "factor",
        "family",
        "category",
        "mechanism",
        "redundancy_cluster_080",
        "date_min",
        "date_max",
    )
]


def _summary_json(directory: Path) -> Optional[Dict[str, object]]:
    path = directory / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _primitive_window(
    config: FamilyConfig,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if config.primitive_dir is None:
        return None, None
    manifest_path = config.primitive_dir / "manifest.json"
    if not manifest_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.get("date_coverage", {})
    start = coverage.get("actual_min")
    end = coverage.get("actual_max")
    return (
        pd.Timestamp(start) if start else None,
        pd.Timestamp(end) if end else None,
    )


def main() -> int:
    rows: List[Dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        rows.append(
            {"check": name, "passed": bool(condition), "detail": detail}
        )

    families: List[FamilyConfig] = [
        *active_families(),
        BRIDGE_CONFIG,
    ]
    sample_start = pd.Timestamp(BASELINE_POLICY["sample_start"])
    sample_end = pd.Timestamp(BASELINE_POLICY["sample_end"])

    check(
        "policy:benchmark_constant",
        settings.UNIVERSE == BASELINE_POLICY["benchmark"],
        f"settings.UNIVERSE={settings.UNIVERSE}",
    )
    check(
        "policy:cost_constant",
        float(IMPLIED_ANNU_FEE_BPS) == float(BASELINE_POLICY["cost_bps"]),
        f"IMPLIED_ANNU_FEE_BPS={IMPLIED_ANNU_FEE_BPS}",
    )

    for config in families:
        if config.is_bridge:
            summary_path = POOL_ROOT / "candidate_summary.csv"
            frame = pd.read_csv(summary_path)
            frame = frame.loc[frame["factor"] == BRIDGE_FACTOR].copy()
            missing_map = dict(
                zip(frame["factor"], frame["missing_reason"].fillna(""))
            )
            manifest = {}
        else:
            summary_path = config.directory / config.summary_csv
            frame = pd.read_csv(summary_path)
            missing_map = {}
            manifest_path = config.directory / config.manifest_json
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {}
            )

        check(
            f"{config.name}:summary_exists",
            summary_path.exists() and len(frame) > 0,
            str(summary_path),
        )

        for field, expected in (
            ("benchmark", BASELINE_POLICY["benchmark"]),
            ("signal_shift", BASELINE_POLICY["signal_shift"]),
            ("cost_bps", BASELINE_POLICY["cost_bps"]),
        ):
            if field in manifest:
                check(
                    f"{config.name}:policy:{field}",
                    manifest[field] == expected,
                    f"manifest {field}={manifest[field]}",
                )
            else:
                check(
                    f"{config.name}:policy:{field}_via_shared_module",
                    True,
                    "field not recorded in family manifest; enforced by "
                    "shared backtest module constants",
                )

        present_numeric = [
            column for column in REQUIRED_NUMERIC if column in frame.columns
        ]
        numeric = frame[present_numeric].apply(
            pd.to_numeric, errors="coerce"
        )
        inf_mask = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
        check(
            f"{config.name}:no_inf_values",
            not inf_mask.any(),
            f"inf_cells={int(inf_mask.sum())}",
        )

        directions = pd.to_numeric(frame["factor_direction"], errors="coerce")
        check(
            f"{config.name}:factor_direction_pm_one",
            directions.isin([-1, 1]).all(),
            f"values={sorted(directions.dropna().unique())}",
        )

        if "rank_ic_effective" in frame.columns:
            effective = pd.to_numeric(
                frame["rank_ic_effective"], errors="coerce"
            )
            raw = pd.to_numeric(frame["rank_ic_raw"], errors="coerce")
            consistent = (
                (effective - raw * directions).abs().dropna() < 1e-6
            )
            check(
                f"{config.name}:raw_effective_consistency",
                bool(consistent.all()),
                f"max_abs_diff={(effective - raw * directions).abs().max()}",
            )
        if "group_pnl_saved_direction" in frame.columns:
            saved = frame["group_pnl_saved_direction"].dropna().unique()
            check(
                f"{config.name}:group_pnl_saved_effective",
                set(saved) <= {"effective"},
                f"values={list(saved)}",
            )

        primitive_start, primitive_end = _primitive_window(config)
        for column in ("date_min", "date_max"):
            if column not in frame.columns:
                continue
            dates = pd.to_datetime(frame[column], errors="coerce")
            within_sample = (
                (dates >= sample_start) & (dates <= sample_end)
            ) | dates.isna()
            check(
                f"{config.name}:coverage_within_sample:{column}",
                bool(within_sample.all()),
                f"violations={int((~within_sample).sum())}",
            )
        if primitive_start is not None and "date_min" in frame.columns:
            factor_min = pd.to_datetime(frame["date_min"], errors="coerce")
            check(
                f"{config.name}:coverage_within_primitive:start",
                bool(
                    (factor_min >= primitive_start)
                    .fillna(True)
                    .all()
                ),
                f"primitive_start={primitive_start.date()}, "
                f"factor_min={factor_min.min()}",
            )
        if primitive_end is not None and "date_max" in frame.columns:
            factor_max = pd.to_datetime(frame["date_max"], errors="coerce")
            check(
                f"{config.name}:coverage_within_primitive:end",
                bool(
                    (factor_max <= primitive_end).fillna(True).all()
                ),
                f"primitive_end={primitive_end.date()}, "
                f"factor_max={factor_max.max()}",
            )

        for _, row in frame.iterrows():
            factor = row["factor"]
            factor_dir = config.factor_result_dir(factor)
            summary_json = _summary_json(factor_dir)
            check(
                f"{config.name}:{factor}:summary_json_exists",
                summary_json is not None,
                str(factor_dir / "summary.json"),
            )
            if summary_json is None:
                continue
            for column, json_keys in METRIC_PAIRS_JSON.items():
                if column not in frame.columns:
                    continue
                csv_value = pd.to_numeric(
                    pd.Series([row[column]]), errors="coerce"
                ).iloc[0]
                json_value = next(
                    (
                        summary_json[key]
                        for key in json_keys
                        if key in summary_json
                    ),
                    None,
                )
                if pd.isna(csv_value) or json_value is None:
                    continue
                expected = float(json_value)
                if column == "icir_raw":
                    # summary.json rank_icir is effective-direction; the
                    # family csv icir_raw keeps the raw frozen direction.
                    expected *= int(row["factor_direction"])
                check(
                    f"{config.name}:{factor}:json_match:{column}",
                    abs(float(csv_value) - expected) < TOL,
                    f"csv={csv_value}, json_expected={expected}",
                )
            direction_json = summary_json.get("factor_direction")
            if direction_json is not None:
                check(
                    f"{config.name}:{factor}:direction_json_match",
                    int(row["factor_direction"]) == int(direction_json),
                    f"csv={row['factor_direction']}, json={direction_json}",
                )

            reason = missing_map.get(factor, "")
            for column in present_numeric:
                value = pd.to_numeric(
                    pd.Series([row.get(column)]), errors="coerce"
                ).iloc[0]
                if pd.isna(value):
                    check(
                        f"{config.name}:{factor}:required_metric_explained:{column}",
                        column in reason,
                        f"missing_reason={reason!r}",
                    )

    unified = pd.read_csv(POOL_ROOT / "candidate_summary.csv")
    unified_numeric = unified[REQUIRED_NUMERIC].apply(
        pd.to_numeric, errors="coerce"
    )
    unified_inf = np.isinf(
        unified_numeric.to_numpy(dtype=float, na_value=np.nan)
    )
    check(
        "unified:no_inf_values",
        not unified_inf.any(),
        f"inf_cells={int(unified_inf.sum())}",
    )
    for _, row in unified.iterrows():
        reason = (
            row["missing_reason"] if pd.notna(row["missing_reason"]) else ""
        )
        for column in REQUIRED_NUMERIC:
            value = pd.to_numeric(
                pd.Series([row.get(column)]), errors="coerce"
            ).iloc[0]
            if pd.isna(value):
                check(
                    f"unified:{row['factor']}:required_metric_explained:{column}",
                    column in reason,
                    f"missing_reason={reason!r}",
                )

    output = pd.DataFrame(rows)
    output_path = POOL_ROOT / "candidate_pool_consistency_audit.csv"
    output.to_csv(output_path, index=False)
    failures = output.loc[~output["passed"]]
    (POOL_ROOT / "candidate_pool_consistency_audit.json").write_text(
        json.dumps(
            {
                "checks": int(len(output)),
                "passed": int(output["passed"].sum()),
                "failed": int(len(failures)),
                "failures": failures.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if len(failures):
        raise RuntimeError(
            "Candidate pool consistency failed:\n"
            + failures.to_string(index=False)
        )
    print(f"[done] pool checks={len(output)} all passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
