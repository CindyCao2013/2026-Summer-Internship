#!/usr/bin/env python
"""Fail-fast completeness audit for Sprint 6 Price Formation artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.candidate_pool_registry import (  # noqa: E402
    POOL_ROOT,
    active_families,
    expected_formula_count,
    get_family,
)
from l2_factor_reproduction.python.price_formation_factors import (  # noqa: E402
    PRICE_FORMATION_FACTOR_NAMES,
)


ROOT = Path(RESULT_ROOT)
FAMILY_CONFIG = get_family("price_formation")
PRIMITIVE = FAMILY_CONFIG.primitive_dir
FAMILY = FAMILY_CONFIG.directory


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def main() -> int:
    rows: List[Dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        rows.append(
            {
                "check": name,
                "passed": bool(condition),
                "detail": detail,
            }
        )

    primitive_required = (
        "source_inventory.md",
        "schema_ddb.csv",
        "schema_ch_sse.csv",
        "schema_ch_szse.csv",
        "coverage_comparison.csv",
        "source_parity_2024_06.csv",
        "source_audit_manifest.json",
        "manifest.json",
        "primitive_coverage.csv",
        "quality_audit.csv",
        "query_metadata.json",
    )
    for name in primitive_required:
        path = PRIMITIVE / name
        check(f"primitive:{name}", path.is_file(), str(path))
    partitions = sorted(
        (PRIMITIVE / "dataset").glob("year=*/*.parquet")
    )
    check(
        "primitive:partition_count",
        len(partitions) == 31,
        f"partitions={len(partitions)}",
    )
    if (PRIMITIVE / "manifest.json").exists() and partitions:
        manifest = json.loads(
            (PRIMITIVE / "manifest.json").read_text(encoding="utf-8")
        )
        parquet_rows = sum(_parquet_rows(path) for path in partitions)
        check(
            "primitive:manifest_row_count",
            parquet_rows == int(manifest["row_count"]),
            f"parquet={parquet_rows}, manifest={manifest['row_count']}",
        )
        check(
            "primitive:no_combined_copy",
            not bool(manifest["storage"]["combined_parquet_written"]),
            str(manifest["storage"]),
        )
        check(
            "primitive:no_raw_minute_copy",
            not bool(manifest["storage"]["raw_minute_panel_written"]),
            str(manifest["storage"]),
        )
        check(
            "primitive:hard_checks",
            int(
                manifest["invalid_row_counts"]["hard_check_failures"]
            )
            == 0,
            str(manifest["invalid_row_counts"]),
        )
        check(
            "primitive:canonical_source",
            manifest["canonical_source"]
            == "dfs://QV_Trade_to_MinuteBar/Stock_one_minute",
            manifest["canonical_source"],
        )
        check(
            "primitive:compression_zstd",
            manifest["storage"]["compression"] == "zstd",
            str(manifest["storage"]),
        )
        lineage = manifest.get("lineage", {})
        for name in (
            "primitive_module",
            "factor_formula_module",
            "evaluation_module",
            "registry",
        ):
            check(
                f"primitive:lineage:{name}",
                name in lineage,
                str(sorted(lineage.keys())),
            )

    family_required = (
        "factor_registry.csv",
        "factor_registry.json",
        "candidate_summary.csv",
        "factor_coverage.csv",
        "yearly_ic_raw.csv",
        "yearly_ic_effective.csv",
        "factor_correlation_spearman.csv",
        "factor_correlation_abs.csv",
        "factor_correlation_pairs.csv",
        "high_corr_pairs.csv",
        "redundancy_annotations.csv",
        "redundancy_clusters_080.csv",
        "aliases.csv",
        "correlation_heatmap.png",
        "price_formation_vs_trade_flow_corr.csv",
        "price_formation_vs_order_size_corr.csv",
        "price_formation_vs_order_book_corr.csv",
        "cross_family_selection.csv",
        "report.md",
        "manifest.json",
        "factor_build_manifest.json",
    )
    for name in family_required:
        path = FAMILY / name
        check(f"family:{name}", path.is_file(), str(path))

    registry_path = FAMILY / "factor_registry.csv"
    summary_path = FAMILY / "candidate_summary.csv"
    coverage_path = FAMILY / "factor_coverage.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        check(
            "family:registry_count",
            len(registry) == len(PRICE_FORMATION_FACTOR_NAMES),
            f"rows={len(registry)}",
        )
        check(
            "family:registry_names",
            set(registry["name"]) == set(PRICE_FORMATION_FACTOR_NAMES),
            "registry matches frozen names",
        )
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        check(
            "family:summary_count",
            len(summary) == len(PRICE_FORMATION_FACTOR_NAMES),
            f"rows={len(summary)}",
        )
        required_metrics = {
            "rank_ic_raw",
            "icir_raw",
            "rank_ic_std",
            "positive_ic_fraction",
            "g10_excess_annu_ret",
            "g10_excess_sharpe",
            "hl_annu_ret",
            "hl_sharpe",
            "hl_mdd",
            "avg_hl_turnover",
            "implied_annu_fee",
            "net_annu_after_fee",
            "sign_consistency",
            "decile_mono_spearman",
            "factor_direction",
            "redundancy_cluster_080",
        }
        check(
            "family:summary_metrics",
            required_metrics.issubset(summary.columns),
            f"missing={sorted(required_metrics - set(summary.columns))}",
        )
    coverage = (
        pd.read_csv(coverage_path).set_index("factor")
        if coverage_path.exists()
        else pd.DataFrame()
    )

    factor_required = (
        "factor_narrow.parquet",
        "rank_ic.csv",
        "rank_ic_raw.csv",
        "yearly_ic.csv",
        "yearly_ic_raw.csv",
        "yearly_ic_effective.csv",
        "yearly_stability.csv",
        "group_pnl.csv",
        "group_turnover.csv",
        "group_to.csv",
        "summary.json",
        "cum_pnl.png",
        "decile_bar.png",
        "yearly_ic.png",
        "ic_series.png",
    )
    for factor in PRICE_FORMATION_FACTOR_NAMES:
        directory = FAMILY / "factors" / factor
        for name in factor_required:
            path = directory / name
            check(f"factor:{factor}:{name}", path.is_file(), str(path))
        narrow_path = directory / "factor_narrow.parquet"
        if narrow_path.exists() and len(coverage):
            expected = int(coverage.loc[factor, "n_factor_rows"])
            actual = _parquet_rows(narrow_path)
            check(
                f"factor:{factor}:narrow_rows",
                actual == expected,
                f"parquet={actual}, coverage={expected}",
            )

    unified_required = (
        "candidate_registry.csv",
        "candidate_registry.json",
        "candidate_summary.csv",
        "manifest.json",
        "README.md",
    )
    for name in unified_required:
        path = POOL_ROOT / name
        check(f"unified:{name}", path.is_file(), str(path))
    unified_registry = POOL_ROOT / "candidate_registry.csv"
    if unified_registry.exists():
        frame = pd.read_csv(unified_registry)
        expected = expected_formula_count()
        check(
            "unified:formula_count_matches_registry",
            len(frame) == expected,
            f"rows={len(frame)}, expected_from_family_registries={expected}",
        )
        check(
            "unified:no_duplicate_names",
            not frame["name"].duplicated().any(),
            f"duplicates={frame.loc[frame['name'].duplicated(), 'name'].tolist()}",
        )
        check(
            "unified:price_formation_present",
            (frame["family"] == "price_formation").sum()
            == len(PRICE_FORMATION_FACTOR_NAMES),
            f"price_formation_rows={(frame['family'] == 'price_formation').sum()}",
        )
        check(
            "unified:families_match_registry_config",
            set(frame["family"].unique())
            == {config.name for config in active_families()}
            | {"trade_flow_mcap_bridge"},
            f"families={sorted(frame['family'].unique())}",
        )
    unified_summary = POOL_ROOT / "candidate_summary.csv"
    if unified_summary.exists():
        summary_frame = pd.read_csv(unified_summary, nrows=5)
        from l2_factor_reproduction.python.candidate_pool_registry import (
            CANDIDATE_SUMMARY_SCHEMA_V1,
        )

        check(
            "unified:summary_schema_v1_columns",
            set(CANDIDATE_SUMMARY_SCHEMA_V1).issubset(summary_frame.columns),
            f"missing={sorted(set(CANDIDATE_SUMMARY_SCHEMA_V1) - set(summary_frame.columns))}",
        )
        check(
            "unified:summary_missing_reason_column",
            "missing_reason" in summary_frame.columns,
            str(list(summary_frame.columns)),
        )

    output = pd.DataFrame(rows)
    FAMILY.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        FAMILY / "artifact_completeness_audit.csv", index=False
    )
    failures = output.loc[~output["passed"]]
    (FAMILY / "artifact_completeness_audit.json").write_text(
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
            "Artifact completeness failed:\n"
            + failures.to_string(index=False)
        )
    print(f"[done] artifact checks={len(output)} all passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
