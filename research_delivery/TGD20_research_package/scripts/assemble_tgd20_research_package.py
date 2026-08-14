#!/usr/bin/env python3
"""Assemble the standalone TGD20 research delivery package.

Small files are copied as immutable snapshots. Large canonical parquet caches are
hard-linked when possible (copy fallback), so the package is complete without
wasting another ~1 GB on the same filesystem.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research_delivery/TGD20_research_package"

CODE_FILES = [
    "core/l2_features/__init__.py",
    "core/l2_features/return_timing.py",
    "core/l2_features/return_distribution.py",
    "core/l2_features/timing_residual.py",
    "core/l2_features/tgd.py",
    "core/l2_features/tgd_panel_builder.py",
    "core/l2_features/test_return_timing.py",
    "core/l2_features/test_timing_residual.py",
    "core/l2_features/test_tgd.py",
    "run_tgd_validation_v1.py",
    "run_tgd_execution_opt_v1.py",
    "run_tgd_replication_integrity.py",
    "run_tgd_flow_orthogonality_v1.py",
    "factor_report_generator_v2.py",
    "tests/test_factor_report_generator_v2.py",
]

DEPENDENCY_FILES = [
    "Factor_Dev_Lib.py",
    "factor_config.py",
    "factor_data_loaders.py",
    "factor_runner.py",
    "factor_attribution.py",
    "alpha_d4_expansion_stack.py",
    "alpha_dimension_density.py",
    "alpha_investability.py",
    "alpha_research_report.py",
    "execution_layer.py",
    "factor_formulas_sue.py",
    "industry_neutral.py",
    "liquidity_normalization.py",
    "intraday_lib.py",
    "data_preheat.py",
]

SPEC_FILES = [
    "factor_specs/TGD20.yaml",
    "factor_specs/TGD20_report_content.yaml",
    "research/alpha_library_v1/research_satellites/TGD20.yaml",
    "docs/milestone_tgd20_research_pack_v1.md",
]

ARTIFACT_DIRS = [
    "research/reports/tgd_v1",
    "research/reports/factors/TGD20",
    "research/reports/factor_orthogonality/TGD20_FlowDensity20",
    "research_delivery/selected_factors/TGD20",
]

EXTRA_REPORT_FILES = [
    "research/reports/alpha_factor_research_report_v1/sections/4_Factor_Research/TGD20.md",
]

CACHE_DIR = ROOT / "research/cache/tgd_timing_daily"
CACHE_FILES = [
    ROOT / "research/cache/tgd_panels/TGD20_20200101_20251231_w20.parquet",
    ROOT / "research/cache/tgd_panels/TGD20_long_20200101_20251231_w20.parquet",
]


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, hardlink: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if hardlink:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    shutil.copy2(source, destination)
    return "snapshot_copy"


def iter_tree_files(path: Path) -> Iterable[Path]:
    for item in sorted(path.rglob("*")):
        if item.is_file():
            yield item


def record_copy(
    source: Path,
    destination: Path,
    records: List[Tuple[str, str, str, int, str]],
    hardlink: bool = False,
) -> None:
    mode = copy_file(source, destination, hardlink=hardlink)
    records.append(
        (
            str(destination.relative_to(PACKAGE)),
            str(source.relative_to(ROOT)),
            mode,
            source.stat().st_size,
            sha256(source),
        )
    )


def copy_tree(
    source: Path,
    destination: Path,
    records: List[Tuple[str, str, str, int, str]],
) -> None:
    for item in iter_tree_files(source):
        record_copy(item, destination / item.relative_to(source), records)


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    records: List[Tuple[str, str, str, int, str]] = []

    for relative in CODE_FILES:
        source = ROOT / relative
        record_copy(source, PACKAGE / "code/canonical" / relative, records)
    for relative in DEPENDENCY_FILES:
        source = ROOT / relative
        record_copy(source, PACKAGE / "code/dependencies" / relative, records)
    for relative in SPEC_FILES:
        source = ROOT / relative
        record_copy(source, PACKAGE / "specification" / relative, records)
    for relative in EXTRA_REPORT_FILES:
        source = ROOT / relative
        record_copy(source, PACKAGE / "prior_research" / relative, records)

    for relative in ARTIFACT_DIRS:
        source = ROOT / relative
        copy_tree(source, PACKAGE / "artifacts" / relative, records)

    for script in (
        ROOT / "research_delivery/scripts/assemble_tgd20_research_package.py",
        ROOT / "research_delivery/scripts/build_tgd20_buyside_diagnostics.py",
        ROOT / "research_delivery/scripts/build_tgd20_junior_qr_diagnostics.py",
        ROOT / "research_delivery/scripts/render_tgd20_report_html.py",
    ):
        record_copy(script, PACKAGE / "scripts" / script.name, records)

    for source in sorted(CACHE_DIR.glob("*.parquet")):
        record_copy(
            source,
            PACKAGE / "data/cache/tgd_timing_daily" / source.name,
            records,
            hardlink=True,
        )
    for source in CACHE_FILES:
        record_copy(
            source,
            PACKAGE / "data/cache/tgd_panels" / source.name,
            records,
            hardlink=True,
        )

    manifest = PACKAGE / "SOURCE_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["package_path", "source_path", "packaging_mode", "bytes", "sha256"]
        )
        writer.writerows(records)

    inventory = PACKAGE / "PACKAGE_INVENTORY.csv"
    inventory_rows = []
    for item in iter_tree_files(PACKAGE):
        if item == inventory:
            continue
        inventory_rows.append(
            (
                str(item.relative_to(PACKAGE)),
                item.stat().st_size,
                sha256(item),
            )
        )
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["package_path", "bytes", "sha256"])
        writer.writerows(inventory_rows)

    total_bytes = sum(row[3] for row in records)
    print(
        f"Packaged {len(inventory_rows)} files ({total_bytes / 1024**3:.2f} GiB source) "
        f"under {PACKAGE}"
    )


if __name__ == "__main__":
    main()
