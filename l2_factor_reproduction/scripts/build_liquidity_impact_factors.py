#!/usr/bin/env python
"""Stream Liquidity/Impact primitive quarters into 24 frozen factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.candidate_pool import (  # noqa: E402
    correlation_pairs,
    redundancy_annotations,
)
from l2_factor_reproduction.python.liquidity_impact_daily import (  # noqa: E402
    COVERAGE_THRESHOLD,
)
from l2_factor_reproduction.python.liquidity_impact_factors import (  # noqa: E402
    LIQUIDITY_IMPACT_FACTOR_NAMES,
    LIQUIDITY_IMPACT_FACTOR_SPECS,
    REQUIRED_PRIMITIVE_COLUMNS,
    build_liquidity_impact_feature_frame,
    feature_to_narrow,
    registry_frame,
)

PRIMITIVE_DIR = Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily"
DATASET_DIR = PRIMITIVE_DIR / "dataset"
POOL_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "liquidity_impact_family"
)
FACTOR_ROOT = POOL_DIR / "factors"
NARROW_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("tradetime", pa.timestamp("ns"), nullable=False),
        pa.field("factorname", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
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


def _parse_names(value: str) -> List[str]:
    if value.strip().lower() in {"", "all"}:
        return list(LIQUIDITY_IMPACT_FACTOR_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(LIQUIDITY_IMPACT_FACTOR_SPECS))
    if unknown:
        raise ValueError(f"Unknown Liquidity/Impact factors: {unknown}")
    if not names:
        raise ValueError("Factor list is empty")
    return names


def _primitive_files(start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    files = sorted(
        DATASET_DIR.glob("quarter=*/liquidity_impact_daily_*.parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"No Liquidity/Impact primitive partitions under {DATASET_DIR}"
        )
    return files


def _open_writers(
    names: List[str], *, overwrite: bool
) -> Tuple[Dict[str, pq.ParquetWriter], Dict[str, Path]]:
    writers: Dict[str, pq.ParquetWriter] = {}
    temporary_paths: Dict[str, Path] = {}
    for name in names:
        directory = FACTOR_ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / "factor_narrow.parquet"
        temporary = directory / "factor_narrow.parquet.building"
        if final_path.exists() and not overwrite:
            raise FileExistsError(
                f"{final_path} exists; pass --overwrite to rebuild"
            )
        if temporary.exists():
            temporary.unlink()
        writers[name] = pq.ParquetWriter(
            temporary,
            NARROW_SCHEMA,
            compression="zstd",
            use_dictionary=["symbol", "factorname"],
        )
        temporary_paths[name] = temporary
    return writers, temporary_paths


def _update_correlation(
    features: pd.DataFrame,
    names: List[str],
    total: pd.DataFrame,
    count: pd.DataFrame,
    min_names: int,
) -> None:
    for _, block in features.groupby("TradeDate", sort=True):
        if len(block) < min_names:
            continue
        corr = block[names].corr(method="spearman", min_periods=min_names)
        valid = corr.notna()
        total.iloc[:, :] += corr.fillna(0.0).to_numpy()
        count.iloc[:, :] += valid.astype(np.int64).to_numpy()


def _write_heatmap(corr: pd.DataFrame) -> None:
    figure_dir = POOL_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    size = max(12, len(corr) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        corr.to_numpy(dtype=float),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(
        "Liquidity / Price Impact Family — mean daily cross-sectional"
        " Spearman"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "correlation_heatmap.png", dpi=150)
    fig.savefig(POOL_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _cluster_table(annotations: pd.DataFrame) -> pd.DataFrame:
    membership = (
        annotations.groupby("redundancy_cluster_080", sort=True)["factor"]
        .agg(list)
        .to_dict()
    )
    output = annotations.copy()
    output["cluster_size"] = output["redundancy_cluster_080"].map(
        lambda cluster: len(membership[cluster])
    )
    output["cluster_members"] = output["redundancy_cluster_080"].map(
        lambda cluster: "|".join(membership[cluster])
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--factors", default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-correlation", action="store_true")
    parser.add_argument("--min-correlation-names", type=int, default=100)
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    names = _parse_names(args.factors)
    files = _primitive_files(start, end)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    registry = registry_frame(names)
    registry.to_csv(POOL_DIR / "factor_registry.csv", index=False)
    (POOL_DIR / "factor_registry.json").write_text(
        json.dumps(
            registry.to_dict("records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    writers, temporary_paths = _open_writers(names, overwrite=args.overwrite)
    coverage = {
        name: {
            "factor": name,
            "n_factor_rows": 0,
            "date_min": None,
            "date_max": None,
            "symbols": set(),
        }
        for name in names
    }
    corr_total = pd.DataFrame(0.0, index=names, columns=names)
    corr_count = pd.DataFrame(
        0, index=names, columns=names, dtype=np.int64
    )
    success = False
    try:
        for position, path in enumerate(files, start=1):
            print(
                f"[factor {position}/{len(files)}] {path.name}",
                flush=True,
            )
            primitive = pd.read_parquet(
                path, columns=list(REQUIRED_PRIMITIVE_COLUMNS)
            )
            primitive["TradeDate"] = pd.to_datetime(primitive["TradeDate"])
            primitive = primitive.loc[
                primitive["TradeDate"].between(start, end)
            ]
            if primitive.empty:
                continue
            features = build_liquidity_impact_feature_frame(primitive)
            if not args.skip_correlation:
                _update_correlation(
                    features,
                    names,
                    corr_total,
                    corr_count,
                    args.min_correlation_names,
                )
            for name in names:
                narrow = feature_to_narrow(features, name)
                if narrow.empty:
                    continue
                table = pa.Table.from_pandas(
                    narrow,
                    schema=NARROW_SCHEMA,
                    preserve_index=False,
                    safe=True,
                )
                writers[name].write_table(table, row_group_size=250_000)
                item = coverage[name]
                item["n_factor_rows"] += len(narrow)
                date_min = pd.Timestamp(narrow["tradetime"].min()).normalize()
                date_max = pd.Timestamp(narrow["tradetime"].max()).normalize()
                item["date_min"] = (
                    date_min
                    if item["date_min"] is None
                    else min(item["date_min"], date_min)
                )
                item["date_max"] = (
                    date_max
                    if item["date_max"] is None
                    else max(item["date_max"], date_max)
                )
                item["symbols"].update(narrow["symbol"].unique())
            del primitive, features
        success = True
    finally:
        for writer in writers.values():
            writer.close()

    if not success:
        raise RuntimeError("Liquidity/Impact factor build did not complete")
    for name, temporary in temporary_paths.items():
        final_path = FACTOR_ROOT / name / "factor_narrow.parquet"
        temporary.replace(final_path)

    coverage_frame = pd.DataFrame(
        [
            {
                "factor": name,
                "n_factor_rows": item["n_factor_rows"],
                "date_min": (
                    str(item["date_min"].date())
                    if item["date_min"] is not None
                    else None
                ),
                "date_max": (
                    str(item["date_max"].date())
                    if item["date_max"] is not None
                    else None
                ),
                "n_symbols": len(item["symbols"]),
            }
            for name, item in coverage.items()
        ]
    )
    coverage_frame.to_csv(POOL_DIR / "factor_coverage.csv", index=False)

    if not args.skip_correlation:
        corr = corr_total.divide(corr_count.where(corr_count > 0))
        corr.to_csv(POOL_DIR / "factor_correlation_spearman.csv")
        corr.abs().to_csv(POOL_DIR / "factor_correlation_abs.csv")
        pairs = correlation_pairs(corr)
        pairs.to_csv(POOL_DIR / "factor_correlation_pairs.csv", index=False)
        pairs.loc[
            pairs["abs_mean_daily_spearman"] >= 0.80
        ].to_csv(POOL_DIR / "high_corr_pairs.csv", index=False)
        annotations = redundancy_annotations(corr, threshold=0.80)
        annotations.to_csv(
            POOL_DIR / "redundancy_annotations.csv", index=False
        )
        clusters = _cluster_table(annotations)
        clusters.to_csv(POOL_DIR / "redundancy_clusters_080.csv", index=False)
        _write_heatmap(corr)

    formula_module = (
        PROJ_ROOT
        / "l2_factor_reproduction/python/liquidity_impact_factors.py"
    )
    primitive_manifest = PRIMITIVE_DIR / "manifest.json"
    manifest = {
        "version": "liquidity_impact_factor_layer_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "primitive_partitions": len(files),
        "primitive_manifest_sha256": (
            _sha256(primitive_manifest)
            if primitive_manifest.exists()
            else "unavailable"
        ),
        "formula_module_sha256": _sha256(formula_module),
        "registry_sha256": hashlib.sha256(
            registry.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "n_frozen_formulas": len(names),
        "factors": names,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "size_buckets_cny": {
            "small": "<= 1e4",
            "mid": "(4e4, 2e5]",
            "large": "> 2e5",
            "super_large": "> 1e6",
        },
        "high_impact_definition": (
            "abs(minute_return) >= per-symbol-day 90th percentile "
            "(frozen top-10%)"
        ),
        "correlation_method": (
            "time mean of daily cross-sectional Spearman"
        ),
        "high_correlation_threshold": 0.80,
        "near_alias_threshold": 0.95,
        "proxy_disclosure": (
            "effective_spread_proxy / realized_spread_proxy_5m and the"
            " *_trade_impact fields are minute approximations; see"
            " primitive manifest proxy_limitations"
        ),
        "direction_policy": (
            "raw formula direction frozen; effective direction is"
            " display-only"
        ),
    }
    (POOL_DIR / "factor_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] factors={len(names)} -> {FACTOR_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
