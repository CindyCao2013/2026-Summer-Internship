#!/usr/bin/env python
"""Alpha mechanism coverage map: universe graph, redundancy heatmap, gap detection."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import factor_config as cfg
import intraday_lib
import pandas as pd

from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import EOD_ENGINE_REGISTRY, build_eod_engine_factor
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_REGISTRY,
    build_liquidity_norm_cache,
    build_liquidity_norm_factor,
)
from factor_selection import compute_alpha_scores
from factor_taxonomy import (
    ALPHA_BUNDLE_V1_LIST,
    EOD_ENGINE_HF_V2_LIST,
    EOD_ENGINE_HF_V3_LIST,
    EOD_ENGINE_PRIORITY_A_LIST,
    FACTOR_TAXONOMY,
    MECHANISM_LAYERS,
    PLANNED_MECHANISM_GAPS,
    mechanism_layer_for,
)
from liquidity_normalization import factor_correlation_matrix

BATCH_SOURCES: List[Tuple[str, str, str]] = [
    ("result/eod_engine", "core", "eod_engine"),
    ("result/eod_engine_priority_a", "priority_a", "eod_engine_priority_a"),
    ("result/eod_engine_hf_v2", "hf_v2", "eod_engine_hf_v2"),
    ("result/eod_engine_hf_v3", "hf_v3", "eod_engine_hf_v3"),
    ("result/eod_liquidity_norm", "core", "eod_liquidity_norm"),
    ("result/eod_pv", "new_eod", "eod_pv"),
    ("result", "classic", "eod_pv"),
    ("result", "c2c_all_factors", "eod_pv"),
]

EOD_ENGINE_FACTORS = set(EOD_ENGINE_REGISTRY.keys())
LIQUIDITY_NORM_FACTORS = set(LIQUIDITY_NORM_REGISTRY.keys())


def discover_batch_summaries() -> pd.DataFrame:
    frames = []
    for root, tag, track in BATCH_SOURCES:
        path = Path(root) / f"batch_summary_{tag}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_track"] = track
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No batch summary CSVs found under result/")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["factor_name", "universe"], keep="last")
    return merged


def build_factor_panel(
    name: str,
    pv_cache,
    norm_cache,
    start_day: dt.datetime,
    end_day: dt.datetime,
) -> pd.DataFrame:
    if name in LIQUIDITY_NORM_FACTORS:
        wide = build_liquidity_norm_factor(name, norm_cache)
    elif name in EOD_ENGINE_FACTORS:
        wide = build_eod_engine_factor(name, pv_cache)
    else:
        wide = build_factor(name, pv_cache)
    return wide.loc[start_day:end_day]


def build_panels(
    factor_names: List[str],
    start_day: dt.datetime,
    end_day: dt.datetime,
    start_preheat: dt.datetime,
) -> Dict[str, pd.DataFrame]:
    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )

    panels = {}
    for name in factor_names:
        try:
            panels[name] = build_factor_panel(
                name, pv_cache, norm_cache, start_day, end_day
            )
        except Exception as exc:
            print(f"[SKIP panel] {name}: {exc}")
    return panels


def layer_coverage_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in MECHANISM_LAYERS:
        subset = scored[scored["mechanism_layer"] == layer]
        strong = subset[
            (subset["abs_rank_ic_mean"] >= 0.02) | (subset["abs_icir_mean"] >= 0.7)
        ]
        rows.append(
            {
                "mechanism_layer": layer,
                "n_factors_tested": len(subset),
                "n_strong_screen": len(strong),
                "best_factor": subset.sort_values("composite_score", ascending=False)[
                    "factor_name"
                ].iloc[0]
                if len(subset)
                else None,
                "best_abs_icir": subset["abs_icir_mean"].max() if len(subset) else None,
                "best_hl_sharpe": subset["hl_sharpe_mean"].max() if len(subset) else None,
                "coverage_status": _layer_status(len(subset), len(strong), layer),
            }
        )
    return pd.DataFrame(rows)


def _layer_status(n_tested: int, n_strong: int, layer: str) -> str:
    gap_layers = {
        "cross_sectional_distortion",
        "nonlinear_liq_vol",
        "multiscale_disagreement",
        "rotation_leadership",
    }
    if layer in gap_layers and n_strong == 0:
        return "GAP"
    if n_strong >= 2:
        return "STRONG"
    if n_strong == 1:
        return "PARTIAL"
    if n_tested > 0:
        return "WEAK"
    return "EMPTY"


def redundancy_clusters(corr: pd.DataFrame, threshold: float = 0.7) -> List[List[str]]:
    names = list(corr.columns)
    visited = set()
    clusters = []
    for name in names:
        if name in visited:
            continue
        cluster = {name}
        queue = [name]
        while queue:
            cur = queue.pop()
            for other in names:
                if other in cluster:
                    continue
                if abs(corr.loc[cur, other]) >= threshold:
                    cluster.add(other)
                    queue.append(other)
        visited.update(cluster)
        if len(cluster) > 1:
            clusters.append(sorted(cluster))
    return clusters


def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4g}" if pd.notna(val) else "")
            else:
                cells.append("" if pd.isna(val) else str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_report(
    path: Path,
    layer_df: pd.DataFrame,
    scored: pd.DataFrame,
    clusters: List[List[str]],
    bundle: List[str],
) -> None:
    lines = [
        "# Alpha Mechanism Coverage Map",
        "",
        "## Layer completeness",
        "",
        _df_to_md_table(layer_df),
        "",
        "## Planned gaps (not yet in EOD universe)",
        "",
    ]
    for gap in PLANNED_MECHANISM_GAPS:
        lines.append(f"- **{gap['id']}** ({gap['layer']}): {gap['mechanism']}")
        lines.append(f"  - blocker: {gap['blocker']}")
    lines.extend(
        [
            "",
            "## Redundancy clusters (|corr| >= threshold)",
            "",
        ]
    )
    if clusters:
        for i, cl in enumerate(clusters, 1):
            lines.append(f"{i}. {', '.join(cl)}")
    else:
        lines.append("_No clusters above threshold._")

    lines.extend(["", "## Alpha bundle v1", ""])
    for name in bundle:
        row = scored[scored["factor_name"] == name]
        if len(row):
            lines.append(
                f"- {name}: layer={row['mechanism_layer'].iloc[0]}, "
                f"abs_icir={row['abs_icir_mean'].iloc[0]:.2f}, "
                f"hl_sharpe={row['hl_sharpe_mean'].iloc[0]:.2f}"
            )
        else:
            lines.append(f"- {name}")

    lines.extend(
        [
            "",
            "## Top 15 by composite score (ALL universes avg)",
            "",
        ]
    )
    top = scored.sort_values("composite_score", ascending=False).head(15)
    lines.append(
        _df_to_md_table(
            top[
                [
                    "factor_name",
                    "mechanism_layer",
                    "family",
                    "abs_icir_mean",
                    "hl_sharpe_mean",
                    "composite_score",
                ]
            ]
        )
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Alpha mechanism coverage map")
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument(
        "--top-n-corr",
        type=int,
        default=35,
        help="Build correlation matrix for top-N factors by composite score",
    )
    parser.add_argument(
        "--skip-corr",
        action="store_true",
        help="Skip panel build / correlation (faster, summaries only)",
    )
    args = parser.parse_args()

    out_dir = cfg.RESEARCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = discover_batch_summaries()
    summary_all = summary[summary["universe"] == "ALL"].copy()
    scored = compute_alpha_scores(summary)
    scored["mechanism_layer"] = scored["factor_name"].map(mechanism_layer_for)
    scored["in_taxonomy"] = scored["factor_name"].isin(FACTOR_TAXONOMY)
    scored["in_bundle_v1"] = scored["factor_name"].isin(ALPHA_BUNDLE_V1_LIST)
    scored["track_hint"] = scored["factor_name"].map(_track_hint)

    universe_path = out_dir / "alpha_coverage_universe.csv"
    scored.sort_values("composite_score", ascending=False).to_csv(
        universe_path, index=False
    )

    layer_df = layer_coverage_table(scored)
    layer_path = out_dir / "alpha_mechanism_coverage.csv"
    layer_df.to_csv(layer_path, index=False)

    gaps_path = out_dir / "alpha_planned_mechanism_gaps.csv"
    pd.DataFrame(PLANNED_MECHANISM_GAPS).to_csv(gaps_path, index=False)

    clusters: List[List[str]] = []
    corr: Optional[pd.DataFrame] = None

    if not args.skip_corr:
        top_names = (
            scored.sort_values("composite_score", ascending=False)["factor_name"]
            .head(args.top_n_corr)
            .tolist()
        )
        for anchor in ALPHA_BUNDLE_V1_LIST:
            if anchor not in top_names:
                top_names.append(anchor)
        for name in EOD_ENGINE_HF_V3_LIST:
            if name not in top_names:
                top_names.append(name)

        start_day = cfg.START_DAY
        end_day = cfg.END_DAY
        start_preheat = start_day - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
        print(f"Building {len(top_names)} factor panels for correlation map...")
        panels = build_panels(top_names, start_day, end_day, start_preheat)
        corr = factor_correlation_matrix(panels)
        corr_path = out_dir / "alpha_coverage_correlation.csv"
        corr.to_csv(corr_path)
        clusters = redundancy_clusters(corr, args.corr_threshold)
        cluster_rows = []
        for i, cl in enumerate(clusters, 1):
            for name in cl:
                cluster_rows.append({"cluster_id": i, "factor_name": name})
        pd.DataFrame(cluster_rows).to_csv(
            out_dir / "alpha_coverage_redundancy_clusters.csv", index=False
        )
        print(f"Saved correlation -> {corr_path}")

    report_path = out_dir / "alpha_coverage_report.md"
    write_report(report_path, layer_df, scored, clusters, ALPHA_BUNDLE_V1_LIST)

    print("\n=== Mechanism layer coverage ===")
    print(layer_df.to_string(index=False))
    print(f"\nSaved universe  -> {universe_path}")
    print(f"Saved layers    -> {layer_path}")
    print(f"Saved report    -> {report_path}")
    if clusters:
        print(f"\nRedundancy clusters (|corr|>={args.corr_threshold}):")
        for i, cl in enumerate(clusters, 1):
            print(f"  {i}. {cl}")


def _track_hint(name: str) -> str:
    if name in EOD_ENGINE_HF_V3_LIST:
        return "eod_engine_hf_v3"
    if name in EOD_ENGINE_HF_V2_LIST:
        return "eod_engine_hf_v2"
    if name in EOD_ENGINE_PRIORITY_A_LIST:
        return "eod_engine_priority_a"
    if name in LIQUIDITY_NORM_FACTORS:
        return "eod_liquidity_norm"
    if name in EOD_ENGINE_FACTORS:
        return "eod_engine"
    return "eod_pv"


if __name__ == "__main__":
    main()
