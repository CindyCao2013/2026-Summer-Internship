"""Batch Discovery Lite pipeline: registry → gates → survivors.

Does not run Full Fast Discovery, NAV, Sharpe, or ML.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.discovery_lite.candidate_matrix import (
    load_candidate_matrix,
    load_candidate_registry,
    panel_on_dates,
)
from l2_factor_reproduction.discovery_lite.contracts import (
    BDL_CONTRACT,
    DRY_RUN_EXISTING_FACTORS,
    DRY_RUN_NOVELTY_REFERENCE,
    LITE_END,
    LITE_START,
    contract_to_json,
)
from l2_factor_reproduction.discovery_lite.diagnostics import (
    lite_vs_full_diagnostic,
    write_manifest,
    write_report,
)
from l2_factor_reproduction.discovery_lite.gates import (
    FULL_DISCOVERY_SURVIVOR,
    NEAR_ALIAS_REVIEW,
    REJECT_LOW_SIGNAL,
    REJECT_MISSING_PRIMITIVE,
    REJECT_NEAR_ALIAS,
    REJECT_REDUNDANT,
    REVIEW_SURVIVOR,
    SPARSE_EVENT_REVIEW,
    classify_survivor,
    discovery_priority_score,
    run_decile_lite,
    run_gate0,
    run_gate1,
    select_cluster_representatives,
    gate3_status,
)
from l2_factor_reproduction.discovery_lite.novelty import novelty_vs_existing
from l2_factor_reproduction.discovery_lite.redundancy import (
    candidate_correlation,
    cluster_candidates,
    singleton_clusters,
)
from l2_factor_reproduction.python.candidate_pool_registry import POOL_ROOT


def print_run_budget(meta: Dict[str, Any], steps: Sequence[str]) -> None:
    print("=" * 72)
    print("Batch Discovery Lite — pre-run budget")
    print(f"  n_candidates:          {meta.get('n_candidates')}")
    print(f"  n_lite_dates:          {meta.get('n_lite_dates')}")
    print(f"  n_trading_dates:       {meta.get('n_trading_dates')}")
    print(f"  estimated factor rows: {meta.get('estimated_factor_rows')}")
    print(f"  db_scans:              {meta.get('db_scans')}")
    print(f"  source_mode:           {meta.get('source_mode')}")
    n_src = len(meta.get("data_sources_loaded") or [])
    print(f"  data sources loaded:   {n_src} paths (parquet/context, not DB)")
    print("  expected major steps:")
    for step in steps:
        print(f"    - {step}")
    print("=" * 72)
    if int(meta.get("db_scans") or 0) > 0:
        raise RuntimeError("BDL must not issue per-factor DB scans; aborting")


def _family_map(registry: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(registry["name"].astype(str), registry["family"].astype(str)))


def _pool_family_map() -> Dict[str, str]:
    path = POOL_ROOT / "candidate_registry.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["name", "family"])
    return dict(zip(frame["name"].astype(str), frame["family"].astype(str)))


def run_batch_discovery_lite(
    *,
    registry: pd.DataFrame,
    out_dir: Path,
    window: str = "discovery",
    source: str = "auto",
    dry_run: bool = False,
    novelty_names: Optional[Sequence[str]] = None,
    verify_hash: bool = True,
    context: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    t_all = time.perf_counter()
    timings: Dict[str, float] = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_to_json(out_dir / "contract.json")

    novelty_names = list(
        DRY_RUN_NOVELTY_REFERENCE if dry_run and novelty_names is None else (novelty_names or [])
    )
    families = _family_map(registry)

    t0 = time.perf_counter()
    matrix = load_candidate_matrix(
        registry,
        window=window,
        source=source,
        context=context,
        start=LITE_START,
        end=LITE_END,
        novelty_names=novelty_names,
        verify_hash=verify_hash,
    )
    timings["data_loading"] = time.perf_counter() - t0
    novelty_wides: Dict[str, pd.DataFrame] = matrix.load_meta.pop("novelty_wides", {})

    steps = [
        "Gate 0 coverage / numeric validity (no returns)",
        "Gate 1 batch RankIC lite on every-5th date (shared mask/ret, T+1)",
        "Gate 2A candidate Spearman redundancy clusters |ρ|≥0.80",
        "Gate 2B novelty vs frozen existing factors",
        "Gate 3 lite decile shape on cluster representatives",
        "survivor ranking + report",
    ]
    print_run_budget(matrix.load_meta, steps)
    if int(matrix.load_meta.get("db_scans") or 0) != 0:
        raise RuntimeError("per-factor DB architecture detected")

    registry.to_csv(out_dir / "candidate_inventory.csv", index=False)

    t0 = time.perf_counter()
    gate0 = run_gate0(
        registry, matrix.wides, matrix.mask, matrix.lite_dates, matrix.availability
    )
    timings["gate0"] = time.perf_counter() - t0
    gate0.to_csv(out_dir / "gate0_coverage.csv", index=False)
    print(f"[bdl] Gate 0 done in {timings['gate0']:.1f}s", flush=True)
    g0_keep = set(
        gate0.loc[gate0["gate0_status"].isin(["PASS", SPARSE_EVENT_REVIEW]), "factor"]
    )

    t0 = time.perf_counter()
    g1_names = [n for n in registry["name"].astype(str) if n in g0_keep]
    gate1 = run_gate1(
        g1_names,
        matrix.wides,
        matrix.mask,
        matrix.ret,
        matrix.lite_dates,
        start=LITE_START,
        end=LITE_END,
        families=families,
    )
    timings["gate1"] = time.perf_counter() - t0
    # Attach skipped Gate-0 rejects so the table is complete.
    skipped = []
    for name in registry["name"].astype(str):
        if name in g0_keep:
            continue
        skipped.append(
            {
                "factor": name,
                "family": families.get(name, ""),
                "gate1_status": "SKIP_GATE0",
                "mean_rank_ic_lite": np.nan,
                "rank_ic_std_lite": np.nan,
                "icir_lite": np.nan,
                "positive_ic_fraction_lite": np.nan,
                "negative_ic_fraction_lite": np.nan,
                "sign_consistency_lite": np.nan,
                "n_ic_dates": 0,
                "abs_rank_ic_lite": np.nan,
                "abs_icir_lite": np.nan,
            }
        )
    if skipped:
        gate1 = pd.concat([gate1, pd.DataFrame(skipped)], ignore_index=True)
    gate1.to_csv(out_dir / "gate1_ic_screen.csv", index=False)
    print(f"[bdl] Gate 1 done in {timings['gate1']:.1f}s", flush=True)
    g1_keep = set(gate1.loc[gate1["gate1_status"] == "PASS", "factor"])

    t0 = time.perf_counter()
    g2_names = [n for n in g1_keep]
    if len(g2_names) <= 1:
        clusters = singleton_clusters(g2_names)
        pair_table = pd.DataFrame()
    else:
        panel = panel_on_dates(
            matrix.wides, matrix.lite_dates, g2_names, symbols=matrix.mask.columns
        )
        corr = candidate_correlation(panel, g2_names)
        clusters = cluster_candidates(corr)
        pair_table = corr.copy()
    if clusters.empty and g2_names:
        clusters = singleton_clusters(g2_names)
    timings["gate2_redundancy"] = time.perf_counter() - t0
    clusters.to_csv(out_dir / "gate2_redundancy_clusters.csv", index=False)
    if isinstance(pair_table, pd.DataFrame) and not pair_table.empty and pair_table.shape[0] == pair_table.shape[1]:
        pair_table.to_csv(out_dir / "gate2_candidate_corr.csv")

    t0 = time.perf_counter()
    existing_family = _pool_family_map()
    ref_names = [n for n in novelty_wides if n not in g2_names]
    novelty_panel_wides = {k: matrix.wides[k] for k in g2_names if k in matrix.wides}
    novelty_panel_wides.update(novelty_wides)
    if g2_names and (ref_names or novelty_wides):
        n_panel = panel_on_dates(
            novelty_panel_wides,
            matrix.lite_dates,
            list(g2_names) + list(ref_names),
            symbols=matrix.mask.columns,
        )
        novelty = novelty_vs_existing(
            n_panel, g2_names, ref_names, existing_family
        )
    else:
        novelty = novelty_vs_existing(
            pd.DataFrame(), g2_names, [], existing_family
        )
    timings["gate2_novelty"] = time.perf_counter() - t0
    novelty.to_csv(out_dir / "gate2_novelty_vs_existing.csv", index=False)
    print(
        f"[bdl] Gate 2 done (redundancy {timings['gate2_redundancy']:.1f}s, "
        f"novelty {timings['gate2_novelty']:.1f}s)",
        flush=True,
    )

    # Representative selection uses Gate0 coverage + Gate1 IC + novelty.
    g0_i = gate0.set_index("factor")
    g1_i = gate1.set_index("factor")
    nov_i = novelty.set_index("factor") if not novelty.empty else pd.DataFrame()
    merged_for_rep = clusters.copy()
    if merged_for_rep.empty:
        merged_for_rep = singleton_clusters([])
    extra_cols = []
    for _, row in merged_for_rep.iterrows():
        name = str(row["factor"])
        extra_cols.append(
            {
                "pit_status": g0_i.loc[name, "pit_status"] if name in g0_i.index else "PASS",
                "row_coverage": (
                    float(g0_i.loc[name, "row_coverage"]) if name in g0_i.index else 0.0
                ),
                "icir_lite": (
                    float(g1_i.loc[name, "icir_lite"]) if name in g1_i.index else 0.0
                ),
                "max_abs_corr_to_existing": (
                    float(nov_i.loc[name, "max_abs_corr_to_existing"])
                    if name in nov_i.index
                    else np.nan
                ),
            }
        )
    if extra_cols:
        merged_for_rep = pd.concat(
            [merged_for_rep.reset_index(drop=True), pd.DataFrame(extra_cols)],
            axis=1,
        )
    representatives = select_cluster_representatives(merged_for_rep, registry)
    if representatives.empty:
        representatives = merged_for_rep.copy()
        representatives["is_representative"] = False
        representatives["representative_rank"] = np.nan

    t0 = time.perf_counter()
    if representatives.empty or "is_representative" not in representatives.columns:
        rep_names: list[str] = []
    else:
        rep_names = [
            str(n)
            for n in representatives.loc[
                representatives["is_representative"].astype(bool), "factor"
            ]
        ]
    decile_rows = []
    for name in registry["name"].astype(str):
        if name not in rep_names:
            decile_rows.append(
                {
                    "factor": name,
                    "family": families.get(name, ""),
                    "gate3_status": "SKIP_NOT_REPRESENTATIVE",
                    "decile_mono_lite": np.nan,
                    "top_bottom_spread_lite": np.nan,
                    "spread_sign_consistency_lite": np.nan,
                    "n_decile_dates": 0,
                }
            )
            continue
        wide = matrix.wides.get(name)
        metrics = run_decile_lite(
            wide if wide is not None else pd.DataFrame(),
            matrix.mask,
            matrix.ret,
            matrix.lite_dates,
            start=LITE_START,
            end=LITE_END,
        )
        g1_ok = name in g1_keep
        mean_ic = (
            float(g1_i.loc[name, "mean_rank_ic_lite"]) if name in g1_i.index else np.nan
        )
        status = gate3_status(g1_ok, metrics, mean_ic)
        decile_rows.append(
            {
                "factor": name,
                "family": families.get(name, ""),
                "gate3_status": status,
                **metrics,
            }
        )
    gate3 = pd.DataFrame(decile_rows)
    timings["gate3"] = time.perf_counter() - t0
    gate3.to_csv(out_dir / "gate3_decile_lite.csv", index=False)
    print(f"[bdl] Gate 3 done in {timings['gate3']:.1f}s", flush=True)

    g3_i = gate3.set_index("factor")
    rep_i = representatives.set_index("factor") if not representatives.empty else pd.DataFrame()
    ranking_rows = []
    for _, rec in registry.iterrows():
        name = str(rec["name"])
        g0s = str(g0_i.loc[name, "gate0_status"]) if name in g0_i.index else REJECT_MISSING_PRIMITIVE
        g1s = str(g1_i.loc[name, "gate1_status"]) if name in g1_i.index else REJECT_LOW_SIGNAL
        is_rep = bool(rep_i.loc[name, "is_representative"]) if name in rep_i.index else False
        novelty_bucket = (
            str(nov_i.loc[name, "novelty_bucket"]) if name in nov_i.index else "UNKNOWN"
        )
        if name in g1_keep and name in rep_i.index:
            g2s = "PASS" if is_rep else REJECT_REDUNDANT
        elif name in g1_keep:
            g2s = "PASS"
        else:
            g2s = "SKIP"
        g3s = str(g3_i.loc[name, "gate3_status"]) if name in g3_i.index else "SKIP"
        exception = bool(rec.get("near_alias_exception", False)) or bool(
            rec.get("replacement_candidate", False)
        )
        if novelty_bucket == "NEAR_ALIAS" and is_rep:
            g2s = NEAR_ALIAS_REVIEW if exception else REJECT_NEAR_ALIAS
        final = classify_survivor(
            gate0=g0s,
            gate1=g1s,
            gate2=g2s,
            gate3=g3s,
            novelty_bucket=novelty_bucket,
            is_representative=is_rep,
            near_alias_exception=exception,
        )
        row = {
            "factor": name,
            "family": rec.get("family", ""),
            "gate0_status": g0s,
            "gate1_status": g1s,
            "gate2_status": g2s,
            "gate3_status": g3s,
            "rank_ic_lite": g1_i.loc[name, "mean_rank_ic_lite"] if name in g1_i.index else np.nan,
            "icir_lite": g1_i.loc[name, "icir_lite"] if name in g1_i.index else np.nan,
            "sign_consistency_lite": (
                g1_i.loc[name, "sign_consistency_lite"] if name in g1_i.index else np.nan
            ),
            "coverage": g0_i.loc[name, "row_coverage"] if name in g0_i.index else np.nan,
            "decile_mono_lite": (
                g3_i.loc[name, "decile_mono_lite"] if name in g3_i.index else np.nan
            ),
            "top_bottom_spread_lite": (
                g3_i.loc[name, "top_bottom_spread_lite"] if name in g3_i.index else np.nan
            ),
            "redundancy_cluster_080": (
                rep_i.loc[name, "redundancy_cluster_080"] if name in rep_i.index else ""
            ),
            "max_abs_corr_to_existing": (
                nov_i.loc[name, "max_abs_corr_to_existing"] if name in nov_i.index else np.nan
            ),
            "closest_existing_factor": (
                nov_i.loc[name, "closest_existing_factor"] if name in nov_i.index else None
            ),
            "novelty_bucket": novelty_bucket,
            "is_representative": is_rep,
            "final_status": final,
        }
        row["discovery_priority_score"] = discovery_priority_score(row)
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows)
    ranking = ranking.sort_values(
        ["final_status", "discovery_priority_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    # Priority rank among FULL_DISCOVERY_SURVIVOR then REVIEW, then others.
    order_key = ranking["final_status"].map(
        {
            FULL_DISCOVERY_SURVIVOR: 0,
            REVIEW_SURVIVOR: 1,
            NEAR_ALIAS_REVIEW: 1,
        }
    ).fillna(9)
    ranking = ranking.assign(_ord=order_key).sort_values(
        ["_ord", "discovery_priority_score"], ascending=[True, False]
    ).drop(columns="_ord").reset_index(drop=True)
    ranking["priority_rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(out_dir / "survivor_ranking.csv", index=False)

    lite_vs_full = pd.DataFrame()
    if dry_run:
        lite_vs_full = lite_vs_full_diagnostic(ranking, gate1)
        lite_vs_full.to_csv(out_dir / "dry_run_lite_vs_full.csv", index=False)

    timings["total"] = time.perf_counter() - t_all
    counts = {
        "n_candidates": int(len(registry)),
        "n_families": int(registry["family"].nunique()),
        "gate0_survivors": int((gate0["gate0_status"].isin(["PASS", SPARSE_EVENT_REVIEW])).sum()),
        "gate1_survivors": int((gate1["gate1_status"] == "PASS").sum()),
        "gate2_representatives": int(len(rep_names)),
        "gate3_survivors": int((gate3["gate3_status"] == "PASS").sum()),
        "full_discovery_survivors": int(
            (ranking["final_status"] == FULL_DISCOVERY_SURVIVOR).sum()
        ),
        "review_survivors": int(
            ranking["final_status"].isin([REVIEW_SURVIVOR, NEAR_ALIAS_REVIEW]).sum()
        ),
    }
    write_report(
        out_dir / "report.md",
        ranking=ranking,
        gate0=gate0,
        gate1=gate1,
        clusters=representatives,
        novelty=novelty,
        gate3=gate3,
        lite_vs_full=lite_vs_full,
        timings=timings,
        counts=counts,
        load_meta=matrix.load_meta,
        dry_run=dry_run,
    )
    write_manifest(
        out_dir / "manifest.json",
        contract=BDL_CONTRACT,
        load_meta=matrix.load_meta,
        timings=timings,
        counts=counts,
        dry_run=dry_run,
        out_dir=out_dir,
    )
    return {
        "out_dir": str(out_dir),
        "counts": counts,
        "timings": timings,
        "ranking": ranking,
        "lite_vs_full": lite_vs_full,
        "load_meta": matrix.load_meta,
    }


def load_dry_run_registry(pool_registry: Optional[Path] = None) -> pd.DataFrame:
    path = Path(pool_registry) if pool_registry else POOL_ROOT / "candidate_registry.csv"
    full = load_candidate_registry(path)
    names = list(DRY_RUN_EXISTING_FACTORS)
    frame = full.loc[full["name"].astype(str).isin(names)].copy()
    # Preserve frozen order.
    frame["_ord"] = frame["name"].map({n: i for i, n in enumerate(names)})
    frame = frame.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    missing = [n for n in names if n not in set(frame["name"].astype(str))]
    if missing:
        raise KeyError(f"dry-run factors missing from candidate_registry: {missing}")
    return frame
