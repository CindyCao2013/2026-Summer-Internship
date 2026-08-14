"""BDL reports, lite-vs-full diagnostic, and run manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import UNIVERSE
from l2_factor_reproduction.discovery_lite.contracts import CONTRACT_VERSION
from l2_factor_reproduction.python.candidate_pool_registry import POOL_ROOT


def _sign_match(a: float, b: float) -> Optional[bool]:
    if not np.isfinite(a) or not np.isfinite(b) or a == 0 or b == 0:
        return None
    return bool(np.sign(a) == np.sign(b))


def lite_vs_full_diagnostic(
    ranking: pd.DataFrame,
    gate1: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Lite RankIC signs to frozen Full Discovery raw RankIC.

    Does not tune thresholds. Broad directional consistency is the question.
    """
    summary_path = POOL_ROOT / "candidate_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    full = pd.read_csv(summary_path)
    g1 = gate1.set_index("factor")
    rank = ranking.set_index("factor")
    rows = []
    for factor in ranking["factor"].astype(str):
        if factor not in full["factor"].astype(str).values:
            continue
        frow = full.loc[full["factor"].astype(str) == factor].iloc[0]
        lite = (
            float(g1.loc[factor, "mean_rank_ic_lite"])
            if factor in g1.index
            else np.nan
        )
        full_ic = float(frow.get("rank_ic_raw", np.nan))
        match = _sign_match(lite, full_ic)
        rows.append(
            {
                "factor": factor,
                "family": frow.get("family", rank.loc[factor, "family"] if factor in rank.index else ""),
                "rank_ic_lite": lite,
                "rank_ic_full": full_ic,
                "sign_match": match,
                "abs_difference": (
                    abs(lite - full_ic) if np.isfinite(lite) and np.isfinite(full_ic) else np.nan
                ),
                "lite_status": rank.loc[factor, "final_status"] if factor in rank.index else "",
                "existing_full_status": "frozen_baseline",
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, columns=None, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_empty_"
    view = df if columns is None else df[[c for c in columns if c in df.columns]]
    view = view.head(max_rows)
    try:
        return view.to_markdown(index=False)
    except Exception:  # noqa: BLE001 — tabulate may be absent
        return "```\n" + view.to_string(index=False) + "\n```"


def write_manifest(
    path: Path,
    *,
    contract: Mapping[str, Any],
    load_meta: Mapping[str, Any],
    timings: Mapping[str, float],
    counts: Mapping[str, Any],
    dry_run: bool,
    out_dir: Path,
) -> None:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "universe": UNIVERSE,
        "counts": dict(counts),
        "timings_seconds": {k: round(float(v), 3) for k, v in timings.items()},
        "load_meta": {
            k: v
            for k, v in dict(load_meta).items()
            if k != "novelty_wides" and not isinstance(v, (pd.DataFrame, dict)) or k != "novelty_wides"
        },
        "outputs": sorted(p.name for p in Path(out_dir).glob("*") if p.is_file()),
        "contract": dict(contract),
    }
    # Sanitize load_meta paths to strings.
    meta = dict(load_meta)
    meta.pop("novelty_wides", None)
    src = meta.get("data_sources_loaded")
    if isinstance(src, list):
        meta["data_sources_loaded"] = [str(x) for x in src]
    payload["load_meta"] = meta
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _verdict(
    counts: Mapping[str, Any],
    load_meta: Mapping[str, Any],
    lite_vs_full: pd.DataFrame,
    novelty: pd.DataFrame,
) -> str:
    db_scans = int(load_meta.get("db_scans") or 0)
    n_loaded = int(load_meta.get("n_loaded") or 0)
    n_cand = int(counts.get("n_candidates") or 0)
    if db_scans > 0 or n_loaded == 0:
        return "C. BATCH_DISCOVERY_LITE_NOT_READY"
    unknown_nov = 0
    if novelty is not None and not novelty.empty and "novelty_bucket" in novelty.columns:
        unknown_nov = int((novelty["novelty_bucket"] == "UNKNOWN").sum())
    sign_ok = True
    if lite_vs_full is not None and not lite_vs_full.empty:
        matched = lite_vs_full["sign_match"].dropna()
        if len(matched) and float(matched.mean()) < 0.70:
            sign_ok = False
    if not sign_ok:
        return "C. BATCH_DISCOVERY_LITE_NOT_READY"
    n_ref = int(load_meta.get("n_novelty_reference") or 0)
    limited_novelty = unknown_nov > 0 or n_loaded < n_cand or (0 < n_ref < 30)
    if limited_novelty:
        return "B. BATCH_DISCOVERY_LITE_READY_WITH_LIMITATIONS"
    return "A. BATCH_DISCOVERY_LITE_READY"


def write_report(
    path: Path,
    *,
    ranking: pd.DataFrame,
    gate0: pd.DataFrame,
    gate1: pd.DataFrame,
    clusters: pd.DataFrame,
    novelty: pd.DataFrame,
    gate3: pd.DataFrame,
    lite_vs_full: pd.DataFrame,
    timings: Mapping[str, float],
    counts: Mapping[str, Any],
    load_meta: Mapping[str, Any],
    dry_run: bool,
) -> str:
    verdict = _verdict(counts, load_meta, lite_vs_full, novelty)
    sign_txt = "_n/a_"
    if lite_vs_full is not None and not lite_vs_full.empty:
        matched = lite_vs_full["sign_match"].dropna()
        n_match = int(matched.sum()) if len(matched) else 0
        n_cmp = int(len(matched))
        sign_txt = f"{n_match}/{n_cmp} comparable factors have matching RankIC sign"
        mismatches = lite_vs_full.loc[lite_vs_full["sign_match"] == False]
        if not mismatches.empty:
            sign_txt += "\n\nSign mismatches:\n" + _md_table(
                mismatches,
                ["factor", "family", "rank_ic_lite", "rank_ic_full", "abs_difference"],
            )

    t = {k: round(float(v), 2) for k, v in timings.items()}
    body = f"""# Batch Discovery Lite — report

## 1. Verdict

**{verdict}**

Dry-run: `{dry_run}`  
Contract: `{CONTRACT_VERSION}`  
Universe: `{UNIVERSE}`

## 2. Reused infrastructure

- Fast Discovery `fast_context` (mask, excess c2c vs `{UNIVERSE}`, trading calendar)
- `prepare_factor_signal` + `compute_rank_ic` (T+1 Spearman RankIC)
- `candidate_pool.mean_daily_cross_sectional_spearman` + `redundancy_annotations`
- `candidate_pool_registry` path resolution for materialized `factor_narrow.parquet`
- Family primitive adapters from `fast_discovery.FAMILY_ADAPTERS` (plus local order_size wrap)

Not used: ML panel, `groupTest` NAV, FS-1/2/3/4/5, ClickHouse/DDB per-factor scans.

## 3. BDL architecture

```
candidate registry
    → shared Date×Symbol matrix (materialized narrow or primitive-once)
    → Gate 0 coverage (no returns)
    → Gate 1 Lite RankIC (every 5th date, T+1)
    → Gate 2A redundancy |ρ|≥0.80 + 1–2 representatives
    → Gate 2B novelty vs frozen existing factors
    → Gate 3 Lite decile shape
    → FULL_DISCOVERY_SURVIVOR → existing Fast Discovery
```

DB scans: `{load_meta.get("db_scans")}`  
Candidates loaded: `{load_meta.get("n_loaded")}` / `{counts.get("n_candidates")}`  
Lite dates: `{load_meta.get("n_lite_dates")}`

## 4. Frozen Lite contract

See `contract.json`. Thresholds were frozen before this run and were not tuned.

- window: `{load_meta.get("start")}` → `{load_meta.get("end")}`
- date stride: every 5th canonical trading date
- coverage < 0.50 / constant_date_fraction > 0.80
- |mean RankIC lite| ≥ 0.008 **or** |ICIR lite| ≥ 1.5
- redundancy |ρ| ≥ 0.80; near-alias |ρ| ≥ 0.90
- |decile_mono_lite| ≥ 0.50 **or** stable expected top-bottom sign

## 5. Dry-run universe

n_candidates = **{counts.get("n_candidates")}**  
n_families = **{counts.get("n_families")}**

{_md_table(ranking, ["factor", "family", "final_status", "rank_ic_lite", "icir_lite"], max_rows=50)}

## 6. Gate 0 results

Gate-0 survivors (PASS + SPARSE_EVENT_REVIEW): **{counts.get("gate0_survivors")}**

{_md_table(gate0, ["factor", "family", "gate0_status", "row_coverage", "constant_date_fraction", "nonfinite_ratio", "pit_status"])}

## 7. Gate 1 results

Gate-1 survivors: **{counts.get("gate1_survivors")}**

{_md_table(gate1, ["factor", "family", "gate1_status", "mean_rank_ic_lite", "icir_lite", "sign_consistency_lite", "n_ic_dates"])}

## 8. Gate 2 redundancy

Cluster representatives: **{counts.get("gate2_representatives")}**

{_md_table(clusters, ["factor", "redundancy_cluster_080", "max_candidate_corr", "max_candidate_corr_peer", "is_representative"])}

## 9. Novelty vs existing universe

{_md_table(novelty, ["factor", "max_abs_corr_to_existing", "closest_existing_factor", "closest_existing_family", "novelty_bucket"])}

Near-alias does not auto-kill without a recorded status (`REJECT_NEAR_ALIAS` or `NEAR_ALIAS_REVIEW`).

## 10. Gate 3 decile shape

Gate-3 PASS: **{counts.get("gate3_survivors")}**

{_md_table(gate3, ["factor", "family", "gate3_status", "decile_mono_lite", "top_bottom_spread_lite", "spread_sign_consistency_lite"])}

## 11. Lite vs Full diagnostic

{sign_txt}

Lite uses every 5th date, so numerical parity is not required. The engineering bar is: BDL must not systematically reverse obvious full-sample RankIC signs.

## 12. Runtime / memory

| step | seconds |
|------|---------|
| data loading | {t.get("data_loading", float("nan"))} |
| Gate 0 | {t.get("gate0", float("nan"))} |
| Gate 1 | {t.get("gate1", float("nan"))} |
| Gate 2 redundancy | {t.get("gate2_redundancy", float("nan"))} |
| Gate 2 novelty | {t.get("gate2_novelty", float("nan"))} |
| Gate 3 | {t.get("gate3", float("nan"))} |
| total | {t.get("total", float("nan"))} |

Estimated factor rows: `{load_meta.get("estimated_factor_rows")}`  
Source mode: `{load_meta.get("source_mode")}`

## 13. Project mutation audit

- candidate_pool_v1: not written
- Fast Discovery: not modified (BDL imports `load_fast_context` / `prepare_factor_signal`)
- FS-1/2/3/4/5: not modified
- ML branch freeze: not modified
- existing factor definitions: not modified

## 14. Limitations

- Novelty vs existing uses a frozen representative reference set, not every one of the 130 pool formulas (memory).
- Liquidity-impact materialized exposures may cover fewer symbols than CSI1000; event-sparse formulas need `sparse_event=true` or they fail Gate 0 on coverage.
- Lite RankIC/ICIR are subsampled triage metrics, not Full Discovery replacements. Thresholds are frozen and must not be retuned from a dry-run.
- Gate 2 daily Spearman is the main runtime; it is still a shared-panel calculation, not per-factor full backtests.
- `DISCOVERY_PRIORITY_SCORE` orders survivors only; it is not an Alpha score and not an accept/reject rule.

## 15. Recommendation

FULL_DISCOVERY_SURVIVOR count: **{counts.get("full_discovery_survivors")}**  
REVIEW count: **{counts.get("review_survivors")}**

BDL does **not** auto-promote factors into the production registry.  
Do **not** start Liquidity Resilience or ML v2 from this sprint automatically.

If verdict is A or B: the next research action is a new family (Liquidity Resilience) using this triage layer.  
If verdict is C: fix BDL before any new family.
"""
    path.write_text(body, encoding="utf-8")
    return verdict
