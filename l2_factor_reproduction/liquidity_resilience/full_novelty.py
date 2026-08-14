"""Post-BDL Stage B novelty: survivors only vs the frozen existing universe.

Does not expand every BDL run to candidate × ~130. Does not mutate
candidate_pool_v1, BDL thresholds, Fast Discovery, or FS/ML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.discovery_lite.candidate_matrix import (
    load_factor_narrow_slice,
    load_trading_calendar,
    narrow_slice_to_wide,
    panel_on_dates,
    resolve_factor_narrow_path,
)
from l2_factor_reproduction.discovery_lite.contracts import (
    DRY_RUN_NOVELTY_REFERENCE,
    LITE_END,
    LITE_START,
    NEAR_ALIAS_THRESHOLD,
    OUTPUT_ROOT,
    lite_trading_dates,
)
from l2_factor_reproduction.discovery_lite.novelty import novelty_bucket
from l2_factor_reproduction.python.candidate_pool_registry import POOL_ROOT

RESULT_ROOT_P = Path(RESULT_ROOT)
LR_ROOT = RESULT_ROOT_P / "liquidity_resilience"
LR_PANEL = LR_ROOT / "lr1_lite_materialization" / "panel.parquet"
BDL_DIR = OUTPUT_ROOT / "liquidity_resilience"
FULL_NOVELTY_CSV = LR_ROOT / "lr1_post_bdl_full_novelty.csv"

# Frozen after Stage B. Full Fast Discovery may run only on these names.
POST_NOVELTY_SURVIVORS: Tuple[str, ...] = (
    "spread_residual_width_5m",
    "spread_resilience_asymmetry_5m",
    "bid_replenishment_efficiency_3m",
)

NEIGHBOR_FAMILIES = (
    "order_book",
    "liquidity_impact",
    "trade_flow",
    "price_formation",
)
# Mechanism neighbors for "same_family" on a family that is not yet in the pool.
RESILIENCE_NEIGHBOR_FAMILIES = ("liquidity_impact", "directional_refill")

DIRECTIONAL_REFILL_DIR = RESULT_ROOT_P / "sprint13_directional_refill" / "factors"
DIRECTIONAL_REFILL_FACTORS = (
    "ask_recovery_5m",
    "bid_recovery_5m",
    "directional_refill_asymmetry",
    "ask_recovery_5d",
    "bid_recovery_5d",
    "directional_refill_asymmetry_5d",
    "refill_strength_asymmetry",
    "shock_weighted_asymmetry",
)

STATUS_PASS = "PASS_INDEPENDENT"
STATUS_REVIEW = "REVIEW_MODERATE_OVERLAP"
STATUS_ALIAS = "REJECT_HIDDEN_ALIAS"
STATUS_LIMITED = "REFERENCE_COVERAGE_LIMITED"


def classify_full_novelty_status(full_bucket: str) -> str:
    """Map full-universe novelty bucket to the Stage B status.

    0.75–0.90 is retained as review when the LR mechanism is distinct;
    only |ρ|≥0.90 is a hidden alias. This function does not predict ML weights.
    """
    bucket = str(full_bucket or "").upper()
    if bucket in {"", "UNKNOWN"}:
        return STATUS_LIMITED
    if bucket == "NEAR_ALIAS":
        return STATUS_ALIAS
    if bucket == "LOW_NOVELTY":
        return STATUS_REVIEW
    if bucket in {"HIGH_NOVELTY", "MEDIUM_NOVELTY"}:
        return STATUS_PASS
    return STATUS_LIMITED


def load_pool_registry() -> pd.DataFrame:
    path = POOL_ROOT / "candidate_registry.csv"
    frame = pd.read_csv(path)
    frame["name"] = frame["name"].astype(str)
    frame["family"] = frame["family"].astype(str)
    return frame


def load_bdl_survivors(bdl_dir: Path = BDL_DIR) -> pd.DataFrame:
    ranking = pd.read_csv(bdl_dir / "survivor_ranking.csv")
    keep = ranking.loc[
        ranking["final_status"].astype(str) == "FULL_DISCOVERY_SURVIVOR"
    ].copy()
    return keep.reset_index(drop=True)


def load_core14_novelty(bdl_dir: Path = BDL_DIR) -> pd.DataFrame:
    path = bdl_dir / "gate2_novelty_vs_existing.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def extra_resilience_paths() -> List[Tuple[str, str, Path]]:
    rows = []
    for name in DIRECTIONAL_REFILL_FACTORS:
        path = DIRECTIONAL_REFILL_DIR / name / "factor_narrow.parquet"
        if path.exists():
            rows.append((name, "directional_refill", path))
    return rows


def iter_existing_references(
    pool: pd.DataFrame,
    *,
    skip_names: Iterable[str] = (),
) -> List[Tuple[str, str, Path]]:
    skip = set(skip_names)
    out: List[Tuple[str, str, Path]] = []
    for _, rec in pool.iterrows():
        name = str(rec["name"])
        family = str(rec["family"])
        if name in skip:
            continue
        path = resolve_factor_narrow_path(name, family)
        if path.exists():
            out.append((name, family, path))
    out.extend(extra_resilience_paths())
    # de-duplicate by name; pool wins over extras
    seen = set()
    uniq = []
    for name, family, path in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, family, path))
    return uniq


def _load_wides(
    specs: Sequence[Tuple[str, str, Path]],
    start,
    end,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], int]:
    wides: Dict[str, pd.DataFrame] = {}
    family_map: Dict[str, str] = {}
    n_loaded = 0
    for name, family, path in specs:
        if not path.exists():
            continue
        wide = narrow_slice_to_wide(load_factor_narrow_slice(path, start, end))
        if wide is None or wide.empty:
            continue
        wides[name] = wide
        family_map[name] = family
        n_loaded += 1
        print(f"[lr-novelty] loaded existing {family}/{name} {wide.shape}", flush=True)
    return wides, family_map, n_loaded


def _closest(
    corr_row: pd.Series,
    names: Sequence[str],
    family_map: Mapping[str, str],
) -> Tuple[float, Optional[str], Optional[str]]:
    names = [n for n in names if n in corr_row.index]
    if not names:
        return float("nan"), None, None
    peers = corr_row.reindex(names).dropna()
    if peers.empty:
        return float("nan"), None, None
    closest = str(peers.abs().idxmax())
    val = float(abs(peers.loc[closest]))
    return val, closest, family_map.get(closest)


def run_full_universe_novelty(
    *,
    survivors: Optional[pd.DataFrame] = None,
    out_path: Path = FULL_NOVELTY_CSV,
) -> pd.DataFrame:
    survivors = load_bdl_survivors() if survivors is None else survivors
    cand_names = [str(n) for n in survivors["factor"]]
    if not cand_names:
        raise RuntimeError("no BDL FULL_DISCOVERY_SURVIVOR rows to audit")

    pool = load_pool_registry()
    refs = iter_existing_references(pool, skip_names=cand_names)
    core14 = list(DRY_RUN_NOVELTY_REFERENCE)

    trading_dates = load_trading_calendar("discovery")
    lite_dates = lite_trading_dates(trading_dates, start=LITE_START, end=LITE_END)

    print("=" * 72)
    print("LR Stage B — targeted full-universe novelty (survivors only)")
    print(f"  survivors:                 {len(cand_names)}")
    print(f"  existing references listed: {len(refs)}")
    print(f"  lite dates:                {len(lite_dates)}")
    print("  db_scans:                  0 (materialized parquet only)")
    print("  not running:               24 × 130, Full Discovery, FS/ML")
    print("=" * 72)
    if len(cand_names) * len(refs) > 24 * 40:
        # Guard: this path is for survivors, not the full LR registry.
        if len(cand_names) > 12:
            raise RuntimeError("refusing to expand novelty to the unsieved LR registry")

    if not LR_PANEL.exists():
        raise FileNotFoundError(LR_PANEL)
    panel_src = pd.read_parquet(LR_PANEL)
    panel_src["TradeDate"] = pd.to_datetime(panel_src["TradeDate"]).dt.normalize()
    cand_wides: Dict[str, pd.DataFrame] = {}
    for name in cand_names:
        if name not in panel_src.columns:
            raise KeyError(f"survivor {name} missing from LR panel")
        wide = panel_src.pivot_table(
            index="TradeDate", columns="Symbol", values=name, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index).normalize()
        cand_wides[name] = wide.astype(np.float32, copy=False)

    existing_wides, family_map, n_loaded = _load_wides(refs, LITE_START, LITE_END)
    print(f"[lr-novelty] loaded {n_loaded}/{len(refs)} existing references", flush=True)

    all_wides = dict(cand_wides)
    all_wides.update(existing_wides)
    long_panel = panel_on_dates(
        all_wides,
        lite_dates,
        list(cand_wides) + list(existing_wides),
    )
    from l2_factor_reproduction.python.candidate_pool import (
        mean_daily_cross_sectional_spearman,
    )

    corr_names = [c for c in cand_names + list(existing_wides) if c in long_panel.columns]
    print(
        f"[lr-novelty] Spearman panel rows={len(long_panel)} "
        f"factors={len(corr_names)} dates={len(lite_dates)}",
        flush=True,
    )
    corr = mean_daily_cross_sectional_spearman(long_panel, corr_names)

    core14_present = [n for n in core14 if n in existing_wides]
    neighbor_names = [
        n for n, fam in family_map.items() if fam in RESILIENCE_NEIGHBOR_FAMILIES
    ]
    bdl_i = survivors.set_index("factor")
    rows = []
    for name in cand_names:
        if name not in corr.index:
            core_corr, core_closest = float("nan"), None
            full_corr, full_closest, full_fam = float("nan"), None, None
            same_corr, same_closest = float("nan"), None
        else:
            core_corr, core_closest, _ = _closest(
                corr.loc[name], core14_present, family_map
            )
            full_corr, full_closest, full_fam = _closest(
                corr.loc[name], list(existing_wides), family_map
            )
            same_corr, same_closest, _ = _closest(
                corr.loc[name], neighbor_names, family_map
            )
        bucket_core = novelty_bucket(core_corr)
        bucket_full = novelty_bucket(full_corr)
        status = classify_full_novelty_status(bucket_full)
        bdl_status = (
            str(bdl_i.loc[name, "final_status"]) if name in bdl_i.index else ""
        )
        rows.append(
            {
                "candidate": name,
                "bdl_status": bdl_status,
                "max_abs_corr_core14": core_corr,
                "closest_core14_factor": core_closest,
                "max_abs_corr_full_existing": full_corr,
                "closest_existing_factor": full_closest,
                "closest_existing_family": full_fam,
                "same_family_max_corr": same_corr,
                "same_family_closest_factor": same_closest,
                "same_family_scope": "liquidity_impact+directional_refill (no frozen LR family in pool)",
                "novelty_bucket_core14": bucket_core,
                "novelty_bucket_full": bucket_full,
                "novelty_status": status,
                "n_existing_references_loaded": n_loaded,
                "n_pool_registry": int(len(pool)),
                "discovery_status": "DISCOVERY_CANDIDATE",
            }
        )
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[lr-novelty] wrote {out_path}", flush=True)
    return out
