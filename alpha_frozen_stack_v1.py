"""Alpha Stack v1 — frozen EOD production space.

Published Jul 2026 after OHLCV dimension map + CN broker + fundamental batch 1 triage.

Structure:
  - 5 OHLCV reps (frozen, production)
  - ≤2 CN broker candidates (strict residual IC pass; pending mono/universe gates)
  - D6 Value deferred until Quality pillar (roe_stability) — EP/BP failed strict test
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import factor_config as cfg

# ---------------------------------------------------------------------------
# Stack specification
# ---------------------------------------------------------------------------
FROZEN_OHLCV_REPS: List[Dict[str, str]] = [
    {"dim": "D1", "factor": "low_vol_liquidity_quality_60d", "source": "ohlcv_frozen"},
    {"dim": "D2", "factor": "volatility_60d", "source": "ohlcv_frozen"},
    {"dim": "D3", "factor": "lower_shadow_support_20d", "source": "ohlcv_frozen"},
    {"dim": "D4", "factor": "winner_sentiment_reversal_5d", "source": "ohlcv_frozen"},
    {"dim": "D5", "factor": "upside_fragility_20d", "source": "ohlcv_frozen"},
]

CN_CANDIDATES: List[Dict[str, str]] = [
    {
        "dim": "CN1",
        "factor": "cn_limit_up_strength_20d",
        "source": "cn_structure",
        "track": "eod_cn_broker_all",
    },
    {
        "dim": "CN2",
        "factor": "cn_volume_surge_moment_20d",
        "source": "cn_liquidity",
        "track": "eod_cn_broker_all",
    },
]

DEFERRED_DIMENSIONS: List[Dict[str, str]] = [
    {
        "dim": "D6",
        "pillar": "value",
        "factors_rejected": "ep_ttm,ep_ttm_ind_neutral,bp,bp_ind_neutral",
        "reason": "ohlcv_redundant_proxy — failed strict orthogonal test vs frozen stack",
        "next": "implement roe_stability (Quality) before re-attempting D6 Value",
    },
]

PENDING_GATES = {
    "mono_score_all": 0.80,
    "universe_stability": 0.50,
    "sign_consistency": 0.75,
    "mean_abs_ic": 0.02,
    "strict_pass": True,
}

STACK_VERSION = "v1"
STACK_STATUS = "eod_frozen_pending_cn_gates"


@dataclass
class StackEntry:
    dim: str
    factor: str
    source: str
    stack_role: str  # frozen | candidate_pending | deferred
    track: str = ""
    notes: str = ""
    pending_checks: List[str] = field(default_factory=list)


def _lookup_ranking(
    ranking: pd.DataFrame, track: str, factor: str
) -> Optional[pd.Series]:
    if ranking.empty:
        return None
    mask = ranking["factor_name"] == factor
    if track and "track" in ranking.columns:
        mask &= ranking["track"] == track
    sub = ranking[mask]
    if sub.empty and track:
        sub = ranking[ranking["factor_name"] == factor]
    return sub.iloc[0] if len(sub) else None


def _lookup_strict(attrib: pd.DataFrame, factor: str) -> Optional[bool]:
    if attrib.empty or "factor_name" not in attrib.columns:
        return None
    sub = attrib[attrib["factor_name"] == factor]
    if sub.empty or "strict_pass" not in sub.columns:
        return None
    return bool(sub.iloc[0]["strict_pass"])


def _pending_checks(row: Optional[pd.Series], strict_pass: Optional[bool]) -> List[str]:
    checks: List[str] = []
    if row is None:
        checks.append("missing_robust_ranking")
        return checks
    if strict_pass is not True:
        checks.append("strict_residual_ic")
    mono = row.get("mono_score_all")
    if pd.isna(mono) or mono < PENDING_GATES["mono_score_all"]:
        checks.append("monotonicity")
    uni = row.get("universe_stability")
    if pd.isna(uni) or uni < PENDING_GATES["universe_stability"]:
        checks.append("universe_stability")
    sign = row.get("sign_consistency")
    if pd.isna(sign) or sign < PENDING_GATES["sign_consistency"]:
        checks.append("sign_consistency")
    ic = row.get("mean_abs_ic")
    if pd.isna(ic) or ic < PENDING_GATES["mean_abs_ic"]:
        checks.append("mean_abs_ic")
    return checks


def build_stack_manifest(
    ranking: pd.DataFrame,
    cn_attribution: Optional[pd.DataFrame] = None,
    fundamental_attribution: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build published stack table with robust metrics and gate status."""
    cn_attrib = cn_attribution if cn_attribution is not None else pd.DataFrame()
    rows: List[Dict] = []

    for spec in FROZEN_OHLCV_REPS:
        row = _lookup_ranking(ranking, track="", factor=spec["factor"])
        rec = {
            "stack_version": STACK_VERSION,
            "stack_status": STACK_STATUS,
            "dim": spec["dim"],
            "factor": spec["factor"],
            "source": spec["source"],
            "stack_role": "frozen",
            "track": row.get("track", "eod_engine_robust") if row is not None else "",
            "notes": "OHLCV production rep — orthogonal return driver confirmed",
            "pending_checks": "",
            "strict_pass": "",
            "promotion_status": "production",
        }
        if row is not None:
            for col in (
                "production_score",
                "mean_abs_ic",
                "universe_stability",
                "sign_consistency",
                "mono_score_all",
                "mean_hl_sharpe",
                "all_universe_ic_hit",
            ):
                if col in row.index:
                    rec[col] = row[col]
        rows.append(rec)

    for spec in CN_CANDIDATES:
        row = _lookup_ranking(ranking, spec["track"], spec["factor"])
        strict = _lookup_strict(cn_attrib, spec["factor"])
        pending = _pending_checks(row, strict)
        rec = {
            "stack_version": STACK_VERSION,
            "stack_status": STACK_STATUS,
            "dim": spec["dim"],
            "factor": spec["factor"],
            "source": spec["source"],
            "stack_role": "candidate_pending",
            "track": spec["track"],
            "notes": "Strict residual IC pass; pending monotonicity + universe gates",
            "pending_checks": "|".join(pending) if pending else "",
            "strict_pass": strict if strict is not None else "",
            "promotion_status": "promote_if_gates_pass" if not pending else "hold",
        }
        if row is not None:
            for col in (
                "production_score",
                "mean_abs_ic",
                "universe_stability",
                "sign_consistency",
                "mono_score_all",
                "mean_hl_sharpe",
                "all_universe_ic_hit",
            ):
                if col in row.index:
                    rec[col] = row[col]
        rows.append(rec)

    for spec in DEFERRED_DIMENSIONS:
        rec = {
            "stack_version": STACK_VERSION,
            "stack_status": STACK_STATUS,
            "dim": spec["dim"],
            "factor": "",
            "source": "fundamental",
            "stack_role": "deferred",
            "track": "fundamental_batch1",
            "notes": spec["reason"],
            "pending_checks": spec["next"],
            "strict_pass": False,
            "promotion_status": "deferred",
            "factors_rejected": spec.get("factors_rejected", ""),
        }
        if fundamental_attribution is not None and not fundamental_attribution.empty:
            rejected = spec.get("factors_rejected", "").split(",")
            sub = fundamental_attribution[
                fundamental_attribution["factor_name"].isin(rejected)
            ]
            if not sub.empty and "conclusion" in sub.columns:
                rec["fundamental_verdict"] = "; ".join(
                    f"{r.factor_name}:{r.conclusion}" for r in sub.itertuples()
                )
        rows.append(rec)

    return pd.DataFrame(rows)


def publish_frozen_stack_v1(
    ranking: pd.DataFrame,
    out_dir: Optional[Path] = None,
    cn_attribution_path: Optional[Path] = None,
    fundamental_attribution_path: Optional[Path] = None,
) -> Path:
    """Write alpha_frozen_stack_v1.csv + .json to research/results."""
    out_dir = out_dir or cfg.RESEARCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cn_path = cn_attribution_path or out_dir / "cn_broker_attribution.csv"
    fund_path = fundamental_attribution_path or out_dir / "fundamental_attribution.csv"
    cn_attrib = pd.read_csv(cn_path) if cn_path.exists() else pd.DataFrame()
    fund_attrib = pd.read_csv(fund_path) if fund_path.exists() else pd.DataFrame()

    manifest = build_stack_manifest(ranking, cn_attrib, fund_attrib)
    csv_path = out_dir / "alpha_frozen_stack_v1.csv"
    manifest.to_csv(csv_path, index=False)

    meta = {
        "stack_version": STACK_VERSION,
        "stack_status": STACK_STATUS,
        "published": pd.Timestamp.now().isoformat(),
        "frozen_ohlcv_count": len(FROZEN_OHLCV_REPS),
        "cn_candidate_count": len(CN_CANDIDATES),
        "pending_gates": PENDING_GATES,
        "deferred": [d["dim"] for d in DEFERRED_DIMENSIONS],
        "entries": manifest.to_dict(orient="records"),
    }
    json_path = out_dir / "alpha_frozen_stack_v1.json"
    json_path.write_text(json.dumps(meta, indent=2, default=str))

    return csv_path
