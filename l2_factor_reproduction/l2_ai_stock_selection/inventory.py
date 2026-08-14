"""Current L2 factor inventory — reads existing registries, does not create a second one."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.contracts import (
    CANDIDATE_POOL_CSV,
    FS1_INVENTORY_CSV,
    classify_time_scale,
)
from l2_factor_reproduction.python.candidate_pool_registry import FAMILY_REGISTRY


INVENTORY_COLUMNS = (
    "factor_name",
    "factor_family",
    "category",
    "economic_interpretation",
    "primitive_source",
    "transform",
    "window",
    "time_scale",
    "signed",
    "availability_timestamp",
    "current_validation_status",
    "eligible_for_fs",
    "ineligible_reason",
    "redundancy_cluster_080",
    "formula",
    "lookback_days",
    "mechanism",
    "registry_status",
)


def _primitive_source(family: str) -> str:
    cfg = FAMILY_REGISTRY.get(family)
    if cfg is None:
        if family == "trade_flow_mcap_bridge":
            return "trade_flow_daily + FloatMktCap"
        return ""
    return cfg.primitive_name or ""


def load_factor_inventory(
    *,
    registry_path: Optional[Path] = None,
    fs1_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Join candidate_pool_v1 registry with FS-1 eligibility. Read-only."""
    registry_path = Path(registry_path or CANDIDATE_POOL_CSV)
    fs1_path = Path(fs1_path or FS1_INVENTORY_CSV)
    reg = pd.read_csv(registry_path)
    if "name" not in reg.columns:
        raise ValueError("candidate_registry.csv missing 'name'")
    reg = reg.rename(columns={"name": "factor_name", "family": "factor_family"})

    fs = pd.DataFrame()
    if fs1_path.exists():
        fs = pd.read_csv(fs1_path)
        fs = fs.rename(columns={"factor": "factor_name", "family": "factor_family"})

    keep_fs = [
        c
        for c in (
            "factor_name",
            "eligible_for_fs",
            "ineligible_reason",
            "redundancy_cluster_080",
            "data_source_path",
        )
        if c in fs.columns
    ]
    if keep_fs:
        merged = reg.merge(fs[keep_fs], on="factor_name", how="left")
    else:
        merged = reg.copy()
        merged["eligible_for_fs"] = True
        merged["ineligible_reason"] = ""
        merged["redundancy_cluster_080"] = ""

    lookback = pd.to_numeric(merged.get("lookback_days", 1), errors="coerce").fillna(1)
    signed_raw = merged.get("signed")
    if signed_raw is None:
        signed = pd.Series([""] * len(merged))
    else:
        signed = signed_raw.map({True: True, False: False, "True": True, "False": False})

    out = pd.DataFrame(
        {
            "factor_name": merged["factor_name"].astype(str),
            "factor_family": merged["factor_family"].astype(str),
            "category": merged["category"].fillna("").astype(str)
            if "category" in merged.columns
            else "",
            "economic_interpretation": merged["positive_value_meaning"].fillna("").astype(str)
            if "positive_value_meaning" in merged.columns
            else merged.get("mechanism", pd.Series([""] * len(merged))).astype(str),
            "primitive_source": merged["factor_family"].map(_primitive_source),
            "transform": merged["formula"].fillna("").astype(str)
            if "formula" in merged.columns
            else "",
            "window": lookback.astype(int),
            "time_scale": lookback.map(classify_time_scale),
            "signed": signed,
            "availability_timestamp": "session_close_T; executable_T_plus_1",
            "current_validation_status": merged.get(
                "registry_status", pd.Series(["unknown"] * len(merged))
            ).astype(str),
            "eligible_for_fs": merged.get("eligible_for_fs", True),
            "ineligible_reason": merged.get("ineligible_reason", "").fillna("").astype(str),
            "redundancy_cluster_080": merged.get("redundancy_cluster_080", "")
            .fillna("")
            .astype(str),
            "formula": merged.get("formula", "").fillna("").astype(str),
            "lookback_days": lookback.astype(int),
            "mechanism": merged.get("mechanism", "").fillna("").astype(str),
            "registry_status": merged.get("registry_status", "").fillna("").astype(str),
        }
    )
    return out[list(INVENTORY_COLUMNS)]


def family_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in inventory.groupby("factor_family", dropna=False):
        elig = g["eligible_for_fs"]
        if elig.dtype != bool:
            elig = elig.astype(str).str.lower().isin(("true", "1"))
        rows.append(
            {
                "factor_family": fam,
                "n_formulas": int(len(g)),
                "n_eligible_for_fs": int(elig.sum()),
                "n_signed": int(g["signed"].eq(True).sum()),
                "n_fast": int((g["time_scale"] == "fast").sum()),
                "n_mid": int((g["time_scale"] == "mid").sum()),
                "n_slow": int((g["time_scale"] == "slow").sum()),
                "primitive_source": g["primitive_source"].iloc[0] if len(g) else "",
            }
        )
    return pd.DataFrame(rows).sort_values("factor_family").reset_index(drop=True)
