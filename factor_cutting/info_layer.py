"""Information Layer API — knife search as information transformation.

Not feature engineering: finds where additive market information should be partitioned.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from factor_cutting.cutting_analysis.knife_family_analysis import (
    FAMILY_OF,
    family_attribution_report,
    family_of,
)
from factor_cutting.knives import available_knives, build_knife

DEFAULT_CANDIDATES = [
    "volume",
    "amount",
    "ats_trade_count",
    "ats_volume",
    "turnover",
    "amihud",
]


def search_knives(
    object_panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    candidates: Optional[Sequence[str]] = None,
    amount: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    trade_count: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    ret_1d: Optional[pd.DataFrame] = None,
    window: int = 20,
    corr_indep_threshold: float = 0.50,
) -> dict:
    """
    Search knife space for a fixed additive object.

    Returns JSON-serializable dict::

        {
          "best_knife": "volume",
          "effectiveness": 0.045,
          "family": "participation",
          "ranking": [...],
          "independent_knives": ["ats_trade_count"],
          "corr": {...},
        }
    """
    cand = list(candidates or DEFAULT_CANDIDATES)
    knives: Dict[str, pd.DataFrame] = {}

    base = available_knives(
        amount=amount,
        volume=volume,
        trade_count=trade_count,
        turnover=turnover,
        ret_1d=ret_1d,
        float_mktcap=float_mktcap,
    )
    for name in cand:
        if name in base:
            knives[name] = base[name].reindex_like(object_panel)
            continue
        try:
            knives[name] = build_knife(
                name,
                amount=amount,
                volume=volume,
                trade_count=trade_count,
                turnover=turnover,
                ret_1d=ret_1d,
                float_mktcap=float_mktcap,
            ).reindex_like(object_panel)
        except (ValueError, KeyError, TypeError):
            continue

    # Drop empty
    knives = {k: v for k, v in knives.items() if v.notna().sum().sum() >= 1000}
    if not knives:
        return {
            "best_knife": None,
            "effectiveness": None,
            "family": None,
            "ranking": [],
            "independent_knives": [],
            "corr": {},
            "error": "no_knives_available",
        }

    eval_df, corr_df, indep_df, _ = family_attribution_report(
        object_panel, knives, ret, window=window
    )

    best = eval_df.iloc[0]
    independent = []
    if indep_df is not None and not indep_df.empty:
        independent = indep_df.loc[indep_df["independent"], "knife"].tolist()
        # best itself always listed separately; peers that remain independent of best
        independent = [k for k in independent if k != best["knife"]]

    ranking = []
    for _, r in eval_df.iterrows():
        ranking.append(
            {
                "knife": r["knife"],
                "family": r["family"],
                "effectiveness": float(r["effectiveness"]) if pd.notna(r["effectiveness"]) else None,
                "ic_spread": float(r["ic_spread"]) if pd.notna(r["ic_spread"]) else None,
                "separation": float(r["separation"]) if pd.notna(r["separation"]) else None,
            }
        )

    corr_dict = {}
    if corr_df is not None and not corr_df.empty:
        for a in corr_df.index:
            corr_dict[str(a)] = {
                str(b): (None if pd.isna(corr_df.loc[a, b]) else float(corr_df.loc[a, b]))
                for b in corr_df.columns
            }

    return {
        "object": "panel",
        "best_knife": str(best["knife"]),
        "effectiveness": float(best["effectiveness"]) if pd.notna(best["effectiveness"]) else None,
        "family": family_of(str(best["knife"])),
        "ranking": ranking,
        "independent_knives": independent,
        "corr": corr_dict,
        "n_candidates": len(knives),
        "corr_indep_threshold": corr_indep_threshold,
        "families": {k: FAMILY_OF.get(k, "other") for k in knives},
    }
