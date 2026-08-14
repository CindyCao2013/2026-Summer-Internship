"""Horizon-aware IC profile and FS consensus.

Does not optimize consensus weights on backtest Sharpe.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.contracts import CANONICAL_HORIZONS
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import rank_ic


def ic_horizon_row(
    factor: pd.DataFrame,
    labels: Mapping[int, pd.DataFrame],
    *,
    horizons: Sequence[int] = CANONICAL_HORIZONS,
    execution_contract: Optional[str] = None,
) -> Dict[str, float]:
    from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
        resolve_execution_contract,
    )

    contract = resolve_execution_contract(execution_contract)
    ics = {}
    for h in horizons:
        if h not in labels:
            ics[h] = float("nan")
            continue
        ics[h] = rank_ic(factor, labels[h])
    finite = {h: v for h, v in ics.items() if np.isfinite(v)}
    peak = max(finite, key=lambda k: abs(finite[k])) if finite else float("nan")
    peak_ic = finite.get(peak, float("nan")) if finite else float("nan")
    signs = [np.sign(v) for v in finite.values() if abs(v) > 1e-12]
    sign_stability = float(abs(sum(signs)) / len(signs)) if signs else float("nan")
    half_life = approx_half_life(ics, peak)
    return {
        **{f"IC_{h}D": ics[h] for h in horizons},
        "peak_horizon": peak,
        "peak_ic": peak_ic,
        "sign_stability": sign_stability,
        "approx_half_life": half_life,
        "execution_contract": contract,
    }


def approx_half_life(ics: Mapping[int, float], peak_horizon) -> float:
    """Smallest horizon >= peak whose |IC| has fallen to half of peak |IC|."""
    if peak_horizon != peak_horizon:
        return float("nan")
    peak = int(peak_horizon)
    peak_abs = abs(float(ics.get(peak, np.nan)))
    if not np.isfinite(peak_abs) or peak_abs < 1e-12:
        return float("nan")
    ordered = sorted(h for h in ics if h >= peak)
    for h in ordered:
        val = ics[h]
        if np.isfinite(val) and abs(val) <= 0.5 * peak_abs:
            return float(h - peak)
    last = ordered[-1] if ordered else peak
    return float(last - peak)


def consensus_selection(
    evidence: pd.DataFrame,
    *,
    min_methods: int = 2,
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Transparent count of independent evidence flags.

    ``evidence`` has boolean/0-1 columns named in ``methods``.
    Selection is by count, not by a Sharpe-tuned weighted score.
    """
    out = evidence.copy()
    if methods is None:
        skip = {
            "factor",
            "family",
            "selected",
            "selection_count",
            "selection_methods",
            "nonlinear_keep_override",
            "nonlinear_review",
            "tree_gain_without_confirmation",
            "jury_state",
        }
        methods = tuple(c for c in out.columns if c not in skip)
    present = [m for m in methods if m in out.columns]
    if not present:
        raise ValueError("evidence table has none of {}".format(methods))
    flags = out[present].fillna(0).astype(float).clip(0, 1)
    out["selection_count"] = flags.sum(axis=1).astype(int)
    out["selection_methods"] = flags.apply(
        lambda row: ",".join(m for m in present if row[m] >= 0.5), axis=1
    )
    out["selected"] = out["selection_count"] >= int(min_methods)
    return out
