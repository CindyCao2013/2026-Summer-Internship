"""Limited ratio / state-normalized L2 feature catalog.

Does not explode the search space. Existing candidate-pool ratios are tagged
ALREADY_IN_POOL. Only economically justified new ratios are PROPOSED.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.inventory import load_factor_inventory

# Huatai-style examples mapped onto this library. Do not re-register aliases.
EXISTING_RATIO_ALIASES = (
    ("active_buy_amount / total_amount", "buy_dominance / net_buy_ratio"),
    ("active_sell_amount / total_amount", "complement of buy_dominance"),
    ("large_order_amount / total_amount", "large_order_ratio_20w"),
    ("mid_order_amount / total_amount", "mid_order_ratio_4w_20w / mid_order_ratio_5w_20w"),
    ("net_active_flow / total_amount", "net_buy_ratio"),
    ("net_active_flow / market_cap", "net_buy_amount_mcap"),
    ("spread / price", "relative_spread_mean"),
    (
        "depth_imbalance = (bid-ask)/(bid+ask)",
        "obi_l1_mean / obi_l5_mean / obi_l10_mean / weighted_obi_mean",
    ),
    (
        "large_buy / (large_buy + large_sell)",
        "large_order_direction",
    ),
    ("cancel_value / trade_value", "cancel_value_intensity"),
    ("price_impact / traded_amount", "signed_amount_impact / return_per_amount"),
)

PROPOSED_RATIOS = (
    {
        "candidate_name": "ofi_over_depth",
        "numerator": "signed_order_flow_or_net_order_change",
        "denominator": "total_depth_l5",
        "family": "order_book",
        "economic_interpretation": (
            "Order-flow imbalance scaled by displayed depth: same OFI is more "
            "informative when the book is thin."
        ),
        "zero_denom_policy": "set NaN when total_depth_l5 <= 0",
        "primitive_available": "PARTIAL",
        "note": "OBI exists; a true OFI/depth ratio is not a frozen pool formula.",
    },
    {
        "candidate_name": "impact_over_realized_vol",
        "numerator": "signed_amount_impact",
        "denominator": "intraday_realized_volatility",
        "family": "liquidity_impact x price_formation",
        "economic_interpretation": (
            "Price impact per unit of realized volatility: separates noisy "
            "high-vol days from true liquidity consumption."
        ),
        "zero_denom_policy": "set NaN when RV <= epsilon",
        "primitive_available": "READY",
        "note": "Both legs exist as frozen formulas; ratio itself is new.",
    },
    {
        "candidate_name": "spread_over_realized_vol",
        "numerator": "relative_spread_mean",
        "denominator": "intraday_realized_volatility",
        "family": "order_book x price_formation",
        "economic_interpretation": "Quoted spread in volatility units (liquidity tightness).",
        "zero_denom_policy": "set NaN when RV <= epsilon",
        "primitive_available": "READY",
        "note": "relative_spread_mean and realized vol both exist.",
    },
    {
        "candidate_name": "net_flow_over_adv",
        "numerator": "active_buy_amt - active_sell_amt",
        "denominator": "trailing_20d_total_amt",
        "family": "trade_flow",
        "economic_interpretation": (
            "Signed active flow scaled by recent ADV rather than same-day amount "
            "or market cap."
        ),
        "zero_denom_policy": "set NaN when ADV <= 0",
        "primitive_available": "PARTIAL",
        "note": "Same-day total_amt scaling exists; ADV scaling does not.",
    },
    {
        "candidate_name": "cancel_over_submitted",
        "numerator": "buy_cancel_qty + sell_cancel_qty",
        "denominator": "submitted_qty",
        "family": "cancel_lifecycle",
        "economic_interpretation": "Cancel intensity vs new submissions, not vs trades.",
        "zero_denom_policy": "set NaN when submitted_qty <= 0",
        "primitive_available": "MISSING",
        "note": "Current intensity denominators are trade value/qty, not submitted volume.",
    },
)


def _is_ratio_like(formula: str) -> bool:
    text = str(formula).lower()
    tokens = ("/", "ratio", "share", "imbalance", "dominance", "intensity")
    return any(tok in text for tok in tokens)


def build_ratio_candidate_registry(
    inventory: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if inventory is None:
        inventory = load_factor_inventory()
    rows: List[dict] = []
    for _, rec in inventory.iterrows():
        formula = str(rec.get("formula", ""))
        if not _is_ratio_like(formula):
            continue
        rows.append(
            {
                "candidate_name": rec["factor_name"],
                "status": "ALREADY_IN_POOL",
                "family": rec["factor_family"],
                "numerator": "",
                "denominator": "",
                "formula": formula,
                "economic_interpretation": rec.get("mechanism", ""),
                "zero_denom_policy": "inherited from frozen formula",
                "primitive_available": "READY",
                "eligible_for_fs": rec.get("eligible_for_fs", True),
                "note": "Do not recompute; reuse candidate_pool_v1.",
            }
        )
    seen = {r["candidate_name"] for r in rows}
    for spec in PROPOSED_RATIOS:
        if spec["candidate_name"] in seen:
            continue
        rows.append(
            {
                "candidate_name": spec["candidate_name"],
                "status": "PROPOSED",
                "family": spec["family"],
                "numerator": spec["numerator"],
                "denominator": spec["denominator"],
                "formula": "{}/{}".format(spec["numerator"], spec["denominator"]),
                "economic_interpretation": spec["economic_interpretation"],
                "zero_denom_policy": spec["zero_denom_policy"],
                "primitive_available": spec["primitive_available"],
                "eligible_for_fs": False,
                "note": spec["note"],
            }
        )
    cols = [
        "candidate_name",
        "status",
        "family",
        "numerator",
        "denominator",
        "formula",
        "economic_interpretation",
        "zero_denom_policy",
        "primitive_available",
        "eligible_for_fs",
        "note",
    ]
    return pd.DataFrame(rows)[cols]


NEAR_ZERO_DENOM = 1e-12
WINSOR_P = (0.01, 0.99)


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    eps: float = NEAR_ZERO_DENOM,
) -> np.ndarray:
    """num/den with zero and near-zero denominators set to NaN. Never Inf."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=float)
    num, den = np.broadcast_arrays(num, den)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > float(eps))
    out[ok] = num[ok] / den[ok]
    out[np.isinf(out)] = np.nan
    return out


def ratio_diagnostics(
    name: str,
    numerator: np.ndarray,
    denominator: np.ndarray,
    ratio: np.ndarray,
    *,
    family: str = "",
    economic_meaning: str = "",
    eps: float = NEAR_ZERO_DENOM,
    coverage_n_dates: int = 0,
    coverage_n_symbols: int = 0,
    n_cs_with_finite: float = float("nan"),
) -> dict:
    num = np.asarray(numerator, dtype=float).ravel()
    den = np.asarray(denominator, dtype=float).ravel()
    r = np.asarray(ratio, dtype=float).ravel()
    n = int(r.size)
    finite_r = r[np.isfinite(r)]
    finite_den = den[np.isfinite(den)]
    zero_den = int(np.isfinite(den).sum() and np.sum(np.isfinite(den) & (den == 0)))
    near_zero = int(np.sum(np.isfinite(den) & (np.abs(den) > 0) & (np.abs(den) <= float(eps))))
    if finite_r.size:
        p1, p50, p99 = np.quantile(finite_r, [0.01, 0.50, 0.99])
        lo, hi = np.quantile(finite_r, list(WINSOR_P))
        clipped = np.clip(finite_r, lo, hi)
        winsor_impact = float(np.mean(np.abs(finite_r - clipped)))
    else:
        p1 = p50 = p99 = winsor_impact = float("nan")
    return {
        "candidate_name": name,
        "family": family,
        "numerator": "",
        "denominator": "",
        "economic_meaning": economic_meaning,
        "n": n,
        "zero_denominator_rate": float(zero_den / n) if n else float("nan"),
        "near_zero_denominator_rate": float(near_zero / n) if n else float("nan"),
        "nan_rate": float(np.mean(~np.isfinite(r))) if n else float("nan"),
        "inf_rate": float(np.mean(np.isinf(r))) if n else 0.0,
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "cross_sectional_coverage": float(n_cs_with_finite),
        "coverage_n_dates": int(coverage_n_dates),
        "coverage_n_symbols": int(coverage_n_symbols),
        "winsorization_impact": float(winsor_impact),
        "n_finite_ratio": int(finite_r.size),
        "n_finite_denominator": int(finite_den.size),
        "pathological": bool(
            (n and float(np.mean(np.isinf(r))) > 0)
            or (n and float(zero_den / n) > 0.50)
        ),
    }
