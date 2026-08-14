"""Style-control names and sequential RankIC stripping (diagnostic, not causal)."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import rank_ic
from l2_factor_reproduction.l2_ai_stock_selection.residual_alpha import residualize_panel


DEFAULT_STRIP_ORDER: Sequence[str] = (
    "industry",
    "size",
    "momentum_20d",
    "residual_volatility",
    "liquidity",
    "turnover_20d",
)


def sequential_ic_after_controls(
    score: pd.DataFrame,
    target: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
    *,
    order: Sequence[str] = DEFAULT_STRIP_ORDER,
    train_dates: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Progressive residualization of the *score*, then RankIC vs target.

    This is an exposure-sensitivity diagnostic. Do not interpret the sequence
    as a causal chain. Residualization uses ``train_dates`` only.
    """
    rows: List[dict] = []
    rows.append(
        {
            "step": "raw",
            "control_added": "",
            "rank_ic": rank_ic(score, target),
        }
    )
    accumulated: Dict[str, pd.DataFrame] = {}
    current = score
    for name in order:
        if name not in controls:
            rows.append(
                {
                    "step": "after_{}".format(name),
                    "control_added": name,
                    "rank_ic": float("nan"),
                    "status": "CONTROL_UNAVAILABLE",
                }
            )
            continue
        accumulated[name] = controls[name]
        current = residualize_panel(
            current, accumulated, train_dates=train_dates
        )
        rows.append(
            {
                "step": "after_{}".format(name),
                "control_added": name,
                "rank_ic": rank_ic(current, target),
                "status": "OK",
            }
        )
    return pd.DataFrame(rows)


def style_control_catalog() -> pd.DataFrame:
    """Where each style series should come from. No silent reconstruction."""
    return pd.DataFrame(
        [
            {
                "style": "industry",
                "source": "Factor_Dev_Lib.get_preheat_ind_data_citics (Citics L1)",
                "status": "READY",
            },
            {
                "style": "size",
                "source": "primitives/mcap_wide_2019-01-01_2026-07-31.parquet log FloatMktCap",
                "status": "READY",
            },
            {
                "style": "momentum_20d",
                "source": "factor_formulas.momentum_20d or 20d cum c2c (test_double_neutralization)",
                "status": "READY",
            },
            {
                "style": "residual_volatility",
                "source": "20d return std (test_double_neutralization) — residual vol vs beta is PARTIAL",
                "status": "PARTIAL",
            },
            {
                "style": "liquidity",
                "source": "intraday_amihud / relative_spread_mean / ADV — choose explicitly",
                "status": "PARTIAL",
            },
            {
                "style": "turnover_20d",
                "source": "WIND S_DQ_TURN 20d mean log (test_double_neutralization)",
                "status": "READY",
            },
            {
                "style": "beta",
                "source": "not a frozen L2 primitive; must be built from ret_matrix vs benchmark",
                "status": "MISSING",
            },
            {
                "style": "short_term_reversal",
                "source": "1d/5d reversal from ret_matrix if used; not a named L2 factor",
                "status": "PARTIAL",
            },
        ]
    )
