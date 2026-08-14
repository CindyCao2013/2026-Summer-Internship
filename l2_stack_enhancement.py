"""L2 stack enhancement test — does L2 improve frozen alpha stack (not standalone IC)?

Primary question for L2 research:
  "Does L2 improve D1–D5?"  NOT  "Does L2 have raw IC?"

L2 can be a conditional alpha enhancer even when strict residual IC fails.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from alpha_information_space import mean_rank_ic
from factor_attribution import (
    align_signal,
    combine_equal_weight,
    cs_zscore,
    hl_sharpe_from_composite,
    incremental_bundle_test,
)

# Enhancement gate thresholds (looser than strict dimension gate)
STACK_IC_DELTA_MIN = 0.003  # +30bp mean rank IC on equal-weight stack
STACK_SHARPE_DELTA_MIN = 0.10  # +0.10 H-L Sharpe on stack + candidate blend


def stack_enhancement_row(
    baseline_panels: List[pd.DataFrame],
    candidate_panel: pd.DataFrame,
    ret: pd.DataFrame,
    factor_name: str,
) -> Dict:
    """Compare frozen stack alone vs stack augmented with L2 candidate."""
    baseline = combine_equal_weight(baseline_panels)
    cand_z = cs_zscore(align_signal(candidate_panel))
    baseline_sig = align_signal(baseline)

    ic_base = mean_rank_ic(baseline_sig, ret)
    ic_cand = mean_rank_ic(cand_z, ret)

    enhanced = cs_zscore(baseline_sig + cand_z)
    ic_enh = mean_rank_ic(enhanced, ret)

    sharpe_base, ret_base, _ = hl_sharpe_from_composite(baseline_sig, ret)
    sharpe_enh, ret_enh, _ = hl_sharpe_from_composite(enhanced, ret)
    sharpe_cand, _, _ = hl_sharpe_from_composite(cand_z, ret)

    ic_delta = ic_enh - ic_base if pd.notna(ic_enh) and pd.notna(ic_base) else np.nan
    sharpe_delta = (
        sharpe_enh - sharpe_base if pd.notna(sharpe_enh) and pd.notna(sharpe_base) else np.nan
    )

    bundle = incremental_bundle_test(baseline_panels, candidate_panel, ret)

    enhancement_pass = (
        (pd.notna(ic_delta) and ic_delta >= STACK_IC_DELTA_MIN)
        or (pd.notna(sharpe_delta) and sharpe_delta >= STACK_SHARPE_DELTA_MIN)
    )

    return {
        "factor_name": factor_name,
        "stack_ic_baseline": ic_base,
        "stack_ic_enhanced": ic_enh,
        "stack_ic_delta": ic_delta,
        "candidate_solo_ic": ic_cand,
        "stack_sharpe_baseline": sharpe_base,
        "stack_sharpe_enhanced": sharpe_enh,
        "stack_sharpe_delta": sharpe_delta,
        "candidate_solo_sharpe": sharpe_cand,
        "stack_enhancement_pass": enhancement_pass,
        **{k: v for k, v in bundle.items() if k != "factor_name"},
    }


def run_stack_enhancement_test(
    baseline_panels: List[pd.DataFrame],
    candidate_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        stack_enhancement_row(baseline_panels, panel, ret, fname)
        for fname, panel in candidate_panels.items()
    ]
    return pd.DataFrame(rows)
