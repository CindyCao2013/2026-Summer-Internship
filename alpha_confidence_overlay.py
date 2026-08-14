"""Alpha confidence overlay — adjust EOD stack signal with L2 state (Signal × State).

  S' = z(S) * (1 + λ * z(L2_state))

Not a new factor; tests whether L2 improves ranking quality of frozen stack.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_information_space import mean_rank_ic
from factor_attribution import align_signal, combine_equal_weight, cs_zscore, hl_sharpe_from_composite


def confidence_overlay_signal(
    baseline_panel: pd.DataFrame,
    l2_state_panel: pd.DataFrame,
    lam: float = 0.25,
    signal_shift: int = 1,
) -> pd.DataFrame:
    """Multiplicative overlay on cross-sectional z-scored baseline."""
    base = cs_zscore(align_signal(baseline_panel, signal_shift))
    l2 = cs_zscore(align_signal(l2_state_panel, signal_shift))
    return cs_zscore(base * (1.0 + lam * l2))


def overlay_metrics(
    baseline_panels: List[pd.DataFrame],
    l2_state_panel: pd.DataFrame,
    ret: pd.DataFrame,
    lam: float = 0.25,
    label: str = "",
) -> Dict:
    baseline = combine_equal_weight(baseline_panels)
    base_sig = align_signal(baseline)
    overlay = confidence_overlay_signal(baseline, l2_state_panel, lam=lam)

    ic_base = mean_rank_ic(base_sig, ret)
    ic_overlay = mean_rank_ic(overlay, ret)
    sharpe_base, _, _ = hl_sharpe_from_composite(base_sig, ret)
    sharpe_overlay, _, _ = hl_sharpe_from_composite(overlay, ret)

    return {
        "overlay_label": label,
        "lambda": lam,
        "ic_baseline": ic_base,
        "ic_overlay": ic_overlay,
        "ic_delta": ic_overlay - ic_base if pd.notna(ic_overlay) and pd.notna(ic_base) else np.nan,
        "sharpe_baseline": sharpe_base,
        "sharpe_overlay": sharpe_overlay,
        "sharpe_delta": sharpe_overlay - sharpe_base if pd.notna(sharpe_overlay) and pd.notna(sharpe_base) else np.nan,
    }


def run_overlay_grid(
    baseline_panels: List[pd.DataFrame],
    l2_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    lambdas: Optional[List[float]] = None,
) -> pd.DataFrame:
    lambdas = lambdas or [0.25]
    rows = []
    for fname, panel in l2_panels.items():
        for lam in lambdas:
            rows.append(overlay_metrics(baseline_panels, panel, ret, lam=lam, label=fname))
    return pd.DataFrame(rows)
