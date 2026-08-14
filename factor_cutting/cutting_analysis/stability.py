"""Stability: yearly IC, regime slices, universe IC table."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from factor_attribution import universe_ic_table
from factor_cutting.cutting_analysis.knife_ic import apply_universe_mask, ic_stats, monthly_ic_stats

DEFAULT_REGIMES: List[Tuple[str, str, str]] = [
    ("2010_2014", "2010-01-01", "2014-12-31"),
    ("2015_2016_bull_crash", "2015-01-01", "2016-12-31"),
    ("2017_2019", "2017-01-01", "2019-12-31"),
    ("2020_2021", "2020-01-01", "2021-12-31"),
    ("2022_bear", "2022-01-01", "2022-12-31"),
    ("2023_2025", "2023-01-01", "2025-12-31"),
]


def yearly_ic_table(panel: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    ic = daily_rank_ic_series(panel, ret)
    rows = []
    for year, sub in ic.groupby(ic.index.year):
        s = sub.dropna()
        if len(s) < 40:
            rows.append({"year": int(year), "n": len(s), "rank_ic": np.nan, "icir": np.nan})
        else:
            rows.append(
                {
                    "year": int(year),
                    "n": len(s),
                    "rank_ic": float(s.mean()),
                    "icir": icir_from_daily(s),
                }
            )
    return pd.DataFrame(rows)


def regime_ic_table(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    regimes: Sequence[Tuple[str, str, str]] = DEFAULT_REGIMES,
) -> pd.DataFrame:
    rows = []
    for label, start_s, end_s in regimes:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        p = panel.loc[(panel.index >= start) & (panel.index <= end)]
        r = ret.loc[(ret.index >= start) & (ret.index <= end)]
        if len(p) < 40:
            rows.append({"regime": label, "n": len(p), "rank_ic": np.nan, "icir": np.nan})
            continue
        st = ic_stats(p, r)
        rows.append(
            {
                "regime": label,
                "n": st["n_days"],
                "rank_ic": st["rank_ic"],
                "icir": st["icir"],
            }
        )
    return pd.DataFrame(rows)


def universe_stability_table(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    universe_masks: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per-universe RankIC / ICIR (masks are 1/NaN membership)."""
    rows = []
    # ALL
    st = ic_stats(panel, ret)
    rows.append({"universe": "ALL", **{k: st[k] for k in ("rank_ic", "icir", "n_days")}})
    for uni, mask in universe_masks.items():
        p = apply_universe_mask(panel, mask)
        st = ic_stats(p, ret)
        rows.append({"universe": uni, **{k: st[k] for k in ("rank_ic", "icir", "n_days")}})
    return pd.DataFrame(rows)


def full_stability_pack(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    universe_masks: Optional[Dict[str, pd.DataFrame]] = None,
    regimes: Sequence[Tuple[str, str, str]] = DEFAULT_REGIMES,
) -> dict:
    out = {
        "full": ic_stats(panel, ret),
        "monthly": monthly_ic_stats(panel, ret),
        "yearly": yearly_ic_table(panel, ret),
        "regime": regime_ic_table(panel, ret, regimes=regimes),
    }
    # drop heavy series from full for JSON friendliness later
    out["full_meta"] = {k: out["full"][k] for k in ("rank_ic", "icir", "ic_pos_ratio", "n_days")}
    out["monthly_meta"] = {
        k: out["monthly"][k] for k in ("monthly_rank_ic", "monthly_icir", "n_months")
    }
    if universe_masks:
        out["universe"] = universe_stability_table(panel, ret, universe_masks)
        # also reuse existing helper for sign consistency attrs when possible
        try:
            uni_df = universe_ic_table(panel, ret, universe_masks)
            out["universe_extra"] = uni_df
        except Exception:
            pass
    return out
