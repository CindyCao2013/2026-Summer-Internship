"""P2 L2 net active-flow factors — cumulative buy−sell amount (not VOI/MPB ratios).

Distinct from cn_voi_shock (vol imbalance z) / cn_mpb_shock (amt imbalance z):
these are size-normalized cumulative flow levels over 5/20d.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from factor_attribution import cs_zscore
from l2_data_loaders import L2DailyWideCache
from l2_microstructure_v2 import time_series_zscore

P2_FLOW_FACTOR_LIST = [
    "net_active_flow_mktcap_20d",
    "net_active_flow_mktcap_5d",
    "net_active_flow_shock",
    "active_buy_share_20d",
]

_EPS = 1e-8


def daily_net_active_amt(cache: L2DailyWideCache) -> pd.DataFrame:
    return cache.active_buy_amt - cache.active_sell_amt


def build_net_active_flow_mktcap(
    cache: L2DailyWideCache,
    float_mktcap: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Parameterized net active flow / mktcap cumulative sum (for ±30% stability)."""
    net = daily_net_active_amt(cache)
    mcap = float_mktcap.reindex(index=net.index, columns=net.columns).replace(0, np.nan)
    flow_over_mcap = net / mcap
    min_p = max(3, window // 2)
    return cs_zscore(flow_over_mcap.rolling(window, min_periods=min_p).sum())


def build_p2_flow_panels(
    cache: L2DailyWideCache,
    float_mktcap: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    net = daily_net_active_amt(cache)
    mcap = float_mktcap.reindex(index=net.index, columns=net.columns).replace(0, np.nan)
    # Wind derivative mktcap often in 万元; active amt typically yuan — scale-robust via cs_z later
    flow_over_mcap = net / mcap
    buy = cache.active_buy_amt
    sell = cache.active_sell_amt
    buy_share = buy / (buy + sell).replace(0, np.nan)

    panels = {
        "net_active_flow_mktcap_20d": flow_over_mcap.rolling(20, min_periods=10).sum(),
        "net_active_flow_mktcap_5d": flow_over_mcap.rolling(5, min_periods=3).sum(),
        "net_active_flow_shock": time_series_zscore(net),
        "active_buy_share_20d": buy_share.rolling(20, min_periods=10).mean(),
    }
    return {k: cs_zscore(v) for k, v in panels.items()}


def filter_p2_flow_factors(names: List[str]) -> List[str]:
    valid = set(P2_FLOW_FACTOR_LIST)
    return [n for n in names if n in valid]
