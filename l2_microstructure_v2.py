"""L2 Microstructure v2 — event-driven bricks (6 factors only).

Groups:
  A. Flow shock     — cn_voi_shock, cn_mpb_shock
  B. Persistence    — cn_flow_persistence, cn_imbalance_duration
  C. Consumption    — cn_liquidity_consumption, cn_cancel_shock

NOT slow 20d level means. Each signal is daily event/state at t.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from l2_data_loaders import L2DailyWideCache
from l2_microstructure import daily_mpb, daily_oir, daily_voi

_EPS = 1e-8
SHOCK_WINDOW = 60
SHOCK_MIN_PERIODS = 20
PERSIST_LAG = 5
PERSIST_WINDOW = 20
PERSIST_MIN_PERIODS = 10


def time_series_zscore(daily: pd.DataFrame, window: int = SHOCK_WINDOW) -> pd.DataFrame:
    mu = daily.rolling(window, min_periods=SHOCK_MIN_PERIODS).mean()
    sd = daily.rolling(window, min_periods=SHOCK_MIN_PERIODS).std()
    return (daily - mu) / sd.replace(0, np.nan)


def flow_persistence(daily: pd.DataFrame, lag: int = PERSIST_LAG, window: int = PERSIST_WINDOW) -> pd.DataFrame:
    """Corr(VOI_t, mean(VOI_{t-lag:t-1})) over trailing window."""
    lagged = daily.rolling(lag, min_periods=max(3, lag // 2)).mean().shift(1)
    return daily.rolling(window, min_periods=PERSIST_MIN_PERIODS).corr(lagged)


def daily_cancel_imbalance(cache: L2DailyWideCache) -> pd.DataFrame:
    return daily_oir(cache)


def daily_liquidity_consumption_ratio(cache: L2DailyWideCache) -> pd.DataFrame:
    """LCR proxy: trade volume / active-flow depth (SSL2 book depth → v2.5)."""
    depth_proxy = cache.active_buy_vol + cache.active_sell_vol
    return cache.volume / depth_proxy.replace(0, np.nan)


def daily_imbalance_duration(cache: L2DailyWideCache) -> pd.DataFrame:
    """Fraction of minutes with minute-VOI > threshold (buy-pressure duration proxy)."""
    if cache.imbalance_duration is not None and not cache.imbalance_duration.empty:
        return cache.imbalance_duration
    # Fallback: daily VOI level as weak proxy if minute query unavailable
    return (daily_voi(cache) > 0.1).astype(float)


def build_l2_v2_factor_panels(cache: L2DailyWideCache) -> Dict[str, pd.DataFrame]:
    voi = daily_voi(cache)
    mpb = daily_mpb(cache)
    cancel = daily_cancel_imbalance(cache)
    return {
        "cn_voi_shock": time_series_zscore(voi),
        "cn_mpb_shock": time_series_zscore(mpb),
        "cn_flow_persistence": flow_persistence(voi),
        "cn_imbalance_duration": daily_imbalance_duration(cache),
        "cn_liquidity_consumption": daily_liquidity_consumption_ratio(cache),
        "cn_cancel_shock": time_series_zscore(cancel),
    }
