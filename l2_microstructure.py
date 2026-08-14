"""L2 microstructure daily bricks — VOI, OIR, MPB (中信建投 高频量价).

Definitions (daily aggregate from minute tick/order-flow labels):
  VOI — volume order imbalance: (active_buy_vol - active_sell_vol) / total active
  OIR — order imbalance ratio:   (bid_cancel_vol - ask_cancel_vol) / total cancel
  MPB — mid-price basis / pressure: (active_buy_amt - active_sell_amt) / total active amt

Each factor = 20d rolling mean of the daily brick (EOD signal, not intraday zoo).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from l2_data_loaders import L2DailyWideCache

_EPS = 1e-8
ROLL_WINDOW = 20
MIN_PERIODS = 10


def daily_voi(cache: L2DailyWideCache) -> pd.DataFrame:
    num = cache.active_buy_vol - cache.active_sell_vol
    den = cache.active_buy_vol + cache.active_sell_vol
    return num / den.replace(0, np.nan)


def daily_oir(cache: L2DailyWideCache) -> pd.DataFrame:
    num = cache.bid_cancel_vol - cache.ask_cancel_vol
    den = cache.bid_cancel_vol + cache.ask_cancel_vol
    return num / den.replace(0, np.nan)


def daily_mpb(cache: L2DailyWideCache) -> pd.DataFrame:
    num = cache.active_buy_amt - cache.active_sell_amt
    den = cache.active_buy_amt + cache.active_sell_amt
    return num / den.replace(0, np.nan)


def rolling_brick(daily: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    return daily.rolling(window, min_periods=MIN_PERIODS).mean()


def build_l2_daily_bricks(cache: L2DailyWideCache) -> Dict[str, pd.DataFrame]:
    return {
        "voi_daily": daily_voi(cache),
        "oir_daily": daily_oir(cache),
        "mpb_daily": daily_mpb(cache),
    }


def build_l2_factor_panels(cache: L2DailyWideCache) -> Dict[str, pd.DataFrame]:
    bricks = build_l2_daily_bricks(cache)
    return {
        "cn_voi_20d": rolling_brick(bricks["voi_daily"]),
        "cn_oir_20d": rolling_brick(bricks["oir_daily"]),
        "cn_mpb_20d": rolling_brick(bricks["mpb_daily"]),
    }
