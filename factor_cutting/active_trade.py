"""APM / active-trading factor — daily proxy + minute upgrade stub.

Paper APM needs overnight vs afternoon residuals vs index, then CS residual vs Ret20.
Full implementation is Stage-4 (session / minute). v1 ships a daily research proxy only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from factor_cutting.engine import CuttingSpec, KnifeSpec, ObjectSpec, OutputSpec

APM_SPEC = CuttingSpec(
    name="apm",
    paper="APM因子模型的进阶版",
    direction_paper="positive_ic",
    status="stub_needs_session_or_minute",
    object=ObjectSpec(variable="overnight_vs_afternoon_residual", additive=True),
    knife=KnifeSpec(variable="time_of_day", method="bucket", window=20),
    output=OutputSpec(op="difference", formula="residualized_stat"),
)


def overnight_return(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return open_ / close.shift(1) - 1.0


def daytime_return(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return close / open_.replace(0, np.nan) - 1.0


def compute_apm_overnight_day_proxy(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """
    Research proxy (NOT paper APM):

        mean(overnight - daytime) / (std / sqrt(n)) over ``window``

    Does not residualize vs index or Ret20. Use only as cutting-framework smoke test.
    """
    delta = overnight_return(open_, close) - daytime_return(open_, close)
    mu = delta.rolling(window, min_periods=max(10, window // 2)).mean()
    sd = delta.rolling(window, min_periods=max(10, window // 2)).std()
    n = delta.rolling(window, min_periods=max(10, window // 2)).count()
    return mu / (sd / np.sqrt(n.replace(0, np.nan)))


def compute_apm(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError(
        "Paper APM requires session/minute returns + index residualization. "
        "Use compute_apm_overnight_day_proxy for a daily smoke-test, "
        "or upgrade MinuteFeatureStore (Stage 4)."
    )
