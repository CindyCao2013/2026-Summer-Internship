"""Financial / derivative reads — thin wrappers (wide tables stay in factor_data_loaders)."""

from __future__ import annotations

import datetime as dt
from typing import Optional, Tuple, Union

import pandas as pd

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]


def fetch_financial_ttm_long(
    start: DateLike,
    end: DateLike,
    *,
    session=None,
    history_years: int = 3,
    statement_type: str = "合并报表",
) -> Tuple[pd.DataFrame, object]:
    """Delegate to ``factor_data_loaders.load_financial_ttmhis_long``."""
    from factor_data_loaders import load_financial_ttmhis_long

    start_ts = pd.Timestamp(start).to_pydatetime()
    end_ts = pd.Timestamp(end).to_pydatetime()
    return load_financial_ttmhis_long(
        start_ts,
        end_ts,
        session=session,
        history_years=history_years,
        statement_type=statement_type,
    )
