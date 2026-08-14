"""Walk-forward / leakage helpers for L2 AI Stock Selection v1.

Reuses FS-3 purge rule: train_label_end < oos_anchor.
Adds an availability rule the FS-4 freeze failed: score dates must be
strictly after train_label_end_max.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.contracts import EXECUTION_CONVENTION


def assert_train_does_not_use_oos(
    train_dates: Sequence,
    oos_start,
    label_end_by_feature_date: pd.Series,
) -> None:
    """Raise if any train feature date has label_end >= oos_start."""
    oos_start = pd.Timestamp(oos_start).normalize()
    for dt in train_dates:
        dt = pd.Timestamp(dt).normalize()
        if dt not in label_end_by_feature_date.index:
            continue
        end = label_end_by_feature_date.loc[dt]
        if pd.isna(end):
            continue
        if pd.Timestamp(end) >= oos_start:
            raise ValueError(
                "lookahead: train feature {} has label_end {} >= oos_start {}".format(
                    dt.date(), pd.Timestamp(end).date(), oos_start.date()
                )
            )


def first_allowed_score_date(train_label_end_max, trading_dates: Sequence):
    """Strict AI-v1 rule: first score date must be > train_label_end_max."""
    cutoff = pd.Timestamp(train_label_end_max).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize().unique().sort_values()
    later = dates[dates > cutoff]
    if len(later) == 0:
        return pd.NaT
    return pd.Timestamp(later[0])


def score_dates_are_available(
    score_dates: Sequence,
    train_label_end_max,
) -> pd.DataFrame:
    """FS-4 gap: scores on dates <= train_label_end_max are retroactive."""
    cutoff = pd.Timestamp(train_label_end_max).normalize()
    rows = []
    for dt in score_dates:
        dt = pd.Timestamp(dt).normalize()
        ok = dt > cutoff
        rows.append(
            {
                "TradeDate": dt,
                "train_label_end_max": cutoff,
                "available": bool(ok),
                "reason": "OK" if ok else "RETROACTIVE_SCORE",
            }
        )
    return pd.DataFrame(rows)


def filter_available_score_dates(
    score_dates: Sequence,
    train_label_end_max,
) -> pd.DatetimeIndex:
    """AI-v1 adapter: keep only score_date > train_label_end_max."""
    audit = score_dates_are_available(score_dates, train_label_end_max)
    ok = audit.loc[audit["available"], "TradeDate"]
    return pd.DatetimeIndex(ok).normalize().unique().sort_values()


def execution_summary() -> str:
    return (
        "signal_shift={shift}; feature T -> returns T+1..T+h; method={method}".format(
            shift=EXECUTION_CONVENTION["signal_shift"],
            method=EXECUTION_CONVENTION["return_method"],
        )
    )
