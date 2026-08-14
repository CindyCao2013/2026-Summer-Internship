"""Continuous-auction session grid. Reuses liquidity_impact_daily timestamps.

Canonical continuous auction (from lid._session_filter / EXPECTED_CONTINUOUS_MINUTES):
  morning: 09:30-11:29  → mkey 570-689
  afternoon: 13:00-14:59 → mkey 780-899

Opening auction, lunch, hour-15 close auction are excluded.
Recovery +h is +h integer mkeys in the same session, never timestamp+timedelta.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python import liquidity_impact_daily as lid

# hour*60+minute, matching lid joined_minute_sql mkey.
AM_MKEY_START = 9 * 60 + 30  # 570 = 09:30
AM_MKEY_END = 11 * 60 + 29  # 689 = 11:29
PM_MKEY_START = 13 * 60  # 780 = 13:00
PM_MKEY_END = 14 * 60 + 59  # 899 = 14:59

AM_KEYS = np.arange(AM_MKEY_START, AM_MKEY_END + 1, dtype=np.int32)
PM_KEYS = np.arange(PM_MKEY_START, PM_MKEY_END + 1, dtype=np.int32)
ALL_SESSION_KEYS = np.concatenate([AM_KEYS, PM_KEYS])

assert len(AM_KEYS) + len(PM_KEYS) == lid.EXPECTED_CONTINUOUS_MINUTES


def mkey_from_hm(hour: int, minute: int) -> int:
    return int(hour) * 60 + int(minute)


def session_of_mkey(mkey: object) -> Optional[str]:
    """'AM' / 'PM' / None. Vectorized for arrays/Series."""
    if isinstance(mkey, (pd.Series, np.ndarray)):
        arr = np.asarray(mkey, dtype=np.int32)
        out = np.full(arr.shape, None, dtype=object)
        out[(arr >= AM_MKEY_START) & (arr <= AM_MKEY_END)] = "AM"
        out[(arr >= PM_MKEY_START) & (arr <= PM_MKEY_END)] = "PM"
        if isinstance(mkey, pd.Series):
            return pd.Series(out, index=mkey.index)
        return out
    k = int(mkey)
    if AM_MKEY_START <= k <= AM_MKEY_END:
        return "AM"
    if PM_MKEY_START <= k <= PM_MKEY_END:
        return "PM"
    return None


def session_keys(session: str) -> np.ndarray:
    if session == "AM":
        return AM_KEYS
    if session == "PM":
        return PM_KEYS
    raise ValueError(f"unknown session {session!r}")


def same_session(mkey: int, other: int) -> bool:
    a = session_of_mkey(mkey)
    return a is not None and a == session_of_mkey(other)


def horizon_in_session(mkey: int, h: int) -> bool:
    """True iff mkey+h is a valid bar in the same continuous session.

    This is the +h valid-trading-minute rule. It rejects lunch crossing,
    close/auction crossing, and end-of-session truncation. It does not
    bridge to the next session or next day.
    """
    if h < 0:
        return False
    target = int(mkey) + int(h)
    return same_session(int(mkey), target)


def pre_in_session(mkey: int) -> bool:
    """Immediately previous minute exists in the same session (causal pre)."""
    return horizon_in_session(int(mkey) - 1, 1)


def eligible_event_mkey(mkey: int, h: int) -> bool:
    """Shock at mkey can use a complete +h recovery path in-session, with causal pre."""
    return pre_in_session(mkey) and horizon_in_session(mkey, h)


def boundary_reason(mkey: int, h: int) -> str:
    """Why a (mkey, h) path is ineligible, or 'ok'."""
    sess = session_of_mkey(mkey)
    if sess is None:
        return "outside_continuous_auction"
    if not pre_in_session(mkey):
        if sess == "AM" and int(mkey) == AM_MKEY_START:
            return "no_pre_at_open"
        if sess == "PM" and int(mkey) == PM_MKEY_START:
            return "lunch_or_open_pre_cross"
        return "missing_pre"
    if horizon_in_session(mkey, h):
        return "ok"
    if sess == "AM":
        return "lunch_or_morning_close_cross"
    return "close_auction_or_session_end_cross"


def assert_not_naive_timedelta(event_ts: pd.Timestamp, h: int, target_mkey: int) -> None:
    """Guard: naive wall-clock +h minutes is not the recovery rule."""
    naive = event_ts + pd.Timedelta(minutes=int(h))
    naive_mkey = naive.hour * 60 + naive.minute
    if naive_mkey != int(target_mkey):
        # Crossing lunch/close is exactly when timedelta disagrees; that is intended.
        return
    if not horizon_in_session(mkey_from_hm(event_ts.hour, event_ts.minute), h):
        raise AssertionError("timedelta matched a path the session rule rejects")
