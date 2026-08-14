"""Clock-time cut operators. Explicit segment contract, not ad-hoc slices."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    AM_MKEY_END,
    AM_MKEY_START,
    AUCTION_MKEY,
    EXPECTED_CONTINUOUS_MINUTES,
    LUNCH_MKEY_END,
    LUNCH_MKEY_START,
    PM_MKEY_END,
    PM_MKEY_START,
    TIME_SEGMENTS,
    time_segment,
)


def mkey_from_hm(hour: int, minute: int) -> int:
    return int(hour) * 60 + int(minute)


def mkey_from_minute_index(minute_index) -> np.ndarray:
    """Map frozen 0-239 continuous index onto mkey. 240 (auction) -> 900."""
    idx = np.asarray(minute_index, dtype=np.int32)
    out = np.full(idx.shape, AUCTION_MKEY, dtype=np.int32)
    am = (idx >= 0) & (idx < 120)
    pm = (idx >= 120) & (idx < 240)
    out[am] = AM_MKEY_START + idx[am]
    out[pm] = PM_MKEY_START + (idx[pm] - 120)
    return out


def mkey_from_timestamp(ts) -> int:
    t = pd.Timestamp(ts)
    return mkey_from_hm(int(t.hour), int(t.minute))


def mkey_series(times: Sequence) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(list(times)))
    return (idx.hour * 60 + idx.minute).to_numpy(dtype=np.int32)


def is_lunch(mkey: int) -> bool:
    return LUNCH_MKEY_START <= int(mkey) <= LUNCH_MKEY_END


def is_continuous(mkey: int) -> bool:
    k = int(mkey)
    return (AM_MKEY_START <= k <= AM_MKEY_END) or (PM_MKEY_START <= k <= PM_MKEY_END)


def is_close_auction(mkey: int) -> bool:
    return int(mkey) == AUCTION_MKEY


def session_of_mkey(mkey: int) -> Optional[str]:
    k = int(mkey)
    if AM_MKEY_START <= k <= AM_MKEY_END:
        return "AM"
    if PM_MKEY_START <= k <= PM_MKEY_END:
        return "PM"
    if k == AUCTION_MKEY:
        return "AUCTION"
    if is_lunch(k):
        return "LUNCH"
    return None


def consecutive_mkeys(a: int, b: int) -> bool:
    """True iff b is the next continuous bar after a (lunch/auction never consecutive)."""
    if not is_continuous(a) or not is_continuous(b):
        return False
    if session_of_mkey(a) != session_of_mkey(b):
        return False
    return int(b) == int(a) + 1


def _in_segment_bounds(mkey: int, spec: Dict[str, object]) -> bool:
    name = str(spec["segment_name"])
    k = int(mkey)
    if name == "FULL":
        return is_continuous(k)
    if name == "CLOSE_AUCTION":
        return is_close_auction(k)
    lo = int(spec["mkey_start"])
    hi = int(spec["mkey_end"])
    return lo <= k <= hi


def time_mask(
    mkeys: Sequence,
    segment_name: str,
    *,
    include_close_auction: bool = False,
) -> np.ndarray:
    """Boolean mask for a named clock-time segment.

    Lunch is never included. CLOSE never includes 15:00 unless the caller
    explicitly uses CLOSE_AUCTION. FULL never includes auction.
    """
    spec = time_segment(segment_name)
    arr = np.asarray(mkeys, dtype=np.int32)
    name = str(spec["segment_name"])
    if name == "FULL":
        mask = ((arr >= AM_MKEY_START) & (arr <= AM_MKEY_END)) | (
            (arr >= PM_MKEY_START) & (arr <= PM_MKEY_END)
        )
    elif name == "CLOSE_AUCTION":
        mask = arr == AUCTION_MKEY
    else:
        lo = int(spec["mkey_start"])
        hi = int(spec["mkey_end"])
        mask = (arr >= lo) & (arr <= hi)
    if name == "CLOSE_AUCTION":
        return mask
    if include_close_auction and bool(spec.get("contains_close_auction")) is False:
        # Caller asked to fold auction into a continuous segment: refuse silently
        # by keeping auction out. Explicit CLOSE_AUCTION is the only opt-in.
        pass
    if not include_close_auction:
        mask = mask & (arr != AUCTION_MKEY)
    return mask


def segment_availability(segment_name: str, *, source_id: str = "") -> Dict[str, object]:
    spec = time_segment(segment_name)
    late_close_unreliable = (
        str(spec["segment_name"]) == "LATE_CLOSE" and source_id == "ddb_stock_one_minute"
    )
    return {
        "segment_name": spec["segment_name"],
        "start_time": spec["start_time"],
        "end_time": spec["end_time"],
        "inclusive_exclusive": (
            "[{}, {}]".format(spec["start_time"], spec["end_time"])
            if spec["end_inclusive"]
            else "[{}, {})".format(spec["start_time"], spec["end_time"])
        ),
        "contains_close_auction": bool(spec["contains_close_auction"]),
        "contains_1456_1500": bool(spec["contains_1456_1500"]),
        "uses_last_5min": bool(spec["uses_last_5min"]),
        "availability_timestamp": spec["availability_timestamp"],
        "source_compatibility": spec["source_compatibility"],
        "source_id": source_id,
        "late_close_reliable": (not late_close_unreliable),
        "execution_contract_compatible": True,  # V2V T+1, never Close[T]
        "production_execution_compatible": True,
        "close_t_execution": False,
        "n_expected_minutes": int(spec["n_expected_minutes"]),
    }


def assert_no_future_day(dates: Sequence, trade_date) -> None:
    """Stock-day cuts may not mix other calendar dates."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    t = pd.Timestamp(trade_date).normalize()
    if len(idx) == 0:
        return
    bad = idx[idx != t]
    if len(bad) > 0:
        raise ValueError(
            "stock-day cut leaked other dates: {}".format(
                sorted({str(x.date()) for x in bad})
            )
        )


def continuous_mkey_grid() -> np.ndarray:
    am = np.arange(AM_MKEY_START, AM_MKEY_END + 1, dtype=np.int32)
    pm = np.arange(PM_MKEY_START, PM_MKEY_END + 1, dtype=np.int32)
    out = np.concatenate([am, pm])
    if out.size != EXPECTED_CONTINUOUS_MINUTES:
        raise AssertionError("continuous grid is not 240 bars")
    return out


def list_segments(*, include_optional: bool = False) -> Tuple[str, ...]:
    names = []
    for row in TIME_SEGMENTS:
        if row.get("optional_v1") and not include_optional:
            continue
        names.append(str(row["segment_name"]))
    return tuple(names)
