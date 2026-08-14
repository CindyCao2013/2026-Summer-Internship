"""Trace actual factor vs return indexes. Do not infer from the word 'shift'."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.contracts import (
    EXECUTION_CONVENTION,
    TIMING_VERDICT,
)


# Source-traced session windows. Paths are repo-relative.
PRIMITIVE_CUTOFFS: List[Dict[str, object]] = [
    {
        "family": "trade_flow",
        "module": "l2_factor_reproduction/python/ch_tick.py",
        "session_window": "09:30:00 <= ExchTime < 15:00:01",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "technically_available": "after 15:00:00 tick is ingested; session close T",
        "note": "Docstring and every CH predicate use < 15:00:01, so 15:00:00 is in-sample.",
    },
    {
        "family": "order_size",
        "module": "l2_factor_reproduction/python/ch_tick.py",
        "session_window": "09:30:00 <= ExchTime < 15:00:01",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "technically_available": "after 15:00:00 tick is ingested; session close T",
        "note": "Same tick filter as trade_flow.",
    },
    {
        "family": "order_book",
        "module": "l2_factor_reproduction/python/ch_order_book.py",
        "session_window": "continuous means: is_close_auction=0; 15:00 stored separately",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "technically_available": "daily means exclude 15:00 but include 14:56-14:59; close_auction_* and closing_obi_l5 need late session / 15:00",
        "note": (
            "toHour(ExchTime)=15 marks close auction. avgIf(..., is_close_auction=0) "
            "still includes minute_index 210-239 (14:30-14:59). close_auction_obi_5 "
            "and close_auction_relative_spread use the 15:00 snapshot. closing_obi_l5 "
            "maps to closing_30m_obi_5 (14:30-14:59)."
        ),
    },
    {
        "family": "price_formation",
        "module": "l2_factor_reproduction/python/price_formation_daily.py",
        "session_window": "continuous [09:30,11:30)+[13:00,15:00); auction at 15:00",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "technically_available": "after 14:59 continuous bar; close_auction_return after 15:00 bar",
        "note": "close_auction_return = log(close_auction_price / continuous_close). No pre-close cutoff.",
    },
    {
        "family": "liquidity_impact",
        "module": "l2_factor_reproduction/python/liquidity_impact_daily.py",
        "session_window": "09:30-11:29 + 13:00-14:59 (240 continuous minutes)",
        "includes_1456_1500": True,
        "includes_close_auction": False,
        "includes_1500_tick": False,
        "technically_available": "after 14:59 minute bar; still after any feasible Close[T] decision if Close[T] is the auction print",
        "note": "Excludes 15:00 auction. Still includes 14:56-14:59. No 14:30/14:55 cutoff.",
    },
    {
        "family": "cancel_lifecycle",
        "module": "l2_factor_reproduction/python/ch_cancel_lifecycle.py",
        "session_window": "minutes-of-day 570-689 and 780-899 = [09:30,11:30)+[13:00,15:00)",
        "includes_1456_1500": True,
        "includes_close_auction": False,
        "includes_1500_tick": False,
        "technically_available": "after 14:59; auction excluded because minute < 900",
        "note": "14:59 is included (899). 15:00 (900) is excluded.",
    },
]


def apply_prepare_factor_signal_shift(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    signal_shift: int = 1,
) -> tuple:
    """Mirror backtest.prepare_factor_signal index logic (no mask).

    1. intersect indexes
    2. signal = factor.shift(signal_shift)
    3. ret reindexed to shifted signal index
    groupTest then multiplies the already-aligned frames; it does not shift again.
    """
    common_idx = factor.index.intersection(ret.index)
    common_cols = factor.columns.intersection(ret.columns)
    signal = factor.loc[common_idx, common_cols].copy()
    ret_a = ret.loc[common_idx, common_cols].copy()
    if signal_shift:
        signal = signal.shift(signal_shift)
    signal = signal.dropna(how="all", axis=1).dropna(how="all")
    ret_a = ret_a.reindex(index=signal.index, columns=signal.columns)
    return signal, ret_a


def map_factor_date_to_c2c_return(
    dates: Sequence,
    factor_date,
    *,
    signal_shift: int = 1,
) -> Dict[str, object]:
    """Given calendar dates, map factor[T] to the return row it actually hits.

    ret[D] is defined as Close[D]/Close[D-1]-1 (get_Ret_Matrix method=c2c).
    After shift(1), signal[D] = factor[D-1], so factor[T] hits ret[T+1]
    = Close[T+1]/Close[T]-1, which *starts* at Close[T].
    """
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    t = pd.Timestamp(factor_date).normalize()
    if t not in dates:
        raise KeyError("factor_date {} not in calendar".format(t.date()))
    pos = int(dates.get_loc(t))
    ret_pos = pos + int(signal_shift)
    if ret_pos >= len(dates):
        return {
            "factor_date": str(t.date()),
            "signal_index_date": None,
            "return_index_date": None,
            "holding_start_close": None,
            "holding_end_close": None,
            "return_formula": None,
            "executable_at_holding_start": False,
        }
    ret_date = pd.Timestamp(dates[ret_pos])
    return {
        "factor_date": str(t.date()),
        "signal_index_date": str(ret_date.date()),
        "return_index_date": str(ret_date.date()),
        "holding_start_close": str(t.date()),
        "holding_end_close": str(ret_date.date()),
        "return_formula": "Close[{}]/Close[{}]-1".format(ret_date.date(), t.date()),
        "executable_at_holding_start": False,
    }


def three_date_walkthrough(dates: Sequence) -> List[Dict[str, object]]:
    """Trace factor dates T0,T1,T2. Prefer a 4-day calendar so T2 has T+1."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    if len(dates) < 3:
        raise ValueError("need 3 consecutive trading dates")
    factor_dates = dates[:3]
    calendar = dates
    rows = []
    for t in factor_dates:
        rec = map_factor_date_to_c2c_return(calendar, t, signal_shift=1)
        rec["calendar"] = [str(pd.Timestamp(d).date()) for d in calendar[: max(3, min(4, len(calendar)))]]
        if rec["return_formula"] is None:
            rec["interpretation"] = (
                "factor[{}] has no T+1 on this calendar".format(rec["factor_date"])
            )
        else:
            rec["interpretation"] = (
                "factor[{}] known after close {} -> paired with {} "
                "(starts at Close[{}], which has already passed)"
            ).format(
                rec["factor_date"],
                rec["factor_date"],
                rec["return_formula"],
                rec["holding_start_close"],
            )
        rows.append(rec)
    return rows


def any_family_uses_post_close_info() -> bool:
    return any(
        bool(r["includes_close_auction"]) or bool(r["includes_1500_tick"])
        for r in PRIMITIVE_CUTOFFS
    )


def uniform_pre_close_cutoff() -> None:
    return None


def execution_timing_contract_dict(
    *,
    walkthrough_dates: Sequence = (),
) -> Dict[str, object]:
    walk = three_date_walkthrough(walkthrough_dates) if len(list(walkthrough_dates)) >= 3 else []
    return {
        "verdict": TIMING_VERDICT,
        "verdict_options": [
            "C2C_TPLUS1_EXECUTABLE",
            "C2C_TPLUS1_REQUIRES_PRE_CLOSE_CUTOFF",
            "C2C_TPLUS1_NOT_EXECUTABLE",
            "TIMING_UNRESOLVED",
        ],
        "frozen_stack": {
            "prepare_factor_signal": (
                "signal = factor.shift(1); ret reindexed to signal; "
                "no second shift in groupTest"
            ),
            "get_Ret_Matrix_c2c": "(S_DQ_CLOSE/S_DQ_PRECLOSE-1) on date D = Close[D]/Close[D-1]-1",
            "signal_shift": int(EXECUTION_CONVENTION["signal_shift"]),
            "factor_T_pairs_with": "Close[T+1]/Close[T]-1",
            "not": "Close[T+1] to Close[T+2]",
            "groupTest_additional_shift": False,
        },
        "information_availability": {
            "uniform_pre_close_cutoff": uniform_pre_close_cutoff(),
            "backtest_docstring": (
                "python/backtest.py: factor is same-session aggregate, "
                "known only after the session close of T"
            ),
            "any_family_includes_1456_1500": True,
            "any_family_includes_close_auction": True,
            "any_family_includes_1500_tick": True,
            "technically_available_timestamp": "session close T / after 15:00 auction print for several families",
        },
        "primitive_cutoffs": PRIMITIVE_CUTOFFS,
        "economic_inconsistency": (
            "A T+1 close-to-close return starts at Close[T]. If the factor is "
            "only known after Close[T], the strategy cannot establish the "
            "position at Close[T]."
        ),
        "required_ai_v1_correction": [
            "T+1 open or VWAP into a subsequent holding window",
            "or shift the c2c holding window one extra session: factor T -> Close[T+1] to Close[T+2]",
        ],
        "frozen_historical_backtests": "UNCHANGED",
        "production_y3_y10": "NOT_MATERIALIZED_AS_PRODUCTION",
        "three_date_walkthrough": walk,
        "answers": {
            "q1_intraday_cutoff": "No uniform cutoff. See primitive_cutoffs.",
            "q2_includes_1456_auction_postclose": True,
            "q3_technically_available": "After session close T; auction families after 15:00 print.",
            "q4_groupTest_execution_timestamp": (
                "groupTest uses already-shifted signal[D] * ret[D]; "
                "implied fill is Close[D-1] for the c2c return that starts there. "
                "No additional lag."
            ),
            "q5_shift1_means": "factor T -> return Close[T] to Close[T+1], NOT Close[T+1] to Close[T+2]",
            "q6_three_dates": walk,
        },
    }
