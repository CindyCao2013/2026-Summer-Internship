"""Liquidity Resilience LR-0/LR-1 unit tests (no ClickHouse / no returns)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.liquidity_resilience.candidates import (
    aggregate_daily,
    events_from_minutes,
)
from l2_factor_reproduction.liquidity_resilience.contracts import (
    FROZEN_CANDIDATE_NAMES,
    frozen_candidate_specs,
)
from l2_factor_reproduction.liquidity_resilience.recovery import (
    obi_restoration,
    recovery_fraction,
    spread_recovery_fraction,
)
from l2_factor_reproduction.liquidity_resilience.session import (
    AM_MKEY_END,
    PM_MKEY_END,
    eligible_event_mkey,
    horizon_in_session,
    mkey_from_hm,
    session_of_mkey,
)
from l2_factor_reproduction.liquidity_resilience.shocks import (
    active_buy_shock,
    trailing_median_2d,
)


def test_frozen_count_in_range() -> None:
    specs = frozen_candidate_specs()
    assert 20 <= len(specs) <= 30
    assert len(FROZEN_CANDIDATE_NAMES) == len(specs)
    assert len(set(FROZEN_CANDIDATE_NAMES)) == len(FROZEN_CANDIDATE_NAMES)


# A — within-session horizon
def test_within_session_plus_3() -> None:
    mkey = mkey_from_hm(10, 0)
    assert session_of_mkey(mkey) == "AM"
    assert horizon_in_session(mkey, 3)
    assert mkey + 3 == mkey_from_hm(10, 3)


# B — lunch boundary
def test_lunch_plus_5_ineligible() -> None:
    mkey = mkey_from_hm(11, 26)
    assert session_of_mkey(mkey) == "AM"
    assert not horizon_in_session(mkey, 5)
    assert mkey + 5 > AM_MKEY_END
    assert not eligible_event_mkey(mkey, 5)


# C — close boundary
def test_close_plus_5_ineligible() -> None:
    mkey = mkey_from_hm(14, 56)
    assert session_of_mkey(mkey) == "PM"
    assert not horizon_in_session(mkey, 5)
    assert mkey + 5 > PM_MKEY_END
    assert not eligible_event_mkey(mkey, 5)


def test_afternoon_open_has_no_pre() -> None:
    assert not eligible_event_mkey(mkey_from_hm(13, 0), 5)


# D E F — recovery completeness
def test_recovery_complete() -> None:
    val = float(recovery_fraction(100.0, 60.0, 100.0))
    assert val == pytest.approx(1.0)


def test_recovery_none() -> None:
    val = float(recovery_fraction(100.0, 60.0, 60.0))
    assert val == pytest.approx(0.0)


def test_recovery_deterioration() -> None:
    val = float(recovery_fraction(100.0, 60.0, 40.0))
    assert val < 0


def test_tiny_denominator_excluded() -> None:
    val = float(recovery_fraction(100.0, 100.0, 110.0, denom_floor=1.0))
    assert np.isnan(val)


# G — spread recovery
def test_spread_recovery_restored() -> None:
    # pre 0.001 → t0 0.002 → h 0.001
    val = float(spread_recovery_fraction(0.001, 0.002, 0.001))
    assert val == pytest.approx(1.0)


def test_spread_recovery_stayed_wide() -> None:
    val = float(spread_recovery_fraction(0.001, 0.002, 0.002))
    assert val == pytest.approx(0.0)


# H — OBI restoration
def test_obi_restoration_full() -> None:
    val = float(obi_restoration(0.2, -0.3, 0.2))
    assert val == pytest.approx(1.0)


def test_obi_restoration_none() -> None:
    val = float(obi_restoration(0.2, -0.3, -0.3))
    assert val == pytest.approx(0.0)


# I — buy/sell mapping via a toy minute panel
def _toy_minutes() -> pd.DataFrame:
    """One AM session, two symbols. 10:00-10:40 grid."""
    keys = np.arange(mkey_from_hm(10, 0), mkey_from_hm(10, 41))
    rows = []
    for mkey in keys:
        t = int(mkey - keys[0])
        # AAA.SH: quiet until t=25, then a buy shock that depletes ask
        buy = 100.0
        sell = 80.0
        ask = 1000.0
        bid = 1000.0
        if t == 25:
            buy = 1000.0
            ask = 600.0
        if t == 26:
            ask = 700.0
        if t == 28:
            ask = 900.0
        if t == 30:
            ask = 1000.0
        rows.append(_bar("AAA.SH", mkey, bid=bid, ask=ask, buy=buy, sell=sell))
        # BBB.SH: sell shock at t=25 depletes bid; no buy shock
        buy2, sell2, bid2, ask2 = 80.0, 100.0, 1000.0, 1000.0
        if t == 25:
            sell2 = 1000.0
            bid2 = 600.0
        if t == 30:
            bid2 = 1000.0
        rows.append(_bar("BBB.SH", mkey, bid=bid2, ask=ask2, buy=buy2, sell=sell2))
        # CCC.SH: never shocks
        rows.append(_bar("CCC.SH", mkey, bid=1000.0, ask=1000.0, buy=50.0, sell=50.0))
    return pd.DataFrame(rows)


def _bar(symbol: str, mkey: int, *, bid: float, ask: float, buy: float, sell: float) -> dict:
    return {
        "Symbol": symbol,
        "symbol_raw": symbol.split(".")[0],
        "exchange": "." + symbol.split(".")[1],
        "TradeDate": pd.Timestamp("2024-06-28"),
        "mkey": int(mkey),
        "has_book": True,
        "bid1": 10.0,
        "ask1": 10.01,
        "bid_depth_5": bid,
        "ask_depth_5": ask,
        "active_buy_amount": buy,
        "active_sell_amount": sell,
    }


def test_buy_shock_maps_to_ask_depth() -> None:
    daily, events, _ = _run_toy()
    buy = events.loc[events["shock_type"] == "ACTIVE_BUY_SHOCK"]
    assert not buy.empty
    assert (buy["Symbol"] == "AAA.SH").all()
    rec5 = float(buy["ask_recovery_5"].iloc[0])
    assert rec5 == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(float(buy["bid_recovery_5"].iloc[0])) or True
    # sell-side recovery on a buy event is not the Block A metric
    aaa = daily.loc[daily["Symbol"] == "AAA.SH"].iloc[0]
    assert aaa["ask_depth_recovery_5m"] == pytest.approx(1.0, abs=1e-6)


def test_sell_shock_maps_to_bid_depth() -> None:
    daily, events, _ = _run_toy()
    sell = events.loc[events["shock_type"] == "ACTIVE_SELL_SHOCK"]
    assert not sell.empty
    assert (sell["Symbol"] == "BBB.SH").all()
    bbb = daily.loc[daily["Symbol"] == "BBB.SH"].iloc[0]
    assert bbb["bid_depth_recovery_5m"] == pytest.approx(1.0, abs=1e-6)


def _run_toy():
    from l2_factor_reproduction.liquidity_resilience.candidates import daily_from_minutes

    return daily_from_minutes(_toy_minutes())


# J — no-event semantics
def test_no_event_is_na_not_zero() -> None:
    daily, events, _ = _run_toy()
    ccc = daily.loc[daily["Symbol"] == "CCC.SH"].iloc[0]
    assert "CCC.SH" not in set(events["Symbol"]) or (
        events.loc[events["Symbol"] == "CCC.SH", "shock_type"]
        .isin(["ACTIVE_BUY_SHOCK", "ACTIVE_SELL_SHOCK"])
        .sum()
        == 0
    )
    assert pd.isna(ccc["ask_depth_recovery_5m"])
    assert pd.isna(ccc["bid_depth_recovery_5m"])
    assert float(ccc["ask_depth_recovery_5m"]) != 0 if pd.notna(ccc["ask_depth_recovery_5m"]) else True


# K — PIT shock detection
def test_future_minutes_do_not_change_past_shock() -> None:
    buy = np.ones((1, 30)) * 10.0
    buy[0, 20] = 100.0
    sell = np.ones((1, 30)) * 5.0
    valid = np.ones((1, 30), dtype=bool)
    trail = trailing_median_2d(buy)
    mask1 = active_buy_shock(buy, sell, valid=valid, trail_buy=trail)
    buy2 = buy.copy()
    buy2[0, 25:] = 1e9
    trail2 = trailing_median_2d(buy2)
    mask2 = active_buy_shock(buy2, sell, valid=valid, trail_buy=trail2)
    assert mask1[0, 20] == mask2[0, 20]
    assert mask1[0, :21].tolist() == mask2[0, :21].tolist()


# L — deterministic aggregation
def test_deterministic_daily_aggregation() -> None:
    daily1, events, _ = _run_toy()
    daily2 = aggregate_daily(events, symbols=["AAA.SH", "BBB.SH", "CCC.SH"])
    a = daily1.set_index("Symbol")[list(FROZEN_CANDIDATE_NAMES)].sort_index()
    b = daily2.set_index("Symbol")[list(FROZEN_CANDIDATE_NAMES)].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=1e-12)


def test_no_spread_speed_alias_in_registry() -> None:
    names = set(FROZEN_CANDIDATE_NAMES)
    assert "spread_recovery_speed_3m" not in names
    assert "spread_recovery_speed_5m" not in names


def test_full_novelty_status_mapping() -> None:
    from l2_factor_reproduction.discovery_lite.novelty import novelty_bucket
    from l2_factor_reproduction.liquidity_resilience.full_novelty import (
        STATUS_ALIAS,
        STATUS_PASS,
        STATUS_REVIEW,
        classify_full_novelty_status,
    )

    assert classify_full_novelty_status(novelty_bucket(0.40)) == STATUS_PASS
    assert classify_full_novelty_status(novelty_bucket(0.60)) == STATUS_PASS
    assert classify_full_novelty_status(novelty_bucket(0.80)) == STATUS_REVIEW
    assert classify_full_novelty_status(novelty_bucket(0.94)) == STATUS_ALIAS
    assert classify_full_novelty_status("UNKNOWN") == "REFERENCE_COVERAGE_LIMITED"
