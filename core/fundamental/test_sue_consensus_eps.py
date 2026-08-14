"""Unit tests for SUE_ConsensusEPS PIT event builder (no Oracle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.fundamental.sue_consensus_eps_panel import (
    build_sue_consensus_eps_events,
    events_to_impulse_panel,
)


def _bundle_express_first():
    notice = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "report_period": ["20231231"],
            "known_date": [pd.Timestamp("2024-01-10")],
            "np_mid": [1e8],
            "source": ["notice"],
        }
    )
    express = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "report_period": ["20231231"],
            "known_date": [pd.Timestamp("2024-01-15")],
            "np_mid": [1.1e8],
            "eps": [1.2],
            "source": ["express"],
        }
    )
    income = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "report_period": ["20231231"],
            "known_date": [pd.Timestamp("2024-03-01")],
            "np_mid": [1.1e8],
            "eps": [1.25],
            "source": ["income"],
        }
    )
    consensus = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "est_dt": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-14")],
            "report_period": ["20231231", "20231231"],
            "eps_avg": [1.0, 1.0],
            "np_avg": [1e8, 1e8],
        }
    )
    return {
        "notice": notice,
        "express": express,
        "income": income,
        "consensus": consensus,
    }


def test_known_dt_is_first_eps_disclosure_not_notice_only():
    ev, audit = build_sue_consensus_eps_events(_bundle_express_first())
    assert audit["pit_hard_pass"] is True
    assert len(ev) == 1
    assert ev.iloc[0]["known_dt"] == pd.Timestamp("2024-01-15")
    assert ev.iloc[0]["notice_dt"] == pd.Timestamp("2024-01-10")
    assert ev.iloc[0]["actual_eps"] == pytest.approx(1.2)
    assert ev.iloc[0]["consensus_eps"] == pytest.approx(1.0)
    assert ev.iloc[0]["sue"] == pytest.approx(0.2)
    assert ev.iloc[0]["est_dt"] < ev.iloc[0]["known_dt"]


def test_rejects_consensus_on_or_after_known():
    b = _bundle_express_first()
    b["consensus"] = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "est_dt": [pd.Timestamp("2024-01-15")],
            "report_period": ["20231231"],
            "eps_avg": [1.0],
            "np_avg": [1e8],
        }
    )
    ev, _audit = build_sue_consensus_eps_events(b)
    assert len(ev) == 0


def test_impulse_maps_to_trading_day_ge_known():
    b = _bundle_express_first()
    ev, _ = build_sue_consensus_eps_events(b)
    ev = ev.copy()
    ev["known_dt"] = pd.Timestamp("2024-01-13")  # Saturday
    idx = pd.DatetimeIndex(["2024-01-12", "2024-01-15", "2024-01-16"])
    wide = events_to_impulse_panel(ev, idx)
    assert wide.loc[pd.Timestamp("2024-01-15"), "600000.SH"] == pytest.approx(0.2)
    assert np.isnan(wide.loc[pd.Timestamp("2024-01-12"), "600000.SH"])
