"""Tests for intraday alpha factors (mock MinuteBarStore, no DolphinDB)."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from core.intraday_alphas import (
    compute_close_vwap_deviation,
    compute_late_session_strength,
    compute_morning_reversal_pressure,
    compute_volume_back_loading,
    compute_volume_front_loading,
    narrow_for_ddb,
)
from minute_bar_store import MinuteBarStore


def _synthetic_raw(n_days: int = 25) -> pd.DataFrame:
    """Build DDB-shaped bars then normalize via MinuteBarStore."""
    store = MinuteBarStore(cache_root=tempfile.mkdtemp(), start_date="2024-01-01")
    symbols = ["600000", "000001"]
    rows = []
    start = dt.datetime(2024, 5, 1)
    for d_i in range(n_days):
        day = start + dt.timedelta(days=d_i)
        if day.weekday() >= 5:
            continue
        for sym in symbols:
            # denser morning + afternoon bars
            minutes = [
                (9, 31),
                (9, 59),
                (10, 0),
                (10, 29),
                (11, 29),
                (13, 1),
                (13, 29),
                (14, 29),
                (14, 30),
                (14, 45),
                (14, 59),
            ]
            for j, (h, m) in enumerate(minutes):
                bt = dt.datetime(day.year, day.month, day.day, h, m)
                close = 10.0 + 0.02 * j
                volume = 1000.0 + 10 * j
                amount = close * volume
                rows.append(
                    {
                        "Symbol": sym,
                        "Date": day,
                        "Bartime": bt,
                        "Open": close - 0.01,
                        "High": close + 0.01,
                        "Low": close - 0.02,
                        "Close": close,
                        "Volume": volume,
                        "Amount": amount,
                        "Adjfactor": 1.0,
                        "Active_buy_volume": volume * 0.6,
                        "Active_sell_volume": volume * 0.4,
                        "Active_buy_amount": amount * 0.6,
                        "Active_sell_amount": amount * 0.4,
                        "Active_buy_count": 10,
                        "Active_sell_count": 8,
                        "Bid_cancel_volume": 0.0,
                        "Bid_cancel_count": 0.0,
                        "Ask_cancel_volume": 0.0,
                        "Ask_cancel_count": 0.0,
                    }
                )
    raw = pd.DataFrame(rows)
    return store.normalize_raw(raw)


class _FakeStore:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_data(self, start, end, symbols=None, fields=None, force_reload=False):
        out = self._df.copy()
        out = out[
            (pd.to_datetime(out["date"]) >= pd.Timestamp(start))
            & (pd.to_datetime(out["date"]) <= pd.Timestamp(end))
        ]
        if symbols is not None:
            out = out[out["symbol"].isin(symbols)]
        return out.reset_index(drop=True)


class TestCloseVwapDeviation(unittest.TestCase):
    def setUp(self):
        self.ddb_flag = mock.patch(
            "factor_config.INTRADAY_CLOSE_VWAP_USE_DDB", False
        )
        self.ddb_flag.start()
        self.addCleanup(self.ddb_flag.stop)
        self.store = _FakeStore(_synthetic_raw(5))

    def test_output_shape(self):
        df = compute_close_vwap_deviation("2024-05-01", "2024-05-10", store=self.store)
        self.assertGreater(len(df), 0)
        self.assertListEqual(
            list(df.columns), ["bartime", "symbol", "factorname", "value"]
        )
        self.assertTrue((df["factorname"] == "close_vwap_deviation").all())

    def test_manual_vwap_one_bar(self):
        # Hand-check: two bars same day
        rows = []
        day = dt.datetime(2024, 5, 6)
        for h, m, close, vol in [(9, 31, 10.0, 100.0), (9, 59, 12.0, 100.0)]:
            rows.append(
                {
                    "symbol": "600000.SH",
                    "date": day,
                    "bartime": dt.datetime(2024, 5, 6, h, m),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": vol,
                    "amount": close * vol,
                    "adjfactor": 1.0,
                    "active_buy_amt": 0.0,
                    "active_sell_amt": 0.0,
                }
            )
        fake = _FakeStore(pd.DataFrame(rows))
        out = compute_close_vwap_deviation(
            "2024-05-06", "2024-05-06", store=fake, return_full_day=True
        )
        # At second bar VWAP=(10*100+12*100)/200=11, deviation=(12-11)/11
        row = out[pd.to_datetime(out["bartime"]).dt.time == dt.time(9, 59)].iloc[0]
        self.assertAlmostEqual(float(row["value"]), (12.0 - 11.0) / 11.0, places=6)

    def test_filter_bartimes(self):
        df = compute_close_vwap_deviation("2024-05-01", "2024-05-10", store=self.store)
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        allowed = {
            dt.time(9, 59),
            dt.time(10, 29),
            dt.time(11, 29),
            dt.time(13, 29),
            dt.time(14, 29),
        }
        self.assertTrue(times.issubset(allowed))

    def test_narrow_for_ddb(self):
        df = compute_close_vwap_deviation("2024-05-01", "2024-05-10", store=self.store)
        narrow = narrow_for_ddb(df)
        self.assertIn("tradetime", narrow.columns)
        self.assertNotIn("bartime", narrow.columns)


class TestLateSessionStrength(unittest.TestCase):
    def setUp(self):
        self.ddb_flag = mock.patch(
            "factor_config.INTRADAY_LATE_SESSION_STRENGTH_USE_DDB", False
        )
        self.ddb_flag.start()
        self.addCleanup(self.ddb_flag.stop)
        self.store = _FakeStore(_synthetic_raw(12))

    def test_next_day_0959_stamp(self):
        df = compute_late_session_strength("2024-05-06", "2024-05-15", store=self.store)
        self.assertGreater(len(df), 0)
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        self.assertEqual(times, {dt.time(9, 59)})

    def test_value_range(self):
        df = compute_late_session_strength("2024-05-06", "2024-05-15", store=self.store)
        self.assertTrue(df["value"].between(0, 1).all())
        self.assertTrue((df["value"] - 0.6).abs().max() < 1e-9)


class TestVolumeFrontLoading(unittest.TestCase):
    def setUp(self):
        self.ddb_flag = mock.patch(
            "factor_config.INTRADAY_VOLUME_FRONT_USE_DDB", False
        )
        self.ddb_flag.start()
        self.addCleanup(self.ddb_flag.stop)
        self.store = _FakeStore(_synthetic_raw(30))

    def test_output_exists(self):
        df = compute_volume_front_loading(
            "2024-05-20", "2024-05-28", store=self.store, lookback_days=10
        )
        self.assertGreater(len(df), 0)
        self.assertTrue((df["value"] > 0).all())
        self.assertTrue((df["factorname"] == "volume_front_loading").all())

    def test_stamp_at_1029(self):
        df = compute_volume_front_loading(
            "2024-05-20", "2024-05-28", store=self.store, lookback_days=10
        )
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        self.assertEqual(times, {dt.time(10, 29)})


class TestVolumeBackLoading(unittest.TestCase):
    def setUp(self):
        self.ddb_flag = mock.patch(
            "factor_config.INTRADAY_VOLUME_BACK_USE_DDB", False
        )
        self.ddb_flag.start()
        self.addCleanup(self.ddb_flag.stop)
        self.store = _FakeStore(_synthetic_raw(30))

    def test_output_exists(self):
        df = compute_volume_back_loading(
            "2024-05-20", "2024-05-28", store=self.store, lookback_days=10
        )
        self.assertGreater(len(df), 0)
        self.assertTrue((df["value"] > 0).all())
        self.assertTrue((df["factorname"] == "volume_back_loading").all())

    def test_stamp_at_next_day_0959(self):
        df = compute_volume_back_loading(
            "2024-05-20", "2024-05-28", store=self.store, lookback_days=10
        )
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        self.assertEqual(times, {dt.time(9, 59)})


class TestMorningReversalPressure(unittest.TestCase):
    def setUp(self):
        self.store = _FakeStore(_synthetic_raw(5))

    def test_stamp_and_sign(self):
        df = compute_morning_reversal_pressure(
            "2024-05-01", "2024-05-10", store=self.store
        )
        self.assertGreater(len(df), 0)
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        self.assertEqual(times, {dt.time(10, 29)})
        self.assertTrue((df["factorname"] == "morning_reversal_pressure").all())
        # synthetic: open≈9.99, close_10 rises with j → value should be negative-ish
        self.assertTrue(np.isfinite(df["value"]).all())


class TestIntradayBackendRegistry(unittest.TestCase):
    def test_panel_compatibility_view_is_derived_from_registry(self):
        from core.intraday_alphas import (
            INTRADAY_FACTOR_BACKEND,
            PANEL_BASED_INTRADAY_FACTORS,
        )

        expected = frozenset(
            name
            for name, backend in INTRADAY_FACTOR_BACKEND.items()
            if backend == "panel"
        )
        self.assertEqual(PANEL_BASED_INTRADAY_FACTORS, expected)
        self.assertEqual(
            INTRADAY_FACTOR_BACKEND["active_buy_sell_imbalance"], "ddb"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
