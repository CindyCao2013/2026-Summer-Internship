"""Unit tests for MinuteBarStore (mocked DolphinDB — no network required)."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from unittest import mock

import pandas as pd

from minute_bar_store import (
    CANONICAL_COLUMNS,
    COLUMN_MAP,
    MinuteBarStore,
    apply_trading_hours,
    resolve_fields,
    to_wind_code,
)


def _fake_raw_month(year: int, month: int, days: int = 3, n_sym: int = 2) -> pd.DataFrame:
    """Synthesize DDB-shaped minute bars for a few days in a month."""
    rows = []
    symbols = ["600000", "000001"][:n_sym]
    for day in range(1, days + 1):
        date = dt.datetime(year, month, day)
        for sym in symbols:
            for hour, minute in [(9, 31), (10, 0), (14, 30), (15, 0)]:
                bt = dt.datetime(1970, 1, 1, hour, minute)
                rows.append(
                    {
                        "Symbol": sym,
                        "Date": date,
                        "Bartime": bt,
                        "Open": 10.0,
                        "High": 10.1,
                        "Low": 9.9,
                        "Close": 10.05,
                        "Volume": 1000.0,
                        "Amount": 10000.0,
                        "Active_buy_volume": 400.0,
                        "Active_sell_volume": 300.0,
                        "Active_buy_amount": 4000.0,
                        "Active_sell_amount": 3000.0,
                        "Active_buy_count": 10.0,
                        "Active_sell_count": 8.0,
                        "Bid_cancel_volume": 50.0,
                        "Bid_cancel_count": 2.0,
                        "Ask_cancel_volume": 40.0,
                        "Ask_cancel_count": 1.0,
                        "Adjfactor": 1.0,
                    }
                )
    return pd.DataFrame(rows)


class TestColumnNormalize(unittest.TestCase):
    def test_column_map_covers_l2_fields(self):
        required = {
            "Active_buy_amount",
            "Active_sell_amount",
            "Active_buy_volume",
            "Bid_cancel_volume",
            "Ask_cancel_volume",
        }
        self.assertTrue(required.issubset(COLUMN_MAP.keys()))
        self.assertIn("active_buy_amt", CANONICAL_COLUMNS)
        self.assertIn("active_buy_count", CANONICAL_COLUMNS)

    def test_normalize_raw_wind_and_columns(self):
        store = MinuteBarStore(start_date="2020-01-01")
        raw = _fake_raw_month(2024, 5, days=1)
        df = store.normalize_raw(raw)
        self.assertIn("active_buy_amt", df.columns)
        self.assertIn("active_buy_vol", df.columns)
        self.assertTrue((df["symbol"].str.endswith((".SH", ".SZ"))).all())
        self.assertTrue(set(df["symbol"].str[0]).issubset({"0", "3", "6"}))
        self.assertTrue((pd.to_datetime(df["bartime"]).dt.year == 2024).all())

    def test_field_aliases_cnt(self):
        self.assertEqual(resolve_fields(["active_buy_cnt"]), ["active_buy_count"])

    def test_to_wind_code(self):
        self.assertEqual(to_wind_code("600000"), "600000.SH")
        self.assertEqual(to_wind_code("000001"), "000001.SZ")
        self.assertEqual(to_wind_code("600000.SH"), "600000.SH")


class TestTradingHours(unittest.TestCase):
    def test_apply_trading_hours(self):
        store = MinuteBarStore()
        df = store.normalize_raw(_fake_raw_month(2024, 5, days=1))
        filtered = apply_trading_hours(df)
        times = pd.to_datetime(filtered["bartime"]).dt.time
        self.assertTrue(
            (
                (times >= dt.time(9, 30))
                & (times <= dt.time(11, 30))
                | (times >= dt.time(13, 0))
                & (times <= dt.time(15, 0))
            ).all()
        )


class TestDdbQueryAndMemCache(unittest.TestCase):
    def setUp(self):
        self.store = MinuteBarStore(start_date="2020-01-01", memory_cache_size=5)
        self.query_count = 0

        def fake_run(script):
            self.query_count += 1
            return _fake_raw_month(2024, 5, days=5)

        self._patcher = mock.patch.object(
            self.store, "_run_with_retry", side_effect=fake_run
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_first_get_queries_ddb_second_uses_mem_cache(self):
        df1 = self.store.get_data("2024-05-01", "2024-05-31")
        self.assertFalse(df1.empty)
        self.assertEqual(self.query_count, 1)

        df2 = self.store.get_data("2024-05-01", "2024-05-31")
        self.assertEqual(self.query_count, 1)
        self.assertEqual(len(df1), len(df2))

    def test_force_reload_bypasses_cache(self):
        self.store.get_data("2024-05-01", "2024-05-10")
        self.store.get_data("2024-05-01", "2024-05-10", force_reload=True)
        self.assertEqual(self.query_count, 2)

    def test_different_ranges_query_separately(self):
        self.store.get_data("2024-05-01", "2024-05-15")
        self.store.get_data("2024-04-01", "2024-04-30")
        self.assertEqual(self.query_count, 2)

    def test_history_start_clamp(self):
        store = MinuteBarStore(start_date="2024-05-01", memory_cache_size=5)
        pulled_scripts = []

        def fake_run(script):
            pulled_scripts.append(script)
            return _fake_raw_month(2024, 5, days=3)

        with mock.patch.object(store, "_run_with_retry", side_effect=fake_run):
            store.get_data("2020-01-01", "2024-05-02")
        self.assertTrue(len(pulled_scripts) >= 1)
        self.assertIn("2024.05.01", pulled_scripts[0])

    def test_build_ddb_script_has_partition_filter(self):
        script = self.store._build_ddb_script(
            pd.Timestamp("2024-05-01"),
            pd.Timestamp("2024-05-31"),
            symbols=["600000.SH"],
            fields=["volume", "close"],
        )
        self.assertIn("Date between 2024.05.01 : 2024.05.31", script)
        self.assertIn("Symbol in", script)
        self.assertIn("Volume", script)
        self.assertIn("Close", script)
        self.assertTrue(script.strip().endswith("result"))

    def test_build_ddb_script_trading_hours(self):
        script = self.store._build_ddb_script(
            pd.Timestamp("2024-05-01"),
            pd.Timestamp("2024-05-31"),
            trading_hours_only=True,
        )
        self.assertIn("second(Bartime) >= 09:30:00", script)


    def test_cache_root_deprecated_accepted(self):
        store = MinuteBarStore(cache_root=tempfile.mkdtemp())
        self.assertIsNotNone(store)


if __name__ == "__main__":
    unittest.main(verbosity=2)
