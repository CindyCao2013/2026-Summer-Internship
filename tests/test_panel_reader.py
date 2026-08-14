"""Tests for unified data access layer (mocked DDB — no network)."""

from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

import pandas as pd

from core.data.panel_reader import (
    DAILY_PANEL_COLUMNS,
    MINUTE_PANEL_COLUMNS,
    get_daily_panel,
    get_minute_panel,
)
from core.ddb.eod import build_eod_long_script


class TestPanelReader(unittest.TestCase):
    def test_eod_script_partition_filter(self):
        script = build_eod_long_script("2024-01-01", "2024-01-31", symbols=["600000.SH"])
        self.assertIn("TRADE_DT between 2024.01.01 : 2024.01.31", script)
        self.assertIn("S_INFO_WINDCODE in", script)
        self.assertIn("S_DQ_CLOSE as close", script)

    def test_get_daily_panel_mock(self):
        fake = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "symbol": ["600000.SH"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1e6],
                "amount": [1e7],
            }
        )
        with mock.patch("core.ddb.eod.fetch_eod_long", return_value=fake):
            out = get_daily_panel("2024-01-01", "2024-01-31")
        self.assertEqual(list(out.columns), list(DAILY_PANEL_COLUMNS))
        self.assertEqual(len(out), 1)

    def test_get_minute_panel_mock(self):
        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-01-02 09:31:00")],
                "symbol": ["600000.SH"],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.05],
                "volume": [1000.0],
                "amount": [10050.0],
            }
        )
        with mock.patch("core.ddb.minute.fetch_minute_long", return_value=fake):
            out = get_minute_panel("2024-01-01", "2024-01-31")
        self.assertEqual(list(out.columns), list(MINUTE_PANEL_COLUMNS))
        self.assertEqual(len(out), 1)

    def test_get_ddb_session_reuse(self):
        from core.ddb.connection import close_shared_ddb_session, get_ddb_session

        close_shared_ddb_session()
        with mock.patch("core.ddb.connection.ddb.session") as mock_sess_cls:
            mock_sess = mock.Mock()
            mock_sess_cls.return_value = mock_sess
            s1 = get_ddb_session(reuse=True)
            s2 = get_ddb_session(reuse=True)
            self.assertIs(s1, s2)
            mock_sess.connect.assert_called_once()
        close_shared_ddb_session()

    def test_get_ddb_session_delegation(self):
        from core.ddb.connection import get_ddb_session

        with mock.patch("core.ddb.connection.ddb.session") as mock_sess_cls:
            mock_sess = mock.Mock()
            mock_sess_cls.return_value = mock_sess
            get_ddb_session()
            mock_sess.connect.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
