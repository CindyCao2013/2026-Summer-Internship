"""Tests for bartime-stamped panel factors + market demean helper."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from core.intraday_alphas import (
    compute_smartmoney_1129_rev,
    compute_tgd20_1429,
    narrow_for_ddb,
)
from intraday_lib import subtract_market_return


class TestSubtractMarketReturn(unittest.TestCase):
    def test_demeans_quintiles_keeps_hml(self):
        rows = []
        for g, r in [("group_0", -0.02), ("group_1", -0.01), ("group_2", 0.0),
                     ("group_3", 0.01), ("group_4", 0.02), ("group_HML", 0.04)]:
            rows.append(
                {
                    "group": g,
                    "Date": pd.Timestamp("2024-05-06"),
                    "Bartime": dt.time(14, 29),
                    "Ret_30": r,
                }
            )
        df = pd.DataFrame(rows)
        out = subtract_market_return(df)
        # quintile mean was 0 → demeaned == original; HML untouched
        self.assertAlmostEqual(float(out.loc[out["group"] == "group_0", "Ret_30"]), -0.02)
        self.assertAlmostEqual(float(out.loc[out["group"] == "group_HML", "Ret_30"]), 0.04)

    def test_nonzero_market_removed(self):
        rows = []
        for g, r in [("group_0", 0.08), ("group_1", 0.10), ("group_2", 0.12),
                     ("group_3", 0.14), ("group_4", 0.16), ("group_HML", 0.08)]:
            rows.append(
                {
                    "group": g,
                    "Date": pd.Timestamp("2024-05-06"),
                    "Bartime": dt.time(9, 59),
                    "Ret_15": r,
                }
            )
        df = pd.DataFrame(rows)
        out = subtract_market_return(df)
        # market = 0.12
        self.assertAlmostEqual(float(out.loc[out["group"] == "group_0", "Ret_15"]), -0.04)
        self.assertAlmostEqual(float(out.loc[out["group"] == "group_4", "Ret_15"]), 0.04)
        self.assertAlmostEqual(float(out.loc[out["group"] == "group_HML", "Ret_15"]), 0.08)


class TestBartimePanelFactors(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-05-01", periods=5)
        cols = ["600000.SH", "000001.SZ"]
        self.panel = pd.DataFrame(
            [[1.0, -2.0], [2.0, -1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 2.0]],
            index=dates,
            columns=cols,
        )

    def test_tgd20_1429_shape_and_time(self):
        with mock.patch(
            "core.intraday_alphas._load_panel",
            return_value=self.panel.shift(1),
        ):
            # bypass real cache: call _narrow via compute with patched load
            with mock.patch(
                "core.intraday_alphas._load_panel",
                return_value=self.panel.copy(),
            ):
                df = compute_tgd20_1429("2024-05-01", "2024-05-10")
        self.assertGreater(len(df), 0)
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        self.assertEqual(times, {dt.time(14, 29)})
        self.assertTrue((df["factorname"] == "TGD20_1429").all())
        narrow = narrow_for_ddb(df)
        self.assertIn("tradetime", narrow.columns)

    def test_smartmoney_rev_negates(self):
        with mock.patch(
            "core.intraday_alphas._load_panel",
            return_value=self.panel.copy(),
        ):
            df = compute_smartmoney_1129_rev("2024-05-01", "2024-05-10")
        times = set(pd.to_datetime(df["bartime"]).dt.time)
        self.assertEqual(times, {dt.time(11, 29)})
        # first non-nan day after... panel not shifted here so values negated raw
        # 600000 first value 1.0 → -1.0
        row = df[(df["symbol"] == "600000.SH")].iloc[0]
        self.assertAlmostEqual(float(row["value"]), -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
