"""Unit tests for intradlay heatmap diagnostics (no DolphinDB)."""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_heatmap_lib import (
    compute_ic_hml_matrices,
    plot_heatmap_matrix,
    run_factor_heatmap_offline,
    stamp_panel_to_narrow,
)


def _make_panel(n_days: int = 30, n_sym: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"{i:06d}.SH" if i % 2 == 0 else f"{i:06d}.SZ" for i in range(n_sym)]
    # use 6xxxxx / 0xxxxx style
    cols = [f"{600000 + i}.SH" if i < n_sym // 2 else f"{1 + i}.SZ" for i in range(n_sym)]
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_days, n_sym))
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_ret_long(panel: pd.DataFrame, bartimes, horizons) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    # Make Ret_60 positively correlated with lagged factor for IC check
    for d in panel.index:
        for lab in bartimes:
            h, m = map(int, lab.split(":"))
            for sym in panel.columns:
                fac = panel.loc[d, sym]
                rec = {
                    "symbol": str(sym),
                    "Date": pd.Timestamp(d),
                    "Bartime": dt.time(h, m),
                    "BartimeLabel": f"{h:02d}:{m:02d}",
                }
                for hz in horizons:
                    noise = float(rng.normal(0, 0.01))
                    if hz == "Ret_60":
                        rec[hz] = 0.05 * float(fac) + noise
                    else:
                        rec[hz] = noise
                rows.append(rec)
    return pd.DataFrame(rows)


class TestStampAndIC(unittest.TestCase):
    def setUp(self):
        self.bartimes = ["09:59", "10:29", "14:29"]
        self.horizons = ["Ret_15", "Ret_60", "Ret_EOD"]
        self.panel = _make_panel()
        self.ret = _make_ret_long(self.panel, self.bartimes, self.horizons)

    def test_stamp_shape(self):
        narrow = stamp_panel_to_narrow(
            self.panel, "DummyFactor", bartimes=self.bartimes, shift_days=0
        )
        self.assertListEqual(
            list(narrow.columns), ["tradetime", "symbol", "factorname", "value"]
        )
        n_expected = self.panel.notna().sum().sum() * len(self.bartimes)
        self.assertEqual(len(narrow), n_expected)

    def test_ic_positive_on_ret60(self):
        narrow = stamp_panel_to_narrow(
            self.panel, "DummyFactor", bartimes=self.bartimes, shift_days=0
        )
        ic, hml = compute_ic_hml_matrices(
            narrow, self.ret, horizons=self.horizons, bartimes=self.bartimes
        )
        self.assertIn("Ret_60", ic.columns)
        # Correlated construction → mean IC on Ret_60 should be clearly > 0
        self.assertGreater(float(ic["Ret_60"].mean()), 0.2)

    def test_manual_ic_matches(self):
        # Single day / bartime hand check
        day = self.panel.index[5]
        bt = "10:29"
        narrow = stamp_panel_to_narrow(
            self.panel.loc[[day]], "F", bartimes=[bt], shift_days=0
        )
        ret = self.ret[
            (self.ret["Date"] == pd.Timestamp(day)) & (self.ret["BartimeLabel"] == bt)
        ]
        merged = narrow.merge(
            ret, left_on=["symbol"], right_on=["symbol"], how="inner"
        )
        manual = merged["value"].corr(merged["Ret_60"], method="spearman")
        ic, _ = compute_ic_hml_matrices(
            narrow, ret, horizons=["Ret_60"], bartimes=[bt]
        )
        self.assertAlmostEqual(float(ic.loc[bt, "Ret_60"]), float(manual), places=6)


class TestOutputs(unittest.TestCase):
    def test_run_offline_writes_files(self):
        tmp = Path(tempfile.mkdtemp())
        panel = _make_panel(n_days=20, n_sym=12)
        bartimes = ["09:59", "14:29"]
        horizons = ["Ret_15", "Ret_60"]
        ret = _make_ret_long(panel, bartimes, horizons)
        paths = run_factor_heatmap_offline(
            "DummyFactor",
            panel,
            ret,
            out_dir=tmp,
            universe="ALL",
            bartimes=bartimes,
            horizons=horizons,
            shift_days=0,
        )
        self.assertTrue(paths["ic_csv"].exists())
        self.assertTrue(paths["hml_csv"].exists())
        self.assertTrue(paths["ic_png"].exists())
        self.assertTrue(paths["ic_png"].with_suffix(".pdf").exists())
        self.assertTrue(paths["summary"].exists())
        ic = pd.read_csv(paths["ic_csv"], index_col=0)
        self.assertTrue(set(horizons).issubset(set(ic.columns)))
        self.assertTrue(set(bartimes).issubset(set(ic.index.astype(str))))


class TestPlot(unittest.TestCase):
    def test_plot_heatmap_matrix(self):
        tmp = Path(tempfile.mkdtemp()) / "hm.png"
        mat = pd.DataFrame(
            [[0.01, -0.02], [0.03, 0.0]],
            index=["09:59", "14:29"],
            columns=["Ret_15", "Ret_60"],
        )
        out = plot_heatmap_matrix(mat, title="t", out_path=tmp)
        self.assertTrue(out.exists())
        self.assertTrue(out.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
