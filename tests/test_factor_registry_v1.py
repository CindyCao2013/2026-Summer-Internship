"""Factor Registry v1 — Milestone 1E validation tests."""

from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
REG_YAML = REPO / "research" / "registry" / "factor_registry.yaml"
REG_CSV = REPO / "research" / "registry" / "factor_registry.csv"

BANNED_SUBSTRINGS = [
    "epsilon_d",
    "epsilon_u",
    "tgd_eps",
    "tau_ma",
    "upsilon",
    "gu_ma20",
    "gd_ma20",
    "m_high",
    "m_low",
    "buffer_5_15",
    "buffer_10_20",
    "daily_buffer",
    "execution_best",
]

ALLOWED_STATUS = {
    "discovery",
    "testing",
    "candidate",
    "validated",
    "production",
    "retired",
}


class TestFactorRegistryV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = yaml.safe_load(REG_YAML.read_text(encoding="utf-8"))
        cls.factors = cls.reg["factors"]
        with REG_CSV.open(encoding="utf-8") as f:
            cls.csv_rows = list(csv.DictReader(f))

    def test_yaml_and_csv_same_ids(self):
        y_ids = [f["factor_id"] for f in self.factors]
        c_ids = [r["factor_id"] for r in self.csv_rows]
        self.assertEqual(y_ids, c_ids)

    def test_unique_factor_ids(self):
        ids = [f["factor_id"] for f in self.factors]
        self.assertEqual(len(ids), len(set(ids)))

    def test_one_identity_per_row(self):
        for f in self.factors:
            self.assertTrue(f.get("factor_id"))
            self.assertIn("status", f)
            self.assertIn("formula_frozen", f)
            self.assertIn("family", f)

    def test_status_enum(self):
        for f in self.factors:
            self.assertIn(f["status"], ALLOWED_STATUS, msg=f["factor_id"])

    def test_diagnostics_cannot_enter_registry(self):
        for f in self.factors:
            fid = f["factor_id"].lower()
            for banned in BANNED_SUBSTRINGS:
                self.assertNotIn(banned, fid, msg=f"{f['factor_id']} looks like diagnostic/execution")

    def test_execution_labels_cannot_enter_registry(self):
        joined = " ".join(f["factor_id"].lower() for f in self.factors)
        self.assertNotRegex(joined, r"buffer_\d+")
        self.assertNotIn("daily_buffer", joined)

    def test_first_inventory_statuses(self):
        by = {f["factor_id"]: f["status"] for f in self.factors}
        self.assertEqual(by["TGD20"], "validated")
        self.assertEqual(by["FlowDensity20"], "candidate")
        self.assertEqual(by["D1_LiquidityQuality60d"], "candidate")
        self.assertEqual(by["D4_WinnerSentimentReversal5d"], "candidate")
        self.assertEqual(by["D5_UpsideFragility20d"], "candidate")
        self.assertEqual(by["IdealReversal"], "testing")
        # IdealAmplitude added in Phase III A2 (may be absent on older checkouts)
        if "IdealAmplitude" in by:
            self.assertEqual(by["IdealAmplitude"], "testing")
        if "ActiveTradeProxy" in by:
            self.assertEqual(by["ActiveTradeProxy"], "testing")

    def test_d1_signal_identity_documented(self):
        d1 = next(f for f in self.factors if f["factor_id"] == "D1_LiquidityQuality60d")
        self.assertEqual(d1.get("evaluation_signal"), "raw_cs_zscore")
        self.assertEqual(d1.get("production_signal"), "size_industry_neutralized")

    def test_no_composite_stack_as_factor_id(self):
        ids = [f["factor_id"] for f in self.factors]
        self.assertNotIn("Base3", ids)
        self.assertFalse(any(re.search(r"composite|stack", i, re.I) for i in ids))


if __name__ == "__main__":
    unittest.main()
