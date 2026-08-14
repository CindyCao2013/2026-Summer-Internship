"""Minimal tests for Factor Research Harness — Milestone 1A."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from factor_research_harness import (
    load_factor_spec,
    list_registered_adapters,
    resolve_benchmark,
    run_factor_research,
)
from run_factor_research import build_parser, main


REPO = Path(__file__).resolve().parents[1]


class TestBenchmarkResolution(unittest.TestCase):
    def test_research_mode(self):
        b = resolve_benchmark("research")
        self.assertEqual(b["benchmark_version"], "research_v1")
        self.assertEqual(b["universe"], "ALL")
        self.assertEqual(b["horizon_days"], 20)
        self.assertEqual(b["neutralization"], "none")
        self.assertIsNone(b["cost_bp"])

    def test_production_mode(self):
        b = resolve_benchmark("production")
        self.assertEqual(b["benchmark_version"], "production_v1")
        self.assertEqual(b["universe"], "CSI1000")
        self.assertEqual(b["horizon_days"], 20)
        self.assertEqual(b["neutralization"], "industry_size")
        self.assertEqual(b["cost_bp"], 15)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            resolve_benchmark("weekly")


class TestFactorSpec(unittest.TestCase):
    def test_load_tgd20_spec(self):
        spec = load_factor_spec("TGD20")
        self.assertEqual(spec.get("factor_id"), "TGD20")
        self.assertTrue(spec.get("frozen_formula"))
        self.assertIn("temporal_information", spec.get("family") or [])

    def test_missing_spec(self):
        self.assertEqual(load_factor_spec("DOES_NOT_EXIST_XYZ"), {})


class TestHarnessTGD20(unittest.TestCase):
    def test_adapter_registered(self):
        self.assertIn("TGD20", list_registered_adapters())

    def test_dry_run(self):
        r = run_factor_research("TGD20", "production", dry_run=True)
        self.assertTrue(r.ok)
        names = [s.name for s in r.stages]
        self.assertIn("load_factor_spec", names)
        self.assertIn("resolve_benchmark", names)
        self.assertIn("dry_run", names)

    def test_tgd20_production_pipeline(self):
        r = run_factor_research("TGD20", "production")
        self.assertTrue(r.ok, msg=json.dumps(r.to_dict(), indent=2))
        by_name = {s.name: s for s in r.stages}
        self.assertEqual(by_name["compute"].status, "skipped")
        self.assertEqual(by_name["evaluate"].status, "ok")
        self.assertEqual(by_name["generate_report_pack"].status, "ok")

    def test_unknown_factor_skeleton(self):
        r = run_factor_research("UNKNOWN_FACTOR_X", "research")
        self.assertTrue(r.ok)
        statuses = {s.name: s.status for s in r.stages}
        self.assertEqual(statuses["compute"], "not_implemented")
        self.assertEqual(statuses["evaluate"], "not_implemented")


class TestCLI(unittest.TestCase):
    def test_parser_defaults(self):
        p = build_parser()
        args = p.parse_args(["--factor", "TGD20"])
        self.assertEqual(args.mode, "production")
        self.assertFalse(args.dry_run)

    def test_main_list_adapters(self):
        self.assertEqual(main(["--list-adapters"]), 0)

    def test_main_tgd20(self):
        self.assertEqual(main(["--factor", "TGD20", "--mode", "research"]), 0)


if __name__ == "__main__":
    unittest.main()
