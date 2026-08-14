"""Unit tests for Report Generator v2 — Milestone 1D.6 schema-driven loading."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from factor_report_generator_v2 import (
    REPO_ROOT,
    SPECS_DIR,
    _load_yaml,
    apply_artifact_copies,
    render_appendix_b,
)


GENERATOR = REPO_ROOT / "factor_report_generator_v2.py"


class TestNoFactorIdBranches(unittest.TestCase):
    def test_generator_has_no_factor_id_equality_branches(self):
        src = GENERATOR.read_text(encoding="utf-8")
        # Ban explicit identity branches that special-case packs in Python
        banned = [
            r'if\s+factor_id\s*==',
            r'if\s+factor_id\s+in\s*\(',
            r'factor_id\s*==\s*["\']TGD20["\']',
            r'factor_id\s*==\s*["\']FlowDensity20["\']',
            r'factor_id\s*==\s*["\']D1_LiquidityQuality60d["\']',
            r'factor_id\s*==\s*["\']IdealReversal["\']',
        ]
        for pat in banned:
            self.assertIsNone(re.search(pat, src), msg=f"banned pattern still present: {pat}")

    def test_generator_does_not_hardcode_tgd_essay_path(self):
        src = GENERATOR.read_text(encoding="utf-8")
        self.assertNotIn("日内分钟收益率时序特征_TGD20因子研究报告.md", src)
        self.assertNotIn("return_timing, timing_residual, tgd", src)


class TestArtifactCopiesSpecDriven(unittest.TestCase):
    def test_tgd_spec_declares_orthogonality_copy(self):
        spec = _load_yaml(SPECS_DIR / "TGD20.yaml")
        copies = (spec.get("artifacts") or {}).get("copy") or []
        dests = [c.get("dest") for c in copies]
        self.assertIn("diagnostics/orthogonality_TGD20_FlowDensity20_summary.md", dests)

    def test_apply_artifact_copies_uses_spec_only(self):
        spec = {
            "artifacts": {
                "copy": [
                    {
                        "src": "docs/schemas/chart_registry.yaml",
                        "dest": "diagnostics/chart_registry_copy.yaml",
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td)
            applied = apply_artifact_copies(pack, spec)
            self.assertTrue(applied)
            self.assertTrue((pack / "diagnostics" / "chart_registry_copy.yaml").exists())


class TestAppendixBFromContent(unittest.TestCase):
    def test_appendix_b_includes_code_map_not_foreign_paths(self):
        content = {
            "code_map": [
                {"item": "Implementation", "path": "path/to/d1.py"},
                {"item": "Essay", "path": "reports/d1/essay.md"},
            ]
        }
        md = render_appendix_b("D1_LiquidityQuality60d", content)
        self.assertIn("path/to/d1.py", md)
        self.assertIn("reports/d1/essay.md", md)
        self.assertNotIn("tgd_v1", md)
        self.assertIn("factor_specs/D1_LiquidityQuality60d_report_content.yaml", md)

    def test_tgd_content_code_map_present(self):
        content = _load_yaml(SPECS_DIR / "TGD20_report_content.yaml")
        self.assertTrue(content.get("code_map"))
        md = render_appendix_b("TGD20", content)
        self.assertIn("core/l2_features/", md)
        self.assertIn("tgd_v1", md)


class TestAllRepresentativeSpecsHaveArtifactsKey(unittest.TestCase):
    def test_four_specs(self):
        for fid in ("TGD20", "FlowDensity20", "D1_LiquidityQuality60d", "IdealReversal"):
            spec = _load_yaml(SPECS_DIR / f"{fid}.yaml")
            self.assertIn("artifacts", spec, msg=fid)
            content = _load_yaml(SPECS_DIR / f"{fid}_report_content.yaml")
            self.assertIn("code_map", content, msg=fid)


if __name__ == "__main__":
    unittest.main()
