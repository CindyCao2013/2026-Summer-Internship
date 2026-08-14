"""Smoke tests for the parameterized standalone report exporter."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.reporting.standalone_report_export import (  # noqa: E402
    build_html,
    build_pdf,
    discover_markdown_files,
)


def _make_report(tmp_path: Path) -> Tuple[Path, Path]:
    report_root = tmp_path / "canonical_report"
    figures = report_root / "figures"
    appendix = report_root / "appendix"
    figures.mkdir(parents=True)
    appendix.mkdir()

    image_path = figures / "diagnostic.png"
    Image.new("RGB", (80, 48), color=(47, 111, 159)).save(image_path)

    (report_root / "README.md").write_text(
        "# Source navigation\n\nThis file is not a numbered chapter.\n",
        encoding="utf-8",
    )
    (report_root / "10_decision.md").write_text(
        "# 10 Research Decision\n\nThe frozen decision is documented here.\n",
        encoding="utf-8",
    )
    (report_root / "02_method.md").write_text(
        "# 02 Method\n\n"
        "| Field | Definition |\n"
        "| --- | --- |\n"
        "| signal | normalized trade amount |\n",
        encoding="utf-8",
    )
    (report_root / "01_summary.md").write_text(
        "# 01 摘要\n\n"
        "The normalized signal is denoted by \\(x_{i,t}\\).\n\n"
        "![diagnostic figure](figures/diagnostic.png)\n\n"
        "See [the method](02_method.md).\n",
        encoding="utf-8",
    )
    (appendix / "evidence.md").write_text(
        "# Appendix Evidence\n\n"
        "\\[\n"
        "z_{i,t} = x_{i,t} / s_{i,t}\n"
        "\\]\n\n"
        "![appendix-relative figure](../figures/diagnostic.png)\n",
        encoding="utf-8",
    )
    return report_root, image_path


def test_discovers_numbered_chapters_then_appendix(tmp_path: Path) -> None:
    report_root, _ = _make_report(tmp_path)

    relative = [
        path.relative_to(report_root).as_posix()
        for path in discover_markdown_files(report_root)
    ]

    assert relative == [
        "01_summary.md",
        "02_method.md",
        "10_decision.md",
        "appendix/evidence.md",
    ]


def test_html_smoke_embeds_local_images_and_mathjax(tmp_path: Path) -> None:
    report_root, image_path = _make_report(tmp_path)
    output_path = tmp_path / "export" / "report.html"

    result = build_html(report_root, output_path, title="Normalized Signal Audit")

    document = result.read_text(encoding="utf-8")
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    assert result == output_path.resolve()
    assert result.stat().st_size > 1_000
    assert "data:image/png;base64,{}".format(payload) in document
    assert document.count("data:image/png;base64,") == 2
    assert 'src="figures/diagnostic.png"' not in document
    assert "window.MathJax" in document
    assert "tex-svg.js" in document
    assert document.index('data-source="01_summary.md"') < document.index(
        'data-source="10_decision.md"'
    )
    assert 'href="#section-02_method"' in document


def test_pdf_smoke_contains_image_and_is_non_empty(tmp_path: Path) -> None:
    report_root, _ = _make_report(tmp_path)
    output_path = tmp_path / "export" / "report.pdf"

    result = build_pdf(report_root, output_path, title="归一化因子研究")

    pdf_bytes = result.read_bytes()
    assert result == output_path.resolve()
    assert pdf_bytes.startswith(b"%PDF")
    assert b"/Subtype /Image" in pdf_bytes
    assert result.stat().st_size > 5_000


def test_explicit_markdown_paths_are_parameterized(tmp_path: Path) -> None:
    report_root, _ = _make_report(tmp_path)
    output_path = tmp_path / "selected.html"

    build_html(
        report_root,
        output_path,
        title="Selected Chapter",
        markdown_files=["02_method.md"],
    )

    document = output_path.read_text(encoding="utf-8")
    assert 'data-source="02_method.md"' in document
    assert 'data-source="01_summary.md"' not in document
