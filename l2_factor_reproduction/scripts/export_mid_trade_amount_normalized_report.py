#!/usr/bin/env python3
"""Export the normalized mid-trade-amount research report to HTML and PDF.

Usage:
    python l2_factor_reproduction/scripts/export_mid_trade_amount_normalized_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from l2_factor_reproduction.reporting.standalone_report_export import (  # noqa: E402
    DEFAULT_MATHJAX_URL,
    export_standalone_report,
)


DEFAULT_REPORT_ROOT = (
    WORKSPACE
    / "research"
    / "reports"
    / "factors"
    / "mid_order_ratio"
    / "normalized_v1"
)
DEFAULT_TITLE = "Mid-Trade-Amount Normalization — Standalone Factor Research Report"
DEFAULT_OUTPUT_STEM = "mid_trade_amount_normalized_report"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="Canonical report directory containing 01..10 and appendix Markdown.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory; defaults to <report-root>/export.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Title used by both HTML and PDF.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem, without an extension.",
    )
    parser.add_argument(
        "--mathjax-url",
        default=DEFAULT_MATHJAX_URL,
        help="MathJax JavaScript URL used by the single-file HTML.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional Chinese-capable TTF/TTC/OTF font for the PDF.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report_root = args.report_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else report_root / "export"
    )
    exported = export_standalone_report(
        report_root,
        output_dir,
        title=args.title,
        output_stem=args.output_stem,
        mathjax_url=args.mathjax_url,
        font_path=args.font_path,
    )
    print("HTML: {}".format(exported.html_path))
    print("PDF:  {}".format(exported.pdf_path))
    print("file://{}".format(exported.html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
