#!/usr/bin/env python3
"""Write and verify the final normalized_v1 report-package manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2_factor_reproduction.reporting.package_manifest import (  # noqa: E402
    build_final_package_manifest,
)


DEFAULT_REPORT_ROOT = (
    ROOT / "research/reports/factors/mid_order_ratio/normalized_v1"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    args = parser.parse_args()
    manifest = build_final_package_manifest(args.report_root)
    print(
        f"Final package verified: {manifest['file_count']} files, "
        f"{manifest['figure_count']} figures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

