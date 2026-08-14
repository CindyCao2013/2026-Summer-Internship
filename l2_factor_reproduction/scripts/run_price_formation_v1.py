#!/usr/bin/env python
"""Sprint 11 — Price Formation inventory-first Fast Discovery.

Usage:
    python -m l2_factor_reproduction.scripts.run_price_formation_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.price_formation_v1 import (  # noqa: E402
    OUT_ROOT,
    run_sprint11,
)


def main() -> int:
    # Close Sprint 10 status already written; run Sprint 11
    manifest = run_sprint11()
    print("\n===== Sprint 11 =====")
    print(f"selection={manifest['selection']} next={manifest.get('next_candidate')}")
    print(f"artifacts -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
