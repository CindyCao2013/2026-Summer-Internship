#!/usr/bin/env python
"""Sprint 9 — Flow × Price Response / Absorption Family v1.

Usage:
    python -m l2_factor_reproduction.scripts.run_flow_price_response_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.flow_price_response_v1 import (  # noqa: E402
    OUT_DIR,
    run_sprint9,
)


def main() -> int:
    summary, contrib, profile, capability = run_sprint9()
    print("\n===== Sprint 9 summary =====")
    cols = [
        "factor",
        "gate",
        "hl_sharpe",
        "decile_mono_spearman",
        "adjacent_violations",
        "g10_gross_excess_annual",
        "short_leg_share_abs",
        "avg_hl_oneway_turnover",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nartifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
