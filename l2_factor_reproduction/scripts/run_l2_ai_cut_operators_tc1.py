#!/usr/bin/env python
"""TC-1 operator smoke: 2024-06 sequence load + 36 recipes. No IC/backtest."""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.tc1 import run_tc1  # noqa: E402
from l2_factor_reproduction.l2_ai_stock_selection.paths import ensure_layout  # noqa: E402


def main() -> None:
    ensure_layout()
    result = run_tc1()
    print("TC-1 done candidates={n} dir={d}".format(
        n=result["n_candidates"], d=result["out_dir"]
    ))


if __name__ == "__main__":
    main()
