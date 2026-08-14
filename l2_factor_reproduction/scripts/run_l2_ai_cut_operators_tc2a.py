#!/usr/bin/env python
"""TC-2A: targeted nonlinear rescue + timing localization on 12 parents.

Does not train LightGBM/XGBoost. Does not expand to all 25 TC-2 parents.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.tc2a import run_tc2a  # noqa: E402
from l2_factor_reproduction.l2_ai_stock_selection.paths import ensure_layout  # noqa: E402


def main() -> None:
    ensure_layout()
    result = run_tc2a()
    print("TC-2A done verdict={v} dir={d}".format(v=result["verdict"], d=result["out_dir"]))


if __name__ == "__main__":
    main()
