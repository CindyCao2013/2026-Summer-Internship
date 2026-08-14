#!/usr/bin/env python
"""PHASE TC-0: write cut-operator audit artifacts. No CH/DDB scan. No TC-1 generation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.diagnostics import (  # noqa: E402
    write_tc0_artifacts,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (  # noqa: E402
    CUT_OPERATORS,
    ensure_layout,
)


def main() -> None:
    ensure_layout()
    written = write_tc0_artifacts(CUT_OPERATORS)
    for key, path in written.items():
        print("{}: {}".format(key, path))


if __name__ == "__main__":
    main()
