"""Unit tests for shared factor-report metric labels and arithmetic."""

from __future__ import annotations

import math
import sys
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import (  # noqa: E402
    format_group_test_stats_title,
    implied_annu_fee,
)


def test_implied_fee_uses_basis_points_and_labels_them_correctly() -> None:
    turnover = 1.6884
    expected = turnover * 7.5 / 10_000 * 250

    implied_fee = implied_annu_fee(turnover, fee_bps=7.5)
    assert math.isclose(implied_fee, expected)

    title = format_group_test_stats_title(
        direction=1,
        annu_ret=0.4,
        sharpe=2.7,
        mdd=-0.1,
        avg_turnover=turnover,
        rank_ic=0.05,
        icir=6.0,
        implied_fee=implied_fee,
        fee_bps=7.5,
    )
    assert "Implied AnnuFee(7.5 bps)" in title
    assert "Implied AnnuFee(7.5%)" not in title
