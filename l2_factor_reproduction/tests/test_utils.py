"""轻量单测：不依赖 DDB 的工具函数。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.utils.date_utils import to_ddb_date, to_yyyymmdd
from l2_factor_reproduction.python.utils.symbol_utils import to_pure_code, to_wind_symbol


def test_symbol_utils():
    assert to_wind_symbol("600000") == "600000.SH"
    assert to_wind_symbol("000001.SZ") == "000001.SZ"
    assert to_pure_code("600000.SH") == "600000"


def test_date_utils():
    assert to_ddb_date("20240102") == "2024.01.02"
    assert to_yyyymmdd("2024.01.02") == "20240102"


if __name__ == "__main__":
    test_symbol_utils()
    test_date_utils()
    print("ok")
