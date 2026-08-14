"""L2 Microstructure Engine v2 — 6 event-driven factors only.

See research/results/l2_v1_closed.md for v1 conclusion (CLOSED).
"""

from typing import Dict, List

import pandas as pd

from l2_data_loaders import L2DailyWideCache
from l2_microstructure_v2 import build_l2_v2_factor_panels

L2_MICROSTRUCTURE_V2_LIST = [
    "cn_voi_shock",
    "cn_mpb_shock",
    "cn_flow_persistence",
    "cn_imbalance_duration",
    "cn_liquidity_consumption",
    "cn_cancel_shock",
]


def build_l2_v2_factor(factor_name: str, cache: L2DailyWideCache) -> pd.DataFrame:
    panels = build_l2_v2_factor_panels(cache)
    if factor_name not in panels:
        valid = sorted(panels.keys())
        raise ValueError(f"Unknown l2_v2 factor: {factor_name}. Valid: {valid}")
    return panels[factor_name]


def filter_l2_v2_factors(factor_names: List[str]) -> List[str]:
    valid = set(L2_MICROSTRUCTURE_V2_LIST)
    out = []
    for name in factor_names:
        if name not in valid:
            print(f"[SKIP] Unknown l2_v2 factor: {name}")
            continue
        out.append(name)
    return out


def build_all_l2_v2_factors(cache: L2DailyWideCache) -> Dict[str, pd.DataFrame]:
    return build_l2_v2_factor_panels(cache)
