"""L2 Microstructure Engine v1 — OIR, VOI, MPB only (daily aggregate)."""

from typing import Callable, Dict, List

import pandas as pd

from l2_data_loaders import L2DailyWideCache
from l2_microstructure import build_l2_factor_panels, rolling_brick, daily_mpb, daily_oir, daily_voi

L2Func = Callable[[L2DailyWideCache], pd.DataFrame]
L2_REGISTRY: Dict[str, L2Func] = {}


def register_l2(name: str):
    def decorator(func: L2Func) -> L2Func:
        if name in L2_REGISTRY:
            raise ValueError(f"Duplicated l2 factor: {name}")
        L2_REGISTRY[name] = func
        return func

    return decorator


@register_l2("cn_voi_20d")
def f_cn_voi_20d(cache: L2DailyWideCache) -> pd.DataFrame:
    return rolling_brick(daily_voi(cache))


@register_l2("cn_oir_20d")
def f_cn_oir_20d(cache: L2DailyWideCache) -> pd.DataFrame:
    return rolling_brick(daily_oir(cache))


@register_l2("cn_mpb_20d")
def f_cn_mpb_20d(cache: L2DailyWideCache) -> pd.DataFrame:
    return rolling_brick(daily_mpb(cache))


def build_l2_factor(factor_name: str, cache: L2DailyWideCache) -> pd.DataFrame:
    if factor_name not in L2_REGISTRY:
        valid = sorted(L2_REGISTRY.keys())
        raise ValueError(f"Unknown l2 factor: {factor_name}. Valid: {valid}")
    return L2_REGISTRY[factor_name](cache)


def filter_l2_factors(factor_names: List[str]) -> List[str]:
    out = []
    for name in factor_names:
        if name not in L2_REGISTRY:
            print(f"[SKIP] Unknown l2 factor: {name}")
            continue
        out.append(name)
    return out


def build_all_l2_factors(cache: L2DailyWideCache) -> Dict[str, pd.DataFrame]:
    return build_l2_factor_panels(cache)
