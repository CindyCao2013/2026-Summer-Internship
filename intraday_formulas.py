"""分钟级因子公式与清单（高频低频化 roadmap）。

Phase 1: 框架占位 + 示例因子名。
Phase 2: 已实现 close_vwap_deviation / late_session_strength /
         volume_front_loading / volume_back_loading / morning_reversal_pressure；其余仍为 roadmap 占位。
"""

from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

# Phase 1: 可先用 EOD 代理或示例窄表验证 intraday 回测链路
INTRADAY_PHASE1_LIST = [
    "intraday_example",
]

# Phase 2 — implemented (true minute alphas + bartime-stamped panel factors)
INTRADAY_PHASE2_IMPLEMENTED = [
    "close_vwap_deviation",
    "active_buy_sell_imbalance",
    "late_session_strength",
    "volume_front_loading",
    "volume_back_loading",
    "morning_reversal_pressure",
    "TGD20_1429",
    "SmartMoney_1129_Rev",
    "bartime_ofi",
    "ofi_persistence",
    "active_buy_shock",
    "average_active_trade_size",
    "large_active_buy_ratio",
    "intraday_amihud",
    "realized_volatility",
    "minute_skew",
]

# Phase 2 roadmap (not yet implemented)
INTRADAY_PHASE2_ROADMAP = [
    "intraday_return_smoothness",
    "intraday_volume_return_corr",
    "liquidity_shock_recovery",
]

INTRADAY_PHASE2_LIST = list(INTRADAY_PHASE2_IMPLEMENTED)

INTRADAY_FACTOR_LIST = (
    INTRADAY_PHASE1_LIST + INTRADAY_PHASE2_IMPLEMENTED + INTRADAY_PHASE2_ROADMAP
)

PHASE2_ROADMAP_SET = set(INTRADAY_PHASE2_ROADMAP)

IntradayFunc = Callable[..., pd.DataFrame]
INTRADAY_REGISTRY: Dict[str, IntradayFunc] = {}

# Lazy import computers to avoid circular imports at module load in light contexts
def _load_computers() -> Dict[str, IntradayFunc]:
    from core.intraday_alphas import INTRADAY_ALPHA_COMPUTERS

    return dict(INTRADAY_ALPHA_COMPUTERS)


INTRADAY_FACTOR_COMPUTERS: Dict[str, IntradayFunc] = {}


def _ensure_computers() -> Dict[str, IntradayFunc]:
    global INTRADAY_FACTOR_COMPUTERS
    if not INTRADAY_FACTOR_COMPUTERS:
        INTRADAY_FACTOR_COMPUTERS = _load_computers()
    return INTRADAY_FACTOR_COMPUTERS


def register_intraday(name: str):
    def decorator(func: IntradayFunc) -> IntradayFunc:
        if name in INTRADAY_REGISTRY:
            raise ValueError(f"Duplicated intraday factor: {name}")
        INTRADAY_REGISTRY[name] = func
        return func

    return decorator


def filter_available_intraday_factors(factor_names: Iterable[str]) -> List[str]:
    computers = _ensure_computers()
    output = []
    for name in factor_names:
        if name in PHASE2_ROADMAP_SET:
            print(f"[SKIP] {name}: Phase 2 roadmap not implemented")
            continue
        if name in computers or name in INTRADAY_REGISTRY or name == "intraday_example":
            output.append(name)
            continue
        if name in INTRADAY_PHASE2_IMPLEMENTED:
            output.append(name)
            continue
        print(f"[SKIP] {name}: not registered")
    return output


def available_intraday_factors() -> List[str]:
    computers = _ensure_computers()
    names = set(INTRADAY_REGISTRY) | set(computers) | {"intraday_example"}
    return sorted(names)


def build_intraday_narrow_table(
    factor_name: str,
    start_date,
    end_date,
    *,
    store=None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Dispatch to Phase2 computers; returns DDB-ready narrow table."""
    from core.intraday_alphas import narrow_for_ddb

    computers = _ensure_computers()
    if factor_name not in computers:
        raise KeyError(f"No computer for intraday factor: {factor_name}")
    raw = computers[factor_name](
        start_date, end_date, store=store, symbols=symbols, return_full_day=False
    )
    return narrow_for_ddb(raw)
