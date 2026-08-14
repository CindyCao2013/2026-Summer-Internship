"""因子脚本组装器：加载 init + 因子 .dos，调用 compute_<name>，返回窄表。"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from l2_factor_reproduction.config.settings import (
    BIG_ORDER_DRIVE_TOP_FRAC,
    BIG_ORDER_TOP_FRAC,
    DDB_SCRIPT_ROOT,
    END_DAY,
    START_DAY,
    UNIVERSE,
)
from l2_factor_reproduction.python.ch_tick import fetch_order_size_narrow
from l2_factor_reproduction.python.ddb_client import DDBFactorClient
from l2_factor_reproduction.python.utils.date_utils import to_ddb_date, to_datetime

logger = logging.getLogger(__name__)

# 因子名 -> 相对 ddb_scripts 的脚本路径（可多个，按顺序 run）
FACTOR_SCRIPT_MAP: Dict[str, tuple] = {
    "avg_outflow_ratio": ("factors/minute_based/avg_order_amount.dos",),
    "avg_outflow_ratio_v2": ("factors/minute_based/avg_order_amount_v2.dos",),
    "avg_outflow_ratio_v3": ("factors/minute_based/avg_order_amount_v2.dos",),
    "big_order_net_inflow": ("factors/minute_based/big_order_flow.dos",),
    "big_order_drive_ret": ("factors/minute_based/big_order_drive.dos",),
    "net_order_change": ("factors/order_book/net_order_change.dos",),
    "order_change_volatility": (
        "factors/order_book/net_order_change.dos",
        "factors/order_book/order_volatility.dos",
    ),
    "order_change_skew": (
        "factors/order_book/net_order_change.dos",
        "factors/order_book/order_skew.dos",
    ),
    "mid_order_ratio": ("factors/trade_flow/order_size_ratio.dos",),
    "small_order_ratio": ("factors/trade_flow/order_size_ratio.dos",),
}

# Phase2：CH 服务端聚合（不再逐日拉全量 Tick）
TICK_AGG_FACTORS = {"mid_order_ratio", "small_order_ratio"}

# 分钟类全样本易 OOM：按年分块
YEARLY_CHUNK_FACTORS = {
    "big_order_net_inflow",
    "big_order_drive_ret",
    "avg_outflow_ratio",
    "avg_outflow_ratio_v2",
    "avg_outflow_ratio_v3",
}


def _ensure_init(client: DDBFactorClient) -> None:
    client.run_file(f"{DDB_SCRIPT_ROOT}/init/init_env.dos")


def _load_factor_scripts(client: DDBFactorClient, factor_name: str) -> None:
    for rel in FACTOR_SCRIPT_MAP[factor_name]:
        client.run_file(f"{DDB_SCRIPT_ROOT}/{rel}")


def _call_kwargs(factor_name: str, start: dt.datetime, end: dt.datetime, univ: str) -> str:
    kwargs = f"startDate={to_ddb_date(start)}, endDate={to_ddb_date(end)}, universe='{univ}'"
    if factor_name == "big_order_net_inflow":
        kwargs += f", top_frac={BIG_ORDER_TOP_FRAC}"
    elif factor_name == "big_order_drive_ret":
        kwargs += f", top_frac={BIG_ORDER_DRIVE_TOP_FRAC}"
    return kwargs


def _year_chunks(start: dt.datetime, end: dt.datetime) -> List[Tuple[dt.datetime, dt.datetime]]:
    chunks: List[Tuple[dt.datetime, dt.datetime]] = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, dt.datetime(year, 1, 1))
        chunk_end = min(end, dt.datetime(year, 12, 31))
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
    return chunks


def _normalize_result(result) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame(columns=["symbol", "tradetime", "factorname", "value"])
    if isinstance(result, pd.DataFrame):
        return result
    return pd.DataFrame(result)


def _stock_pool_vector(
    client: DDBFactorClient, start: dt.datetime, end: dt.datetime, universe: str
) -> List[str]:
    stocks = client.run_script(
        f"get_stock_pool({to_ddb_date(start)}, {to_ddb_date(end)}, '{universe}', true)"
    )
    if stocks is None:
        return []
    if isinstance(stocks, pd.Series):
        return [str(x) for x in stocks.tolist()]
    if isinstance(stocks, pd.DataFrame):
        return [str(x) for x in stocks.iloc[:, 0].tolist()]
    return [str(x) for x in list(stocks)]


def _build_tick_order_size_factor(
    client: DDBFactorClient,
    factor_name: str,
    start: dt.datetime,
    end: dt.datetime,
    universe: str,
) -> pd.DataFrame:
    """Phase2：ClickHouse 内 GROUP BY，Python 只读 (symbol,date) 聚合结果。"""
    _ensure_init(client)
    stocks = _stock_pool_vector(client, start, end, universe)
    logger.info(
        "%s CH server-side agg %s~%s universe=%s stocks=%d",
        factor_name,
        start.date(),
        end.date(),
        universe,
        len(stocks),
    )

    ranges = [(start, end)] if start.year == end.year else _year_chunks(start, end)
    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in ranges:
        logger.info("%s agg chunk %s~%s", factor_name, chunk_start.date(), chunk_end.date())
        narrow = fetch_order_size_narrow(
            chunk_start,
            chunk_end,
            factor_name,
            symbols=stocks if stocks else None,
        )
        if narrow is not None and not narrow.empty:
            parts.append(narrow)
            logger.info("  -> rows=%d", len(narrow))
        else:
            logger.warning("  -> empty chunk %s~%s", chunk_start.date(), chunk_end.date())

    if not parts:
        return pd.DataFrame(columns=["symbol", "tradetime", "factorname", "value"])
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["symbol", "tradetime", "factorname"], keep="last")


def _compute_once(
    client: DDBFactorClient,
    factor_name: str,
    start: dt.datetime,
    end: dt.datetime,
    univ: str,
) -> pd.DataFrame:
    script = f"compute_{factor_name}({_call_kwargs(factor_name, start, end, univ)})"
    logger.info("DDB call: %s", script)
    return _normalize_result(client.run_script(script))


def build_factor(
    client: DDBFactorClient,
    factor_name: str,
    *,
    start_day=None,
    end_day=None,
    universe: Optional[str] = None,
    chunk_by_year: Optional[bool] = None,
) -> pd.DataFrame:
    """计算单个因子，返回窄表 columns: symbol, tradetime, factorname, value。"""
    if factor_name not in FACTOR_SCRIPT_MAP:
        raise KeyError(f"未注册因子: {factor_name}")

    start = to_datetime(start_day or START_DAY)
    end = to_datetime(end_day or END_DAY)
    univ = universe or UNIVERSE

    if factor_name in TICK_AGG_FACTORS:
        return _build_tick_order_size_factor(client, factor_name, start, end, univ)

    do_chunk = (
        YEARLY_CHUNK_FACTORS.__contains__(factor_name) if chunk_by_year is None else chunk_by_year
    )

    _ensure_init(client)
    _load_factor_scripts(client, factor_name)

    if not do_chunk or start.year == end.year:
        return _compute_once(client, factor_name, start, end, univ)

    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        logger.info("%s chunk %s ~ %s", factor_name, chunk_start.date(), chunk_end.date())
        try:
            _ensure_init(client)
            _load_factor_scripts(client, factor_name)
            part = _compute_once(client, factor_name, chunk_start, chunk_end, univ)
            if part is not None and not part.empty:
                parts.append(part)
        except Exception:
            logger.exception(
                "%s failed on chunk %s~%s",
                factor_name,
                chunk_start.date(),
                chunk_end.date(),
            )
            raise
        finally:
            try:
                client.run_script("undef(`t, VAR); undef(`t_top, VAR); undef(`result, VAR)")
            except Exception:  # noqa: BLE001
                pass

    if not parts:
        return pd.DataFrame(columns=["symbol", "tradetime", "factorname", "value"])
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["symbol", "tradetime", "factorname"], keep="last")


def _make_builder(name: str) -> Callable[[DDBFactorClient], pd.DataFrame]:
    return lambda client: build_factor(client, name)


FACTOR_BUILDERS: Dict[str, Callable[[DDBFactorClient], pd.DataFrame]] = {
    name: _make_builder(name) for name in FACTOR_SCRIPT_MAP
}
