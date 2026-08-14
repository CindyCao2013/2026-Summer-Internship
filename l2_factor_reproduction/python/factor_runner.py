"""因子运行器：按 FACTOR_LIST 遍历计算并落盘窄表。"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional

import pandas as pd

from l2_factor_reproduction.config.settings import FACTOR_LIST, LOG_LEVEL, RESULT_ROOT
from l2_factor_reproduction.python.ddb_client import DDBFactorClient
from l2_factor_reproduction.python.factor_builder import FACTOR_BUILDERS, build_factor

logging.basicConfig(
    level=getattr(logging, str(LOG_LEVEL).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _save_narrow(factor_name: str, df: pd.DataFrame) -> str:
    out_dir = os.path.join(RESULT_ROOT, factor_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "factor_narrow.parquet")
    df.to_parquet(out_path, index=False)
    logger.info("%s saved -> %s (%d rows)", factor_name, out_path, len(df))
    return out_path


def run_all_factors(
    factor_list: Optional[Iterable[str]] = None,
    client: Optional[DDBFactorClient] = None,
) -> Dict[str, pd.DataFrame]:
    """计算配置中的全部因子，写入 RESULT_ROOT/<name>/factor_narrow.parquet。"""
    names = list(factor_list) if factor_list is not None else list(FACTOR_LIST)
    own_client = client is None
    client = client or DDBFactorClient()
    results: Dict[str, pd.DataFrame] = {}
    try:
        for fname in names:
            if fname not in FACTOR_BUILDERS:
                logger.warning("因子 %s 无构建函数，跳过", fname)
                continue
            logger.info("计算因子 %s ...", fname)
            try:
                df = build_factor(client, fname)
                if df is None or df.empty:
                    logger.warning("%s 返回空数据", fname)
                    continue
                _save_narrow(fname, df)
                results[fname] = df
            except Exception as exc:  # noqa: BLE001
                logger.exception("计算 %s 失败: %s", fname, exc)
    finally:
        if own_client:
            client.close()
    return results


def run_single_factor(
    factor_name: str,
    *,
    save: bool = True,
    start_day=None,
    end_day=None,
    universe: Optional[str] = None,
) -> pd.DataFrame:
    """计算单个因子；可选覆盖日期/股票池（冒烟测试用短区间）。"""
    client = DDBFactorClient()
    try:
        df = build_factor(
            client,
            factor_name,
            start_day=start_day,
            end_day=end_day,
            universe=universe,
        )
        if save and df is not None and not df.empty:
            _save_narrow(factor_name, df)
        return df
    finally:
        client.close()
