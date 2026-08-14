"""通用二次（多因子）中性化：对已中性化/原始因子再剥离额外风格因子。

逐日截面 OLS 取残差，解释变量可为一张或多张外生因子宽表（如动量、换手率）。
与 ``Factor_Dev_Lib.panel_neutral_size_ind`` 互补：该函数只做市值/行业哑变量回归，
本模块用于在其残差上继续剥离任意连续风格因子。
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd


def neutralize_again(
    signal_wide: pd.DataFrame,
    extra_factors: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]],
    min_obs: int = 30,
) -> pd.DataFrame:
    """逐日截面回归 y ~ 1 + x1 + x2 ...，返回残差宽表。

    Parameters
    ----------
    signal_wide : DataFrame
        日期 x 股票 的因子宽表（可以是已中性化过的残差）。
    extra_factors : DataFrame 或 dict/list
        额外解释变量宽表；dict 时 key 为因子名（仅用于日志）。
    min_obs : int
        当日有效样本不足该值时返回 NaN。
    """
    if isinstance(extra_factors, pd.DataFrame):
        factors = {"x0": extra_factors}
    elif isinstance(extra_factors, (list, tuple)):
        factors = {f"x{i}": f for i, f in enumerate(extra_factors)}
    else:
        factors = dict(extra_factors)

    aligned = {}
    for name, f in factors.items():
        aligned[name] = f.reindex(index=signal_wide.index, columns=signal_wide.columns)

    res = pd.DataFrame(np.nan, index=signal_wide.index, columns=signal_wide.columns, dtype=float)
    x_names = list(aligned)
    for dt in signal_wide.index:
        y = signal_wide.loc[dt]
        xs = [aligned[n].loc[dt] for n in x_names]
        valid = y.notna()
        for x in xs:
            valid &= x.notna()
        if int(valid.sum()) < min_obs:
            continue
        Y = y[valid].astype(float).values
        X = np.column_stack([x[valid].astype(float).values for x in xs])
        X = np.column_stack([np.ones(len(Y)), X])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res.loc[dt, y.index[valid]] = Y - X @ beta
    return res
