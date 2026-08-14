"""财务衍生因子适配层（reference repo 契约 → 本项目 Wide Date×Symbol）。

所有财务字段按公告日 ann_date backward merge_asof：每个交易日只使用
当时已公告的最新财务指标，避免未来信息。

与 factor_formulas_fundamental.py 的分工：
  - Phase 1 (EP/BP/市值): DERIVATIVEINDICATOR 日频宽表，无需 ann_date
  - Phase 2 (ROE/质量):  本模块 + ASHARETTMHIS 公告对齐
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SYMBOL_COL = "symbol"
DEFAULT_DATE_COL = "trade_date"
DEFAULT_ANN_COL = "ann_date"


def _pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def prices_wide_to_long(
    close: pd.DataFrame,
    *,
    date_col: str = DEFAULT_DATE_COL,
    symbol_col: str = DEFAULT_SYMBOL_COL,
) -> pd.DataFrame:
    """Close 宽表 → 长表 (trade_date, symbol, close)。"""
    px = close.sort_index().sort_index(axis=1).astype(float)
    long_df = px.stack(dropna=False).rename("close").reset_index()
    long_df.columns = [date_col, symbol_col, "close"]
    long_df[date_col] = pd.to_datetime(long_df[date_col])
    return long_df


def panel_long_to_wide(series: pd.Series) -> pd.DataFrame:
    """PanelLong (date, symbol) → Wide Date×Symbol。"""
    if series.empty:
        return pd.DataFrame()
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise TypeError("Expected MultiIndex(date, symbol) Series")
    wide = series.unstack(level=1)
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "Date"
    return wide.sort_index()


def finance_ann_to_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
    *,
    metric_col: str,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    ann_col: str = DEFAULT_ANN_COL,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    ann_date 财务序列 → Wide Date×Symbol（等价 backward merge_asof + 日频网格）。

    比 PanelLong merge_asof 更适合本项目宽表 pipeline，且兼容 pandas 1.2。
    """
    fin = finance_df[[symbol_col, ann_col, metric_col]].copy()
    fin[ann_col] = pd.to_datetime(fin[ann_col], errors="coerce")
    fin[metric_col] = pd.to_numeric(fin[metric_col], errors="coerce")
    fin = (
        fin.dropna(subset=[ann_col, metric_col])
        .sort_values([symbol_col, ann_col])
        .drop_duplicates(subset=[symbol_col, ann_col], keep="last")
    )

    idx = pd.DatetimeIndex(close.index).sort_values()
    cols = list(close.columns)
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)

    sym_groups = {sym: grp for sym, grp in fin.groupby(symbol_col, sort=False)}
    for sym in cols:
        grp = sym_groups.get(sym)
        if grp is None or grp.empty:
            continue
        ann_series = grp.set_index(ann_col)[metric_col].sort_index()
        union_idx = idx.union(ann_series.index).sort_values()
        aligned = ann_series.reindex(union_idx).ffill().reindex(idx)
        if not higher_is_better:
            aligned = -aligned
        out[sym] = aligned.values
    return out


def merge_asof_finance_to_daily(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    metric_col: str,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    date_col: str = DEFAULT_DATE_COL,
    ann_col: str = DEFAULT_ANN_COL,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    单个财务指标 → PanelLong (date, symbol)。

    finance_df 须含 symbol_col, ann_col, metric_col。
    prices_long 须含 date_col, symbol_col, close（close 仅用于交易日网格）。
    """
    need_px = {date_col, symbol_col, "close"}
    missing_px = need_px - set(prices_long.columns)
    if missing_px:
        raise ValueError(f"prices_long missing columns: {missing_px}")
    need_f = {symbol_col, ann_col, metric_col}
    missing_f = need_f - set(finance_df.columns)
    if missing_f:
        raise ValueError(f"finance_df missing columns: {missing_f}")

    px = prices_long[[date_col, symbol_col, "close"]].copy()
    px[date_col] = pd.to_datetime(px[date_col])
    fin = finance_df[[symbol_col, ann_col, metric_col]].copy()
    fin[ann_col] = pd.to_datetime(fin[ann_col], errors="coerce")
    fin[metric_col] = pd.to_numeric(fin[metric_col], errors="coerce")
    fin = (
        fin.dropna(subset=[ann_col, metric_col])
        .sort_values([symbol_col, ann_col])
        .drop_duplicates(subset=[symbol_col, ann_col], keep="last")
    )
    px = px.sort_values([symbol_col, date_col])

    merged = pd.merge_asof(
        px,
        fin,
        left_on=date_col,
        right_on=ann_col,
        by=symbol_col,
        direction="backward",
    )
    score = pd.to_numeric(merged[metric_col], errors="coerce")
    if not higher_is_better:
        score = -score
    midx = pd.MultiIndex.from_arrays(
        [merged[date_col].values, merged[symbol_col].values],
        names=["date", "symbol"],
    )
    out = pd.Series(score.values, index=midx)
    return out[~out.index.duplicated(keep="last")].sort_index()


def calc_finance_metric(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    metric_candidates: Tuple[str, ...],
    higher_is_better: bool = True,
    ts_code_col: str = DEFAULT_SYMBOL_COL,
    date_col: str = DEFAULT_DATE_COL,
    ann_col: str = DEFAULT_ANN_COL,
) -> pd.Series:
    """Reference 契约：按 metric_candidates 优先级选列并对齐到日频。"""
    metric_col = _pick_column(finance_df, metric_candidates)
    if metric_col is None:
        return pd.Series(dtype=float)
    return merge_asof_finance_to_daily(
        finance_df,
        prices_long,
        metric_col=metric_col,
        symbol_col=ts_code_col,
        date_col=date_col,
        ann_col=ann_col,
        higher_is_better=higher_is_better,
    )


def calc_roe(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    ts_code_col: str = DEFAULT_SYMBOL_COL,
    date_col: str = DEFAULT_DATE_COL,
    ann_col: str = DEFAULT_ANN_COL,
) -> pd.Series:
    """ROE TTM，ann_date backward 对齐；越高越好。"""
    return calc_finance_metric(
        finance_df,
        prices_long,
        metric_candidates=("roe", "S_FA_ROE_TTM", "ROE_DILUTED", "S_FA_ROE_YEARLY"),
        higher_is_better=True,
        ts_code_col=ts_code_col,
        date_col=date_col,
        ann_col=ann_col,
    )


def _roe_rolling_instability(
    finance_df: pd.DataFrame,
    *,
    metric_col: str = "roe",
    symbol_col: str = DEFAULT_SYMBOL_COL,
    ann_col: str = DEFAULT_ANN_COL,
    window: int = 8,
    min_periods: int = 4,
) -> pd.DataFrame:
    """
    在公告频率上计算 ROE 滚动波动（std）。
    返回 long: symbol, ann_date, roe_instability。
    """
    fin = finance_df[[symbol_col, ann_col, metric_col]].copy()
    fin[ann_col] = pd.to_datetime(fin[ann_col], errors="coerce")
    fin[metric_col] = pd.to_numeric(fin[metric_col], errors="coerce")
    fin = fin.dropna(subset=[ann_col, metric_col])
    fin = fin.sort_values([symbol_col, ann_col]).drop_duplicates(subset=[symbol_col, ann_col], keep="last")

    rows = []
    for sym, grp in fin.groupby(symbol_col, sort=False):
        g = grp.sort_values(ann_col)
        instability = g[metric_col].rolling(window, min_periods=min_periods).std()
        tmp = pd.DataFrame(
            {
                symbol_col: sym,
                ann_col: g[ann_col].values,
                "roe_instability": instability.values,
            }
        )
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=[symbol_col, ann_col, "roe_instability"])
    return pd.concat(rows, ignore_index=True)


def calc_roe_stability(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    window: int = 8,
    min_periods: int = 4,
    ts_code_col: str = DEFAULT_SYMBOL_COL,
    date_col: str = DEFAULT_DATE_COL,
    ann_col: str = DEFAULT_ANN_COL,
) -> pd.Series:
    """
    ROE 稳定性：公告频率上 -rolling_std(ROE)，再 merge_asof 到日频。
    越高 = ROE 越稳定（Quality / D7 候选）。
    """
    metric_col = _pick_column(finance_df, ("roe", "S_FA_ROE_TTM", "ROE_DILUTED"))
    if metric_col is None:
        return pd.Series(dtype=float)

    instability = _roe_rolling_instability(
        finance_df,
        metric_col=metric_col,
        symbol_col=ts_code_col,
        ann_col=ann_col,
        window=window,
        min_periods=min_periods,
    )
    if instability.empty:
        return pd.Series(dtype=float)

    # 转为 stability score = -instability，再按 ann_date 对齐
    stab_df = instability.copy()
    stab_df["roe_stability"] = -pd.to_numeric(stab_df["roe_instability"], errors="coerce")
    stab_df = stab_df.dropna(subset=["roe_stability"])
    return merge_asof_finance_to_daily(
        stab_df.rename(columns={"roe_stability": "metric"}),
        prices_long,
        metric_col="metric",
        symbol_col=ts_code_col,
        date_col=date_col,
        ann_col=ann_col,
        higher_is_better=True,
    )


def calc_finance_metric_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
    *,
    metric_candidates: Tuple[str, ...],
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """便捷入口：finance long + close wide → factor wide。"""
    long_px = prices_wide_to_long(close)
    series = calc_finance_metric(
        finance_df,
        long_px,
        metric_candidates=metric_candidates,
        higher_is_better=higher_is_better,
    )
    return panel_long_to_wide(series)


def calc_roe_wide(finance_df: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    metric_col = _pick_column(finance_df, ("roe", "S_FA_ROE_TTM", "ROE_DILUTED", "S_FA_ROE_YEARLY"))
    if metric_col is None:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    if metric_col != "roe":
        finance_df = finance_df.rename(columns={metric_col: "roe"})
        metric_col = "roe"
    return finance_ann_to_wide(finance_df, close, metric_col=metric_col)


def calc_roe_stability_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int = 8,
    min_periods: int = 4,
) -> pd.DataFrame:
    metric_col = _pick_column(finance_df, ("roe", "S_FA_ROE_TTM", "ROE_DILUTED"))
    if metric_col is None:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    if metric_col != "roe":
        finance_df = finance_df.rename(columns={metric_col: "roe"})
    instability = _roe_rolling_instability(
        finance_df,
        metric_col="roe",
        window=window,
        min_periods=min_periods,
    )
    if instability.empty:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    stab_df = instability.copy()
    stab_df["roe_stability"] = -pd.to_numeric(stab_df["roe_instability"], errors="coerce")
    return finance_ann_to_wide(
        stab_df.rename(columns={"roe_stability": "metric"}),
        close,
        metric_col="metric",
    )


def normalize_finance_long(
    df: pd.DataFrame,
    *,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    ann_col: str = DEFAULT_ANN_COL,
    roe_col_candidates: Sequence[str] = ("roe", "S_FA_ROE_TTM"),
) -> pd.DataFrame:
    """Wind DDB 列名 → reference 契约列名。"""
    out = df.copy()
    sym = _pick_column(out, ("symbol", "S_INFO_WINDCODE", "ts_code", "WindCode"))
    ann = _pick_column(out, ("ann_date", "ANN_DT", "ann_dt"))
    roe = _pick_column(out, tuple(roe_col_candidates))
    if sym is None or ann is None:
        raise ValueError("finance panel missing symbol or ann_date column")
    rename = {sym: symbol_col, ann: ann_col}
    if roe and roe != "roe":
        rename[roe] = "roe"

    col_map = {
        "gross_profit_ttm": ("gross_profit_ttm", "S_FA_GROSSMARGIN_TTM"),
        "total_assets_mrq": ("total_assets_mrq", "S_FA_ASSET_MRQ"),
        "cfo_ttm": ("cfo_ttm", "NET_CASH_FLOWS_OPER_ACT_TTM"),
        "net_profit_ttm": ("net_profit_ttm", "NET_PROFIT_PARENT_COMP_TTM"),
    }
    for target, candidates in col_map.items():
        src = _pick_column(out, candidates)
        if src and src != target:
            rename[src] = target

    out = out.rename(columns=rename)
    out[ann_col] = pd.to_datetime(out[ann_col], errors="coerce")
    out["roe"] = pd.to_numeric(out.get("roe"), errors="coerce")
    for col in col_map:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    keep = [symbol_col, ann_col, "roe"]
    for col in col_map:
        if col in out.columns:
            keep.append(col)
    extra = [c for c in out.columns if c not in keep]
    return out[keep + extra]


def _ratio_at_ann(
    finance_df: pd.DataFrame,
    *,
    numerator_col: str,
    denominator_col: str,
    metric_col: str,
    min_denominator: float = 1e-6,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    ann_col: str = DEFAULT_ANN_COL,
) -> pd.DataFrame:
    """公告频率上计算 ratio，返回 long: symbol, ann_date, metric_col。"""
    need = {symbol_col, ann_col, numerator_col, denominator_col}
    missing = need - set(finance_df.columns)
    if missing:
        return pd.DataFrame(columns=[symbol_col, ann_col, metric_col])

    fin = finance_df[[symbol_col, ann_col, numerator_col, denominator_col]].copy()
    fin[ann_col] = pd.to_datetime(fin[ann_col], errors="coerce")
    num = pd.to_numeric(fin[numerator_col], errors="coerce")
    den = pd.to_numeric(fin[denominator_col], errors="coerce")
    fin[metric_col] = num / den.where(den.abs() >= min_denominator)
    fin = (
        fin.dropna(subset=[ann_col, metric_col])
        .sort_values([symbol_col, ann_col])
        .drop_duplicates(subset=[symbol_col, ann_col], keep="last")
    )
    return fin[[symbol_col, ann_col, metric_col]]


def calc_gross_profitability_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """GP/A = gross profit TTM / total assets MRQ (Novy-Marx quality brick)."""
    ratio_df = _ratio_at_ann(
        finance_df,
        numerator_col="gross_profit_ttm",
        denominator_col="total_assets_mrq",
        metric_col="gross_profitability",
    )
    if ratio_df.empty:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    return finance_ann_to_wide(ratio_df, close, metric_col="gross_profitability")


def calc_cfo_quality_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
    *,
    min_net_profit: float = 1e6,
) -> pd.DataFrame:
    """CFO/NI = operating cash flow TTM / parent net profit TTM (accruals proxy)."""
    ratio_df = _ratio_at_ann(
        finance_df,
        numerator_col="cfo_ttm",
        denominator_col="net_profit_ttm",
        metric_col="cfo_quality",
        min_denominator=min_net_profit,
    )
    if ratio_df.empty:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    return finance_ann_to_wide(ratio_df, close, metric_col="cfo_quality")


def calc_cfp_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    *,
    mktcap_yuan_per_unit: float = 10000.0,
) -> pd.DataFrame:
    """
    CFP = operating cash flow TTM / float market cap.

    CFO from ann_date-aligned ASHARETTMHIS; market cap from daily derivative (万元).
    """
    if "cfo_ttm" not in finance_df.columns:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    fin = finance_df[["symbol", "ann_date", "cfo_ttm"]].copy()
    fin["ann_date"] = pd.to_datetime(fin["ann_date"], errors="coerce")
    fin["cfo_ttm"] = pd.to_numeric(fin["cfo_ttm"], errors="coerce")
    fin = (
        fin.dropna(subset=["ann_date", "cfo_ttm"])
        .sort_values(["symbol", "ann_date"])
        .drop_duplicates(subset=["symbol", "ann_date"], keep="last")
    )
    if fin.empty:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    cfo_daily = finance_ann_to_wide(fin, close, metric_col="cfo_ttm")
    mv = float_mktcap.reindex(index=close.index, columns=close.columns)
    denom = mv * mktcap_yuan_per_unit
    return cfo_daily / denom.replace(0, np.nan)


def calc_quality_composite_wide(
    finance_df: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int = 8,
    min_periods: int = 4,
) -> pd.DataFrame:
    """
    D7 Quality composite: equal-weight cross-sectional z of
    roe_stability + gross_profitability + cfo_quality.
    """
    from factor_attribution import cs_zscore

    bricks = [
        calc_roe_stability_wide(
            finance_df, close, window=window, min_periods=min_periods
        ),
        calc_gross_profitability_wide(finance_df, close),
        calc_cfo_quality_wide(finance_df, close),
    ]
    z_parts = [cs_zscore(b) for b in bricks]
    valid = [z for z in z_parts if z is not None and not z.empty]
    if not valid:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    composite = sum(valid) / len(valid)
    return composite.reindex(index=close.index, columns=close.columns)
