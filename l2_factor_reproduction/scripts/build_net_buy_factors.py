#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sprint 2：NetBuy 因子层——从 l2_primitive_trade_flow_daily 派生三因子并回测。

因子（全部日频、全日聚合 -> shift(1) 标准口径）：
- net_buy_ratio        = (active_buy_amt - active_sell_amt) / total_amt
- net_buy_amount_mcap  = (active_buy_amt - active_sell_amt) / S_VAL_MV（总市值，WIND）
- net_buy_count_ratio  = (active_buy_cnt - active_sell_cnt) / (active_buy_cnt + active_sell_cnt)

流程：
1. 股票池过滤（60/68/000/001/002/003/300/301/302 前缀）
2. 窄表落盘 research/results/l2_reproduction/<factor>/factor_narrow.parquet
3. backtest_factor（2019-01-01 ~ 2026-07-31，有效方向管线）
4. 年度 IC 稳定性表
5. 与 mid_order_ratio 正交检验（重叠窗口因子截面相关 + IC 序列相关）

用法:
  python l2_factor_reproduction/scripts/build_net_buy_factors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from COMMON_CONST import DATA_DB_CONN  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    _save_backtest_outputs,
    backtest_factor,
    narrow_to_wide,
)

START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2026-07-31")
STOCK_PREFIXES = ("60", "68", "000", "001", "002", "003", "300", "301", "302")

PRIMITIVE = (
    Path(RESULT_ROOT) / "primitives" / "trade_flow_daily"
    / "trade_flow_daily_2019-01-01_2026-07-31.parquet"
)


def _get_mcap_wide(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """WIND 总市值宽表（与 panel_neutral_size_ind 同表同口径）。"""
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    s.run(f"""
        A_stocks_value = select S_INFO_WINDCODE, TRADE_DT, S_VAL_MV
            from loadTable('dfs://WIND.ASHAREEODDERIVATIVEINDICATOR', 'data')
            where TRADE_DT >= {start.strftime('%Y.%m.%d')} and TRADE_DT <= {end.strftime('%Y.%m.%d')}
            context by TRADE_DT, S_INFO_WINDCODE csort OPDATE limit 1
    """)
    data_cap = s.run("A_stocks_value")
    s.close()
    wide = data_cap.pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values="S_VAL_MV")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _to_narrow(df: pd.DataFrame, value: pd.Series, factor_name: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str),
        "tradetime": pd.to_datetime(df["TradeDate"]) + pd.Timedelta(hours=9, minutes=30),
        "factorname": factor_name,
        "value": value.astype(float),
    })
    return out.dropna(subset=["value"])


def build_factors(flow: pd.DataFrame, mcap: pd.DataFrame) -> dict:
    df = flow.copy()
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["bare"] = df["symbol"].str.split(".").str[0]
    df = df[df["bare"].str.startswith(STOCK_PREFIXES)]

    total = pd.to_numeric(df["total_amt"], errors="coerce").replace(0, np.nan)
    buy = pd.to_numeric(df["active_buy_amt"], errors="coerce")
    sell = pd.to_numeric(df["active_sell_amt"], errors="coerce")
    net = buy - sell
    # CH countIf columns are UInt64. Convert before subtraction to prevent
    # unsigned underflow when active_buy_cnt < active_sell_cnt.
    buy_cnt = pd.to_numeric(
        df["active_buy_cnt"], errors="coerce"
    ).astype("float64")
    sell_cnt = pd.to_numeric(
        df["active_sell_cnt"], errors="coerce"
    ).astype("float64")
    cnt_total = (buy_cnt + sell_cnt).replace(0, np.nan)
    cnt_net = buy_cnt - sell_cnt
    count_ratio = cnt_net / cnt_total
    if (count_ratio.abs() > 1.0 + 1e-12).any():
        raise ValueError("net_buy_count_ratio outside [-1, 1]")

    factors = {
        "net_buy_ratio": net / total,
        "net_buy_count_ratio": count_ratio,
    }

    # net_buy_amount_mcap：symbol-day 对齐市值（MultiIndex map 对齐，避免 merge 索引错位）
    lookup = mcap.stack()
    lookup.index.names = ["TradeDate", "symbol"]
    lookup = lookup[~lookup.index.duplicated(keep="last")]
    key = pd.MultiIndex.from_arrays([df["TradeDate"], df["symbol"]])
    mc = pd.Series(lookup.reindex(key).to_numpy(), index=df.index)
    factors["net_buy_amount_mcap"] = net / mc.replace(0, np.nan) * 1e4  # 万元/亿市值量级

    return {
        name: _to_narrow(df, val, name) for name, val in factors.items()
    }


def yearly_ic_table(rank_ic: pd.Series) -> pd.DataFrame:
    ic = rank_ic.dropna()
    by_year = ic.groupby(ic.index.year)
    tbl = by_year.agg(["mean", "std", "count"])
    tbl["icir_annu"] = tbl["mean"] / tbl["std"] * (250 ** 0.5)
    return tbl.round(4)


def main() -> None:
    result_root = Path(RESULT_ROOT)
    print(f"加载 primitive: {PRIMITIVE}")
    flow = pd.read_parquet(PRIMITIVE)

    mcap_cache = result_root / "primitives" / f"mcap_wide_{START.date()}_{END.date()}.parquet"
    if mcap_cache.exists():
        mcap = pd.read_parquet(mcap_cache)
    else:
        print("DDB 拉取总市值宽表 ...")
        mcap = _get_mcap_wide(START, END)
        mcap.to_parquet(mcap_cache)

    print("构建三因子窄表 ...")
    factors = build_factors(flow, mcap)

    summaries = {}
    for name, narrow in factors.items():
        out_dir = result_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        narrow.to_parquet(out_dir / "factor_narrow.parquet")
        print(f"\n===== {name}: {len(narrow)} rows，回测 {START.date()} ~ {END.date()} =====")
        group_pnl, group_to, rank_ic, summary = backtest_factor(
            narrow, start_day=START, end_day=END
        )
        _save_backtest_outputs(str(out_dir), group_pnl, group_to, rank_ic, summary, factor_name=name)
        yt = yearly_ic_table(rank_ic)
        yt.to_csv(out_dir / "yearly_ic.csv")
        summary["net_annu_after_fee"] = summary["hl_annu_ret_flipped"] - summary["implied_annu_fee"]
        summaries[name] = summary
        print(
            f"factor_direction={summary['factor_direction']} | "
            f"RankIC={summary['rank_ic_mean']:.4f} ICIR={summary['rank_icir']:.2f} | "
            f"H-L年化={summary['hl_annu_ret_flipped']:.2%} Sharpe={summary['hl_sharpe_flipped']:.2f} "
            f"MDD={summary['hl_mdd_flipped']:.2%} | 日均换手={summary['avg_hl_turnover']:.2f} "
            f"成本={summary['implied_annu_fee']:.2%} 净年化={summary['net_annu_after_fee']:.2%}"
        )
        print("年度 IC:")
        print(yt.to_string())

    # --- 与 mid_order_ratio 正交检验（重叠窗口）---
    mid_path = result_root / "mid_order_ratio" / "factor_narrow.parquet"
    if mid_path.exists():
        print("\n===== 与 mid_order_ratio 正交检验（重叠窗口 2023-01 ~ 2024-06）=====")
        mid_wide = narrow_to_wide(pd.read_parquet(mid_path))
        mid_ic = pd.read_csv(result_root / "mid_order_ratio" / "rank_ic.csv",
                             index_col=0, parse_dates=True).iloc[:, 0]
        corr_rows = []
        for name in factors:
            nb_wide = narrow_to_wide(pd.read_parquet(result_root / name / "factor_narrow.parquet"))
            common_idx = nb_wide.index.intersection(mid_wide.index)
            common_cols = nb_wide.columns.intersection(mid_wide.columns)
            a, b = nb_wide.loc[common_idx, common_cols], mid_wide.loc[common_idx, common_cols]
            xs_corr = a.corrwith(b, axis=1, method="spearman")
            nb_ic = pd.read_csv(result_root / name / "rank_ic.csv",
                                index_col=0, parse_dates=True).iloc[:, 0]
            ic_pair = pd.concat([nb_ic, mid_ic], axis=1, keys=["nbr", "mid"]).dropna()
            ic_corr = ic_pair["nbr"].corr(ic_pair["mid"], method="spearman")
            corr_rows.append({
                "factor": name,
                "xs_corr_mean": round(float(xs_corr.mean()), 4),
                "xs_corr_std": round(float(xs_corr.std()), 4),
                "ic_series_corr": round(float(ic_corr), 4),
                "overlap_days": int(len(ic_pair)),
            })
            print(f"{name}: 截面相关={xs_corr.mean():.3f} | IC序列相关={ic_corr:.3f}")
        pd.DataFrame(corr_rows).to_csv(
            result_root / "flow_vs_size_orthogonality.csv", index=False
        )

    print("\n全部完成。")


if __name__ == "__main__":
    main()
