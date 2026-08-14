"""日频分组回测：复用 Factor_Dev_Lib.groupTest。

对齐约定（与 ``factor_runner.prepare_signal`` 一致）：
- 因子由 **当日全日** 分钟数据聚合 → 当日收盘后才可知
- 因此 ``signal.shift(1)`` 后再与当日 ``c2c`` 收益做截面相关 / 分组
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Factor_Dev_Lib import (
    calAnnuRet,
    calMDD,
    calSharpe,
    format_group_test_stats_title,
    get_EOD_Not_Limit,
    get_EOD_Not_ST,
    get_Ret_Matrix,
    get_TradeStatus,
    groupTest,
    implied_annu_fee,
)
from l2_factor_reproduction.config.settings import (
    BACKTEST_SILENT,
    END_DAY,
    FACTOR_LIST,
    N_GROUPS,
    RESULT_ROOT,
    START_DAY,
    UNIVERSE,
)

logger = logging.getLogger(__name__)


def narrow_to_wide(factor_narrow: pd.DataFrame) -> pd.DataFrame:
    """symbol/tradetime/value 窄表 -> 日频宽表。"""
    df = factor_narrow.copy()
    if "tradetime" not in df.columns or "symbol" not in df.columns or "value" not in df.columns:
        raise ValueError("窄表需包含 symbol, tradetime, value 列")
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    wide = df.pivot_table(index="tradetime", columns="symbol", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index).normalize()
    wide = wide.sort_index()
    wide = wide[~wide.index.duplicated(keep="last")]
    return wide


def prepare_factor_signal(
    factor_wide: pd.DataFrame,
    *,
    start,
    end,
    mask: pd.DataFrame,
    signal_shift: int = 1,
    ret_matrix: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """过滤 + shift，返回 (signal, ret)。ret 为相对 UNIVERSE 的超额 c2c。"""
    univ = UNIVERSE
    factor_wide = factor_wide.loc[start:end]
    ret = (
        get_Ret_Matrix(start, end, base_index=univ)
        if ret_matrix is None
        else ret_matrix
    )

    common_idx = factor_wide.index.intersection(ret.index).intersection(mask.index)
    common_cols = factor_wide.columns.intersection(ret.columns).intersection(mask.columns)
    signal = factor_wide.loc[common_idx, common_cols]
    ret = ret.reindex(index=common_idx, columns=common_cols)
    m = mask.reindex(index=common_idx, columns=common_cols)

    signal = signal.where(m == 1)
    if signal_shift:
        signal = signal.shift(signal_shift)
    signal = signal.dropna(how="all", axis=1).dropna(how="all")
    ret = ret.reindex(index=signal.index, columns=signal.columns)
    return signal, ret


def compute_rank_ic(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    return signal.corrwith(ret, axis=1, method="spearman")


def summarize_backtest(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    rank_ic: pd.Series,
) -> Dict[str, Any]:
    """汇总指标；H-L 若均值为负则翻向（与 groupTest 展示一致）。"""
    hl = group_pnl["H-L"]
    direction = 1 if hl.mean() > 0 else -1
    hl_adj = hl * direction

    ic_mean = float(rank_ic.mean())
    ic_std = float(rank_ic.std())
    icir = ic_mean / ic_std * (250 ** 0.5) if ic_std and ic_std > 0 else float("nan")

    # 原始方向 IC（未翻向）：用于和研报「高值越好」对标
    mdd, _ = calMDD(hl_adj)
    avg_to = float(group_to["H-L"].mean())
    group_columns = [column for column in group_pnl.columns if column != "H-L"]
    try:
        top_group = max(group_columns, key=lambda column: int(column))
    except (TypeError, ValueError):
        top_group = group_columns[-1]
    g10_excess = group_pnl[top_group]
    return {
        "n_days": int(len(rank_ic.dropna())),
        "n_names_avg": float(signal.notna().sum(axis=1).mean()),
        "rank_ic_mean": ic_mean,
        "rank_ic_std": ic_std,
        "rank_icir": float(icir),
        "hl_direction_flip": int(direction),  # -1 表示原始 H-L=G10-G1 为负，展示时已翻向
        "hl_annu_ret_flipped": float(calAnnuRet(hl_adj)),
        "hl_sharpe_flipped": float(calSharpe(hl_adj)),
        "hl_mdd_flipped": float(mdd),
        "hl_annu_ret_raw": float(calAnnuRet(hl)),
        "hl_sharpe_raw": float(calSharpe(hl)),
        "g10_excess_annu_ret": float(calAnnuRet(g10_excess)),
        "g10_excess_sharpe": float(calSharpe(g10_excess)),
        "avg_hl_turnover": avg_to,
        "implied_annu_fee": float(implied_annu_fee(avg_to)),
        "group_mean_annu": {
            str(c): float(calAnnuRet(group_pnl[c])) for c in group_pnl.columns if c != "H-L"
        },
    }


def backtest_factor(
    factor_narrow: pd.DataFrame,
    n_groups: int = N_GROUPS,
    *,
    start_day=None,
    end_day=None,
    universe: Optional[str] = None,
    signal_shift: int = 1,
    mask: Optional[pd.DataFrame] = None,
    ret_matrix: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """对因子窄表做十分组回测。

    Returns
    -------
    group_pnl, group_turnover, rank_ic, summary
    """
    start = start_day or START_DAY
    end = end_day or END_DAY
    # universe 参数保留接口；收益基准仍用 settings.UNIVERSE（与配置一致）
    _ = universe

    factor_wide = narrow_to_wide(factor_narrow)
    if mask is None:
        not_limit = get_EOD_Not_Limit(start, end)
        not_st = get_EOD_Not_ST(start, end)
        trade_status = get_TradeStatus(start, end)
        mask = not_limit * not_st * trade_status

    signal, ret = prepare_factor_signal(
        factor_wide,
        start=start,
        end=end,
        mask=mask,
        signal_shift=signal_shift,
        ret_matrix=ret_matrix,
    )
    rank_ic_raw = compute_rank_ic(signal, ret)

    info = "silent" if BACKTEST_SILENT else f"L2_{UNIVERSE}"
    _r, pnl_raw, to_raw = groupTest(signal, ret, n=n_groups, info=info)

    # 方向在因子层面统一处理：若原始 H-L 均值为负，则取反信号重新分组，
    # 之后 G1→G10 单调性、H-L、IC 全部处于「有效方向」，无需事后翻 H-L。
    direction = 1 if pnl_raw["H-L"].mean() > 0 else -1
    if direction < 0:
        signal = -signal
        _r, group_pnl, group_to = groupTest(signal, ret, n=n_groups, info=info)
        rank_ic = -rank_ic_raw
    else:
        group_pnl, group_to, rank_ic = pnl_raw, to_raw, rank_ic_raw

    summary = summarize_backtest(signal, ret, group_pnl, group_to, rank_ic)
    summary["factor_direction"] = int(direction)  # -1 = 生产使用时应取 -factor
    summary["rank_ic_mean_raw"] = float(rank_ic_raw.mean())  # 原始口径 IC（研报对标用）
    summary["rank_ic_std_raw"] = float(rank_ic_raw.std())
    summary["positive_ic_fraction_raw"] = float(
        (rank_ic_raw.dropna() > 0).mean()
    )
    summary["group_pnl_saved_direction"] = "effective"
    return group_pnl, group_to, rank_ic, summary


def load_backtest_context(
    start,
    end,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared investability mask and benchmark-relative returns once."""
    not_limit = get_EOD_Not_Limit(start, end)
    not_st = get_EOD_Not_ST(start, end)
    trade_status = get_TradeStatus(start, end)
    mask = not_limit * not_st * trade_status
    ret = get_Ret_Matrix(start, end, base_index=UNIVERSE)
    return mask, ret


def _to_effective_direction(pnl: pd.DataFrame, direction: int) -> pd.DataFrame:
    """将原始方向（G1=低因子 … G10=高因子，H-L=G10-G1）的分组收益表整体
    转到「有效方向」。direction=-1 时：十分组列顺序倒转并重标 1..n
    （原 G10 变成新 G1），H-L 取负。所有曲线由此处于同一方向。"""
    if direction == 1:
        return pnl
    group_cols = [c for c in pnl.columns if c != "H-L"]
    try:
        group_cols = sorted(group_cols, key=lambda x: int(x))
    except Exception:  # noqa: BLE001
        group_cols = list(group_cols)
    n = len(group_cols)
    out = pd.DataFrame(index=pnl.index)
    for i, c in enumerate(group_cols):
        out[str(n - i)] = pnl[c]
    if "H-L" in pnl.columns:
        out["H-L"] = -pnl["H-L"]
    return out


def save_group_plots(
    out_dir: str,
    factor_name: str,
    group_pnl: pd.DataFrame,
    group_to: Optional[pd.DataFrame] = None,
    rank_ic: Optional[pd.Series] = None,
    direction: int = 1,
) -> Tuple[str, str]:
    """落盘两张标准图（全部为有效方向：高因子值组在上，H-L 上行）：

    1. ``cum_pnl.png``：G1..G10 + H-L 累计收益曲线
    2. ``decile_bar.png``：各组日均收益柱状图（单调性）

    Parameters
    ----------
    direction : int
        输入 ``group_pnl`` 的方向。1 = 已是有效方向（新管线默认）；
        -1 = 仍是原始方向（旧 CSV），展示前整体倒转分组 + H-L 取负。
    """
    os.makedirs(out_dir, exist_ok=True)
    pnl = _to_effective_direction(group_pnl, direction)
    pnl.index = pd.to_datetime(pnl.index)

    hl = pnl["H-L"]
    ic_sign = direction
    avg_to = float(group_to["H-L"].mean()) if group_to is not None and "H-L" in group_to else np.nan
    if rank_ic is not None and len(rank_ic.dropna()) > 0:
        ic_mean = float(rank_ic.mean()) * ic_sign
        ic_std = float(rank_ic.std())
        icir = ic_mean / ic_std * (250 ** 0.5) if ic_std and ic_std > 0 else np.nan
    else:
        ic_mean, icir = np.nan, np.nan
    mdd, _ = calMDD(hl)
    caption = format_group_test_stats_title(
        direction=1,
        annu_ret=calAnnuRet(hl),
        sharpe=calSharpe(hl),
        mdd=mdd,
        avg_turnover=avg_to,
        rank_ic=ic_mean,
        icir=icir,
        implied_fee=implied_annu_fee(avg_to) if pd.notna(avg_to) else np.nan,
    )

    # --- 图1：累计收益（有效方向，G10 在上，H-L 上行）---
    cum = pnl.cumsum()
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    for col in cum.columns:
        lw = 2.5 if col == "H-L" else 1.2
        ax1.plot(cum.index, cum[col], label=str(col), linewidth=lw)
        ax1.text(cum.index[-1], cum[col].iloc[-1], str(col), fontsize=11, va="bottom")
    ax1.legend(loc="upper left", ncol=2, fontsize=10)
    ax1.set_title(f"{factor_name} — Decile + H-L Cumulative (effective direction)", fontsize=14)
    ax1.set_xlabel(caption, fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path_cum = os.path.join(out_dir, "cum_pnl.png")
    fig1.savefig(path_cum, dpi=120)
    plt.close(fig1)

    # --- 图2：日均收益柱状图（单调性，有效方向）---
    means = pnl.mean()
    group_cols = [c for c in means.index if c != "H-L"]
    try:
        group_cols = sorted(group_cols, key=lambda x: int(x))
    except Exception:  # noqa: BLE001
        group_cols = list(group_cols)
    order = group_cols + (["H-L"] if "H-L" in means.index else [])
    means = means.reindex(order)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0"] * len(group_cols) + (["#C44E52"] if "H-L" in order else [])
    ax2.bar([str(c) for c in means.index], means.values, color=colors[: len(means)])
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title(f"{factor_name} — Decile Mean Daily Return (monotonicity, effective)", fontsize=14)
    ax2.set_xlabel("Decile (G1=low signal ... G10=high signal, effective direction)", fontsize=11)
    ax2.set_ylabel("Mean daily return", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    path_bar = os.path.join(out_dir, "decile_bar.png")
    fig2.savefig(path_bar, dpi=120)
    plt.close(fig2)

    logger.info("%s plots -> %s | %s", factor_name, path_cum, path_bar)
    return path_cum, path_bar


def _save_backtest_outputs(
    out_dir: str,
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    rank_ic: pd.Series,
    summary: Dict[str, Any],
    factor_name: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    group_pnl.to_csv(os.path.join(out_dir, "group_pnl.csv"))
    group_to.to_csv(os.path.join(out_dir, "group_turnover.csv"))
    group_to.to_csv(os.path.join(out_dir, "group_to.csv"))
    rank_ic.to_frame("rank_ic").to_csv(os.path.join(out_dir, "rank_ic.csv"))
    direction = int(summary.get("factor_direction", 1))
    rank_ic_raw = rank_ic * direction
    rank_ic_raw.to_frame("rank_ic_raw").to_csv(
        os.path.join(out_dir, "rank_ic_raw.csv")
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    name = factor_name or os.path.basename(out_dir.rstrip("/"))
    save_group_plots(out_dir, name, group_pnl, group_to=group_to, rank_ic=rank_ic)
    _save_ic_plots(out_dir, name, rank_ic_raw)


def _save_ic_plots(
    out_dir: str,
    factor_name: str,
    rank_ic_raw: pd.Series,
) -> Tuple[str, str]:
    """Save raw-direction daily IC series and yearly-mean IC bars."""
    ic = rank_ic_raw.dropna().copy()
    ic.index = pd.to_datetime(ic.index)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ic.index, ic.values, linewidth=0.8, alpha=0.65)
    if len(ic):
        rolling = ic.rolling(60, min_periods=20).mean()
        ax.plot(
            rolling.index,
            rolling.values,
            linewidth=1.8,
            label="60-day mean",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{factor_name} — Raw-direction daily RankIC")
    ax.set_ylabel("RankIC")
    ax.grid(True, alpha=0.3)
    if len(ic):
        ax.legend(loc="best")
    fig.tight_layout()
    series_path = os.path.join(out_dir, "ic_series.png")
    fig.savefig(series_path, dpi=120)
    plt.close(fig)

    yearly = ic.groupby(ic.index.year).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0" if value >= 0 else "#C44E52" for value in yearly]
    ax.bar(yearly.index.astype(str), yearly.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{factor_name} — Yearly mean RankIC (raw direction)")
    ax.set_ylabel("Mean RankIC")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    yearly_path = os.path.join(out_dir, "yearly_ic.png")
    fig.savefig(yearly_path, dpi=120)
    plt.close(fig)
    return series_path, yearly_path


def backtest_all_factors(factor_list=None) -> pd.DataFrame:
    """批量回测 RESULT_ROOT 下已落盘的因子窄表，写 CSV/JSON，并返回摘要表。"""
    names = list(factor_list) if factor_list is not None else list(FACTOR_LIST)
    rows = []
    for fname in names:
        path = os.path.join(RESULT_ROOT, fname, "factor_narrow.parquet")
        try:
            factor_narrow = pd.read_parquet(path)
            group_pnl, group_to, rank_ic, summary = backtest_factor(factor_narrow)
            out_dir = os.path.join(RESULT_ROOT, fname)
            _save_backtest_outputs(out_dir, group_pnl, group_to, rank_ic, summary, factor_name=fname)
            row = {"factor": fname, **{k: v for k, v in summary.items() if k != "group_mean_annu"}}
            rows.append(row)
            logger.info(
                "%s 回测完成 | RankIC=%.4f ICIR=%.2f | H-L年化(翻向)=%.2f%% Sharpe=%.2f",
                fname,
                summary["rank_ic_mean"],
                summary["rank_icir"],
                100 * summary["hl_annu_ret_flipped"],
                summary["hl_sharpe_flipped"],
            )
        except FileNotFoundError:
            logger.warning("%s 因子窄表不存在，跳过回测: %s", fname, path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s 回测失败: %s", fname, exc)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        os.makedirs(RESULT_ROOT, exist_ok=True)
        out_summary = os.path.join(RESULT_ROOT, "phase1_summary.csv")
        summary_df.to_csv(out_summary, index=False)
        logger.info("Phase1 摘要 -> %s", out_summary)
        print(summary_df.to_string(index=False))
    return summary_df
