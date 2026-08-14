#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案A：分钟因子独立子策略回测（不经过 Transformer）。

- 数据源：ClickHouse 分钟 K 线 → 日频 minute_amplitude / price_jump
- 信号：截面 Z-Score + 可选行业中性，等权合成
- 回测：复用 RotationBacktester（与 G4 相同执行口径）

用法::

    cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_minute_alpha_standalone
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402
from sideprojects.f2_agent_lite.data import db_connector as db  # noqa: E402
from sideprojects.f2_agent_lite.data.industry import SYMBOL_INDUSTRY  # noqa: E402


def _cs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cross-sectional z-score (ddof=0); constant rows → 0."""
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
    out = wide.sub(mu, axis=0).div(sd, axis=0)
    return out.fillna(0.0)


def _industry_demean(score_wide: pd.DataFrame) -> pd.DataFrame:
    """Demean scores within industry on each day (singleton industries unchanged)."""
    out = score_wide.copy()
    for d in out.index:
        row = out.loc[d]
        industries = {s: SYMBOL_INDUSTRY.get(s, "其他") for s in row.index}
        adj = row.copy()
        for ind in set(industries.values()):
            cols = [s for s, i in industries.items() if i == ind]
            if len(cols) >= 2:
                adj[cols] = row[cols] - float(row[cols].mean())
        out.loc[d] = adj
    return out.fillna(0.0)


def _load_exec_panel(symbols: Sequence[str], start, end) -> pd.DataFrame:
    """OHLCV + tradability → long frame with next-open execution fields."""
    frames = []
    for sym in symbols:
        ohlcv = db.get_ohlcv(sym, start, end)
        if ohlcv is None or ohlcv.empty:
            print("[minute-alpha] SKIP OHLCV", sym, flush=True)
            continue
        trad = db.compute_tradability_from_ohlcv(ohlcv)
        daily = ohlcv.set_index("date").sort_index()
        if not trad.empty:
            daily = daily.join(trad.set_index("date")[["tradable"]], how="left")
        else:
            daily["tradable"] = np.nan
        daily["next_open"] = daily["open"].shift(-1)
        daily["next_close"] = daily["close"].shift(-1)
        daily["next_date"] = daily.index.to_series().shift(-1)
        daily["next_tradable"] = daily["tradable"].shift(-1)
        part = daily.reset_index()
        part["symbol"] = sym
        frames.append(
            part[
                [
                    "date",
                    "symbol",
                    "close",
                    "next_open",
                    "next_close",
                    "next_date",
                    "next_tradable",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["next_date"] = pd.to_datetime(out["next_date"])
    return out


def build_minute_score_panel(
    cfg: Config,
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Build RotationBacktester signal table from minute factors only."""
    print("[minute-alpha] 从 ClickHouse 拉取分钟因子...", flush=True)
    minute_df = db.get_minute_factors(
        symbols,
        start_date,
        end_date,
        lookback=int(getattr(cfg, "minute_factor_lookback", 10)),
        use_local_tables=bool(getattr(cfg, "minute_use_local_tables", False)),
        config=cfg,
    )
    if minute_df.empty:
        print("[minute-alpha] 警告：未获取到分钟因子数据", flush=True)
        return pd.DataFrame()
    print("[minute-alpha] 分钟因子 shape={}".format(minute_df.shape), flush=True)

    print("[minute-alpha] 加载执行价 / 可交易性 (DolphinDB EOD)...", flush=True)
    # Extra day so next_open/next_close exist through end_date
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    exec_df = _load_exec_panel(symbols, start_date, end_plus)
    if exec_df.empty:
        print("[minute-alpha] 无基础市场数据", flush=True)
        return pd.DataFrame()

    minute_df = minute_df.copy()
    minute_df["date"] = pd.to_datetime(minute_df["date"]).dt.normalize()
    amp_wide = minute_df.pivot(index="date", columns="symbol", values="minute_amplitude")
    jump_wide = minute_df.pivot(index="date", columns="symbol", values="price_jump")

    close_wide = exec_df.pivot(index="date", columns="symbol", values="close")
    common = amp_wide.index.intersection(jump_wide.index).intersection(close_wide.index)
    if len(common) == 0:
        print("[minute-alpha] 日期未对齐", flush=True)
        return pd.DataFrame()

    amp_wide = amp_wide.loc[common]
    jump_wide = jump_wide.loc[common]

    print("[minute-alpha] 构建截面排序分数...", flush=True)
    # minute_amplitude 在 factor_minute 中已取负（高分=低振幅=文献正向）
    # price_jump 仍为有符号跳跃强度；按用户口径对 jump 取负后等权合成
    amp_z = _cs_zscore(amp_wide)
    jump_z = _cs_zscore(jump_wide)
    score_wide = 0.5 * amp_z - 0.5 * jump_z

    if getattr(cfg, "industry_neutral_rank", True):
        print("[minute-alpha] 应用行业中性化...", flush=True)
        score_wide = _industry_demean(score_wide)

    score_long = score_wide.stack(future_stack=True).rename("score").reset_index()
    score_long.columns = ["date", "symbol", "score"]
    score_long["date"] = pd.to_datetime(score_long["date"]).dt.normalize()

    merged = score_long.merge(exec_df, on=["date", "symbol"], how="inner")
    merged = merged.rename(
        columns={
            "close": "close_px",
            "next_open": "open_px",
            "next_close": "next_close_px",
        }
    )
    merged["tradable_exec"] = merged["next_tradable"].notna()
    merged = merged.dropna(subset=["score", "open_px", "next_close_px", "next_date"])
    merged = merged[merged["tradable_exec"]].copy()
    merged = merged[
        [
            "date",
            "next_date",
            "symbol",
            "score",
            "open_px",
            "close_px",
            "next_close_px",
            "tradable_exec",
        ]
    ]
    print("[minute-alpha] 信号面板: {} 行 / {} 日".format(
        len(merged), merged["date"].nunique()
    ), flush=True)
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def run_minute_alpha_backtest(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[方案A] 分钟因子独立子策略回测")
    print("=" * 60)

    panel = build_minute_score_panel(
        cfg=cfg,
        symbols=list(cfg.symbols),
        start_date=cfg.train_start,
        end_date=cfg.test_end,
    )
    if panel.empty:
        return {"error": "信号面板为空"}

    test_start = pd.Timestamp(cfg.test_start).normalize()
    test_end = pd.Timestamp(cfg.test_end).normalize()
    test_panel = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    if test_panel.empty:
        return {"error": "测试期无数据"}
    print(
        "[minute-alpha] 测试期 {} ~ {} 样本={} 日={}".format(
            test_start.date(),
            test_end.date(),
            len(test_panel),
            test_panel["date"].nunique(),
        ),
        flush=True,
    )

    top_k = cfg.rotation_top_k
    bottom_k = cfg.rotation_bottom_k
    if top_k is None:
        top_k = max(1, int(round(len(cfg.symbols) * float(cfg.rotation_top_frac))))
    if bottom_k is None:
        bottom_k = max(1, int(round(len(cfg.symbols) * float(cfg.rotation_bottom_frac))))

    backtester = RotationBacktester(
        initial_cash=cfg.initial_cash,
        cost_rate=cfg.cost_rate,
        top_k=int(top_k),
        bottom_k=int(bottom_k),
        long_gross=cfg.rotation_long_gross,
        short_gross=cfg.rotation_short_gross,
        rebalance_every=int(cfg.rebalance_every or cfg.pred_horizon or 1),
        use_vol_scaling=False,
    )
    result = backtester.run(test_panel)
    metrics = result.metrics or {}
    bh = result.equal_weight_bh_metrics or {}
    avg_to = float(result.equity["turnover"].mean()) if not result.equity.empty else 0.0

    print("\n[分钟因子独立回测结果]", flush=True)
    print("  年化收益: {:.2f}%".format(100.0 * float(metrics.get("annualized_return", 0.0))))
    print("  夏普比率: {:.3f}".format(float(metrics.get("sharpe", 0.0))))
    print("  最大回撤: {:.2f}%".format(100.0 * float(metrics.get("max_drawdown", 0.0))))
    print("  日均换手: {:.2f}%".format(100.0 * avg_to))
    print(
        "  EW-BH 年化: {:.2f}% 夏普: {:.3f}".format(
            100.0 * float(bh.get("annualized_return", 0.0)),
            float(bh.get("sharpe", 0.0)),
        )
    )

    long_top = sorted(
        (result.selection_stats.get("long_pick_counts") or {}).items(),
        key=lambda x: -x[1],
    )[:3]
    short_top = sorted(
        (result.selection_stats.get("short_pick_counts") or {}).items(),
        key=lambda x: -x[1],
    )[:3]
    print("  常做多: {}".format([s for s, _ in long_top]))
    print("  常做空: {}".format([s for s, _ in short_top]))

    sharpe = float(metrics.get("sharpe", 0.0))
    ann = float(metrics.get("annualized_return", 0.0))
    if sharpe > 0.5 and ann > 0.05:
        verdict = "优秀"
    elif sharpe >= 0.1 and ann >= 0.0:
        verdict = "合格"
    else:
        verdict = "无效"
    print("  判定: {}".format(verdict), flush=True)

    result_dir = Path(cfg.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scheme": "minute_alpha_standalone",
        "verdict": verdict,
        "config": cfg.to_dict(),
        "minute_alpha": {
            "annualized_return": metrics.get("annualized_return"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "total_return": metrics.get("total_return"),
            "avg_daily_turnover": avg_to,
            "equal_weight_bh": bh,
            "selection_stats": result.selection_stats,
            "n_signal_rows": int(len(test_panel)),
            "n_signal_days": int(test_panel["date"].nunique()),
            "score_formula": "0.5*cs_z(minute_amplitude_negated) - 0.5*cs_z(price_jump); industry demean",
        },
    }
    if not result.equity.empty:
        result.equity.to_csv(result_dir / "minute_alpha_equity.csv")
    test_panel.to_csv(result_dir / "minute_alpha_signals.csv", index=False)
    with open(result_dir / "minute_alpha_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("\n[结果] {}".format(result_dir / "minute_alpha_summary.json"), flush=True)
    return summary


if __name__ == "__main__":
    cfg = Config()
    cfg.use_minute_factors = True
    cfg.use_north_money = False
    cfg.use_fundamentals = False
    cfg.use_advanced_alpha = False
    cfg.use_market_risk = False
    cfg.use_vol_scaling = False
    cfg.industry_neutral_rank = True
    run_minute_alpha_backtest(cfg)
