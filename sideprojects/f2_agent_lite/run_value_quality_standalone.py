#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第三条腿：独立价值/质量因子策略（ROE + EP + 北向）。

- 数据：DolphinDB Wind 估值 / TTMHIS ROE（PIT）/ 陆股通持仓
- 信号：截面 Z-Score + 行业中性，等权合成
- 回测：RotationBacktester（与分钟腿同口径）

用法::

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_value_quality_standalone
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)


def _industry_demean(score_wide: pd.DataFrame) -> pd.DataFrame:
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
    frames = []
    for sym in symbols:
        ohlcv = db.get_ohlcv(sym, start, end)
        if ohlcv is None or ohlcv.empty:
            print("[vq-alpha] SKIP OHLCV", sym, flush=True)
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


def _load_factor_long(symbols: Sequence[str], start, end) -> pd.DataFrame:
    """Pull EP / ROE / north_share_ratio per symbol into long frame."""
    rows = []
    for sym in symbols:
        print("[vq-alpha] fundamentals", sym, "...", flush=True)
        try:
            val = db.get_valuation(sym, start, end)
        except Exception as exc:
            print("[vq-alpha] valuation failed {}: {}".format(sym, exc), flush=True)
            val = pd.DataFrame(columns=["date", "ep_ttm"])
        try:
            fund = db.get_fundamentals_pit(sym, start, end)
        except Exception as exc:
            print("[vq-alpha] fundamentals_pit failed {}: {}".format(sym, exc), flush=True)
            fund = pd.DataFrame(columns=["date", "roe"])
        try:
            north = db.get_northbound(sym, start, end)
        except Exception as exc:
            print("[vq-alpha] northbound failed {}: {}".format(sym, exc), flush=True)
            north = pd.DataFrame(columns=["date", "north_share_ratio", "north_share_chg"])

        # Align on valuation calendar (daily)
        if val is None or val.empty:
            continue
        daily = val[["date", "ep_ttm"]].copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        daily = daily.sort_values("date").drop_duplicates("date", keep="last")

        if fund is not None and not fund.empty:
            f = fund[["date", "roe"]].copy()
            f["date"] = pd.to_datetime(f["date"]).dt.normalize()
            f = f.sort_values("date").drop_duplicates("date", keep="last")
            # PIT frame already daily-ffilled; merge + ffill defensively
            daily = daily.merge(f, on="date", how="left")
            daily["roe"] = daily["roe"].ffill()
        else:
            daily["roe"] = np.nan

        if north is not None and not north.empty:
            n = north[["date", "north_share_ratio", "north_share_chg"]].copy()
            n["date"] = pd.to_datetime(n["date"]).dt.normalize()
            n = n.sort_values("date").drop_duplicates("date", keep="last")
            daily = daily.merge(n, on="date", how="left")
            daily["north_share_ratio"] = daily["north_share_ratio"].ffill()
            daily["north_share_chg"] = daily["north_share_chg"].fillna(0.0)
        else:
            daily["north_share_ratio"] = np.nan
            daily["north_share_chg"] = np.nan

        daily["symbol"] = sym
        rows.append(
            daily[
                [
                    "date",
                    "symbol",
                    "ep_ttm",
                    "roe",
                    "north_share_ratio",
                    "north_share_chg",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=["date", "symbol", "ep_ttm", "roe", "north_share_ratio", "north_share_chg"]
        )
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for c in ["ep_ttm", "roe", "north_share_ratio", "north_share_chg"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def orthogonalize_to_minute(
    score_wide: pd.DataFrame,
    minute_score_wide: pd.DataFrame,
    window: int = 60,
    min_obs: int = 30,
    method: str = "cs_daily",
) -> pd.DataFrame:
    """Strip minute-factor collinearity from VQ scores.

    method:
      - ``cs_daily``: same-day cross-sectional residual (guarantees ρ≈0)
      - ``rolling``: causal pooled β over past ``window`` days, then VQ−β·Minute
    """
    common_dates = score_wide.index.intersection(minute_score_wide.index)
    common_symbols = score_wide.columns.intersection(minute_score_wide.columns)
    if len(common_dates) == 0 or len(common_symbols) == 0:
        return score_wide

    score_aligned = score_wide.reindex(columns=common_symbols)
    minute_aligned = minute_score_wide.reindex(
        index=score_wide.index, columns=common_symbols
    )
    residuals = score_aligned.copy()

    if method == "cs_daily":
        n_ok = 0
        for d in common_dates:
            y = score_aligned.loc[d].astype(float)
            x = minute_aligned.loc[d].astype(float)
            mask = y.notna() & x.notna()
            if int(mask.sum()) < 3:
                continue
            yy = y[mask].to_numpy()
            xx = x[mask].to_numpy()
            xx_c = xx - xx.mean()
            yy_c = yy - yy.mean()
            var_x = float(np.dot(xx_c, xx_c))
            if var_x < 1e-12:
                residuals.loc[d, mask] = yy_c  # demean only
            else:
                beta = float(np.dot(xx_c, yy_c) / var_x)
                resid = yy_c - beta * xx_c
                residuals.loc[d, mask] = resid
            n_ok += 1
        print(
            "[vq-alpha] 正交化完成(cs_daily): overlap_days={} residualized={}".format(
                len(common_dates), n_ok
            ),
            flush=True,
        )
    else:
        dates = list(score_aligned.index)
        date_to_i = {d: i for i, d in enumerate(dates)}
        betas = []
        for d in common_dates:
            i = date_to_i[d]
            start_i = max(0, i - int(window))
            if i - start_i < 5:
                continue
            y_block = score_aligned.iloc[start_i:i]
            x_block = minute_aligned.iloc[start_i:i]
            mask = y_block.notna() & x_block.notna()
            y = y_block.where(mask).to_numpy(dtype=float).ravel()
            x = x_block.where(mask).to_numpy(dtype=float).ravel()
            ok = np.isfinite(y) & np.isfinite(x)
            y, x = y[ok], x[ok]
            if len(y) < int(min_obs) or float(np.std(x)) < 1e-8:
                continue
            x_c = x - x.mean()
            y_c = y - y.mean()
            var_x = float(np.dot(x_c, x_c))
            if var_x < 1e-12:
                continue
            beta = float(np.dot(x_c, y_c) / var_x)
            betas.append(beta)
            residuals.loc[d] = score_aligned.loc[d] - beta * minute_aligned.loc[d]
        mean_beta = float(np.mean(betas)) if betas else float("nan")
        print(
            "[vq-alpha] 正交化完成(rolling): overlap_days={} betas={} mean_β={:.4f}".format(
                len(common_dates), len(betas), mean_beta
            ),
            flush=True,
        )

    out = score_wide.copy()
    out.loc[:, common_symbols] = residuals.loc[:, common_symbols]
    return out


def _load_minute_score_wide(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.results_dir) / "minute_alpha_signals.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.pivot(index="date", columns="symbol", values="score")


def build_value_quality_panel(
    cfg: Config,
    symbols: List[str],
    start_date: str,
    end_date: str,
    *,
    orthogonalize: bool = True,
    ortho_window: int = 60,
) -> pd.DataFrame:
    """Build VQ signal table for RotationBacktester."""
    print("[vq-alpha] 加载基本面与北向数据...", flush=True)
    hist_start = (pd.Timestamp(start_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    factor_df = _load_factor_long(symbols, hist_start, end_date)
    if factor_df.empty:
        print("[vq-alpha] 警告：无基本面或北向数据", flush=True)
        return pd.DataFrame()
    print("[vq-alpha] factor long shape={}".format(factor_df.shape), flush=True)

    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    exec_df = _load_exec_panel(symbols, start_date, end_plus)
    if exec_df.empty:
        print("[vq-alpha] 无执行价数据", flush=True)
        return pd.DataFrame()

    ep_wide = factor_df.pivot(index="date", columns="symbol", values="ep_ttm")
    roe_wide = factor_df.pivot(index="date", columns="symbol", values="roe")
    north_wide = factor_df.pivot(index="date", columns="symbol", values="north_share_ratio")
    north_chg_wide = factor_df.pivot(index="date", columns="symbol", values="north_share_chg")

    close_idx = exec_df.pivot(index="date", columns="symbol", values="close").index
    common = (
        ep_wide.index.intersection(roe_wide.index)
        .intersection(north_wide.index)
        .intersection(close_idx)
    )
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    common = common[(common >= start_ts) & (common <= end_ts)]
    if len(common) == 0:
        print("[vq-alpha] 日期未对齐", flush=True)
        return pd.DataFrame()

    ep_z = _cs_zscore(ep_wide.loc[common])
    roe_z = _cs_zscore(roe_wide.loc[common])
    north_z = _cs_zscore(north_wide.loc[common])
    north_chg_z = _cs_zscore(north_chg_wide.reindex(common).fillna(0.0))

    score_wide = (ep_z + roe_z + north_z) / 3.0 + 0.25 * north_chg_z

    ortho_applied = False
    if orthogonalize:
        minute_wide = _load_minute_score_wide(cfg)
        if minute_wide.empty:
            print("[vq-alpha] 警告：未找到分钟因子信号，跳过正交化", flush=True)
        else:
            score_wide = orthogonalize_to_minute(
                score_wide,
                minute_wide,
                window=int(ortho_window),
                method="cs_daily",
            )
            ortho_applied = True

    if getattr(cfg, "industry_neutral_rank", True):
        print("[vq-alpha] 应用行业中性化...", flush=True)
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
    out = merged[
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
    out.attrs["orthogonalized"] = ortho_applied
    print(
        "[vq-alpha] 信号面板: {} 行 / {} 日 (orthogonalized={})".format(
            len(out), out["date"].nunique(), ortho_applied
        ),
        flush=True,
    )
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def run_value_quality_backtest(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[第三条腿] 价值/质量因子独立子策略回测")
    print("=" * 60)

    panel = build_value_quality_panel(
        cfg=cfg,
        symbols=list(cfg.symbols),
        start_date=cfg.train_start,
        end_date=cfg.test_end,
        orthogonalize=("--ortho" in sys.argv),
        ortho_window=60,
    )
    if panel.empty:
        return {"error": "信号面板为空"}
    ortho_flag = bool(panel.attrs.get("orthogonalized", False))

    test_start = pd.Timestamp(cfg.test_start).normalize()
    test_end = pd.Timestamp(cfg.test_end).normalize()
    test_panel = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    if test_panel.empty:
        return {"error": "测试期无数据"}
    print(
        "[vq-alpha] 测试期 {} ~ {} 样本={} 日={}".format(
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

    sharpe = float(metrics.get("sharpe", 0.0))
    ann = float(metrics.get("annualized_return", 0.0))
    if sharpe > 0.5 and ann > 0.05:
        verdict = "优秀"
    elif sharpe >= 0.1 and ann >= 0.0:
        verdict = "合格"
    else:
        verdict = "无效"

    print("\n[价值/质量因子独立回测结果]", flush=True)
    print("  年化收益: {:.2f}%".format(100.0 * ann))
    print("  夏普比率: {:.3f}".format(sharpe))
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
    print("  判定: {}".format(verdict), flush=True)
    print("  正交化: {}".format(ortho_flag), flush=True)

    result_dir = Path(cfg.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if not result.equity.empty:
        result.equity.to_csv(result_dir / "value_quality_equity.csv")
    test_panel.to_csv(result_dir / "value_quality_signals.csv", index=False)
    summary = {
        "scheme": "value_quality_standalone",
        "orthogonalized_to_minute": ortho_flag,
        "ortho_window": 60,
        "verdict": verdict,
        "strategy": metrics,
        "equal_weight_bh": bh,
        "avg_daily_turnover": avg_to,
        "selection_stats": result.selection_stats,
        "score_formula": "(cs_z(ep)+cs_z(roe)+cs_z(north_ratio))/3 + 0.25*cs_z(north_chg); ortho vs minute (cs_daily residual); industry demean",
        "ortho_method": "cs_daily",
        "n_signal_rows": int(len(test_panel)),
        "n_signal_days": int(test_panel["date"].nunique()),
    }
    with open(result_dir / "value_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("\n[结果] {}".format(result_dir / "value_quality_summary.json"), flush=True)
    return summary


if __name__ == "__main__":
    cfg = Config()
    cfg.use_north_money = True
    cfg.use_fundamentals = True
    cfg.use_advanced_alpha = False
    cfg.use_market_risk = False
    cfg.use_minute_factors = False
    cfg.use_vol_scaling = False
    cfg.industry_neutral_rank = True
    cfg.label_threshold = 0.015
    cfg.rebalance_every = 7
    run_value_quality_backtest(cfg)
