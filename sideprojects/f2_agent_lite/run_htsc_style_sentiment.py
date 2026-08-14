#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""华泰风格新闻情感调整因子（无外部 ML 依赖）.

研报启发（华泰金融工程 · 文本情感因子）：
  1. senti_adj：负面情感权重 × NEG_MULT（默认 3），抑制正面泛滥噪音
  2. lookback decay：过去 LOOKBACK 个交易日线性衰减加权（近大远小）

数据：``results/finbert_titles_cache.csv`` + 已有词典打分
信号：adj 日分 → 20 日衰减 → CS-Z → 行业中性 → RotationBacktester

用法::

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_htsc_style_sentiment
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
from sideprojects.f2_agent_lite.factors.cn_finance_lexicon import lexicon_stats  # noqa: E402
from sideprojects.f2_agent_lite.run_lexicon_sentiment_standalone import score_text  # noqa: E402

NEG_MULT = 3.0
LOOKBACK = 20
MIN_PERIODS = 5
THREE_LEG_BASELINE = 1.714


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
            print("[htsc] SKIP OHLCV", sym, flush=True)
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


def _ensure_title_scores(result_dir: Path) -> pd.DataFrame:
    """Reuse lexicon title scores if present; else score from titles cache."""
    scored = result_dir / "lexicon_title_scores_cache.csv"
    if scored.exists():
        print("[htsc] 复用词典标题打分 {}".format(scored), flush=True)
        df = pd.read_csv(scored, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "sentiment_score" not in df.columns:
            raise ValueError("lexicon cache missing sentiment_score")
        return df

    cache = result_dir / "finbert_titles_cache.csv"
    if not cache.exists():
        raise FileNotFoundError("missing {}".format(cache))
    print("[htsc] 从标题缓存重新打分...", flush=True)
    df = pd.read_csv(cache, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sentiment_score"] = df["title"].fillna("").map(score_text)
    df.to_csv(scored, index=False)
    return df


def _daily_senti_adj(titles: pd.DataFrame, neg_mult: float = NEG_MULT) -> pd.DataFrame:
    """Huatai-style adjustment: amplify negative title contributions by neg_mult."""
    t = titles.copy()
    raw = pd.to_numeric(t["sentiment_score"], errors="coerce").fillna(0.0)
    # intensity-aware adj: positive ×1, negative ×neg_mult
    t["adj_title"] = np.where(raw >= 0.0, raw, raw * float(neg_mult))
    t["is_pos"] = (raw > 0).astype(int)
    t["is_neg"] = (raw < 0).astype(int)

    if "related_score" in t.columns:
        w = np.log1p(pd.to_numeric(t["related_score"], errors="coerce").fillna(0).clip(lower=0))
        t["w"] = 1.0 + 0.25 * w
    else:
        t["w"] = 1.0
    t["w_adj"] = t["adj_title"] * t["w"]

    g = t.groupby(["date", "symbol"], as_index=False).agg(
        w_adj_sum=("w_adj", "sum"),
        w_sum=("w", "sum"),
        pos_count=("is_pos", "sum"),
        neg_count=("is_neg", "sum"),
        n_titles=("adj_title", "size"),
        raw_mean=("sentiment_score", "mean"),
    )
    g["adj_sum"] = g["w_adj_sum"] / g["w_sum"].clip(lower=1e-8)
    # count-based secondary adj: mild extra penalty when neg share is high
    tot = (g["pos_count"] + g["neg_count"]).clip(lower=1)
    neg_share = g["neg_count"] / tot
    g["daily_adj"] = g["adj_sum"] - 0.5 * neg_share
    return g


def _decay_weighted(series: pd.Series, window: int = LOOKBACK, min_periods: int = MIN_PERIODS) -> pd.Series:
    """Linear decay over trailing window; newest weight=1, oldest≈1/window."""

    def _avg(x: np.ndarray) -> float:
        n = len(x)
        if n < min_periods:
            return np.nan
        w = np.linspace(1.0 / window, 1.0, n)
        return float(np.average(x, weights=w))

    return series.rolling(window, min_periods=min_periods).apply(_avg, raw=True)


def build_htsc_panel(
    cfg: Config,
    symbols: List[str],
    start_date: str,
    end_date: str,
    neg_mult: float = NEG_MULT,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    result_dir = Path(cfg.results_dir)
    titles = _ensure_title_scores(result_dir)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    titles = titles[(titles["date"] >= start_ts) & (titles["date"] <= end_ts)].copy()
    titles = titles[titles["symbol"].isin(symbols)]
    if titles.empty:
        print("[htsc] 标题为空", flush=True)
        return pd.DataFrame()

    print(
        "[htsc] titles={} lexicon={} neg_mult={} lookback={}".format(
            len(titles), lexicon_stats()["n_total"], neg_mult, lookback
        ),
        flush=True,
    )

    daily = _daily_senti_adj(titles, neg_mult=neg_mult)
    daily_path = result_dir / "htsc_daily_adj.csv"
    daily.to_csv(daily_path, index=False)
    print("[htsc] 日频 adj 已写 {}".format(daily_path), flush=True)

    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    exec_df = _load_exec_panel(symbols, start_date, end_plus)
    if exec_df.empty:
        print("[htsc] 无执行价", flush=True)
        return pd.DataFrame()

    # trading calendar per symbol → reindex news days (0 if no news) → decay
    frames = []
    for sym in symbols:
        cal = (
            exec_df.loc[exec_df["symbol"] == sym, ["date"]]
            .drop_duplicates()
            .sort_values("date")
        )
        if cal.empty:
            continue
        sub = daily.loc[daily["symbol"] == sym, ["date", "daily_adj"]].copy()
        merged = cal.merge(sub, on="date", how="left")
        merged["daily_adj"] = merged["daily_adj"].fillna(0.0)
        merged["score_raw"] = _decay_weighted(
            merged["daily_adj"], window=lookback, min_periods=MIN_PERIODS
        )
        merged["symbol"] = sym
        frames.append(merged[["date", "symbol", "score_raw", "daily_adj"]])

    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot(index="date", columns="symbol", values="score_raw").sort_index()
    for sym in symbols:
        if sym not in wide.columns:
            wide[sym] = np.nan
    wide = wide[list(symbols)]
    # early NaNs from min_periods → 0 after CS prep
    wide = wide.fillna(0.0)

    score_wide = _cs_zscore(wide)
    if getattr(cfg, "industry_neutral_rank", True):
        print("[htsc] 应用行业中性化...", flush=True)
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
    print(
        "[htsc] 信号面板: {} 行 / {} 日".format(len(out), out["date"].nunique()),
        flush=True,
    )
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _make_bt(cfg: Config) -> RotationBacktester:
    top_k = cfg.rotation_top_k
    bottom_k = cfg.rotation_bottom_k
    if top_k is None:
        top_k = max(1, int(round(len(cfg.symbols) * float(cfg.rotation_top_frac))))
    if bottom_k is None:
        bottom_k = max(1, int(round(len(cfg.symbols) * float(cfg.rotation_bottom_frac))))
    return RotationBacktester(
        initial_cash=cfg.initial_cash,
        cost_rate=cfg.cost_rate,
        top_k=int(top_k),
        bottom_k=int(bottom_k),
        long_gross=cfg.rotation_long_gross,
        short_gross=cfg.rotation_short_gross,
        rebalance_every=int(cfg.rebalance_every or cfg.pred_horizon or 1),
        use_vol_scaling=False,
    )


def run_htsc_backtest(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[华泰风格] senti_adj×{} + {}日衰减 独立回测".format(int(NEG_MULT), LOOKBACK))
    print("=" * 60)

    panel = build_htsc_panel(cfg, list(cfg.symbols), cfg.train_start, cfg.test_end)
    if panel.empty:
        return {"error": "信号面板为空"}

    test_start = pd.Timestamp(cfg.test_start).normalize()
    test_end = pd.Timestamp(cfg.test_end).normalize()
    test_panel = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    if test_panel.empty:
        return {"error": "测试期无数据"}

    result = _make_bt(cfg).run(test_panel)
    metrics = result.metrics or {}
    bh = result.equal_weight_bh_metrics or {}
    avg_to = float(result.equity["turnover"].mean()) if not result.equity.empty else 0.0
    sr = float(metrics.get("sharpe", 0.0))
    ann = float(metrics.get("annualized_return", 0.0))

    if sr >= 0.4 and ann > 0.0:
        verdict = "有基础Alpha"
    elif sr >= 0.2 and ann >= 0.0:
        verdict = "微弱信号"
    else:
        verdict = "无效"

    print("\n[华泰风格情感独立回测结果]", flush=True)
    print("  年化收益: {:.2f}%".format(100 * ann))
    print("  夏普比率: {:.3f}".format(sr))
    print("  最大回撤: {:.2f}%".format(100 * float(metrics.get("max_drawdown", 0.0))))
    print("  日均换手: {:.2f}%".format(100 * avg_to))
    print(
        "  EW-BH 年化: {:.2f}% 夏普: {:.3f}".format(
            100 * float(bh.get("annualized_return", 0.0)), float(bh.get("sharpe", 0.0))
        )
    )
    print("  判定: {}".format(verdict), flush=True)

    result_dir = Path(cfg.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if not result.equity.empty:
        result.equity.to_csv(result_dir / "htsc_style_equity.csv")
    test_panel.to_csv(result_dir / "htsc_style_sentiment_signals.csv", index=False)
    summary = {
        "scheme": "htsc_style_senti_adj_decay",
        "verdict": verdict,
        "params": {
            "neg_mult": NEG_MULT,
            "lookback_trading_days": LOOKBACK,
            "min_periods": MIN_PERIODS,
            "lexicon": lexicon_stats(),
        },
        "strategy": metrics,
        "equal_weight_bh": bh,
        "avg_daily_turnover": avg_to,
        "selection_stats": result.selection_stats,
        "score_formula": "lexicon → neg×{} → {}d linear-decay → cs_z → industry demean".format(
            int(NEG_MULT), LOOKBACK
        ),
        "n_signal_rows": int(len(test_panel)),
        "n_signal_days": int(test_panel["date"].nunique()),
    }
    with open(result_dir / "htsc_style_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("\n[结果] {}".format(result_dir / "htsc_style_summary.json"), flush=True)
    return summary


def _load_rets(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df["ret"].astype(float) if "ret" in df.columns else df["equity"].pct_change().fillna(0.0)
    s.index = pd.to_datetime(s.index).normalize()
    return s


def _ann(r: pd.Series) -> float:
    r = r.dropna().astype(float)
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0) if len(r) else 0.0


def _sharpe(r: pd.Series) -> float:
    r = r.dropna().astype(float)
    if len(r) < 5 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _cs_z_col(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd is None or sd < 1e-12 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def run_four_leg_htsc_combination(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[四源组合-华泰风格] G4 + 分钟 + VQ + HTSC-adj")
    print("=" * 60)
    result_dir = Path(cfg.results_dir)
    equity_paths = {
        "g4": result_dir / "grid_G4_equity.csv",
        "minute": result_dir / "minute_alpha_equity.csv",
        "vq": result_dir / "value_quality_equity.csv",
        "htsc": result_dir / "htsc_style_equity.csv",
    }
    for name, p in equity_paths.items():
        if not p.exists():
            return {"error": "missing equity {}".format(p)}

    series = {k: _load_rets(p) for k, p in equity_paths.items()}
    idx = series["g4"].index
    for s in series.values():
        idx = idx.intersection(s.index)
    aligned = {k: v.loc[idx] for k, v in series.items()}
    corr = pd.DataFrame(aligned).corr()
    eqw = sum(aligned.values()) / 4.0
    vols = {k: max(float(v.std()), 1e-8) for k, v in aligned.items()}
    inv = {k: 1.0 / vols[k] for k in vols}
    zsum = sum(inv.values())
    w = {k: inv[k] / zsum for k in inv}
    rp = sum(w[k] * aligned[k] for k in aligned)

    print("  重叠日: {}".format(len(idx)))
    print("  相关矩阵:\n{}".format(corr.round(4).to_string()))
    legs = {}
    for name, rr in [
        *[(k, aligned[k]) for k in aligned],
        ("equal_weight", eqw),
        ("risk_parity", rp),
    ]:
        legs[name] = {"ann": _ann(rr), "sharpe": _sharpe(rr), "mdd": _mdd(rr)}
        print(
            "  {:>12}: ann={:+.2f}% sharpe={:.3f} mdd={:.2f}%".format(
                name, 100 * legs[name]["ann"], legs[name]["sharpe"], 100 * legs[name]["mdd"]
            )
        )
    legs["risk_parity"]["w"] = w
    legs["equal_weight"]["w"] = {k: 0.25 for k in aligned}

    out_eq = pd.DataFrame({**{f"ret_{k}": v for k, v in aligned.items()}, "ret_eqw": eqw, "ret_rp": rp})
    out_eq["equity_eqw"] = (1.0 + eqw.fillna(0.0)).cumprod()
    out_eq.to_csv(result_dir / "four_leg_htsc_equity_level.csv")

    signal_files = {
        "g4": result_dir / "g4_signals.csv",
        "minute": result_dir / "minute_alpha_signals.csv",
        "vq": result_dir / "value_quality_signals.csv",
        "htsc": result_dir / "htsc_style_sentiment_signals.csv",
    }
    frames = {k: pd.read_csv(p, parse_dates=["date", "next_date"]) for k, p in signal_files.items()}
    for df in frames.values():
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    m = frames["g4"][["date", "symbol", "score"]].rename(columns={"score": "score_g4"})
    for key, col in [
        ("minute", "score_minute"),
        ("vq", "score_vq"),
        ("htsc", "score_htsc"),
    ]:
        m = m.merge(
            frames[key][["date", "symbol", "score"]].rename(columns={"score": col}),
            on=["date", "symbol"],
            how="inner",
        )
    for col in ["score_g4", "score_minute", "score_vq", "score_htsc"]:
        m["z_" + col] = m.groupby("date")[col].transform(_cs_z_col)
    m["score"] = (
        m["z_score_g4"] + m["z_score_minute"] + m["z_score_vq"] + m["z_score_htsc"]
    ) / 4.0

    score_corrs = {}
    for a, b in [
        ("g4", "minute"),
        ("g4", "vq"),
        ("g4", "htsc"),
        ("minute", "vq"),
        ("minute", "htsc"),
        ("vq", "htsc"),
    ]:
        c = (
            m.groupby("date")
            .apply(
                lambda g, x=a, y=b: g["z_score_" + x].corr(g["z_score_" + y]),
                include_groups=False,
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .mean()
        )
        score_corrs["{}_{}".format(a, b)] = float(c) if pd.notna(c) else float("nan")

    full = m.merge(
        frames["minute"][
            ["date", "next_date", "symbol", "open_px", "close_px", "next_close_px", "tradable_exec"]
        ],
        on=["date", "symbol"],
        how="inner",
    )
    signal = full[
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
    ].dropna(subset=["score", "open_px", "next_close_px", "next_date"])
    signal = signal[signal["tradable_exec"].astype(bool)]

    res = _make_bt(cfg).run(signal)
    metrics = res.metrics or {}
    avg_to = float(res.equity["turnover"].mean()) if not res.equity.empty else 0.0
    sr = float(metrics.get("sharpe", 0.0))

    print("\n[四源信号组合-华泰风格] equal z-average")
    print("  分数相关: {}".format({k: round(v, 4) for k, v in score_corrs.items()}))
    print(
        "  ann={:+.2f}% sharpe={:.3f} mdd={:.2f}% turnover={:.2f}%".format(
            100 * float(metrics.get("annualized_return", 0.0)),
            sr,
            100 * float(metrics.get("max_drawdown", 0.0)),
            100 * avg_to,
        )
    )
    if not res.equity.empty:
        res.equity.to_csv(result_dir / "four_leg_htsc_signal_equity.csv")
    signal.to_csv(result_dir / "four_leg_htsc_signals.csv", index=False)

    eqw_sr = float(legs["equal_weight"]["sharpe"])
    summary = {
        "scheme": "four_leg_g4_minute_vq_htsc",
        "equity_level": {"n_days": int(len(idx)), "corr": corr.to_dict(), "legs": legs},
        "signal_level": {
            "score_corrs": score_corrs,
            "strategy": metrics,
            "equal_weight_bh": res.equal_weight_bh_metrics,
            "selection_stats": res.selection_stats,
            "avg_daily_turnover": avg_to,
        },
        "headline": {
            "equity_equal_weight_sharpe": eqw_sr,
            "equity_risk_parity_sharpe": legs["risk_parity"]["sharpe"],
            "signal_equal_weight_sharpe": sr,
            "hit_sharpe_1_8": bool(eqw_sr >= 1.8 or sr >= 1.8),
            "hit_sharpe_2_0": bool(eqw_sr >= 2.0 or sr >= 2.0),
            "three_leg_baseline_sharpe": THREE_LEG_BASELINE,
            "beats_three_leg": bool(eqw_sr > THREE_LEG_BASELINE),
        },
    }
    with open(result_dir / "four_leg_htsc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("[总览]")
    print("  权益等权夏普 = {:.3f}  (三源基线 {:.3f})".format(eqw_sr, THREE_LEG_BASELINE))
    print("  权益风险平价 = {:.3f}".format(legs["risk_parity"]["sharpe"]))
    print("  信号等权夏普 = {:.3f}".format(sr))
    print(
        "  ≥1.8? {}  ≥2.0? {}  优于三源? {}".format(
            summary["headline"]["hit_sharpe_1_8"],
            summary["headline"]["hit_sharpe_2_0"],
            summary["headline"]["beats_three_leg"],
        )
    )
    print("[结果] {}".format(result_dir / "four_leg_htsc_summary.json"))
    return summary


if __name__ == "__main__":
    cfg = Config()
    cfg.use_minute_factors = False
    cfg.use_north_money = False
    cfg.use_fundamentals = False
    cfg.use_advanced_alpha = False
    cfg.use_market_risk = False
    cfg.use_vol_scaling = False
    cfg.industry_neutral_rank = True
    cfg.label_threshold = 0.015
    cfg.rebalance_every = 7

    out = run_htsc_backtest(cfg)
    if "error" not in out and (
        Path(cfg.results_dir) / "htsc_style_sentiment_signals.csv"
    ).exists():
        run_four_leg_htsc_combination(cfg)
    elif "error" in out:
        print("[阻断] {}".format(out), flush=True)
        sys.exit(2)
