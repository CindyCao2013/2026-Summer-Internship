#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""替代方案：中文金融情感词典日频 Alpha（零外部 ML 依赖）.

输入：``results/finbert_titles_cache.csv``（已缓存的新闻标题）
打分：最长匹配词典 + 否定反转 + 程度副词
信号：日均 → ffill(3) → CS-Z → 行业中性 → RotationBacktester

用法::

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_lexicon_sentiment_standalone
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
from sideprojects.f2_agent_lite.factors.cn_finance_lexicon import (  # noqa: E402
    INTENSIFIERS,
    INT_TERMS_SORTED,
    LEX_TERMS_SORTED,
    NEG_TERMS_SORTED,
    SENTIMENT_LEXICON,
    lexicon_stats,
)


def score_text(text: str) -> float:
    """最长匹配词典打分，输出约在 [-1, 1]."""
    if not text or not isinstance(text, str):
        return 0.0
    s = text.strip()
    if not s:
        return 0.0

    n = len(s)
    used = [False] * n
    hits: List[tuple] = []  # (start, end, score)

    for term in LEX_TERMS_SORTED:
        tlen = len(term)
        if tlen == 0 or tlen > n:
            continue
        start = 0
        while True:
            i = s.find(term, start)
            if i < 0:
                break
            j = i + tlen
            if not any(used[i:j]):
                for k in range(i, j):
                    used[k] = True
                hits.append((i, j, float(SENTIMENT_LEXICON[term])))
            start = i + 1

    if not hits:
        return 0.0

    total = 0.0
    for i, j, raw in hits:
        window = s[max(0, i - 6) : i]
        polarity = 1.0
        for neg in NEG_TERMS_SORTED:
            if window.endswith(neg):
                polarity = -1.0
                break
        intens = 1.0
        for it in INT_TERMS_SORTED:
            if window.endswith(it) or it in window[-len(it) - 2 :]:
                intens = float(INTENSIFIERS[it])
                break
        total += polarity * intens * raw

    return float(np.clip(total / 8.0, -1.0, 1.0))


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
            print("[lexicon] SKIP OHLCV", sym, flush=True)
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


def build_lexicon_panel(cfg: Config, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    result_dir = Path(cfg.results_dir)
    cache = result_dir / "finbert_titles_cache.csv"
    if not cache.exists():
        print("[lexicon] 缺少标题缓存: {}".format(cache), flush=True)
        return pd.DataFrame()

    titles = pd.read_csv(cache, parse_dates=["date"])
    titles["date"] = pd.to_datetime(titles["date"]).dt.normalize()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    titles = titles[(titles["date"] >= start_ts) & (titles["date"] <= end_ts)].copy()
    if titles.empty:
        print("[lexicon] 过滤后标题为空", flush=True)
        return pd.DataFrame()

    stats = lexicon_stats()
    print(
        "[lexicon] 词典 size={} (pos={}, neg={}); 标题={}".format(
            stats["n_total"], stats["n_pos"], stats["n_neg"], len(titles)
        ),
        flush=True,
    )

    print("[lexicon] 打分中...", flush=True)
    titles["sentiment_score"] = titles["title"].fillna("").map(score_text)
    if "related_score" in titles.columns:
        w = np.log1p(pd.to_numeric(titles["related_score"], errors="coerce").fillna(0).clip(lower=0))
        titles["sent_w"] = titles["sentiment_score"] * (1.0 + 0.25 * w)
        value_col = "sent_w"
    else:
        value_col = "sentiment_score"

    nz = float((titles["sentiment_score"] != 0).mean())
    print(
        "[lexicon] 非零标题占比={:.1%}  score_mean={:.4f} std={:.4f}".format(
            nz,
            float(titles["sentiment_score"].mean()),
            float(titles["sentiment_score"].std()),
        ),
        flush=True,
    )

    # persist title-level scores for QA
    scored_path = result_dir / "lexicon_title_scores_cache.csv"
    titles.to_csv(scored_path, index=False)
    print("[lexicon] 标题打分缓存 {}".format(scored_path), flush=True)

    agg = (
        titles.groupby(["date", "symbol"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "sentiment"})
    )
    wide = agg.pivot(index="date", columns="symbol", values="sentiment").sort_index()
    # align to full calendar of symbols via reindex columns
    for sym in symbols:
        if sym not in wide.columns:
            wide[sym] = np.nan
    wide = wide[list(symbols)]
    wide = wide.ffill(limit=3).fillna(0.0)

    score_wide = _cs_zscore(wide)
    if getattr(cfg, "industry_neutral_rank", True):
        print("[lexicon] 应用行业中性化...", flush=True)
        score_wide = _industry_demean(score_wide)

    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    exec_df = _load_exec_panel(symbols, start_date, end_plus)
    if exec_df.empty:
        print("[lexicon] 无执行价数据", flush=True)
        return pd.DataFrame()

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
        "[lexicon] 信号面板: {} 行 / {} 日".format(len(out), out["date"].nunique()),
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


def run_lexicon_backtest(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[词典方案] 中文金融情感独立回测")
    print("=" * 60)

    panel = build_lexicon_panel(
        cfg, list(cfg.symbols), cfg.train_start, cfg.test_end
    )
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
    elif sr >= 0.1 and ann >= 0.0:
        verdict = "合格偏弱"
    else:
        verdict = "无效"

    print("\n[词典情感独立回测结果]", flush=True)
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
        result.equity.to_csv(result_dir / "lexicon_equity.csv")
    test_panel.to_csv(result_dir / "lexicon_signals.csv", index=False)
    summary = {
        "scheme": "lexicon_sentiment_standalone",
        "verdict": verdict,
        "lexicon": lexicon_stats(),
        "strategy": metrics,
        "equal_weight_bh": bh,
        "avg_daily_turnover": avg_to,
        "selection_stats": result.selection_stats,
        "score_formula": "lexicon_longest_match + neg/intens + daily_mean + ffill3 + cs_z + industry demean",
        "n_signal_rows": int(len(test_panel)),
        "n_signal_days": int(test_panel["date"].nunique()),
        "decision_thresholds": {
            "keep_if_sharpe_ge": 0.4,
            "weak_if_sharpe_ge": 0.2,
            "drop_if_sharpe_lt": 0.2,
        },
    }
    with open(result_dir / "lexicon_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("\n[结果] {}".format(result_dir / "lexicon_summary.json"), flush=True)
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


def run_four_leg_lexicon_combination(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[四源组合-词典] G4 + 分钟 + VQ + Lexicon")
    print("=" * 60)
    result_dir = Path(cfg.results_dir)
    equity_paths = {
        "g4": result_dir / "grid_G4_equity.csv",
        "minute": result_dir / "minute_alpha_equity.csv",
        "vq": result_dir / "value_quality_equity.csv",
        "lexicon": result_dir / "lexicon_equity.csv",
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
    out_eq.to_csv(result_dir / "four_leg_lexicon_equity_level.csv")

    signal_files = {
        "g4": result_dir / "g4_signals.csv",
        "minute": result_dir / "minute_alpha_signals.csv",
        "vq": result_dir / "value_quality_signals.csv",
        "lexicon": result_dir / "lexicon_signals.csv",
    }
    frames = {k: pd.read_csv(p, parse_dates=["date", "next_date"]) for k, p in signal_files.items()}
    for df in frames.values():
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    m = frames["g4"][["date", "symbol", "score"]].rename(columns={"score": "score_g4"})
    for key, col in [
        ("minute", "score_minute"),
        ("vq", "score_vq"),
        ("lexicon", "score_lexicon"),
    ]:
        m = m.merge(
            frames[key][["date", "symbol", "score"]].rename(columns={"score": col}),
            on=["date", "symbol"],
            how="inner",
        )
    for col in ["score_g4", "score_minute", "score_vq", "score_lexicon"]:
        m["z_" + col] = m.groupby("date")[col].transform(_cs_z_col)
    m["score"] = (
        m["z_score_g4"] + m["z_score_minute"] + m["z_score_vq"] + m["z_score_lexicon"]
    ) / 4.0

    score_corrs = {}
    for a, b in [
        ("g4", "minute"),
        ("g4", "vq"),
        ("g4", "lexicon"),
        ("minute", "vq"),
        ("minute", "lexicon"),
        ("vq", "lexicon"),
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

    print("\n[四源信号组合-词典] equal z-average")
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
        res.equity.to_csv(result_dir / "four_leg_lexicon_signal_equity.csv")
    signal.to_csv(result_dir / "four_leg_lexicon_signals.csv", index=False)

    eqw_sr = float(legs["equal_weight"]["sharpe"])
    summary = {
        "scheme": "four_leg_g4_minute_vq_lexicon",
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
            "three_leg_baseline_sharpe": 1.714,
            "beats_three_leg": bool(eqw_sr > 1.714),
        },
    }
    with open(result_dir / "four_leg_lexicon_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("[总览]")
    print("  权益等权夏普 = {:.3f}  (三源基线 1.714)".format(eqw_sr))
    print("  权益风险平价 = {:.3f}".format(legs["risk_parity"]["sharpe"]))
    print("  信号等权夏普 = {:.3f}".format(sr))
    print(
        "  ≥1.8? {}  ≥2.0? {}  优于三源? {}".format(
            summary["headline"]["hit_sharpe_1_8"],
            summary["headline"]["hit_sharpe_2_0"],
            summary["headline"]["beats_three_leg"],
        )
    )
    print("[结果] {}".format(result_dir / "four_leg_lexicon_summary.json"))
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

    out = run_lexicon_backtest(cfg)
    if "error" not in out and (Path(cfg.results_dir) / "lexicon_signals.csv").exists():
        run_four_leg_lexicon_combination(cfg)
    elif "error" in out:
        print("[阻断] {}".format(out), flush=True)
        sys.exit(2)
