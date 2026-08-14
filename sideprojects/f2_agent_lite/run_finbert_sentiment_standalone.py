#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案B：FinBERT 日频情感因子独立回测 + 四源组合.

数据：通联 ``news_content_flash.NEWS_TITLE``（经 ``db.get_news_titles``）
模型：本地目录优先（``FINBERT_MODEL_PATH`` / ``--model-path``），
      否则 HuggingFace id（默认 ``yiyanghkust/finbert-tone``；中文标题建议换中文金融情感权重）

信号：title → FinBERT score ∈ [-1,1] → 日均 → ffill(3) → CS-Z → 行业中性

用法::

    export FINBERT_MODEL_PATH=/path/to/local/finbert
    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_finbert_sentiment_standalone
    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_finbert_sentiment_standalone --cache-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402
from sideprojects.f2_agent_lite.data import db_connector as db  # noqa: E402
from sideprojects.f2_agent_lite.data.industry import SYMBOL_INDUSTRY  # noqa: E402

DEFAULT_MODEL_ID = "yiyanghkust/finbert-tone"
_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None
_FINBERT_ERR: Optional[str] = None


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
            print("[finbert] SKIP OHLCV", sym, flush=True)
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


def _resolve_model_source(model_path: Optional[str], model_id: str) -> str:
    candidates = []
    if model_path:
        candidates.append(model_path)
    env = os.environ.get("FINBERT_MODEL_PATH", "").strip()
    if env:
        candidates.append(env)
    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return str(p)
        print("[finbert] 本地路径不存在，跳过: {}".format(p), flush=True)
    return model_id


def _get_finbert(model_source: str) -> Tuple[Optional[object], Optional[object]]:
    global _FINBERT_MODEL, _FINBERT_TOKENIZER, _FINBERT_ERR
    if _FINBERT_MODEL is not None and _FINBERT_TOKENIZER is not None:
        return _FINBERT_MODEL, _FINBERT_TOKENIZER
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch  # noqa: F401
    except ImportError as exc:
        _FINBERT_ERR = (
            "transformers 未安装: {}. "
            "本机外网 DNS 不可用，请离线安装 whl "
            "(transformers/tokenizers/safetensors/sentencepiece/huggingface_hub) "
            "或提供内网镜像。".format(exc)
        )
        print("[finbert] {}".format(_FINBERT_ERR), flush=True)
        return None, None
    try:
        local_files_only = Path(model_source).exists()
        print(
            "[finbert] 加载模型 source={} local_files_only={}".format(
                model_source, local_files_only
            ),
            flush=True,
        )
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(
            model_source, local_files_only=local_files_only
        )
        _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_source, local_files_only=local_files_only
        )
        _FINBERT_MODEL.eval()
        import torch

        if torch.cuda.is_available():
            _FINBERT_MODEL = _FINBERT_MODEL.cuda()
            print("[finbert] 使用 CUDA", flush=True)
        else:
            print("[finbert] 使用 CPU", flush=True)
        return _FINBERT_MODEL, _FINBERT_TOKENIZER
    except Exception as exc:
        _FINBERT_ERR = "模型加载失败 (source={}): {}".format(model_source, exc)
        print("[finbert] {}".format(_FINBERT_ERR), flush=True)
        return None, None


def _infer_sentiment_scores(
    texts: Sequence[str],
    model,
    tokenizer,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    import torch

    n = len(texts)
    out = np.zeros(n, dtype=float)
    device = next(model.parameters()).device
    id2label = {int(k): str(v).lower() for k, v in getattr(model.config, "id2label", {}).items()}
    # Prefer label-name mapping; fallback to FinBERT-tone order [neg, neu, pos]
    pos_idx = neg_idx = None
    for i, lab in id2label.items():
        if "pos" in lab:
            pos_idx = i
        elif "neg" in lab:
            neg_idx = i
    if pos_idx is None or neg_idx is None:
        # common 3-class: 0=neg, 1=neu, 2=pos
        neg_idx, pos_idx = 0, min(2, model.config.num_labels - 1)

    for start in range(0, n, batch_size):
        chunk = [str(t or "").strip() for t in texts[start : start + batch_size]]
        valid = [(j, t) for j, t in enumerate(chunk) if t]
        if not valid:
            continue
        idxs, vals = zip(*valid)
        enc = tokenizer(
            list(vals),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        scores = probs[:, pos_idx] - probs[:, neg_idx]
        for local_i, score in zip(idxs, scores):
            out[start + local_i] = float(score)
        done = min(start + batch_size, n)
        if done % (batch_size * 8) == 0 or done == n:
            print("[finbert] 推理进度 {}/{}".format(done, n), flush=True)
    return out


def _titles_cache_path(result_dir: Path) -> Path:
    return result_dir / "finbert_titles_cache.csv"


def _scores_cache_path(result_dir: Path) -> Path:
    return result_dir / "finbert_title_scores_cache.csv"


def load_or_fetch_titles(
    symbols: Sequence[str],
    start: str,
    end: str,
    result_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    cache = _titles_cache_path(result_dir)
    if cache.exists() and not refresh:
        print("[finbert] 读取标题缓存 {}".format(cache), flush=True)
        df = pd.read_csv(cache, parse_dates=["date", "effective_time"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df

    rows = []
    for sym in symbols:
        print("[finbert] titles", sym, "...", flush=True)
        try:
            part = db.get_news_titles(sym, start, end)
        except Exception as exc:
            print("[finbert] titles failed {}: {}".format(sym, exc), flush=True)
            continue
        if part is None or part.empty:
            continue
        part = part.copy()
        part["symbol"] = sym
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["date", "effective_time", "related_score", "title", "symbol"])
    out = pd.concat(rows, ignore_index=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print("[finbert] 标题缓存写入 {} rows={}".format(cache, len(out)), flush=True)
    return out


def score_titles(
    titles: pd.DataFrame,
    model_source: str,
    result_dir: Path,
    batch_size: int = 64,
    refresh_scores: bool = False,
) -> pd.DataFrame:
    cache = _scores_cache_path(result_dir)
    if cache.exists() and not refresh_scores:
        print("[finbert] 读取打分缓存 {}".format(cache), flush=True)
        df = pd.read_csv(cache, parse_dates=["date", "effective_time"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df

    model, tokenizer = _get_finbert(model_source)
    if model is None or tokenizer is None:
        raise RuntimeError(_FINBERT_ERR or "FinBERT unavailable")

    scored = titles.copy()
    scored["sentiment_score"] = _infer_sentiment_scores(
        scored["title"].tolist(), model, tokenizer, batch_size=batch_size
    )
    # related_score intensity weight (optional mild boost)
    w = np.log1p(scored["related_score"].clip(lower=0).astype(float))
    scored["sent_w"] = scored["sentiment_score"] * (1.0 + 0.25 * w)
    result_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(cache, index=False)
    print("[finbert] 打分缓存写入 {} rows={}".format(cache, len(scored)), flush=True)
    return scored


def build_finbert_sentiment_panel(
    cfg: Config,
    symbols: List[str],
    start_date: str,
    end_date: str,
    model_source: str,
    batch_size: int = 64,
    refresh_titles: bool = False,
    refresh_scores: bool = False,
) -> pd.DataFrame:
    result_dir = Path(cfg.results_dir)
    titles = load_or_fetch_titles(
        symbols, start_date, end_date, result_dir, refresh=refresh_titles
    )
    if titles.empty:
        print("[finbert] 错误：未获取到任何新闻标题", flush=True)
        return pd.DataFrame()

    print("[finbert] 标题条数={} symbols={}".format(len(titles), titles["symbol"].nunique()), flush=True)
    try:
        scored = score_titles(
            titles,
            model_source=model_source,
            result_dir=result_dir,
            batch_size=batch_size,
            refresh_scores=refresh_scores,
        )
    except RuntimeError as exc:
        print("[finbert] 打分失败: {}".format(exc), flush=True)
        return pd.DataFrame()

    value_col = "sent_w" if "sent_w" in scored.columns else "sentiment_score"
    agg = (
        scored.groupby(["date", "symbol"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "sentiment"})
    )
    wide = agg.pivot(index="date", columns="symbol", values="sentiment").sort_index()
    wide = wide.ffill(limit=3).fillna(0.0)
    print(
        "[finbert] 情感范围: {} ~ {} stocks={}".format(
            wide.index.min().date(), wide.index.max().date(), wide.shape[1]
        ),
        flush=True,
    )

    score_wide = _cs_zscore(wide)
    if getattr(cfg, "industry_neutral_rank", True):
        print("[finbert] 应用行业中性化...", flush=True)
        score_wide = _industry_demean(score_wide)

    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    exec_df = _load_exec_panel(symbols, start_date, end_plus)
    if exec_df.empty:
        print("[finbert] 无执行价数据", flush=True)
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
        "[finbert] 信号面板: {} 行 / {} 日".format(len(out), out["date"].nunique()),
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


def run_finbert_backtest(
    cfg: Config,
    model_source: str,
    batch_size: int = 64,
    refresh_titles: bool = False,
    refresh_scores: bool = False,
) -> Dict:
    print("\n" + "=" * 60)
    print("[方案B] FinBERT 情感因子独立回测")
    print("=" * 60)

    panel = build_finbert_sentiment_panel(
        cfg=cfg,
        symbols=list(cfg.symbols),
        start_date=cfg.train_start,
        end_date=cfg.test_end,
        model_source=model_source,
        batch_size=batch_size,
        refresh_titles=refresh_titles,
        refresh_scores=refresh_scores,
    )
    if panel.empty:
        return {"error": "信号面板为空", "detail": _FINBERT_ERR}

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

    if sr > 0.5 and ann > 0.05:
        verdict = "优秀"
    elif sr >= 0.1 and ann >= 0.0:
        verdict = "合格"
    else:
        verdict = "无效"

    print("\n[FinBERT 情感因子独立回测结果]", flush=True)
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
        result.equity.to_csv(result_dir / "finbert_equity.csv")
    test_panel.to_csv(result_dir / "finbert_signals.csv", index=False)
    summary = {
        "scheme": "finbert_standalone",
        "model_source": model_source,
        "verdict": verdict,
        "strategy": metrics,
        "equal_weight_bh": bh,
        "avg_daily_turnover": avg_to,
        "selection_stats": result.selection_stats,
        "score_formula": "FinBERT(title) mean + ffill3 + cs_z + industry demean",
        "n_signal_rows": int(len(test_panel)),
        "n_signal_days": int(test_panel["date"].nunique()),
    }
    with open(result_dir / "finbert_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("\n[结果] {}".format(result_dir / "finbert_summary.json"), flush=True)
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


def run_four_leg_finbert_combination(cfg: Config) -> Dict:
    print("\n" + "=" * 60)
    print("[四源组合-FinBERT] G4 + 分钟 + VQ + FinBERT")
    print("=" * 60)
    result_dir = Path(cfg.results_dir)

    equity_paths = {
        "g4": result_dir / "grid_G4_equity.csv",
        "minute": result_dir / "minute_alpha_equity.csv",
        "vq": result_dir / "value_quality_equity.csv",
        "finbert": result_dir / "finbert_equity.csv",
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
    out_eq.to_csv(result_dir / "four_leg_finbert_equity_level.csv")

    signal_files = {
        "g4": result_dir / "g4_signals.csv",
        "minute": result_dir / "minute_alpha_signals.csv",
        "vq": result_dir / "value_quality_signals.csv",
        "finbert": result_dir / "finbert_signals.csv",
    }
    frames = {k: pd.read_csv(p, parse_dates=["date", "next_date"]) for k, p in signal_files.items()}
    for df in frames.values():
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    m = frames["g4"][["date", "symbol", "score"]].rename(columns={"score": "score_g4"})
    for key, col in [
        ("minute", "score_minute"),
        ("vq", "score_vq"),
        ("finbert", "score_finbert"),
    ]:
        m = m.merge(
            frames[key][["date", "symbol", "score"]].rename(columns={"score": col}),
            on=["date", "symbol"],
            how="inner",
        )
    for col in ["score_g4", "score_minute", "score_vq", "score_finbert"]:
        m["z_" + col] = m.groupby("date")[col].transform(_cs_z_col)
    m["score"] = (
        m["z_score_g4"] + m["z_score_minute"] + m["z_score_vq"] + m["z_score_finbert"]
    ) / 4.0

    score_corrs = {}
    for a, b in [
        ("g4", "minute"),
        ("g4", "vq"),
        ("g4", "finbert"),
        ("minute", "vq"),
        ("minute", "finbert"),
        ("vq", "finbert"),
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

    print("\n[四源信号组合-FinBERT] equal z-average")
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
        res.equity.to_csv(result_dir / "four_leg_finbert_signal_equity.csv")
    signal.to_csv(result_dir / "four_leg_finbert_signals.csv", index=False)

    eqw_sr = float(legs["equal_weight"]["sharpe"])
    summary = {
        "scheme": "four_leg_g4_minute_vq_finbert",
        "equity_level": {
            "n_days": int(len(idx)),
            "corr": corr.to_dict(),
            "legs": legs,
        },
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
        },
    }
    with open(result_dir / "four_leg_finbert_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("[总览]")
    print("  权益等权夏普 = {:.3f}".format(eqw_sr))
    print("  权益风险平价 = {:.3f}".format(legs["risk_parity"]["sharpe"]))
    print("  信号等权夏普 = {:.3f}".format(sr))
    print(
        "  ≥1.8? {}  ≥2.0? {}".format(
            summary["headline"]["hit_sharpe_1_8"], summary["headline"]["hit_sharpe_2_0"]
        )
    )
    print("[结果] {}".format(result_dir / "four_leg_finbert_summary.json"))
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FinBERT news sentiment alpha")
    p.add_argument("--model-path", default="", help="Local FinBERT directory")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id fallback")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--refresh-titles", action="store_true")
    p.add_argument("--refresh-scores", action="store_true")
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Only fetch/cache news titles (no model / no backtest)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
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

    model_source = _resolve_model_source(args.model_path or None, args.model_id)

    if args.cache_only:
        load_or_fetch_titles(
            list(cfg.symbols),
            cfg.train_start,
            cfg.test_end,
            Path(cfg.results_dir),
            refresh=args.refresh_titles,
        )
        sys.exit(0)

    out = run_finbert_backtest(
        cfg,
        model_source=model_source,
        batch_size=args.batch_size,
        refresh_titles=args.refresh_titles,
        refresh_scores=args.refresh_scores,
    )
    if "error" not in out and (Path(cfg.results_dir) / "finbert_signals.csv").exists():
        run_four_leg_finbert_combination(cfg)
    elif "error" in out:
        print("\n[阻断] {}".format(out), flush=True)
        sys.exit(2)
