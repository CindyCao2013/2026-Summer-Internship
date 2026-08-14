#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G4 主策略 + 分钟因子独立子策略 组合回测.

两层验证：
1. **权益层组合**：用已落盘的 ``grid_G4_equity.csv`` + ``minute_alpha_equity.csv``
   日收益等权 / 风险平价，直接量相关与组合夏普。
2. **信号层组合**：重训无分钟特征的 G4 截面模型 → 保存 ``g4_signals.csv``，
   与 ``minute_alpha_signals.csv`` 做等权分数 / 排名平均后，再喂 RotationBacktester。

用法::

    cd /home/SiYangCao/factor_dev/factor_research0703/factor_dev
    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_strategy_combination
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402
from sideprojects.f2_agent_lite.data.cross_sectional_dataset import (  # noqa: E402
    apply_score_smoothing,
    cs_predictions_to_signal_frame,
    prepare_cs_splits,
)
from sideprojects.f2_agent_lite.train_cross_sectional import (  # noqa: E402
    predict_cs_proba,
    train_cs_model,
)
from sideprojects.f2_agent_lite.utils.metrics import performance_summary  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sharpe(rets: pd.Series) -> float:
    r = rets.dropna().astype(float)
    if len(r) < 5 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _ann(rets: pd.Series) -> float:
    r = rets.dropna().astype(float)
    if len(r) == 0:
        return 0.0
    growth = float((1.0 + r).prod())
    return float(growth ** (252.0 / len(r)) - 1.0)


def _max_dd_from_rets(rets: pd.Series) -> float:
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _load_equity_rets(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "ret" in df.columns:
        s = df["ret"].astype(float)
    else:
        s = df["equity"].astype(float).pct_change().fillna(0.0)
    s.index = pd.to_datetime(s.index).normalize()
    s.name = path.stem
    return s


def _make_backtester(cfg: Config) -> RotationBacktester:
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


def _attach_exec_cols(scores: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Merge score with execution columns from a template signal frame."""
    keys = ["date", "next_date", "symbol"]
    cols = ["open_px", "close_px", "next_close_px", "tradable_exec"]
    out = scores.merge(template[keys + cols], on=keys, how="inner")
    out = out.dropna(subset=["score", "open_px", "next_close_px", "next_date"])
    out = out[out["tradable_exec"].astype(bool)].copy()
    return out


# ---------------------------------------------------------------------------
# 1) equity-level combination
# ---------------------------------------------------------------------------


def run_equity_combination(cfg: Config) -> Dict:
    result_dir = Path(cfg.results_dir)
    g4_path = result_dir / "grid_G4_equity.csv"
    m_path = result_dir / "minute_alpha_equity.csv"
    if not g4_path.exists():
        raise FileNotFoundError("缺少 {} — 请先跑 G4 / smooth grid".format(g4_path))
    if not m_path.exists():
        raise FileNotFoundError("缺少 {} — 请先跑 run_minute_alpha_standalone".format(m_path))

    r_g4 = _load_equity_rets(g4_path)
    r_m = _load_equity_rets(m_path)
    idx = r_g4.index.intersection(r_m.index)
    r_g4, r_m = r_g4.loc[idx], r_m.loc[idx]
    corr = float(r_g4.corr(r_m))

    eqw = 0.5 * r_g4 + 0.5 * r_m
    v1, v2 = float(r_g4.std()), float(r_m.std())
    inv1, inv2 = 1.0 / max(v1, 1e-8), 1.0 / max(v2, 1e-8)
    w1, w2 = inv1 / (inv1 + inv2), inv2 / (inv1 + inv2)
    rp = w1 * r_g4 + w2 * r_m

    # theoretical SR under ρ
    s1, s2 = _sharpe(r_g4), _sharpe(r_m)
    theo = float(np.sqrt(s1 ** 2 + s2 ** 2)) if abs(corr) < 1e-9 else float(
        np.sqrt(s1 ** 2 + s2 ** 2 + 2 * corr * s1 * s2)
    )
    # equal-weight theoretical (vol-matched approx)
    theo_eqw = float((s1 + s2) / np.sqrt(2 + 2 * corr)) if (2 + 2 * corr) > 0 else theo

    rows = {
        "g4": {"ann": _ann(r_g4), "sharpe": s1, "mdd": _max_dd_from_rets(r_g4)},
        "minute": {"ann": _ann(r_m), "sharpe": s2, "mdd": _max_dd_from_rets(r_m)},
        "equal_weight": {
            "ann": _ann(eqw),
            "sharpe": _sharpe(eqw),
            "mdd": _max_dd_from_rets(eqw),
            "w_g4": 0.5,
            "w_minute": 0.5,
        },
        "risk_parity": {
            "ann": _ann(rp),
            "sharpe": _sharpe(rp),
            "mdd": _max_dd_from_rets(rp),
            "w_g4": w1,
            "w_minute": w2,
        },
    }

    print("\n" + "=" * 60)
    print("[权益层组合] G4 equity × 分钟因子 equity")
    print("=" * 60)
    print("  重叠交易日: {}".format(len(idx)))
    print("  日收益相关 ρ = {:.4f}".format(corr))
    print("  理论等权夏普(近似) ≈ {:.3f}  |  sqrt(s1²+s2²)≈{:.3f}".format(theo_eqw, theo))
    for name, m in rows.items():
        print(
            "  {:>12}: ann={:+.2f}%  sharpe={:.3f}  mdd={:.2f}%".format(
                name, 100 * m["ann"], m["sharpe"], 100 * m["mdd"]
            )
        )

    # save combined equity path (equal weight)
    eq_path = (1.0 + eqw.fillna(0.0)).cumprod()
    rp_path = (1.0 + rp.fillna(0.0)).cumprod()
    out_eq = pd.DataFrame(
        {
            "ret_g4": r_g4,
            "ret_minute": r_m,
            "ret_equal": eqw,
            "ret_risk_parity": rp,
            "equity_equal": eq_path,
            "equity_risk_parity": rp_path,
            "equity_g4": (1.0 + r_g4.fillna(0.0)).cumprod(),
            "equity_minute": (1.0 + r_m.fillna(0.0)).cumprod(),
        }
    )
    out_eq.to_csv(result_dir / "combined_equity_level.csv")

    return {
        "n_days": int(len(idx)),
        "corr": corr,
        "theoretical_sqrt_ss": theo,
        "theoretical_equal_weight_sr": theo_eqw,
        "legs": rows,
    }


# ---------------------------------------------------------------------------
# 2) signal-level: ensure G4 signals exist
# ---------------------------------------------------------------------------


def ensure_g4_signals(cfg: Config, force_retrain: bool = False) -> pd.DataFrame:
    """Train pure G4 (no minute features) and save g4_signals.csv."""
    result_dir = Path(cfg.results_dir)
    out_path = result_dir / "g4_signals.csv"
    if out_path.exists() and not force_retrain:
        print("[信号] 复用已有 {}".format(out_path), flush=True)
        return pd.read_csv(out_path, parse_dates=["date", "next_date"])

    print("\n[信号] 重训纯 G4（use_minute_factors=False）...", flush=True)
    g4_cfg = deepcopy(cfg)
    g4_cfg.use_minute_factors = False
    g4_cfg.use_north_money = True
    g4_cfg.use_fundamentals = True
    g4_cfg.use_advanced_alpha = False
    g4_cfg.use_market_risk = False
    g4_cfg.use_vol_scaling = False
    g4_cfg.label_threshold = 0.015
    g4_cfg.rebalance_every = 7
    g4_cfg.score_smooth_window = 3
    g4_cfg.scheme = "B"

    splits = prepare_cs_splits(g4_cfg)
    model, train_metrics, device = train_cs_model(g4_cfg, splits.train, splits.val)
    probs = predict_cs_proba(model, splits.test, device, batch_size=min(16, g4_cfg.batch_size))
    signals = cs_predictions_to_signal_frame(
        splits.test,
        probs,
        industry_neutral=bool(g4_cfg.industry_neutral_rank),
    )
    w = int(getattr(g4_cfg, "score_smooth_window", 3) or 0)
    if w > 1:
        signals = apply_score_smoothing(signals, window=w)
        print("[信号] G4 score smoothed window={}".format(w), flush=True)

    keep = [
        "date",
        "next_date",
        "symbol",
        "score",
        "open_px",
        "close_px",
        "next_close_px",
        "tradable_exec",
    ]
    signals = signals[keep].copy()
    signals.to_csv(out_path, index=False)
    print(
        "[信号] 写入 {} rows={} days={} best_val_rank_hit={}".format(
            out_path,
            len(signals),
            signals["date"].nunique(),
            train_metrics.get("best_val_rank_hit"),
        ),
        flush=True,
    )

    # also backtest pure G4 for reference
    bt = _make_backtester(g4_cfg)
    res = bt.run(signals)
    print(
        "[信号] 纯 G4 回测 ann={:.2%} sharpe={:.3f} mdd={:.2%}".format(
            res.metrics.get("annualized_return", 0.0),
            res.metrics.get("sharpe", 0.0),
            res.metrics.get("max_drawdown", 0.0),
        ),
        flush=True,
    )
    if not res.equity.empty:
        res.equity.to_csv(result_dir / "g4_standalone_equity.csv")
    with open(result_dir / "g4_standalone_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": train_metrics,
                "strategy": res.metrics,
                "equal_weight_bh": res.equal_weight_bh_metrics,
                "selection_stats": res.selection_stats,
                "n_features": splits.train.n_features,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    return signals


def combine_signals(
    g4: pd.DataFrame,
    minute: pd.DataFrame,
    method: str = "equal_weight",
) -> pd.DataFrame:
    g = g4[["date", "next_date", "symbol", "score"]].copy()
    m = minute[["date", "next_date", "symbol", "score"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    g["next_date"] = pd.to_datetime(g["next_date"])
    m["next_date"] = pd.to_datetime(m["next_date"])
    g = g.rename(columns={"score": "score_g4"})
    m = m.rename(columns={"score": "score_minute"})

    # align on date+symbol; prefer minute next_date (same calendar)
    combined = g.merge(m, on=["date", "symbol"], how="inner", suffixes=("_g4", "_minute"))
    if combined.empty:
        return pd.DataFrame()
    # use G4 next_date when available
    combined["next_date"] = combined["next_date_g4"].fillna(combined["next_date_minute"])

    # CS z-score each score before blending (scale-invariant)
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        if sd is None or sd < 1e-12 or np.isnan(sd):
            return s * 0.0
        return (s - s.mean()) / sd

    combined["z_g4"] = combined.groupby("date")["score_g4"].transform(_z)
    combined["z_minute"] = combined.groupby("date")["score_minute"].transform(_z)

    if method == "equal_weight":
        combined["score"] = 0.5 * combined["z_g4"] + 0.5 * combined["z_minute"]
    elif method == "rank_average":
        combined["rank_g4"] = combined.groupby("date")["score_g4"].rank(pct=True)
        combined["rank_minute"] = combined.groupby("date")["score_minute"].rank(pct=True)
        combined["score"] = 0.5 * (combined["rank_g4"] + combined["rank_minute"])
    elif method == "g4_only":
        combined["score"] = combined["z_g4"]
    elif method == "minute_only":
        combined["score"] = combined["z_minute"]
    else:
        raise ValueError("unknown method {}".format(method))

    return combined[["date", "next_date", "symbol", "score", "z_g4", "z_minute"]]


def run_signal_combination(cfg: Config, force_retrain_g4: bool = False) -> Dict:
    result_dir = Path(cfg.results_dir)
    minute_path = result_dir / "minute_alpha_signals.csv"
    if not minute_path.exists():
        raise FileNotFoundError("缺少 {} — 请先跑 run_minute_alpha_standalone".format(minute_path))
    minute = pd.read_csv(minute_path, parse_dates=["date", "next_date"])
    g4 = ensure_g4_signals(cfg, force_retrain=force_retrain_g4)

    print("\n" + "=" * 60)
    print("[信号层组合] G4 scores × 分钟因子 scores")
    print("=" * 60)

    # score correlation (daily cross-section mean of pairwise)
    tmp = combine_signals(g4, minute, method="equal_weight")
    daily_corr = (
        tmp.groupby("date")
        .apply(lambda g: g["z_g4"].corr(g["z_minute"]), include_groups=False)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    score_corr = float(daily_corr.mean()) if len(daily_corr) else float("nan")
    print("  截面分数日均相关 ρ_score = {:.4f}".format(score_corr), flush=True)

    methods = ["g4_only", "minute_only", "equal_weight", "rank_average"]
    out: Dict[str, dict] = {"score_corr_mean": score_corr}
    bt = _make_backtester(cfg)

    for method in methods:
        scores = combine_signals(g4, minute, method=method)
        if scores.empty:
            print("  [{}] 空".format(method), flush=True)
            continue
        full = _attach_exec_cols(scores, minute)
        if full.empty:
            # fallback to g4 exec cols
            full = _attach_exec_cols(scores, g4)
        res = bt.run(full)
        metrics = res.metrics or {}
        avg_to = float(res.equity["turnover"].mean()) if not res.equity.empty else 0.0
        print(
            "  [{:>13}] ann={:+.2f}% sharpe={:.3f} mdd={:.2f}% turnover={:.2f}%".format(
                method,
                100 * float(metrics.get("annualized_return", 0.0)),
                float(metrics.get("sharpe", 0.0)),
                100 * float(metrics.get("max_drawdown", 0.0)),
                100 * avg_to,
            ),
            flush=True,
        )
        out[method] = {
            "strategy": metrics,
            "equal_weight_bh": res.equal_weight_bh_metrics,
            "selection_stats": res.selection_stats,
            "avg_daily_turnover": avg_to,
            "n_rows": int(len(full)),
            "n_days": int(full["date"].nunique()),
        }
        if not res.equity.empty:
            res.equity.to_csv(result_dir / "combined_signal_{}_equity.csv".format(method))
            full.to_csv(result_dir / "combined_signal_{}_signals.csv".format(method), index=False)

    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_combined_backtest(cfg: Config, force_retrain_g4: bool = False) -> Dict:
    equity_block = run_equity_combination(cfg)
    signal_block = run_signal_combination(cfg, force_retrain_g4=force_retrain_g4)

    best_signal = None
    best_sr = -1e9
    for k, v in signal_block.items():
        if not isinstance(v, dict) or "strategy" not in v:
            continue
        sr = float(v["strategy"].get("sharpe") or -1e9)
        if sr > best_sr:
            best_sr = sr
            best_signal = k

    eqw_sr = float(equity_block["legs"]["equal_weight"]["sharpe"])
    summary = {
        "scheme": "combined_g4_minute",
        "equity_level": equity_block,
        "signal_level": signal_block,
        "headline": {
            "return_corr": equity_block["corr"],
            "equity_equal_weight_sharpe": eqw_sr,
            "equity_risk_parity_sharpe": equity_block["legs"]["risk_parity"]["sharpe"],
            "best_signal_method": best_signal,
            "best_signal_sharpe": best_sr,
            "hit_sharpe_1_5": bool(eqw_sr >= 1.5 or best_sr >= 1.5),
            "hit_sharpe_2_0": bool(eqw_sr >= 2.0 or best_sr >= 2.0),
        },
    }

    out_path = Path(cfg.results_dir) / "combined_strategy_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("[总览]")
    print("  收益相关 ρ = {:.4f}".format(equity_block["corr"]))
    print(
        "  权益等权夏普 = {:.3f}  |  风险平价 = {:.3f}".format(
            eqw_sr, equity_block["legs"]["risk_parity"]["sharpe"]
        )
    )
    print("  最佳信号组合 = {} 夏普 = {:.3f}".format(best_signal, best_sr))
    print("  突破 1.5? {}   突破 2.0? {}".format(
        summary["headline"]["hit_sharpe_1_5"], summary["headline"]["hit_sharpe_2_0"]
    ))
    print("[结果] {}".format(out_path))
    return summary


if __name__ == "__main__":
    force = "--force-retrain-g4" in sys.argv
    cfg = Config()
    cfg.use_minute_factors = False  # G4 训练不用分钟；分钟信号来自已落盘 CSV
    cfg.use_north_money = True
    cfg.use_fundamentals = True
    cfg.use_advanced_alpha = False
    cfg.use_market_risk = False
    cfg.use_vol_scaling = False
    cfg.industry_neutral_rank = True
    cfg.label_threshold = 0.015
    cfg.rebalance_every = 7
    cfg.score_smooth_window = 3
    run_combined_backtest(cfg, force_retrain_g4=force)
