#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三腿组合：历史 G4 权益 + 分钟因子 + 价值/质量.

1. 权益层：grid_G4_equity × minute_alpha_equity × value_quality_equity
2. 信号层：g4_signals × minute_alpha_signals × value_quality_signals

用法::

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_three_leg_combination
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402


def _load_rets(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "ret" in df.columns:
        s = df["ret"].astype(float)
    else:
        s = df["equity"].astype(float).pct_change().fillna(0.0)
    s.index = pd.to_datetime(s.index).normalize()
    return s


def _ann(r: pd.Series) -> float:
    r = r.dropna().astype(float)
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _sharpe(r: pd.Series) -> float:
    r = r.dropna().astype(float)
    if len(r) < 5 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _cs_z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd is None or sd < 1e-12 or np.isnan(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def equity_three_leg(res_dir: Path) -> Dict:
    paths = {
        "g4": res_dir / "grid_G4_equity.csv",
        "minute": res_dir / "minute_alpha_equity.csv",
        "vq": res_dir / "value_quality_equity.csv",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError("缺少 {}".format(p))

    series = {k: _load_rets(p) for k, p in paths.items()}
    idx = series["g4"].index
    for s in series.values():
        idx = idx.intersection(s.index)
    aligned = {k: v.loc[idx] for k, v in series.items()}

    corr = pd.DataFrame({k: v for k, v in aligned.items()}).corr()
    eqw = sum(aligned.values()) / 3.0
    # risk parity by inverse vol
    vols = {k: max(float(v.std()), 1e-8) for k, v in aligned.items()}
    inv = {k: 1.0 / vols[k] for k in vols}
    z = sum(inv.values())
    w = {k: inv[k] / z for k in inv}
    rp = sum(w[k] * aligned[k] for k in aligned)

    print("\n" + "=" * 60)
    print("[三腿权益组合]")
    print("=" * 60)
    print("  重叠日: {}".format(len(idx)))
    print("  相关矩阵:\n{}".format(corr.round(4).to_string()))
    legs = {}
    for k, r in aligned.items():
        legs[k] = {"ann": _ann(r), "sharpe": _sharpe(r), "mdd": _mdd(r)}
        print(
            "  {:>8}: ann={:+.2f}% sharpe={:.3f} mdd={:.2f}%".format(
                k, 100 * legs[k]["ann"], legs[k]["sharpe"], 100 * legs[k]["mdd"]
            )
        )
    for name, r, extra in [
        ("equal_weight", eqw, {"w": {k: 1 / 3 for k in aligned}}),
        ("risk_parity", rp, {"w": w}),
    ]:
        legs[name] = {
            "ann": _ann(r),
            "sharpe": _sharpe(r),
            "mdd": _mdd(r),
            **extra,
        }
        print(
            "  {:>8}: ann={:+.2f}% sharpe={:.3f} mdd={:.2f}%".format(
                name, 100 * legs[name]["ann"], legs[name]["sharpe"], 100 * legs[name]["mdd"]
            )
        )

    out = pd.DataFrame({**{f"ret_{k}": v for k, v in aligned.items()}, "ret_eqw": eqw, "ret_rp": rp})
    out["equity_eqw"] = (1.0 + eqw.fillna(0.0)).cumprod()
    out["equity_rp"] = (1.0 + rp.fillna(0.0)).cumprod()
    out.to_csv(res_dir / "three_leg_equity_level.csv")
    return {
        "n_days": int(len(idx)),
        "corr": corr.to_dict(),
        "legs": legs,
    }


def signal_three_leg(cfg: Config, res_dir: Path) -> Dict:
    files = {
        "g4": res_dir / "g4_signals.csv",
        "minute": res_dir / "minute_alpha_signals.csv",
        "vq": res_dir / "value_quality_signals.csv",
    }
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError("缺少 {}".format(p))

    frames = {k: pd.read_csv(p, parse_dates=["date", "next_date"]) for k, p in files.items()}
    for k, df in frames.items():
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Merge scores
    m = frames["g4"][["date", "symbol", "score"]].rename(columns={"score": "score_g4"})
    m = m.merge(
        frames["minute"][["date", "symbol", "score"]].rename(columns={"score": "score_minute"}),
        on=["date", "symbol"],
        how="inner",
    )
    m = m.merge(
        frames["vq"][["date", "symbol", "score"]].rename(columns={"score": "score_vq"}),
        on=["date", "symbol"],
        how="inner",
    )
    m["z_g4"] = m.groupby("date")["score_g4"].transform(_cs_z)
    m["z_minute"] = m.groupby("date")["score_minute"].transform(_cs_z)
    m["z_vq"] = m.groupby("date")["score_vq"].transform(_cs_z)
    m["score"] = (m["z_g4"] + m["z_minute"] + m["z_vq"]) / 3.0

    # Attach exec from minute (richest overlap)
    exec_cols = ["date", "next_date", "symbol", "open_px", "close_px", "next_close_px", "tradable_exec"]
    full = m.merge(frames["minute"][exec_cols], on=["date", "symbol"], how="inner")
    full = full.dropna(subset=["score", "open_px", "next_close_px", "next_date"])
    full = full[full["tradable_exec"].astype(bool)].copy()

    # pairwise score corr
    score_corrs = {}
    for a, b in [("g4", "minute"), ("g4", "vq"), ("minute", "vq")]:
        c = (
            m.groupby("date")
            .apply(lambda g, x=a, y=b: g["z_" + x].corr(g["z_" + y]), include_groups=False)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .mean()
        )
        score_corrs["{}_{}".format(a, b)] = float(c) if pd.notna(c) else float("nan")

    top_k = cfg.rotation_top_k or max(1, int(round(len(cfg.symbols) * cfg.rotation_top_frac)))
    bottom_k = cfg.rotation_bottom_k or max(1, int(round(len(cfg.symbols) * cfg.rotation_bottom_frac)))
    bt = RotationBacktester(
        initial_cash=cfg.initial_cash,
        cost_rate=cfg.cost_rate,
        top_k=int(top_k),
        bottom_k=int(bottom_k),
        long_gross=cfg.rotation_long_gross,
        short_gross=cfg.rotation_short_gross,
        rebalance_every=int(cfg.rebalance_every or 7),
        use_vol_scaling=False,
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
    ]
    res = bt.run(signal)
    metrics = res.metrics or {}
    avg_to = float(res.equity["turnover"].mean()) if not res.equity.empty else 0.0

    print("\n" + "=" * 60)
    print("[三腿信号组合] equal z-score average")
    print("=" * 60)
    print("  分数相关: {}".format({k: round(v, 4) for k, v in score_corrs.items()}))
    print(
        "  ann={:+.2f}% sharpe={:.3f} mdd={:.2f}% turnover={:.2f}%".format(
            100 * float(metrics.get("annualized_return", 0.0)),
            float(metrics.get("sharpe", 0.0)),
            100 * float(metrics.get("max_drawdown", 0.0)),
            100 * avg_to,
        )
    )
    if not res.equity.empty:
        res.equity.to_csv(res_dir / "three_leg_signal_equity.csv")
    signal.to_csv(res_dir / "three_leg_signals.csv", index=False)
    return {
        "score_corrs": score_corrs,
        "strategy": metrics,
        "equal_weight_bh": res.equal_weight_bh_metrics,
        "selection_stats": res.selection_stats,
        "avg_daily_turnover": avg_to,
        "n_rows": int(len(signal)),
        "n_days": int(signal["date"].nunique()),
    }


def main() -> int:
    cfg = Config()
    cfg.rebalance_every = 7
    res_dir = Path(cfg.results_dir)
    equity = equity_three_leg(res_dir)
    signal = signal_three_leg(cfg, res_dir)

    eqw_sr = float(equity["legs"]["equal_weight"]["sharpe"])
    sig_sr = float(signal["strategy"].get("sharpe") or 0.0)
    summary = {
        "scheme": "three_leg_g4_minute_vq",
        "equity_level": equity,
        "signal_level": signal,
        "headline": {
            "equity_equal_weight_sharpe": eqw_sr,
            "equity_risk_parity_sharpe": equity["legs"]["risk_parity"]["sharpe"],
            "signal_equal_weight_sharpe": sig_sr,
            "hit_sharpe_1_5": bool(eqw_sr >= 1.5 or sig_sr >= 1.5),
            "hit_sharpe_1_8": bool(eqw_sr >= 1.8 or sig_sr >= 1.8),
            "hit_sharpe_2_0": bool(eqw_sr >= 2.0 or sig_sr >= 2.0),
        },
    }
    out = res_dir / "three_leg_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("[总览]")
    print("  权益等权夏普 = {:.3f}".format(eqw_sr))
    print("  权益风险平价 = {:.3f}".format(equity["legs"]["risk_parity"]["sharpe"]))
    print("  信号等权夏普 = {:.3f}".format(sig_sr))
    print(
        "  ≥1.5? {}  ≥1.8? {}  ≥2.0? {}".format(
            summary["headline"]["hit_sharpe_1_5"],
            summary["headline"]["hit_sharpe_1_8"],
            summary["headline"]["hit_sharpe_2_0"],
        )
    )
    print("[结果] {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
