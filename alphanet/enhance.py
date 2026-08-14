"""Industry/size-neutral CSI500 enhancement (guide §7).

Maximizes weighted factor exposure subject to:
- long-only, 0 <= w_i <= max_weight
- active weight vs benchmark <= cap
- two-way turnover <= 30%
- industry and size exposure match the benchmark (least-squares penalty)

Uses SLSQP when scipy is available; otherwise a clipped proportional tilt.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from alphanet.config import EnhanceConfig
from alphanet.evaluate import universe_equal_weight_return
from alphanet.metrics import information_ratio, summarize_return_series
from alphanet.paths import ENHANCE, ensure_result_dirs


def _scipy_optimize():
    from scipy.optimize import minimize

    return minimize


def benchmark_weights(
    members: pd.Series,
    log_mcap: Optional[pd.Series] = None,
    cap_weight: bool = False,
) -> pd.Series:
    idx = members.dropna()
    idx = idx[idx == 1].index if (idx == 1).any() else idx.index
    w = pd.Series(0.0, index=members.index)
    if len(idx) == 0:
        return w
    if cap_weight and log_mcap is not None:
        mv = np.exp(pd.to_numeric(log_mcap.reindex(idx), errors="coerce")).replace(0, np.nan)
        mv = mv.fillna(mv.median())
        w.loc[idx] = (mv / mv.sum()).astype(float)
    else:
        w.loc[idx] = 1.0 / len(idx)
    return w


def _industry_matrix(industry: pd.Series, index) -> np.ndarray:
    dummies = pd.get_dummies(industry.reindex(index), dummy_na=False)
    return dummies.reindex(index=index).fillna(0.0).to_numpy(dtype=float)


def optimize_weights(
    factor: pd.Series,
    bench_w: pd.Series,
    prev_w: Optional[pd.Series],
    industry: Optional[pd.Series],
    log_mcap: Optional[pd.Series],
    cfg: EnhanceConfig,
    active_cap: float,
) -> pd.Series:
    names = bench_w.index.intersection(factor.dropna().index)
    if len(names) < 10:
        return bench_w.reindex(factor.index).fillna(0.0)
    f = pd.to_numeric(factor.reindex(names), errors="coerce").fillna(0.0).to_numpy()
    b = bench_w.reindex(names).fillna(0.0).to_numpy()
    prev = (
        prev_w.reindex(names).fillna(0.0).to_numpy()
        if prev_w is not None
        else b.copy()
    )
    n = len(names)
    lo = np.maximum(0.0, b - active_cap)
    hi = np.minimum(float(cfg.max_weight), b + active_cap)

    def objective(w):
        return -float(w @ f)

    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    if cfg.industry_neutral and industry is not None:
        A = _industry_matrix(industry, names)
        b_ind = A.T @ b

        def _ind(w, A=A, b_ind=b_ind):
            return A.T @ w - b_ind

        cons.append({"type": "eq", "fun": _ind})
    if cfg.size_neutral and log_mcap is not None:
        size = pd.to_numeric(log_mcap.reindex(names), errors="coerce").fillna(0.0).to_numpy()

        def _sz(w, size=size, b=b):
            return float(w @ size - b @ size)

        cons.append({"type": "eq", "fun": _sz})

    def _to(w, prev=prev):
        return 0.5 * np.abs(w - prev).sum() - float(cfg.two_way_turnover_cap)

    cons.append({"type": "ineq", "fun": lambda w: -_to(w)})
    bounds = list(zip(lo, hi))
    try:
        minimize = _scipy_optimize()
        res = minimize(
            objective,
            x0=b,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-8, "disp": False},
        )
        w_star = res.x if res.success else b
    except Exception:
        tilt = np.clip(f, -3, 3)
        raw = b * (1.0 + tilt)
        raw = np.clip(raw, lo, hi)
        s = raw.sum()
        w_star = raw / s if s > 0 else b
    out = pd.Series(0.0, index=factor.index)
    out.loc[names] = w_star
    total = out.sum()
    if total > 0:
        out = out / total
    return out


def enhance_backtest(
    factor: pd.DataFrame,
    ret_1d: pd.DataFrame,
    members: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    log_mcap: Optional[pd.DataFrame] = None,
    cfg: Optional[EnhanceConfig] = None,
    active_cap: float = 0.01,
) -> Dict[str, object]:
    cfg = cfg or EnhanceConfig()
    dates = factor.index.intersection(ret_1d.index).intersection(members.index)
    weights_hist = []
    prev = None
    port = []
    bench = []
    to_list = []
    for dt in dates:
        f = factor.loc[dt]
        mem = members.loc[dt]
        bw = benchmark_weights(mem, None if log_mcap is None else log_mcap.loc[dt])
        w = optimize_weights(
            f,
            bw,
            prev,
            None if industry is None else industry.loc[dt],
            None if log_mcap is None else log_mcap.loc[dt],
            cfg,
            active_cap,
        )
        if prev is None:
            to = float(w.fillna(0).abs().sum())
        else:
            to = float((w.fillna(0) - prev.reindex(w.index).fillna(0)).abs().sum())
        # next-day return: factor T trades into T+1 close-to-close
        r = ret_1d.shift(-1).loc[dt].reindex(w.index).fillna(0.0)
        br = float((bw.reindex(w.index).fillna(0.0) * r).sum())
        pr = float((w.fillna(0.0) * r).sum()) - to * float(cfg.fee_one_way)
        port.append(pr)
        bench.append(br)
        to_list.append(to)
        weights_hist.append(w)
        prev = w
    port_s = pd.Series(port, index=dates, name="port")
    bench_s = pd.Series(bench, index=dates, name="bench")
    excess = port_s - bench_s
    summary = summarize_return_series(port_s)
    summary.update(
        {
            "excess_annu_ret": summarize_return_series(excess)["annu_ret"],
            "information_ratio": information_ratio(excess),
            "avg_l1_turnover": float(np.mean(to_list)) if to_list else float("nan"),
            "active_cap": float(active_cap),
        }
    )
    return {
        "port": port_s,
        "bench": bench_s,
        "excess": excess,
        "summary": summary,
        "weights": weights_hist,
    }


def run_enhance_grid(
    factor: pd.DataFrame,
    ret_1d: pd.DataFrame,
    members: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    cfg: EnhanceConfig = kwargs.pop("cfg", None) or EnhanceConfig()
    rows = []
    curves = {}
    for cap in cfg.active_weight_caps:
        result = enhance_backtest(factor, ret_1d, members, cfg=cfg, active_cap=cap, **kwargs)
        row = dict(result["summary"])
        row["active_cap"] = cap
        rows.append(row)
        curves[cap] = result["excess"]
    table = pd.DataFrame(rows)
    ensure_result_dirs()
    table.to_csv(ENHANCE / "grid_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(ENHANCE / "excess_curves.csv")
    return table
