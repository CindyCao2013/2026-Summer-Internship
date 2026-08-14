"""Modified evaluation protocol: RankIC + 10-layer excess returns + 7.5 bps cost.

G1 is the *highest* factor group (guide §6). H-L = G1 − G10, not flipped.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from alphanet.config import EvalConfig, FEE_ONE_WAY
from alphanet.metrics import (
    monotonicity_spearman,
    rank_ic_daily,
    summarize_rank_ic,
    summarize_return_series,
)
from alphanet.paths import DECILES, IC, REPORTS, ensure_result_dirs


def universe_equal_weight_return(ret: pd.DataFrame, mask: Optional[pd.DataFrame] = None) -> pd.Series:
    r = ret.copy()
    if mask is not None:
        r = r.where(mask.reindex_like(r) == 1)
    count = r.notna().sum(axis=1).replace(0, np.nan)
    return r.mean(axis=1)


def assign_deciles(signal: pd.DataFrame, n: int = 10, g1_is_top: bool = True) -> pd.DataFrame:
    """Approximate equal-count groups. G1 = top if ``g1_is_top``."""
    ranks = signal.rank(axis=1, method="first", ascending=not g1_is_top)
    # high factor → rank 1 when g1_is_top (ascending=False)
    def _qcut_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return row
        try:
            cat = pd.qcut(valid, q=n, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=row.index)
        return (cat + 1).reindex(row.index)

    grouped = ranks.apply(_qcut_row, axis=1)
    return grouped


def _group_weights(group_id: pd.DataFrame, g: int) -> pd.DataFrame:
    mask = group_id.eq(g)
    count = mask.sum(axis=1).replace(0, np.nan)
    return mask.div(count, axis=0)


def holding_period_return(ret_1d: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple return over the next ``horizon`` sessions, aligned on T."""
    one = 1.0 + ret_1d
    stacked = np.ones_like(one.to_numpy(), dtype=float)
    for k in range(1, int(horizon) + 1):
        stacked = stacked * one.shift(-k).to_numpy()
    return pd.DataFrame(stacked - 1.0, index=ret_1d.index, columns=ret_1d.columns)


def decile_backtest(
    signal: pd.DataFrame,
    ret_1d: pd.DataFrame,
    *,
    eval_cfg: Optional[EvalConfig] = None,
    mask: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    cfg = eval_cfg or EvalConfig()
    n = int(cfg.n_groups)
    horizon = int(cfg.rebalance_every)
    sig = signal.copy()
    if mask is not None:
        sig = sig.where(mask.reindex_like(sig) == 1)
    # keep rebalance dates only
    if horizon > 1:
        sig = sig.iloc[::horizon]
    hp = holding_period_return(ret_1d, horizon).reindex(index=sig.index, columns=sig.columns)
    bench_full = universe_equal_weight_return(hp, mask=mask.reindex_like(hp) if mask is not None else None)
    groups = assign_deciles(sig, n=n, g1_is_top=cfg.g1_is_top)

    group_pnl = pd.DataFrame(index=sig.index)
    group_to = pd.DataFrame(index=sig.index)
    prev_w = None
    for g in range(1, n + 1):
        w = _group_weights(groups, g)
        gross = w.mul(hp).sum(axis=1)
        w0 = w.fillna(0.0)
        to = w0.diff().abs().sum(axis=1)
        to.iloc[0] = w0.iloc[0].abs().sum()
        fee = to * float(cfg.fee_one_way)
        net = gross - fee
        excess = net - bench_full.reindex(net.index)
        group_pnl[g] = excess
        group_to[g] = to
        prev_w = w0
    group_pnl["H-L"] = group_pnl[1] - group_pnl[n]
    group_to["H-L"] = group_to[1] + group_to[n]

    rows = []
    means = {}
    for col in list(range(1, n + 1)) + ["H-L"]:
        stats = summarize_return_series(group_pnl[col])
        stats["group"] = "G{}".format(col) if col != "H-L" else "H-L"
        stats["avg_turnover"] = float(group_to[col].mean())
        rows.append(stats)
        if col != "H-L":
            means[int(col)] = stats["annu_ret"]
    table = pd.DataFrame(rows)
    table["monotonicity_spearman"] = monotonicity_spearman(means)
    # strict decreasing G1..G10 on annualized excess
    annu = [means[g] for g in range(1, n + 1)]
    table.attrs["strict_decreasing"] = all(annu[i] > annu[i + 1] for i in range(n - 1))
    return {
        "group_pnl": group_pnl,
        "group_turnover": group_to,
        "table": table,
        "bench": bench_full,
        "groups": groups,
        "strict_decreasing": table.attrs["strict_decreasing"],
        "monotonicity_spearman": monotonicity_spearman(means),
    }


def ic_test(
    signal: pd.DataFrame,
    ret_1d: pd.DataFrame,
    *,
    horizon: int,
    mask: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    sig = signal.copy()
    if mask is not None:
        sig = sig.where(mask.reindex_like(sig) == 1)
    fwd = holding_period_return(ret_1d, horizon).reindex(index=sig.index, columns=sig.columns)
    # section every horizon days
    sig = sig.iloc[:: max(horizon, 1)]
    fwd = fwd.reindex(index=sig.index)
    ic = rank_ic_daily(sig, fwd)
    summary = summarize_rank_ic(ic)
    return {"ic": ic, "summary": summary}


def write_eval_artifacts(
    variant: str,
    ic_result: Dict[str, object],
    decile_result: Dict[str, object],
) -> Dict[str, str]:
    ensure_result_dirs()
    ic_path = IC / "{}_rankic.csv".format(variant)
    ic_result["ic"].to_csv(ic_path)
    table_path = DECILES / "{}_decile_table.csv".format(variant)
    decile_result["table"].to_csv(table_path, index=False)
    pnl_path = DECILES / "{}_group_pnl.csv".format(variant)
    decile_result["group_pnl"].to_csv(pnl_path)
    report = REPORTS / "{}_eval.md".format(variant)
    s = ic_result["summary"]
    lines = [
        "# AlphaNet {} evaluation".format(variant),
        "",
        "## RankIC",
        "- mean: {:.4f}".format(s["rank_ic_mean"]),
        "- std: {:.4f}".format(s["rank_ic_std"]),
        "- ICIR: {:.4f}".format(s["icir"]),
        "- IC>0: {:.2%}".format(s["ic_positive_frac"]),
        "",
        "## 10-layer excess (one-way fee = {:.3%} / {:.1f} bps)".format(
            FEE_ONE_WAY, FEE_ONE_WAY * 1e4
        ),
        decile_result["table"].to_string(index=False),
        "",
        "strict decreasing G1..G10: {}".format(decile_result["strict_decreasing"]),
        "monotonicity Spearman: {:.3f}".format(decile_result["monotonicity_spearman"]),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return {"ic": str(ic_path), "table": str(table_path), "pnl": str(pnl_path), "report": str(report)}
