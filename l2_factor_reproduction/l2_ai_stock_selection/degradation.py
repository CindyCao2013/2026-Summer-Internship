"""Legacy C2C vs executable V2V factor diagnostics. No auto-DROP."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    IC_ABS_FLOOR,
    classify_degradation,
    ic_preservation,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import rank_ic


def daily_rank_ic_series(factor: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    common = factor.index.intersection(y.index)
    cols = factor.columns.intersection(y.columns)
    if len(common) == 0 or len(cols) == 0:
        return pd.Series(dtype=float)
    return factor.loc[common, cols].corrwith(y.loc[common, cols], axis=1, method="spearman")


def ic_summary(factor: pd.DataFrame, y: pd.DataFrame, mask: Optional[pd.DataFrame] = None) -> dict:
    f = factor
    yy = y.reindex_like(f)
    if mask is not None:
        m = mask.reindex_like(f)
        f = f.where(m == 1)
        yy = yy.where(m == 1)
    ic = daily_rank_ic_series(f, yy)
    ic = ic.dropna()
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std()) if n > 1 else float("nan")
    icir = float(mean / std * np.sqrt(250.0)) if np.isfinite(std) and std > 0 else float("nan")
    pos = float((ic > 0).mean()) if n else float("nan")
    cov = float(np.isfinite(f.to_numpy()).mean()) if f.size else float("nan")
    return {
        "rank_ic_mean": mean,
        "icir": icir,
        "positive_ic_fraction": pos,
        "coverage": cov,
        "n_ic_days": n,
    }


def hl_stats(factor: pd.DataFrame, y: pd.DataFrame, mask: Optional[pd.DataFrame] = None, n_groups: int = 10) -> dict:
    """Overlapping daily-rebalance H-L on the label (diagnostic, not a tradable book)."""
    f = factor
    yy = y.reindex_like(f)
    if mask is not None:
        m = mask.reindex_like(f)
        f = f.where(m == 1)
        yy = yy.where(m == 1)
    ranks = f.rank(axis=1, method="first")
    q = ranks.quantile(np.linspace(0, 1, n_groups + 1), axis=1)
    # per-date qcut via rank thresholds
    n = f.shape[1]
    if n < n_groups * 3:
        return {"hl_annu_ret": float("nan"), "hl_sharpe": float("nan"), "turnover": float("nan")}
    lo = (n / n_groups)
    hi = n - lo
    low = ranks <= lo
    high = ranks > hi
    def _ew(mask_g):
        c = mask_g.sum(axis=1).replace(0, np.nan)
        w = mask_g.div(c, axis=0)
        return w
    w_h = _ew(high)
    w_l = _ew(low)
    pnl = (w_h * yy).sum(axis=1) - (w_l * yy).sum(axis=1)
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pnl) < 20:
        return {"hl_annu_ret": float("nan"), "hl_sharpe": float("nan"), "turnover": float("nan")}
    annu = float(pnl.mean() * 250.0)
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(250.0)) if pnl.std() > 0 else float("nan")
    w = (w_h - w_l).fillna(0.0)
    to = w.diff().abs().sum(axis=1)
    to.iloc[0] = w.iloc[0].abs().sum()
    return {
        "hl_annu_ret": annu,
        "hl_sharpe": sharpe,
        "turnover": float(to.mean()),
    }


def factor_horizon_row(
    factor_name: str,
    family: str,
    horizon: int,
    legacy: dict,
    executable: dict,
    n_factors: int,
) -> dict:
    leg = float(legacy.get("rank_ic_mean", np.nan))
    ex = float(executable.get("rank_ic_mean", np.nan))
    rec = {
        "factor": factor_name,
        "family": family,
        "horizon": int(horizon),
        "legacy_ic": leg,
        "exec_ic": ex,
        "ic_delta": ex - leg if np.isfinite(ex) and np.isfinite(leg) else float("nan"),
        "ic_abs_preservation": ic_preservation(leg, ex),
        "sign_preserved": bool(
            np.isfinite(leg)
            and np.isfinite(ex)
            and abs(leg) >= IC_ABS_FLOOR
            and np.sign(leg) == np.sign(ex)
            and abs(ex) > 0
        ),
        "legacy_icir": legacy.get("icir", float("nan")),
        "exec_icir": executable.get("icir", float("nan")),
        "legacy_pos_ic": legacy.get("positive_ic_fraction", float("nan")),
        "exec_pos_ic": executable.get("positive_ic_fraction", float("nan")),
        "legacy_coverage": legacy.get("coverage", float("nan")),
        "exec_coverage": executable.get("coverage", float("nan")),
        "legacy_hl_annu": legacy.get("hl_annu_ret", float("nan")),
        "exec_hl_annu": executable.get("hl_annu_ret", float("nan")),
        "legacy_hl_sharpe": legacy.get("hl_sharpe", float("nan")),
        "exec_hl_sharpe": executable.get("hl_sharpe", float("nan")),
        "legacy_turnover": legacy.get("turnover", float("nan")),
        "exec_turnover": executable.get("turnover", float("nan")),
        "class": classify_degradation(leg, ex),
    }
    return rec


def add_ranks(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["legacy_rank"] = np.nan
    out["exec_rank"] = np.nan
    out["rank_change"] = np.nan
    for h, g in out.groupby("horizon"):
        idx = g.index
        out.loc[idx, "legacy_rank"] = g["legacy_ic"].abs().rank(ascending=False, method="first")
        out.loc[idx, "exec_rank"] = g["exec_ic"].abs().rank(ascending=False, method="first")
        out.loc[idx, "rank_change"] = out.loc[idx, "exec_rank"] - out.loc[idx, "legacy_rank"]
    return out


def ranking_spearman(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in table.groupby("horizon"):
        a = g["legacy_ic"].abs()
        b = g["exec_ic"].abs()
        ok = a.notna() & b.notna()
        if int(ok.sum()) < 5:
            rho = float("nan")
        else:
            rho = float(a[ok].rank().corr(b[ok].rank(), method="spearman"))
        rows.append({"horizon": int(h), "spearman_abs_ic_ranking": rho, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def fs_survival_table(
    table: pd.DataFrame,
    selected_names: Sequence[str],
    *,
    horizon: int = 5,
) -> pd.DataFrame:
    """Audit frozen FS selected names under executable V2V. Not a refit."""
    names = list(selected_names)
    g_all = table.loc[table["horizon"] == int(horizon)].copy()
    n_univ = int(len(g_all))
    top_q = max(1, int(np.ceil(n_univ * 0.25))) if n_univ else 0
    g = g_all.loc[g_all["factor"].isin(names)].copy()
    still_q = int((g["exec_rank"] <= top_q).sum()) if len(g) else 0
    still_pos = int((g["exec_ic"] > 0).sum()) if len(g) else 0
    flip = int(
        (
            np.sign(g["legacy_ic"].astype(float)) != np.sign(g["exec_ic"].astype(float))
        ).sum()
    ) if len(g) else 0
    rec = {
        "selector": "F_KBEST_60_XGB_Y5",
        "horizon": int(horizon),
        "legacy_selected_count": int(len(names)),
        "n_found_in_audit": int(len(g)),
        "still_top_quartile_exec": still_q,
        "still_positive_exec_ic": still_pos,
        "sign_flip_count": flip,
        "median_exec_ic": float(g["exec_ic"].median()) if len(g) else float("nan"),
        "median_legacy_ic": float(g["legacy_ic"].median()) if len(g) else float("nan"),
        "top_quartile_rank_cutoff": top_q,
        "note": (
            "audit only; do not treat the frozen selected set as production-valid "
            "merely because some names survive"
        ),
    }
    return pd.DataFrame([rec])


def family_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, h), g in table.groupby(["family", "horizon"]):
        n = len(g)
        rows.append(
            {
                "family": fam,
                "horizon": int(h),
                "n_factors": n,
                "median_legacy_ic": float(g["legacy_ic"].median()),
                "median_exec_ic": float(g["exec_ic"].median()),
                "median_ic_delta": float(g["ic_delta"].median()),
                "fraction_sign_preserved": float(g["sign_preserved"].mean()),
                "fraction_ROBUST_EXECUTABLE": float((g["class"] == "ROBUST_EXECUTABLE").mean()),
                "fraction_DECAY_SENSITIVE": float((g["class"] == "DECAY_SENSITIVE").mean()),
                "fraction_TIMING_SENSITIVE": float((g["class"] == "TIMING_SENSITIVE").mean()),
                "fraction_INCONCLUSIVE": float((g["class"] == "INCONCLUSIVE").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "family"]).reset_index(drop=True)
