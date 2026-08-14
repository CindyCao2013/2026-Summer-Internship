"""Robust Alpha Score Engine: rank factors by IC strength × cross-universe/regime stability.

Production research target:
    max IC × Stability   (not max IC alone)

Universe groups: CSI300, CSI500, CSI1000, ALL
Regime groups (when rank_ic_daily available): bull/bear × high-vol/low-vol on CSI300 index
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

UNIVERSE_ORDER = ["CSI300", "CSI500", "CSI1000", "ALL"]
REGIME_NAMES = ["bull", "bear", "high_vol", "low_vol"]


def collect_factor_summaries(result_glob: str = "result/**/summary.csv") -> pd.DataFrame:
    """Load all per-universe summary.csv files under result/."""
    rows = []
    for path in sorted(Path(".").glob(result_glob)):
        try:
            row = pd.read_csv(path).iloc[0].to_dict()
            row["summary_path"] = str(path)
            row["result_dir"] = str(path.parent)
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("rank_ic_mean", "abs_rank_ic_mean", "icir", "abs_icir", "hl_sharpe"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def dispersion_stability(values: pd.Series) -> float:
    """
    Stability = 1 - std(|values|) / |mean(values)|, clipped to [0, 1].
    Higher = more consistent across groups (universe or regime).
    """
    v = values.dropna()
    if len(v) < 2:
        return np.nan
    mean_v = v.mean()
    std_v = v.std()
    denom = abs(mean_v)
    if denom < 1e-8:
        return 0.0
    return float(np.clip(1.0 - std_v / denom, 0.0, 1.0))


def sign_consistency(values: pd.Series) -> float:
    """Fraction of groups sharing the same IC sign as the mean."""
    v = values.dropna()
    if len(v) == 0:
        return np.nan
    mean_sign = np.sign(v.mean())
    if mean_sign == 0:
        return 0.5
    return float((np.sign(v) == mean_sign).mean())


def universe_robustness_metrics(grp: pd.DataFrame) -> Dict[str, float]:
    """Aggregate cross-universe IC / Sharpe stability for one (track, factor)."""
    sub = grp.set_index("universe").reindex(UNIVERSE_ORDER).dropna(subset=["rank_ic_mean"])
    ic = sub["rank_ic_mean"]
    abs_ic = sub["abs_rank_ic_mean"] if "abs_rank_ic_mean" in sub else ic.abs()
    icir = sub["icir"] if "icir" in sub else pd.Series(dtype=float)
    sharpe = sub["hl_sharpe"] if "hl_sharpe" in sub else pd.Series(dtype=float)

    return {
        "n_universes": float(len(sub)),
        "mean_ic": float(ic.mean()),
        "mean_abs_ic": float(abs_ic.mean()),
        "std_ic": float(ic.std()) if len(ic) > 1 else 0.0,
        "min_ic": float(ic.min()),
        "max_ic": float(ic.max()),
        "ic_range": float(ic.max() - ic.min()),
        "universe_stability": dispersion_stability(ic),
        "sign_consistency": sign_consistency(ic),
        "mean_icir": float(icir.mean()) if len(icir.dropna()) else np.nan,
        "mean_abs_icir": float(icir.abs().mean()) if len(icir.dropna()) else np.nan,
        "mean_hl_sharpe": float(sharpe.mean()) if len(sharpe.dropna()) else np.nan,
        "std_hl_sharpe": float(sharpe.std()) if len(sharpe.dropna()) > 1 else np.nan,
        "all_universe_ic_hit": float((abs_ic >= 0.02).mean()),
    }


def load_rank_ic_daily(result_dir: str) -> Optional[pd.Series]:
    path = Path(result_dir) / "rank_ic_daily.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def classify_regime_days(
    market_ret: pd.Series,
    market_vol: pd.Series,
    vol_window: int = 60,
    vol_median_window: int = 252,
) -> pd.DataFrame:
    """Label each day: bull/bear from 60d cum ret; high_vol/low_vol from rolling vol vs median."""
    mret = market_ret.sort_index()
    cum_60 = (1 + mret).rolling(60, min_periods=30).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    vol = market_vol.reindex(mret.index).fillna(method="ffill")
    vol_med = vol.rolling(vol_median_window, min_periods=60).median()

    labels = pd.DataFrame(index=mret.index)
    labels["bull"] = cum_60 >= 0
    labels["bear"] = cum_60 < 0
    labels["high_vol"] = vol >= vol_med
    labels["low_vol"] = vol < vol_med
    return labels


def regime_ic_table(rank_ic_daily: pd.Series, regime_labels: pd.DataFrame) -> pd.Series:
    """Mean daily rank IC in each regime bucket."""
    aligned = pd.concat([rank_ic_daily.rename("ic"), regime_labels], axis=1).dropna(subset=["ic"])
    out = {}
    for name in REGIME_NAMES:
        if name not in aligned.columns:
            continue
        mask = aligned[name].astype(bool)
        if mask.sum() < 20:
            out[name] = np.nan
        else:
            out[name] = float(aligned.loc[mask, "ic"].mean())
    return pd.Series(out)


def regime_robustness_metrics(regime_ic: pd.Series) -> Dict[str, float]:
    ic = regime_ic.dropna()
    if len(ic) == 0:
        return {
            "n_regimes": 0.0,
            "mean_regime_ic": np.nan,
            "regime_stability": np.nan,
            "regime_sign_consistency": np.nan,
        }
    return {
        "n_regimes": float(len(ic)),
        "mean_regime_ic": float(ic.mean()),
        "regime_stability": dispersion_stability(ic),
        "regime_sign_consistency": sign_consistency(ic),
    }


def production_score(row: pd.Series) -> float:
    """
    Robust production score = strength × universe_stability × sign_consistency.
    Optional regime_stability multiplier when available.
    """
    strength = row.get("mean_abs_ic", 0) or 0
    uni_stab = row.get("universe_stability", 0) or 0
    sign_cons = row.get("sign_consistency", 0) or 0
    ic_hit = row.get("all_universe_ic_hit", 0) or 0

    base = strength * uni_stab * sign_cons * (0.5 + 0.5 * ic_hit)

    regime_stab = row.get("regime_stability", np.nan)
    if pd.notna(regime_stab):
        base *= 0.5 + 0.5 * regime_stab
    return float(base)


def build_robust_ranking(
    summaries: pd.DataFrame,
    market_ret: Optional[pd.Series] = None,
    market_vol: Optional[pd.Series] = None,
    min_universes: int = 3,
) -> pd.DataFrame:
    """
    Build production ranking table keyed by (track, factor_name).
    Optionally attach regime stability from rank_ic_daily.csv (ALL universe preferred).
    """
    if summaries.empty:
        return pd.DataFrame()

    regime_labels = None
    if market_ret is not None and market_vol is not None:
        regime_labels = classify_regime_days(market_ret, market_vol)

    records = []
    key_cols = ["track", "factor_name"]
    for key, grp in summaries.groupby(key_cols, dropna=False):
        track, fname = key
        metrics = universe_robustness_metrics(grp)
        if metrics["n_universes"] < min_universes:
            continue

        rec = {"track": track, "factor_name": fname, **metrics}

        # Regime IC: prefer ALL universe daily IC series
        regime_ic = pd.Series(dtype=float)
        for uni in ["ALL", "CSI1000", "CSI500", "CSI300"]:
            sub = grp[grp["universe"] == uni]
            if sub.empty:
                continue
            result_dir = sub.iloc[0].get("result_dir", "")
            daily = load_rank_ic_daily(result_dir)
            if daily is not None and regime_labels is not None:
                regime_ic = regime_ic_table(daily, regime_labels)
                rec["regime_ic_source_universe"] = uni
                break

        rec.update({f"ic_{k}": v for k, v in regime_ic.items()})
        rec.update(regime_robustness_metrics(regime_ic))
        rec["production_score"] = production_score(pd.Series(rec))
        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values("production_score", ascending=False).reset_index(drop=True)


def monotonicity_score(group_pnl_path: Path, direction: int) -> float:
    """Fraction of adjacent decile pairs in the expected direction."""
    if not group_pnl_path.exists():
        return np.nan
    g = pd.read_csv(group_pnl_path, index_col=0)
    groups = [str(i) for i in range(1, 11)]
    if not all(c in g.columns for c in groups):
        return np.nan
    means = g[groups].mean().values
    diffs = np.diff(means)
    if direction == -1:
        return float((diffs <= 0).mean())
    return float((diffs >= 0).mean())


def attach_monotonicity(ranking: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """Add ALL-universe group-test monotonicity to ranking table."""
    mono_scores = []
    for _, row in ranking.iterrows():
        sub = summaries[
            (summaries["track"] == row["track"])
            & (summaries["factor_name"] == row["factor_name"])
            & (summaries["universe"] == "ALL")
        ]
        if sub.empty:
            mono_scores.append(np.nan)
            continue
        result_dir = Path(sub.iloc[0].get("result_dir", ""))
        direction = int(sub.iloc[0].get("direction", 1))
        mono_scores.append(monotonicity_score(result_dir / "group_pnl.csv", direction))
    ranking = ranking.copy()
    ranking["mono_score_all"] = mono_scores
    return ranking
