"""BDL gates 0/1/3 and survivor classification.

Lite metrics are named ``*_lite``. Direction stays raw/frozen — no sign flip
to make IC positive. Gate 0 uses no future returns.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.discovery_lite.contracts import (
    CONSTANT_DATE_THRESHOLD,
    COVERAGE_THRESHOLD,
    DECILE_MONO_THRESHOLD,
    ICIR_ANNUALIZATION,
    ICIR_THRESHOLD,
    MAX_CLUSTER_REPRESENTATIVES,
    MIN_CROSS_SECTION,
    MIN_IC_DATES,
    N_DECILES,
    NEAR_ALIAS_THRESHOLD,
    NONFINITE_RATIO_THRESHOLD,
    PRIORITY_WEIGHTS,
    RANK_IC_THRESHOLD,
    SIGNAL_SHIFT,
    SPREAD_SIGN_CONSISTENCY_THRESHOLD,
)
from l2_factor_reproduction.python.backtest import compute_rank_ic, prepare_factor_signal

GATE0_PASS = "PASS"
REJECT_LOW_COVERAGE = "REJECT_LOW_COVERAGE"
REJECT_CONSTANT = "REJECT_CONSTANT"
REJECT_NONFINITE = "REJECT_NONFINITE"
REJECT_PIT = "REJECT_PIT"
REJECT_MISSING_PRIMITIVE = "REJECT_MISSING_PRIMITIVE"
SPARSE_EVENT_REVIEW = "SPARSE_EVENT_REVIEW"
REJECT_FORMULA = "REJECT_FORMULA"

FULL_DISCOVERY_SURVIVOR = "FULL_DISCOVERY_SURVIVOR"
REVIEW_SURVIVOR = "REVIEW_SURVIVOR"
REJECT_LOW_SIGNAL = "REJECT_LOW_SIGNAL"
REJECT_REDUNDANT = "REJECT_REDUNDANT"
REJECT_NEAR_ALIAS = "REJECT_NEAR_ALIAS"
REJECT_BAD_SHAPE = "REJECT_BAD_SHAPE"
NEAR_ALIAS_REVIEW = "NEAR_ALIAS_REVIEW"


def _finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def coverage_metrics(
    wide: pd.DataFrame,
    mask: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> Dict[str, float]:
    """Coverage on ``dates`` vs tradable universe. NA is not treated as 0."""
    empty = {
        "n_rows": 0,
        "n_dates": 0,
        "n_symbols": 0,
        "date_coverage": 0.0,
        "symbol_coverage": 0.0,
        "row_coverage": 0.0,
        "nonfinite_ratio": 0.0,
        "missing_ratio": 1.0,
        "zero_ratio": 0.0,
        "cross_section_std_median": float("nan"),
        "cross_section_unique_ratio_median": float("nan"),
        "constant_date_fraction": 1.0,
    }
    if wide is None or wide.empty or len(dates) == 0:
        return empty

    idx = wide.index.intersection(dates).intersection(mask.index)
    cols = wide.columns.intersection(mask.columns)
    if len(idx) == 0 or len(cols) == 0:
        return empty

    factor = wide.loc[idx, cols]
    eligible = mask.reindex(index=idx, columns=cols).fillna(0).eq(1)
    observed = factor.notna() & eligible
    values = factor.where(eligible)

    n_eligible = int(eligible.to_numpy().sum())
    n_obs = int(observed.to_numpy().sum())
    arr = values.to_numpy(dtype=float, copy=False)
    finite = _finite_mask(arr)
    non_missing = ~np.isnan(arr)
    n_non_missing = int(non_missing.sum())
    n_nonfinite = int((non_missing & ~finite).sum())
    n_zero = int((finite & (arr == 0)).sum())

    date_has = observed.any(axis=1)
    symbol_has = observed.any(axis=0)
    n_dates_cal = int(len(dates))
    n_symbols_univ = int(mask.columns.nunique()) if mask.shape[1] else int(len(cols))

    stds = []
    unique_ratios = []
    constant_flags = []
    for dt in idx:
        row = values.loc[dt]
        finite_row = row[np.isfinite(row.to_numpy(dtype=float))]
        n = int(len(finite_row))
        if n <= 1:
            constant_flags.append(True)
            continue
        std = float(finite_row.std(ddof=0))
        nunique = int(finite_row.nunique(dropna=True))
        stds.append(std)
        unique_ratios.append(nunique / n)
        constant_flags.append(nunique <= 1 or std == 0.0)

    return {
        "n_rows": n_obs,
        "n_dates": int(date_has.sum()),
        "n_symbols": int(symbol_has.sum()),
        "date_coverage": float(date_has.sum() / n_dates_cal) if n_dates_cal else 0.0,
        "symbol_coverage": float(symbol_has.sum() / n_symbols_univ) if n_symbols_univ else 0.0,
        "row_coverage": float(n_obs / n_eligible) if n_eligible else 0.0,
        "nonfinite_ratio": float(n_nonfinite / n_non_missing) if n_non_missing else 0.0,
        "missing_ratio": float(1.0 - n_obs / n_eligible) if n_eligible else 1.0,
        "zero_ratio": float(n_zero / int(finite.sum())) if int(finite.sum()) else 0.0,
        "cross_section_std_median": float(np.median(stds)) if stds else float("nan"),
        "cross_section_unique_ratio_median": (
            float(np.median(unique_ratios)) if unique_ratios else float("nan")
        ),
        "constant_date_fraction": (
            float(np.mean(constant_flags)) if constant_flags else 1.0
        ),
    }


def infer_pit_status(row: pd.Series) -> str:
    explicit = str(row.get("pit_status", "") or "").strip().upper()
    if explicit in {"PASS", "FAIL", "INVALID", "UNKNOWN"}:
        return "PASS" if explicit == "PASS" else "FAIL"
    status = str(row.get("registry_status", "") or "").strip().lower()
    if status in {"frozen_baseline", "frozen", "validated"}:
        return "PASS"
    if status in {"pit_invalid", "lookahead"}:
        return "FAIL"
    return "PASS" if bool(row.get("formula", "")) else "FAIL"


def gate0_status(
    metrics: Mapping[str, float],
    *,
    formula_valid: bool,
    primitive_available: bool,
    pit_status: str,
    sparse_event: bool = False,
) -> str:
    if not formula_valid:
        return REJECT_FORMULA
    if str(pit_status).upper() != "PASS":
        return REJECT_PIT
    if not primitive_available:
        return REJECT_MISSING_PRIMITIVE
    if float(metrics.get("nonfinite_ratio", 0.0)) > NONFINITE_RATIO_THRESHOLD:
        return REJECT_NONFINITE
    low_cov = float(metrics.get("row_coverage", 0.0)) < COVERAGE_THRESHOLD
    constant = float(metrics.get("constant_date_fraction", 1.0)) > CONSTANT_DATE_THRESHOLD
    if sparse_event and (low_cov or constant):
        return SPARSE_EVENT_REVIEW
    if constant:
        return REJECT_CONSTANT
    if low_cov:
        return REJECT_LOW_COVERAGE
    return GATE0_PASS


def run_gate0(
    registry: pd.DataFrame,
    wides: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    lite_dates: pd.DatetimeIndex,
    availability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    avail = {}
    if availability is not None and not availability.empty:
        avail = availability.set_index("name").to_dict("index")
    rows = []
    for _, rec in registry.iterrows():
        name = str(rec["name"])
        wide = wides.get(name)
        metrics = coverage_metrics(wide if wide is not None else pd.DataFrame(), mask, lite_dates)
        info = avail.get(name, {})
        primitive_available = bool(info.get("primitive_available", name in wides))
        if info.get("load_status") == REJECT_MISSING_PRIMITIVE:
            primitive_available = False
        formula_valid = bool(str(rec.get("formula", "") or "").strip()) and bool(name.strip())
        pit = infer_pit_status(rec)
        sparse = bool(rec.get("sparse_event", False))
        status = gate0_status(
            metrics,
            formula_valid=formula_valid,
            primitive_available=primitive_available,
            pit_status=pit,
            sparse_event=sparse,
        )
        rows.append(
            {
                "factor": name,
                "family": rec.get("family", ""),
                "formula_valid": formula_valid,
                "primitive_available": primitive_available,
                "pit_status": pit,
                "gate0_status": status,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def aggregate_rank_ic_lite(rank_ic: pd.Series) -> Dict[str, float]:
    ic = pd.to_numeric(rank_ic, errors="coerce").dropna()
    n = int(len(ic))
    if n == 0:
        return {
            "mean_rank_ic_lite": float("nan"),
            "rank_ic_std_lite": float("nan"),
            "icir_lite": float("nan"),
            "positive_ic_fraction_lite": float("nan"),
            "negative_ic_fraction_lite": float("nan"),
            "sign_consistency_lite": float("nan"),
            "n_ic_dates": 0,
            "abs_rank_ic_lite": float("nan"),
            "abs_icir_lite": float("nan"),
        }
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = (
        mean / std * (ICIR_ANNUALIZATION ** 0.5)
        if std and np.isfinite(std) and std > 0
        else float("nan")
    )
    pos = float((ic > 0).mean())
    neg = float((ic < 0).mean())
    if mean == 0 or not np.isfinite(mean):
        sign_cons = float((ic == 0).mean())
    else:
        sign_cons = float((np.sign(ic) == np.sign(mean)).mean())
    return {
        "mean_rank_ic_lite": mean,
        "rank_ic_std_lite": std,
        "icir_lite": float(icir),
        "positive_ic_fraction_lite": pos,
        "negative_ic_fraction_lite": neg,
        "sign_consistency_lite": sign_cons,
        "n_ic_dates": n,
        "abs_rank_ic_lite": abs(mean),
        "abs_icir_lite": abs(icir) if np.isfinite(icir) else float("nan"),
    }


def rank_ic_lite_for_wide(
    wide: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
    lite_dates: pd.DatetimeIndex,
    *,
    start,
    end,
    signal_shift: int = SIGNAL_SHIFT,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Reuse prepare_factor_signal + Spearman, then restrict to lite dates."""
    signal, ret_a = prepare_factor_signal(
        wide,
        start=start,
        end=end,
        mask=mask,
        signal_shift=signal_shift,
        ret_matrix=ret,
    )
    keep = signal.index.intersection(lite_dates)
    if len(keep) == 0:
        empty = pd.Series(dtype=float, name="rank_ic_lite")
        return empty, aggregate_rank_ic_lite(empty)
    ic = compute_rank_ic(signal.loc[keep], ret_a.loc[keep])
    ic.name = "rank_ic_lite"
    return ic, aggregate_rank_ic_lite(ic)


def gate1_status(metrics: Mapping[str, float]) -> str:
    n = int(metrics.get("n_ic_dates", 0) or 0)
    if n < MIN_IC_DATES:
        return REJECT_LOW_SIGNAL
    abs_ic = abs(float(metrics.get("mean_rank_ic_lite", 0.0) or 0.0))
    abs_icir = abs(float(metrics.get("icir_lite", 0.0) or 0.0))
    if not np.isfinite(abs_ic):
        abs_ic = 0.0
    if not np.isfinite(abs_icir):
        abs_icir = 0.0
    if abs_ic >= RANK_IC_THRESHOLD or abs_icir >= ICIR_THRESHOLD:
        return "PASS"
    return REJECT_LOW_SIGNAL


def run_gate1(
    names: Sequence[str],
    wides: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    ret: pd.DataFrame,
    lite_dates: pd.DatetimeIndex,
    *,
    start,
    end,
    families: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    rows = []
    families = families or {}
    for name in names:
        wide = wides.get(name)
        if wide is None or wide.empty:
            metrics = aggregate_rank_ic_lite(pd.Series(dtype=float))
            status = REJECT_LOW_SIGNAL
        else:
            _ic, metrics = rank_ic_lite_for_wide(
                wide, mask, ret, lite_dates, start=start, end=end
            )
            status = gate1_status(metrics)
        rows.append(
            {
                "factor": name,
                "family": families.get(name, ""),
                "gate1_status": status,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def assign_deciles(values: pd.Series, n_groups: int = N_DECILES) -> pd.Series:
    """Deterministic equal-count deciles. Ties broken by rank method='first'."""
    s = pd.to_numeric(values, errors="coerce")
    valid = s.dropna()
    if len(valid) < n_groups:
        return pd.Series(index=values.index, dtype=float)
    ranks = valid.rank(method="first")
    bins = pd.cut(ranks, bins=n_groups, labels=False, include_lowest=True)
    out = pd.Series(index=values.index, dtype=float)
    out.loc[bins.index] = bins.to_numpy(dtype=float) + 1.0
    return out


def decile_lite_one_day(
    signal: pd.Series,
    ret: pd.Series,
    n_groups: int = N_DECILES,
    min_names: int = MIN_CROSS_SECTION,
) -> Optional[Dict[str, float]]:
    aligned = pd.concat({"signal": signal, "ret": ret}, axis=1).dropna()
    if len(aligned) < min_names:
        return None
    decile = assign_deciles(aligned["signal"], n_groups=n_groups)
    aligned = aligned.loc[decile.notna()].copy()
    aligned["decile"] = decile.loc[aligned.index]
    means = aligned.groupby("decile")["ret"].mean()
    if len(means) < 2:
        return None
    g1 = float(means.get(1.0, np.nan))
    g10 = float(means.get(float(n_groups), np.nan))
    spread = g10 - g1
    order = pd.Series(means.index.astype(float))
    mono = float(order.corr(means.reset_index(drop=True).astype(float), method="spearman"))
    return {
        "spread": spread,
        "mono": mono,
        "g1": g1,
        "g10": g10,
    }


def run_decile_lite(
    wide: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
    lite_dates: pd.DatetimeIndex,
    *,
    start,
    end,
) -> Dict[str, float]:
    """Raw-direction lite deciles. No NAV / Sharpe / turnover / cost."""
    signal, ret_a = prepare_factor_signal(
        wide,
        start=start,
        end=end,
        mask=mask,
        signal_shift=SIGNAL_SHIFT,
        ret_matrix=ret,
    )
    keep = signal.index.intersection(lite_dates)
    spreads = []
    monos = []
    for dt in keep:
        day = decile_lite_one_day(signal.loc[dt], ret_a.loc[dt])
        if day is None:
            continue
        spreads.append(day["spread"])
        if np.isfinite(day["mono"]):
            monos.append(day["mono"])
    if not spreads:
        return {
            "decile_mono_lite": float("nan"),
            "top_bottom_spread_lite": float("nan"),
            "spread_sign_consistency_lite": float("nan"),
            "n_decile_dates": 0,
        }
    spread = pd.Series(spreads, dtype=float)
    mean_spread = float(spread.mean())
    if mean_spread == 0 or not np.isfinite(mean_spread):
        sign_cons = float((spread == 0).mean())
    else:
        sign_cons = float((np.sign(spread) == np.sign(mean_spread)).mean())
    return {
        "decile_mono_lite": float(np.nanmean(monos)) if monos else float("nan"),
        "top_bottom_spread_lite": mean_spread,
        "spread_sign_consistency_lite": sign_cons,
        "n_decile_dates": int(len(spread)),
    }


def gate3_status(
    gate1_ok: bool,
    decile_metrics: Mapping[str, float],
    mean_rank_ic_lite: float,
) -> str:
    if not gate1_ok:
        return REJECT_LOW_SIGNAL
    mono = float(decile_metrics.get("decile_mono_lite", np.nan))
    spread_cons = float(decile_metrics.get("spread_sign_consistency_lite", np.nan))
    spread = float(decile_metrics.get("top_bottom_spread_lite", np.nan))
    mono_ok = np.isfinite(mono) and abs(mono) >= DECILE_MONO_THRESHOLD
    expected = np.sign(mean_rank_ic_lite) if np.isfinite(mean_rank_ic_lite) else 0.0
    spread_ok = (
        np.isfinite(spread_cons)
        and spread_cons >= SPREAD_SIGN_CONSISTENCY_THRESHOLD
        and np.isfinite(spread)
        and expected != 0
        and np.sign(spread) == expected
    )
    if mono_ok or spread_ok:
        return "PASS"
    return REJECT_BAD_SHAPE


def n_primitives(row: pd.Series) -> int:
    deps = str(row.get("primitive_dependencies", "") or "").strip()
    if not deps:
        return 1
    parts = [p for p in deps.replace(";", ",").split(",") if p.strip()]
    return max(len(parts), 1)


def select_cluster_representatives(
    cluster_df: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    max_keep: int = MAX_CLUSTER_REPRESENTATIVES,
) -> pd.DataFrame:
    """Deterministic 1–2 representatives. Does not use H-L Sharpe."""
    if cluster_df is None or cluster_df.empty:
        return pd.DataFrame(
            columns=list(cluster_df.columns) + ["is_representative", "representative_rank"]
            if cluster_df is not None
            else ["factor", "is_representative", "representative_rank"]
        )
    reg = registry.set_index("name") if "name" in registry.columns else registry
    out_rows = []
    for cluster, block in cluster_df.groupby("redundancy_cluster_080", sort=True):
        scored = []
        for _, row in block.iterrows():
            name = str(row["factor"])
            rec = reg.loc[name] if name in reg.index else pd.Series(dtype=object)
            pit_pass = str(row.get("pit_status", rec.get("pit_status", "PASS"))).upper() == "PASS"
            coverage = float(row.get("row_coverage", row.get("coverage", 0.0)) or 0.0)
            formula = str(rec.get("formula", "") or "")
            scored.append(
                (
                    name,
                    int(pit_pass),
                    coverage,
                    -n_primitives(rec),
                    -len(formula),
                    abs(float(row.get("icir_lite", 0.0) or 0.0)),
                    -float(row.get("max_abs_corr_to_existing", 0.5) or 0.5),
                )
            )
        scored.sort(key=lambda t: t[1:], reverse=True)
        keep = {t[0] for t in scored[:max_keep]}
        for _, row in block.iterrows():
            item = dict(row)
            item["is_representative"] = str(row["factor"]) in keep
            item["representative_rank"] = (
                [t[0] for t in scored].index(str(row["factor"])) + 1
            )
            out_rows.append(item)
    return pd.DataFrame(out_rows)


def discovery_priority_score(row: Mapping[str, float]) -> float:
    """Ordering aid only. Not an Alpha score. Not an accept/reject rule."""
    signal = min(1.0, abs(float(row.get("rank_ic_lite", 0.0) or 0.0)) / 0.05)
    stability = min(1.0, abs(float(row.get("icir_lite", 0.0) or 0.0)) / 4.0)
    shape_raw = float(row.get("decile_mono_lite", np.nan))
    shape = abs(shape_raw) if np.isfinite(shape_raw) else 0.0
    coverage = float(row.get("coverage", row.get("row_coverage", 0.0)) or 0.0)
    corr = float(row.get("max_abs_corr_to_existing", np.nan))
    if not np.isfinite(corr):
        novelty = 0.5
    else:
        novelty = max(0.0, 1.0 - min(1.0, corr / NEAR_ALIAS_THRESHOLD))
    w = PRIORITY_WEIGHTS
    return float(
        w["signal"] * signal
        + w["stability"] * stability
        + w["shape"] * shape
        + w["coverage"] * coverage
        + w["novelty"] * novelty
    )


def classify_survivor(
    *,
    gate0: str,
    gate1: str,
    gate2: str,
    gate3: str,
    novelty_bucket: str,
    is_representative: bool,
    near_alias_exception: bool,
) -> str:
    if gate0 == REJECT_PIT:
        return REJECT_PIT
    if gate0 in {REJECT_LOW_COVERAGE, REJECT_CONSTANT, REJECT_NONFINITE, REJECT_FORMULA}:
        return gate0 if gate0 != REJECT_FORMULA else REJECT_LOW_COVERAGE
    if gate0 == REJECT_MISSING_PRIMITIVE:
        return REJECT_MISSING_PRIMITIVE
    if gate0 == SPARSE_EVENT_REVIEW:
        return REVIEW_SURVIVOR
    if gate1 != "PASS":
        return REJECT_LOW_SIGNAL
    if not is_representative:
        return REJECT_REDUNDANT
    if novelty_bucket == "NEAR_ALIAS":
        if near_alias_exception:
            return NEAR_ALIAS_REVIEW
        return REJECT_NEAR_ALIAS
    if gate3 != "PASS":
        return REJECT_BAD_SHAPE
    if gate2 == NEAR_ALIAS_REVIEW:
        return NEAR_ALIAS_REVIEW
    return FULL_DISCOVERY_SURVIVOR
