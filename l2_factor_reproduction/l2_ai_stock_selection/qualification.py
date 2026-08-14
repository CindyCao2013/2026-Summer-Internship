"""Executable secondary qualification of the 127 frozen L2 features.

Gates are module constants, frozen before factor names are inspected.
Do not retune after seeing the surviving set. Do not train trees here.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import IC_ABS_FLOOR


# --- Pre-registered gates (do not optimize after seeing names) ---
CORE_ABS_IC = 0.02
CORE_ABS_HL_SHARPE = 3.0
CORE_ABS_MONO = 0.70
CORE_MIN_COVERAGE = 0.50
CORE_MIN_SIGN_CONSISTENCY = 0.55
CORE_MIN_IC_DAYS = 120

AUX_ABS_IC = 0.01
AUX_ABS_HL_SHARPE = 2.0
AUX_ABS_MONO = 0.60
AUX_MIN_EVIDENCE = 2
AUX_MIN_COVERAGE = 0.50
AUX_MIN_SIGN_CONSISTENCY = 0.52

NL_ABS_IC_WEAK = 0.02
NL_MI_FLOOR_LINEAR_WEAK = 0.01
NL_MI_FLOOR_NONMONO = 0.015
NL_ABS_MONO_NONMONO = 0.40
NL_IC_ABS_VERY_WEAK = 0.008

DECAY_LEGACY_ABS_IC = 0.02
DECAY_LEGACY_ABS_IC_ALT = 0.008
DECAY_LEGACY_ABS_SHARPE = 2.0
DECAY_PRESERVATION = 0.50

LOW_COVERAGE = 0.50
ONE_PERIOD_ABS_IC = 0.02
ONE_PERIOD_OTHER_ABS_IC = 0.008
DOMINATED_YEAR_RATIO = 3.0

N_GROUPS = 10
MI_MAX_SAMPLES = 8000

PRIMARY_CLASSES = (
    "CORE_ALPHA",
    "AUXILIARY_ALPHA",
    "NONLINEAR_REVIEW",
    "DECAY_TIMING_SENSITIVE",
    "DROP",
)

# TC-1 cuttable parents used only as positive controls, not cherry-picked CORE.
POSITIVE_CONTROL_PARENTS = (
    "obi_l5_mean",
    "signed_amount_impact",
    "cancel_value_pressure",
)

PERIODS = (
    ("full", "2023-01-01", "2024-12-31"),
    ("y2023", "2023-01-01", "2023-12-31"),
    ("y2024", "2024-01-01", "2024-12-31"),
    ("h2023h1", "2023-01-01", "2023-06-30"),
    ("h2023h2", "2023-07-01", "2023-12-31"),
    ("h2024h1", "2024-01-01", "2024-06-30"),
    ("h2024h2", "2024-07-01", "2024-12-31"),
)


def gates_dict() -> Dict[str, object]:
    return {
        "frozen_before_names": True,
        "execution_contract": "EXEC_V2V_TPLUS1_V1",
        "core": {
            "abs_rank_ic": CORE_ABS_IC,
            "abs_hl_sharpe": CORE_ABS_HL_SHARPE,
            "abs_monotonicity": CORE_ABS_MONO,
            "min_coverage": CORE_MIN_COVERAGE,
            "min_sign_consistency": CORE_MIN_SIGN_CONSISTENCY,
            "min_ic_days": CORE_MIN_IC_DAYS,
            "all_three_metrics_required": True,
        },
        "auxiliary": {
            "abs_rank_ic": AUX_ABS_IC,
            "abs_hl_sharpe": AUX_ABS_HL_SHARPE,
            "abs_monotonicity": AUX_ABS_MONO,
            "min_evidence_pieces": AUX_MIN_EVIDENCE,
            "min_coverage": AUX_MIN_COVERAGE,
            "min_sign_consistency": AUX_MIN_SIGN_CONSISTENCY,
            "no_one_metric_auto_pass": True,
        },
        "nonlinear_review": {
            "abs_rank_ic_weak": NL_ABS_IC_WEAK,
            "mi_floor_if_linear_very_weak": NL_MI_FLOOR_LINEAR_WEAK,
            "mi_floor_if_nonmonotonic": NL_MI_FLOOR_NONMONO,
            "abs_mono_nonmonotonic": NL_ABS_MONO_NONMONO,
            "tree_gain_not_used": True,
        },
        "decay_timing": {
            "legacy_abs_ic": DECAY_LEGACY_ABS_IC,
            "legacy_abs_ic_alt": DECAY_LEGACY_ABS_IC_ALT,
            "legacy_abs_sharpe": DECAY_LEGACY_ABS_SHARPE,
            "exec_preservation_below": DECAY_PRESERVATION,
            "not_equivalent_to_drop": True,
        },
        "one_period_dominated": {
            "abs_ic_floor": ONE_PERIOD_ABS_IC,
            "other_abs_ic": ONE_PERIOD_OTHER_ABS_IC,
            "year_ratio": DOMINATED_YEAR_RATIO,
        },
        "positive_control_parents": list(POSITIVE_CONTROL_PARENTS),
        "do_not_optimize_after_names": True,
    }


def _finite_abs(x) -> float:
    v = float(x) if x is not None and np.isfinite(x) else float("nan")
    return abs(v) if np.isfinite(v) else float("nan")


def ic_from_daily(ic: pd.Series, coverage: float) -> dict:
    ic = pd.Series(ic).dropna()
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std()) if n > 1 else float("nan")
    icir = float(mean / std * np.sqrt(250.0)) if np.isfinite(std) and std > 0 else float("nan")
    pos = float((ic > 0).mean()) if n else float("nan")
    if n and np.isfinite(mean) and abs(mean) > 0:
        sign_cons = float((np.sign(ic) == np.sign(mean)).mean())
    else:
        sign_cons = float("nan")
    return {
        "rank_ic_mean": mean,
        "icir": icir,
        "positive_ic_fraction": pos,
        "sign_consistency": sign_cons,
        "coverage": float(coverage) if np.isfinite(coverage) else float("nan"),
        "n_ic_days": n,
    }


def hl_from_daily(pnl: pd.Series) -> dict:
    pnl = pd.Series(pnl).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pnl) < 20:
        return {"hl_annu_ret": float("nan"), "hl_sharpe": float("nan")}
    annu = float(pnl.mean() * 250.0)
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(250.0)) if pnl.std() > 0 else float("nan")
    return {"hl_annu_ret": annu, "hl_sharpe": sharpe}


def daily_hl_and_deciles(
    factor: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_groups: int = N_GROUPS,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Daily H-L (G10-G1, raw direction) and daily mean y by decile."""
    f = factor
    yy = y.reindex_like(f)
    ranks = f.rank(axis=1, method="first")
    n = float(f.shape[1])
    if n < n_groups * 3:
        empty = pd.Series(index=f.index, dtype=float)
        return empty, pd.DataFrame(index=f.index)
    lo = n / n_groups
    hi = n - lo
    high = ranks > hi
    low = ranks <= lo
    c_h = high.sum(axis=1).replace(0, np.nan)
    c_l = low.sum(axis=1).replace(0, np.nan)
    hl = (high.astype(float).mul(yy).sum(axis=1) / c_h) - (
        low.astype(float).mul(yy).sum(axis=1) / c_l
    )
    dec = np.ceil(ranks.to_numpy(dtype=float) * n_groups / max(n, 1.0))
    dec = np.clip(dec, 1, n_groups)
    yv = yy.to_numpy(dtype=float)
    cols = ["d{}".format(g) for g in range(1, n_groups + 1)]
    means = np.full((f.shape[0], n_groups), np.nan, dtype=float)
    for g in range(1, n_groups + 1):
        m = dec == g
        with np.errstate(invalid="ignore"):
            num = np.nansum(np.where(m, yv, np.nan), axis=1)
            den = np.sum(m & np.isfinite(yv), axis=1).astype(float)
            den[den == 0] = np.nan
            means[:, g - 1] = num / den
    dec_df = pd.DataFrame(means, index=f.index, columns=cols)
    return hl, dec_df


def monotonicity_from_deciles(dec_df: pd.DataFrame) -> float:
    if dec_df is None or dec_df.empty:
        return float("nan")
    avg = dec_df.mean(axis=0, skipna=True).astype(float)
    if int(avg.notna().sum()) < 5:
        return float("nan")
    order = np.arange(1, len(avg) + 1, dtype=float)
    return float(pd.Series(order).corr(avg.reset_index(drop=True), method="spearman"))


def one_period_dominated(period_ics: Dict[str, float]) -> bool:
    halves = [period_ics.get(k, float("nan")) for k in ("h2023h1", "h2023h2", "h2024h1", "h2024h2")]
    abs_h = [abs(v) if np.isfinite(v) else 0.0 for v in halves]
    if max(abs_h) >= ONE_PERIOD_ABS_IC and sum(a >= ONE_PERIOD_OTHER_ABS_IC for a in abs_h) <= 1:
        return True
    a = abs(period_ics.get("y2023", float("nan"))) if np.isfinite(period_ics.get("y2023", np.nan)) else 0.0
    b = abs(period_ics.get("y2024", float("nan"))) if np.isfinite(period_ics.get("y2024", np.nan)) else 0.0
    hi, lo = max(a, b), min(a, b)
    if hi >= ONE_PERIOD_ABS_IC and lo < ONE_PERIOD_OTHER_ABS_IC and hi >= DOMINATED_YEAR_RATIO * max(lo, 1e-12):
        return True
    return False


def core_metric_ok(row: dict) -> bool:
    return (
        _finite_abs(row.get("rank_ic_mean")) >= CORE_ABS_IC
        and _finite_abs(row.get("hl_sharpe")) >= CORE_ABS_HL_SHARPE
        and _finite_abs(row.get("monotonicity")) >= CORE_ABS_MONO
    )


def core_stability_ok(row: dict) -> Tuple[bool, List[str]]:
    reasons = []
    cov = float(row.get("coverage", np.nan))
    sc = float(row.get("sign_consistency", np.nan))
    n = int(row.get("n_ic_days", 0) or 0)
    if not (np.isfinite(cov) and cov >= CORE_MIN_COVERAGE):
        reasons.append("LOW_COVERAGE")
    if not (np.isfinite(sc) and sc >= CORE_MIN_SIGN_CONSISTENCY):
        reasons.append("UNSTABLE_SIGN")
    if n < CORE_MIN_IC_DAYS:
        reasons.append("SHORT_SAMPLE")
    if bool(row.get("one_period_dominated", False)):
        reasons.append("ONE_PERIOD_DOMINATED")
    return (len(reasons) == 0, reasons)


def aux_evidence_count(row: dict) -> Tuple[int, List[str]]:
    bits = []
    if _finite_abs(row.get("rank_ic_mean")) >= AUX_ABS_IC:
        bits.append("IC")
    if _finite_abs(row.get("hl_sharpe")) >= AUX_ABS_HL_SHARPE:
        bits.append("SHARPE")
    if _finite_abs(row.get("monotonicity")) >= AUX_ABS_MONO:
        bits.append("MONO")
    return len(bits), bits


def nonlinear_ok(row: dict) -> bool:
    ic = _finite_abs(row.get("rank_ic_mean"))
    mi = float(row.get("mutual_information", np.nan))
    mono = _finite_abs(row.get("monotonicity"))
    if not np.isfinite(ic):
        return False
    if ic >= NL_ABS_IC_WEAK:
        return False
    if np.isfinite(mi) and mi >= NL_MI_FLOOR_LINEAR_WEAK and ic < NL_IC_ABS_VERY_WEAK:
        return True
    if np.isfinite(mi) and mi >= NL_MI_FLOOR_NONMONO and (not np.isfinite(mono) or mono < NL_ABS_MONO_NONMONO):
        return True
    spread = float(row.get("bin_spread", np.nan))
    if np.isfinite(mi) and mi >= NL_MI_FLOOR_NONMONO and np.isfinite(spread) and abs(spread) > 0:
        # U-shape proxy: MI strong, linear mono weak already required above
        return False
    return False


def decay_ok(legacy: dict, executable: dict) -> bool:
    leg_ic = float(legacy.get("rank_ic_mean", np.nan))
    ex_ic = float(executable.get("rank_ic_mean", np.nan))
    leg_sh = float(legacy.get("hl_sharpe", np.nan))
    ex_sh = float(executable.get("hl_sharpe", np.nan))
    if not np.isfinite(leg_ic):
        return False
    legacy_strong = abs(leg_ic) >= DECAY_LEGACY_ABS_IC or (
        abs(leg_ic) >= DECAY_LEGACY_ABS_IC_ALT
        and np.isfinite(leg_sh)
        and abs(leg_sh) >= DECAY_LEGACY_ABS_SHARPE
    )
    if not legacy_strong:
        return False
    if not np.isfinite(ex_ic):
        return True
    sign_flip = np.sign(leg_ic) != np.sign(ex_ic) and abs(leg_ic) >= IC_ABS_FLOOR
    ic_drop = abs(leg_ic) >= IC_ABS_FLOOR and abs(ex_ic) < DECAY_PRESERVATION * abs(leg_ic)
    sh_drop = (
        np.isfinite(leg_sh)
        and abs(leg_sh) >= DECAY_LEGACY_ABS_SHARPE
        and (not np.isfinite(ex_sh) or abs(ex_sh) < DECAY_PRESERVATION * abs(leg_sh))
    )
    return bool(sign_flip or ic_drop or sh_drop)


def classify_factor(
    name: str,
    family: str,
    horizon_rows: Sequence[dict],
    *,
    cluster: str = "",
    cluster_core_aux_count: int = 1,
) -> dict:
    """Assign one primary class plus secondary flags. Gates are module constants."""
    rows = list(horizon_rows)
    if not rows:
        return {
            "factor_name": name,
            "family": family,
            "best_horizon": np.nan,
            "classification_primary": "DROP",
            "secondary_flags": "NO_METRICS",
            "core_horizons": "",
            "exceptions": "NO_METRICS",
        }

    def _score(r):
        return _finite_abs(r.get("rank_ic_mean"))

    best = max(rows, key=lambda r: _score(r) if np.isfinite(_score(r)) else -1.0)
    best_h = int(best["horizon"])

    core_hs = []
    exceptions = []
    flags = []
    decay_hs = []
    nl_hs = []
    aux_hs = []

    for r in rows:
        h = int(r["horizon"])
        metric_ok = core_metric_ok(r)
        stab_ok, reasons = core_stability_ok(r)
        if metric_ok and stab_ok:
            core_hs.append(h)
        elif metric_ok and not stab_ok:
            exceptions.append("h{}:{}".format(h, "+".join(reasons)))
            flags.extend(reasons)
        n_ev, _ = aux_evidence_count(r)
        cov_ok = np.isfinite(r.get("coverage", np.nan)) and float(r["coverage"]) >= AUX_MIN_COVERAGE
        sc_ok = np.isfinite(r.get("sign_consistency", np.nan)) and float(r["sign_consistency"]) >= AUX_MIN_SIGN_CONSISTENCY
        if n_ev >= AUX_MIN_EVIDENCE and cov_ok and sc_ok:
            aux_hs.append(h)
        if nonlinear_ok(r):
            nl_hs.append(h)
        if decay_ok(r.get("legacy", {}), r):
            decay_hs.append(h)
        if bool(r.get("one_period_dominated", False)):
            flags.append("ONE_PERIOD_DOMINATED")
        if np.isfinite(r.get("coverage", np.nan)) and float(r["coverage"]) < LOW_COVERAGE:
            flags.append("LOW_COVERAGE")

    if 5 in core_hs:
        flags.append("CORE_AT_5D")
    if nl_hs:
        flags.append("NONLINEAR_CANDIDATE")
    if decay_hs:
        flags.append("DECAY_SENSITIVE")
    if cluster and cluster_core_aux_count >= 2:
        flags.append("REDUNDANCY_REVIEW")

    if core_hs:
        primary = "CORE_ALPHA"
    elif aux_hs:
        primary = "AUXILIARY_ALPHA"
    elif nl_hs:
        primary = "NONLINEAR_REVIEW"
    elif decay_hs:
        primary = "DECAY_TIMING_SENSITIVE"
    else:
        primary = "DROP"

    # Deduplicate flags, keep order
    seen = set()
    flag_list = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            flag_list.append(f)

    return {
        "factor_name": name,
        "family": family,
        "best_horizon": best_h,
        "best_abs_rank_ic": _score(best),
        "best_rank_ic": float(best.get("rank_ic_mean", np.nan)),
        "best_hl_sharpe": float(best.get("hl_sharpe", np.nan)),
        "best_monotonicity": float(best.get("monotonicity", np.nan)),
        "best_mi": float(best.get("mutual_information", np.nan)),
        "best_coverage": float(best.get("coverage", np.nan)),
        "best_sign_consistency": float(best.get("sign_consistency", np.nan)),
        "classification_primary": primary,
        "secondary_flags": "|".join(flag_list),
        "core_horizons": ",".join(str(h) for h in sorted(core_hs)),
        "aux_horizons": ",".join(str(h) for h in sorted(aux_hs)),
        "nl_horizons": ",".join(str(h) for h in sorted(nl_hs)),
        "decay_horizons": ",".join(str(h) for h in sorted(decay_hs)),
        "exceptions": ";".join(exceptions),
        "redundancy_cluster_080": cluster,
    }


def build_tc2_parent_pool(class_table: pd.DataFrame) -> pd.DataFrame:
    """NONLINEAR_REVIEW + DECAY_TIMING_SENSITIVE flag + small positive controls.

    Does not generate descendants.
    """
    rows = []
    for _, r in class_table.iterrows():
        name = str(r["factor_name"])
        primary = str(r["classification_primary"])
        flags = str(r.get("secondary_flags") or "")
        reasons = []
        if primary == "NONLINEAR_REVIEW" or "NONLINEAR_CANDIDATE" in flags:
            if primary == "NONLINEAR_REVIEW":
                reasons.append("NONLINEAR_STRUCTURAL_RESCUE")
        if primary == "DECAY_TIMING_SENSITIVE" or "DECAY_SENSITIVE" in flags:
            if primary != "CORE_ALPHA":
                reasons.append("TIMING_LOCALIZATION")
        if name in POSITIVE_CONTROL_PARENTS and primary in ("CORE_ALPHA", "AUXILIARY_ALPHA"):
            reasons.append("POSITIVE_CONTROL")
        # Pool membership: NL primary, decay primary, decay flag on non-CORE, positive controls
        keep = False
        if primary == "NONLINEAR_REVIEW":
            keep = True
        if primary == "DECAY_TIMING_SENSITIVE":
            keep = True
        if "DECAY_SENSITIVE" in flags and primary != "CORE_ALPHA":
            keep = True
        if "POSITIVE_CONTROL" in reasons:
            keep = True
        if not keep:
            continue
        reason = "+".join(reasons) if reasons else "NONLINEAR_STRUCTURAL_RESCUE"
        rows.append(
            {
                "factor_name": name,
                "family": r.get("family", ""),
                "classification_primary": primary,
                "best_horizon": r.get("best_horizon", np.nan),
                "reason_for_tc2": reason,
                "secondary_flags": flags,
                "best_abs_rank_ic": r.get("best_abs_rank_ic", np.nan),
            }
        )
    return pd.DataFrame(rows)


def family_qualification_summary(
    class_table: pd.DataFrame,
    horizon_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fam, g in class_table.groupby("family"):
        hg = horizon_table.loc[horizon_table["family"] == fam]
        rows.append(
            {
                "family": fam,
                "n_factors": int(len(g)),
                "CORE_ALPHA": int((g["classification_primary"] == "CORE_ALPHA").sum()),
                "AUXILIARY_ALPHA": int((g["classification_primary"] == "AUXILIARY_ALPHA").sum()),
                "NONLINEAR_REVIEW": int((g["classification_primary"] == "NONLINEAR_REVIEW").sum()),
                "DECAY_TIMING_SENSITIVE": int(
                    (g["classification_primary"] == "DECAY_TIMING_SENSITIVE").sum()
                ),
                "DROP": int((g["classification_primary"] == "DROP").sum()),
                "n_decay_flag": int(g["secondary_flags"].fillna("").str.contains("DECAY_SENSITIVE").sum()),
                "median_exec_ic": float(g["best_rank_ic"].median()) if len(g) else float("nan"),
                "median_abs_exec_ic": float(g["best_abs_rank_ic"].median()) if len(g) else float("nan"),
                "median_hl_sharpe": float(g["best_hl_sharpe"].abs().median()) if len(g) else float("nan"),
                "median_monotonicity": float(g["best_monotonicity"].abs().median()) if len(g) else float("nan"),
                "median_ic_delta": float(hg["ic_delta"].median()) if len(hg) and "ic_delta" in hg.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)


def slice_dates(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    return dates[(dates >= lo) & (dates <= hi)]
