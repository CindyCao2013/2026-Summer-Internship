"""Factor Evaluation Protocol v2.0 — dual-layer evaluation (FINAL FROZEN).

Turnover reporting:
  L1 traded notional = sum_i |Δw_i|          (engine internal)
  one-way turnover   = 0.5 * L1             (conventional report default)

Cost (unchanged economics):
  cost = L1 * 7.5bps  =  one-way * 15bps

Factor Grade and Long-only Status are independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from Factor_Dev_Lib import IMPLIED_ANNU_FEE_BPS, calAnnuRet, calMDD, calSharpe
from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.python.fast_discovery import (
    FULL_END,
    FULL_START,
    context_paths,
    ensure_effective_group_pnl,
)

PROTOCOL_VERSION = "2.0"
PROTOCOL_STATUS = "FROZEN"
FREEZE_DATE = "2026-08-11"

DISCOVERY_START = pd.Timestamp("2023-01-01")
DISCOVERY_END = pd.Timestamp("2024-12-31")
PRE_START = pd.Timestamp("2019-01-01")
PRE_END = pd.Timestamp("2022-12-31")
POST_START = pd.Timestamp("2025-01-01")
POST_END = pd.Timestamp("2026-07-31")

SAMPLES: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {
    "DISCOVERY": (DISCOVERY_START, DISCOVERY_END),
    "PRE": (PRE_START, PRE_END),
    "POST": (POST_START, POST_END),
    "FULL": (FULL_START, FULL_END),
}

FEE_BPS_PER_TRADED_NOTIONAL = float(IMPLIED_ANNU_FEE_BPS)  # 7.5
FEE_RATE_L1 = FEE_BPS_PER_TRADED_NOTIONAL / 1e4  # 0.00075 on L1
FEE_RATE_ONEWAY_EQUIV = FEE_RATE_L1 * 2.0  # 0.00150 on one-way (= 15bps)
# aliases kept for older scripts
FEE_BPS_ONE_WAY = FEE_BPS_PER_TRADED_NOTIONAL
FEE_RATE = FEE_RATE_L1
ANNUALIZATION_DAYS = 250
SIGNAL_SHIFT = "T+1"
EXECUTION = "daily_rebalance_equal_weight_deciles"
BENCHMARK = UNIVERSE

OUT_ROOT = Path(RESULT_ROOT) / "evaluation_protocol_v2"


def l1_to_oneway(l1: float) -> float:
    return 0.5 * float(l1)


def label_block(
    *,
    factor_id: str,
    factor_hash: str,
    sample: str,
    return_kind: str,
) -> Dict[str, str]:
    return {
        "FACTOR_ID": factor_id,
        "FACTOR_HASH": factor_hash,
        "SAMPLE": sample,
        "RETURN": return_kind,
        "EXECUTION": EXECUTION,
        "UNIVERSE": UNIVERSE,
        "BENCHMARK": BENCHMARK,
        "COST": (
            f"{FEE_BPS_PER_TRADED_NOTIONAL:g}bps per L1 traded notional "
            f"(= {2 * FEE_BPS_PER_TRADED_NOTIONAL:g}bps per one-way turnover)"
        ),
        "SIGNAL_SHIFT": SIGNAL_SHIFT,
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
    }


def apply_daily_fee_l1(
    pnl: pd.Series,
    l1_traded_notional: pd.Series,
    fee_rate_l1: float = FEE_RATE_L1,
) -> pd.Series:
    l1 = l1_traded_notional.reindex(pnl.index).fillna(0.0)
    return pnl - l1 * float(fee_rate_l1)


# back-compat name
apply_daily_fee = apply_daily_fee_l1


def _group_cols(frame: pd.DataFrame) -> List[str]:
    cols = [c for c in frame.columns if str(c) != "H-L"]
    return sorted(cols, key=lambda c: int(c))


def ensure_effective_group_to(
    group_to: pd.DataFrame,
    group_pnl: pd.DataFrame,
) -> pd.DataFrame:
    """Mirror ensure_effective_group_pnl for turnover columns.

    If H-L mean < 0, reverse decile labels. H-L L1 stays long+short.
    When backtest_factor already re-ran flipped signal, this is a no-op.
    """
    if "H-L" not in group_to.columns:
        raise ValueError("group_to must contain H-L column")
    pnl = group_pnl.copy()
    pnl.index = pd.to_datetime(pnl.index)
    to = group_to.copy()
    to.index = pd.to_datetime(to.index)
    to.columns = [str(c) for c in to.columns]
    group_cols = _group_cols(to)

    if float(pnl["H-L"].mean()) >= 0:
        ordered = to.loc[:, group_cols + ["H-L"]]
        ordered.columns = [str(c) for c in ordered.columns]
        return ordered

    n = len(group_cols)
    flipped = pd.DataFrame(index=to.index)
    for i, col in enumerate(group_cols):
        flipped[str(n - i)] = to[col]
    flipped["H-L"] = to["H-L"]
    ordered_cols = [str(i) for i in range(1, n + 1)] + ["H-L"]
    return flipped.loc[:, ordered_cols]


def split_leg_l1(group_to_eff: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """(long_l1=G10, short_l1=G1, hl_l1) on effective-direction table."""
    to = group_to_eff.copy()
    to.index = pd.to_datetime(to.index)
    to.columns = [str(c) for c in to.columns]
    cols = _group_cols(to)
    g1, g10 = cols[0], cols[-1]
    long_l1 = to[g10].astype(float)
    short_l1 = to[g1].astype(float)
    if "H-L" in to.columns:
        hl_l1 = to["H-L"].astype(float)
    else:
        hl_l1 = long_l1 + short_l1
    return long_l1, short_l1, hl_l1


# back-compat
def split_leg_turnovers(group_to: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    return split_leg_l1(group_to)


def _mono_violations(pnl: pd.DataFrame) -> Tuple[float, int]:
    group_cols = _group_cols(pnl)
    annu = np.array([calAnnuRet(pnl[c]) for c in group_cols], dtype=float)
    ranks = pd.Series(np.arange(1, len(annu) + 1), dtype=float)
    mono = float(ranks.corr(pd.Series(annu), method="spearman"))
    violations = int(np.sum(annu[1:] < annu[:-1]))
    return mono, violations


def _pos_month_frac_sum(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    s.index = pd.to_datetime(s.index)
    monthly = s.resample("ME").sum()
    return float((monthly > 0).mean()) if len(monthly) else float("nan")


def _pos_month_frac(series: pd.Series) -> float:
    """Protocol v2.0: monthly_return = prod(1+r) - 1."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    s.index = pd.to_datetime(s.index)
    monthly = (1.0 + s).resample("ME").prod() - 1.0
    return float((monthly > 0).mean()) if len(monthly) else float("nan")


def _tracking_error(active: pd.Series) -> float:
    a = active.dropna()
    if len(a) < 2:
        return float("nan")
    return float(a.std() * np.sqrt(252))


def _ir(active: pd.Series) -> float:
    a = active.dropna()
    if len(a) < 2 or float(a.std()) == 0.0:
        return float("nan")
    return float(a.mean() / a.std() * np.sqrt(252))


def load_benchmark_return(window: str = "full") -> pd.Series:
    path = context_paths(window)["benchmark_return"]
    frame = pd.read_parquet(path)
    if isinstance(frame, pd.DataFrame):
        series = (
            frame["benchmark_ret"]
            if "benchmark_ret" in frame.columns
            else frame.iloc[:, 0]
        )
    else:
        series = frame
    series.index = pd.to_datetime(series.index)
    return series.astype(float).sort_index()


def fee_adjusted_group_pnl(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    *,
    fee_rate_l1: float = FEE_RATE_L1,
) -> pd.DataFrame:
    pnl = ensure_effective_group_pnl(group_pnl)
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl.index)
    out = pd.DataFrame(index=pnl.index)
    for col in pnl.columns:
        if col == "H-L":
            continue
        out[col] = apply_daily_fee_l1(pnl[col], to[col], fee_rate_l1=fee_rate_l1)
    long_l1, short_l1, hl_l1 = split_leg_l1(to)
    out["H-L"] = apply_daily_fee_l1(pnl["H-L"], hl_l1, fee_rate_l1=fee_rate_l1)
    out.attrs["long_l1"] = long_l1.reindex(pnl.index)
    out.attrs["short_l1"] = short_l1.reindex(pnl.index)
    out.attrs["hl_l1"] = hl_l1.reindex(pnl.index)
    return out


def factor_layer_metrics(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    rank_ic_raw: pd.Series,
    *,
    sample: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Any]:
    pnl_g = ensure_effective_group_pnl(group_pnl).loc[start:end]
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl_g.index)
    ic = rank_ic_raw.copy()
    ic.index = pd.to_datetime(ic.index)
    ic = ic.loc[start:end]

    long_l1, short_l1, hl_l1 = split_leg_l1(to)
    pnl_n = fee_adjusted_group_pnl(pnl_g, to)

    hl_g = pnl_g["H-L"].dropna()
    hl_n = pnl_n["H-L"].reindex(hl_g.index).dropna()
    mono_g, viol_g = _mono_violations(pnl_g)
    mono_n, viol_n = _mono_violations(pnl_n)

    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    icir = (
        ic_mean / ic_std * np.sqrt(250) if ic_std and ic_std > 0 else float("nan")
    )

    avg_hl_l1 = float(hl_l1.reindex(hl_g.index).mean())
    avg_hl_ow = l1_to_oneway(avg_hl_l1)
    avg_long_l1 = float(long_l1.reindex(hl_g.index).mean())
    avg_short_l1 = float(short_l1.reindex(hl_g.index).mean())
    fee_annu = avg_hl_l1 * FEE_RATE_L1 * ANNUALIZATION_DAYS

    mdd_g, _ = calMDD(hl_g)
    mdd_n, _ = calMDD(hl_n)

    return {
        "sample": sample,
        "n_days": int(len(hl_g)),
        "rank_ic": ic_mean,
        "icir": float(icir),
        "gross_hl_annual": float(calAnnuRet(hl_g)),
        "gross_hl_sharpe": float(calSharpe(hl_g)),
        "gross_hl_mdd": float(mdd_g),
        "net_hl_annual": float(calAnnuRet(hl_n)),
        "net_hl_sharpe": float(calSharpe(hl_n)),
        "net_hl_mdd": float(mdd_n),
        "decile_mono_gross": mono_g,
        "decile_mono_net": mono_n,
        "adjacent_violations_gross": viol_g,
        "adjacent_violations_net": viol_n,
        "positive_hl_month_fraction": _pos_month_frac(hl_g),
        "positive_hl_month_fraction_sum_legacy": _pos_month_frac_sum(hl_g),
        "avg_daily_long_l1_traded_notional": avg_long_l1,
        "avg_daily_short_l1_traded_notional": avg_short_l1,
        "avg_daily_hl_l1_traded_notional": avg_hl_l1,
        "avg_daily_long_oneway_turnover": l1_to_oneway(avg_long_l1),
        "avg_daily_short_oneway_turnover": l1_to_oneway(avg_short_l1),
        "avg_daily_hl_oneway_turnover": avg_hl_ow,
        "annualized_hl_l1_traded_notional": avg_hl_l1 * ANNUALIZATION_DAYS,
        "annualized_hl_oneway_turnover": avg_hl_ow * ANNUALIZATION_DAYS,
        "fee_annualized": fee_annu,
        "cost_convention": "cost=L1*7.5bps = oneway*15bps",
        "avg_daily_hl_turnover": avg_hl_l1,
        "annualized_hl_turnover": avg_hl_l1 * ANNUALIZATION_DAYS,
        "avg_daily_long_turnover": avg_long_l1,
        "avg_daily_short_turnover": avg_short_l1,
    }


def long_only_metrics(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    benchmark: pd.Series,
    *,
    sample: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Any]:
    pnl = ensure_effective_group_pnl(group_pnl).loc[start:end]
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl.index)

    group_cols = _group_cols(pnl)
    g10 = group_cols[-1]
    long_l1, _, _ = split_leg_l1(to)

    g10_excess = pnl[g10].astype(float)
    bench = benchmark.reindex(g10_excess.index).astype(float)
    g10_abs = g10_excess + bench
    long_net = g10_abs - long_l1.reindex(g10_excess.index).fillna(0.0) * FEE_RATE_L1
    active = long_net - bench

    valid = active.notna() & long_net.notna() & g10_excess.notna()
    long_net = long_net.loc[valid]
    active = active.loc[valid]
    long_l1_v = long_l1.reindex(long_net.index).fillna(0.0)

    mdd_ex, _ = calMDD(active)
    avg_long_l1 = float(long_l1_v.mean())
    avg_long_ow = l1_to_oneway(avg_long_l1)

    return {
        "sample": sample,
        "n_days": int(len(active)),
        "long_portfolio": "G10_equal_weight",
        "long_net_annual_return": float(calAnnuRet(long_net)),
        "excess_annual_return": float(calAnnuRet(active)),
        "IR": _ir(active),
        "tracking_error": _tracking_error(active),
        "excess_mdd": float(mdd_ex),
        "positive_excess_month_fraction": _pos_month_frac(active),
        "positive_excess_month_fraction_sum_legacy": _pos_month_frac_sum(active),
        "avg_daily_long_l1_traded_notional": avg_long_l1,
        "avg_daily_long_oneway_turnover": avg_long_ow,
        "annualized_long_l1_traded_notional": avg_long_l1 * ANNUALIZATION_DAYS,
        "annualized_long_oneway_turnover": avg_long_ow * ANNUALIZATION_DAYS,
        "long_fee_annualized": avg_long_l1 * FEE_RATE_L1 * ANNUALIZATION_DAYS,
        "g10_gross_excess_annual": float(calAnnuRet(g10_excess.loc[valid])),
        "long_daily_turnover": avg_long_l1,
        "long_annualized_turnover": avg_long_l1 * ANNUALIZATION_DAYS,
    }


STRONG_FACTOR = {
    "abs_rank_ic": 0.03,
    "gross_hl_sharpe": 2.5,
    "net_hl_sharpe": 1.2,
    "net_hl_annual": 0.15,
    "mono": 0.80,
    "violations": 2,
}
ACCEPTABLE_FACTOR = {
    "abs_rank_ic": 0.02,
    "gross_hl_sharpe": 2.0,
    "net_hl_sharpe": 0.8,
    "net_hl_annual": 0.10,
    "mono": 0.70,
    "violations": 3,
}
STRATEGY_FULL_PROVISIONAL = {
    "net_excess_annual": 0.08,
    "IR": 0.5,
    "positive_excess_month_fraction": 0.55,
}
HIGH_TURNOVER_L1_DAILY = 1.20
HIGH_TURNOVER_ONEWAY_DAILY = 0.60
HIGH_TURNOVER_DAILY = HIGH_TURNOVER_L1_DAILY  # legacy alias


def decay_and_risk_flags(
    full: Dict[str, Any],
    post: Dict[str, Any],
) -> List[str]:
    flags: List[str] = []
    ic_full = float(full["rank_ic"])
    ic_post = float(post["rank_ic"])
    if (
        np.sign(ic_post) != np.sign(ic_full)
        and abs(ic_post) > 1e-12
        and abs(ic_full) > 1e-12
    ):
        flags.append("POST_IC_SIGN_BREAK")
    ic_ret = abs(ic_post) / abs(ic_full) if abs(ic_full) > 1e-12 else float("nan")
    sh_ret = (
        float(post["gross_hl_sharpe"]) / float(full["gross_hl_sharpe"])
        if abs(float(full["gross_hl_sharpe"])) > 1e-12
        else float("nan")
    )
    if np.isfinite(ic_ret) and ic_ret < 0.30:
        flags.append("POST_IC_RETENTION_LT_30")
    if np.isfinite(sh_ret) and sh_ret < 0.40:
        flags.append("POST_SHARPE_RETENTION_LT_40")
    if float(post["decile_mono_gross"]) < 0.50:
        flags.append("POST_MONO_LT_0_50")
    if int(post["adjacent_violations_gross"]) > 3:
        flags.append("POST_VIOLATIONS_GT_3")

    if abs(float(full["net_hl_mdd"])) > 0.30:
        flags.append("NET_MDD_GT_30")
    elif abs(float(full["net_hl_mdd"])) > 0.20:
        flags.append("NET_MDD_GT_20")

    ow = float(full.get("avg_daily_hl_oneway_turnover", float(full["avg_daily_hl_turnover"]) / 2.0))
    if ow >= HIGH_TURNOVER_ONEWAY_DAILY:
        flags.append("HIGH_TURNOVER")
    return flags


def strategy_warnings(long_full: Dict[str, Any], long_post: Dict[str, Any]) -> List[str]:
    warns: List[str] = []
    if float(long_full["excess_annual_return"]) < STRATEGY_FULL_PROVISIONAL["net_excess_annual"]:
        warns.append("STRATEGY_FULL_EXCESS_LT_8")
    if float(long_full["IR"]) < STRATEGY_FULL_PROVISIONAL["IR"]:
        warns.append("STRATEGY_FULL_IR_LT_0_5")
    if (
        float(long_full["positive_excess_month_fraction"])
        < STRATEGY_FULL_PROVISIONAL["positive_excess_month_fraction"]
    ):
        warns.append("STRATEGY_FULL_POS_MONTH_LT_55")
    if float(long_post["IR"]) < 0.3:
        warns.append("STRATEGY_POST_IR_LT_0_3")
    if float(long_post["excess_annual_return"]) < 0.05:
        warns.append("STRATEGY_POST_EXCESS_LT_5")
    if float(long_post["positive_excess_month_fraction"]) < 0.50:
        warns.append("STRATEGY_POST_POS_MONTH_LT_50")
    return warns


def assign_factor_grade(
    full: Dict[str, Any],
    post: Dict[str, Any],
) -> Tuple[str, List[str], Dict[str, float]]:
    """Factor Grade from Factor Layer FULL + POST decay/risk only."""
    flags = decay_and_risk_flags(full, post)
    ic_ret = (
        abs(float(post["rank_ic"])) / abs(float(full["rank_ic"]))
        if abs(float(full["rank_ic"])) > 1e-12
        else float("nan")
    )
    sh_ret = (
        float(post["gross_hl_sharpe"]) / float(full["gross_hl_sharpe"])
        if abs(float(full["gross_hl_sharpe"])) > 1e-12
        else float("nan")
    )
    retention = {"IC_retention": float(ic_ret), "Sharpe_retention": float(sh_ret)}

    strong_ok = (
        abs(float(full["rank_ic"])) >= STRONG_FACTOR["abs_rank_ic"]
        and float(full["gross_hl_sharpe"]) >= STRONG_FACTOR["gross_hl_sharpe"]
        and float(full["net_hl_sharpe"]) >= STRONG_FACTOR["net_hl_sharpe"]
        and float(full["net_hl_annual"]) >= STRONG_FACTOR["net_hl_annual"]
        and float(full["decile_mono_gross"]) >= STRONG_FACTOR["mono"]
        and int(full["adjacent_violations_gross"]) <= STRONG_FACTOR["violations"]
    )
    accept_ok = (
        abs(float(full["rank_ic"])) >= ACCEPTABLE_FACTOR["abs_rank_ic"]
        and float(full["gross_hl_sharpe"]) >= ACCEPTABLE_FACTOR["gross_hl_sharpe"]
        and float(full["net_hl_sharpe"]) >= ACCEPTABLE_FACTOR["net_hl_sharpe"]
        and float(full["net_hl_annual"]) >= ACCEPTABLE_FACTOR["net_hl_annual"]
        and float(full["decile_mono_gross"]) >= ACCEPTABLE_FACTOR["mono"]
        and int(full["adjacent_violations_gross"]) <= ACCEPTABLE_FACTOR["violations"]
    )
    if strong_ok:
        grade = "A_strong_candidate"
    elif accept_ok:
        grade = "B_research_candidate"
    else:
        grade = "C_not_confirmed"

    severe_post = (
        "POST_IC_SIGN_BREAK" in flags
        or "POST_MONO_LT_0_50" in flags
        or "POST_VIOLATIONS_GT_3" in flags
    )
    if grade == "A_strong_candidate" and severe_post:
        grade = "B_research_candidate"
    return grade, flags, retention


def assign_long_only_status(
    long_full: Dict[str, Any],
    long_post: Dict[str, Any],
) -> Tuple[str, List[str]]:
    warns = strategy_warnings(long_full, long_post)
    full_fail = any(w.startswith("STRATEGY_FULL_") for w in warns)
    status = "PROVISIONAL_FAIL" if full_fail else "PROVISIONAL_PASS"
    return status, warns


def assign_verdict(
    full: Dict[str, Any],
    post: Dict[str, Any],
    long_full: Dict[str, Any],
    long_post: Dict[str, Any],
) -> Tuple[str, List[str], List[str], Dict[str, float]]:
    grade, flags, retention = assign_factor_grade(full, post)
    _status, strat_warns = assign_long_only_status(long_full, long_post)
    return grade, flags, strat_warns, retention


@dataclass(frozen=True)
class FrozenFactorSpec:
    factor_id: str
    factor_hash: str
    exact_formula: str
    source_primitive: str
    expected_direction: int


def check_effective_turnover_parity(
    *,
    factor_id: str,
    raw_direction: int,
    to_raw: pd.DataFrame,
    to_eff: pd.DataFrame,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Verify effective G1/G10 L1 match raw legs under direction flip.

    ``backtest_factor`` re-ranks with ``-signal`` when direction=-1, so means
    can differ slightly from a pure column-swap of raw TO (rank ties). Gate
    uses relative tolerance; also records exact column-swap expectation.
    """
    raw = to_raw.copy()
    raw.columns = [str(c) for c in raw.columns]
    eff = to_eff.copy()
    eff.columns = [str(c) for c in eff.columns]
    raw_cols = _group_cols(raw)
    eff_cols = _group_cols(eff)
    raw_g1 = float(raw[raw_cols[0]].mean())
    raw_g10 = float(raw[raw_cols[-1]].mean())
    eff_g1 = float(eff[eff_cols[0]].mean())
    eff_g10 = float(eff[eff_cols[-1]].mean())

    if int(raw_direction) >= 0:
        exp_g1, exp_g10 = raw_g1, raw_g10
        swap_correct = True  # identity
    else:
        exp_g1, exp_g10 = raw_g10, raw_g1
        # Structural check: long/short legs must swap (eff_g10 near raw_g1)
        swap_correct = True

    pass_g1 = bool(np.isclose(eff_g1, exp_g1, rtol=rtol, atol=atol))
    pass_g10 = bool(np.isclose(eff_g10, exp_g10, rtol=rtol, atol=atol))
    hl_raw = float(raw["H-L"].mean()) if "H-L" in raw.columns else raw_g1 + raw_g10
    hl_eff = float(eff["H-L"].mean()) if "H-L" in eff.columns else eff_g1 + eff_g10
    pass_hl = bool(np.isclose(hl_raw, hl_eff, rtol=rtol, atol=atol))

    # Directional swap intent (even under tiny re-rank noise)
    if int(raw_direction) < 0:
        legs_swapped = abs(eff_g10 - raw_g1) < abs(eff_g10 - raw_g10)
    else:
        legs_swapped = abs(eff_g10 - raw_g10) <= abs(eff_g10 - raw_g1)

    return {
        "factor": factor_id,
        "raw_direction": int(raw_direction),
        "raw_g1_l1": raw_g1,
        "raw_g10_l1": raw_g10,
        "effective_g1_l1": eff_g1,
        "effective_g10_l1": eff_g10,
        "expected_effective_g1_l1": exp_g1,
        "expected_effective_g10_l1": exp_g10,
        "raw_hl_l1": hl_raw,
        "effective_hl_l1": hl_eff,
        "abs_err_g1": abs(eff_g1 - exp_g1),
        "abs_err_g10": abs(eff_g10 - exp_g10),
        "pass_g1": pass_g1,
        "pass_g10": pass_g10,
        "pass_hl_invariant": pass_hl,
        "pass_legs_swapped_intent": bool(legs_swapped and swap_correct),
        "pass": bool(pass_g1 and pass_g10 and pass_hl and legs_swapped),
        "note": (
            "backtest_factor re-runs groupTest on flipped signal when direction=-1; "
            "tiny mean diffs vs pure column-swap come from rank(method=first) ties; "
            "rtol/atol gate + swap-intent check"
        ),
    }
