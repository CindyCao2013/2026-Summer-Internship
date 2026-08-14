"""Sprint 16 — Liquidity Impact execution layer (horizon / turnover).

Status: FROZEN_COMPLETE. Do not further optimize impact_per_trade.
Phase E / cvxpy is CANCELLED for this sprint.

Frozen boundary:
- Read already-materialized candidate-pool factor_narrow only.
- Reuse Fast Discovery context (mask, excess c2c vs 000852.SH).
- Do NOT modify primitive formulas, size buckets, recovery windows,
  signed-amount definitions, or factor direction.
- Phase A/C/B/D1/D2 done. SLOW branch closed. MEDIUM names are backlog.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.python.backtest import (
    compute_rank_ic,
    narrow_to_wide,
    prepare_factor_signal,
)
from l2_factor_reproduction.python.fast_discovery import (
    FULL_END,
    FULL_START,
    load_fast_context,
)

CONTRACT_VERSION = "LIQ_IMPACT_EXEC_V1"
SPRINT_ID = "sprint16_liquidity_impact_execution"
FACTOR_FAMILY = "liquidity_impact"

HORIZONS: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
SMOOTHING_WINDOWS: Tuple[int, ...] = (3, 5, 10)
REBALANCE_DAYS: Tuple[int, ...] = (1, 2, 3, 5, 10)
BUFFER_WIDTHS: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)

# Half-life class on |IC(h)| / |IC(1)| crossing 0.5. Frozen before Phase A.
FAST_HALF_LIFE_MAX = 2.5
MEDIUM_HALF_LIFE_MAX = 6.0

H1_PARITY_ATOL = 5e-4

TIER1: Tuple[str, ...] = (
    "impact_per_trade",
    "permanent_impact_1m",
    "signed_amount_impact",
    "signed_sqrt_amount_impact",
)
TIER2: Tuple[str, ...] = (
    "depth_recovery_5m",
    "impact_asymmetry",
    "mid_trade_impact",
)
FACTORS: Tuple[str, ...] = TIER1 + TIER2

# Frozen family-baseline redundancy (candidate pool |ρ|≥0.80 cluster R3).
R3_NEAR_ALIASES: Tuple[str, ...] = (
    "signed_amount_impact",
    "signed_sqrt_amount_impact",
    "impact_per_trade",
)
R3_MAX_ABS_CORR = {
    ("signed_amount_impact", "signed_sqrt_amount_impact"): 0.9469824839028169,
    ("impact_per_trade", "signed_sqrt_amount_impact"): 0.9197117587959853,
}

# Frozen from Phase A decay class. MEDIUM names are parked this round.
SLOW_FACTORS: Tuple[str, ...] = (
    "impact_per_trade",
    "signed_amount_impact",
    "signed_sqrt_amount_impact",
    "depth_recovery_5m",
)
MEDIUM_PARKED: Tuple[str, ...] = (
    "permanent_impact_1m",
    "impact_asymmetry",
    "mid_trade_impact",
)
PHASE_C_WINDOWS: Tuple[int, ...] = (1, 3, 5, 10)  # 1 = RAW
PHASE_C_IC_RETENTION_MIN = 0.60
PHASE_C_MONO_MIN = 0.70
PHASE_B_SLOW_GRID: Tuple[str, ...] = ("daily", "staggered_5d", "staggered_10d")
PHASE_B_HOLDS: Tuple[int, ...] = (1, 5, 10)
PHASE_B_CANDIDATES: Tuple[Tuple[str, int], ...] = (
    ("impact_per_trade", 1),
    ("signed_amount_impact", 3),
    ("signed_sqrt_amount_impact", 1),
    ("depth_recovery_5m", 10),
)
R3_PHASE_B: Tuple[str, ...] = (
    "impact_per_trade",
    "signed_amount_impact",
    "signed_sqrt_amount_impact",
)
PHASE_C_PROMOTE_MAX = 2
PHASE_D_BUFFER_WIDTHS: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
PHASE_D_FACTOR = "impact_per_trade"
PHASE_D_MA_WINDOW = 1
PHASE_D_HOLD = 5
PHASE_D1_IC_RETENTION_MIN = 0.90
PHASE_D_NEAR_NETSR = 0.05
PHASE_D2_TAILS: Tuple[str, ...] = ("G10-G1", "G9:G10-G1", "G8:G10-G1")
PHASE_E_OPTIONAL = False
PHASE_E_STATUS = "CANCELLED"
SPRINT_STATUS = "FROZEN_COMPLETE"

INVENTORY = {
    "impact_per_trade": {
        "factor_status": "VALID_NON_MONOTONIC",
        "predictive_factor": "PASS",
        "monotonicity_gate": "FAIL",
        "strategy_status": "TRADABLE",
        "grade": "TAIL_STRATEGY_GRADE",
        "portfolio_expression": "LONG_G8_G10_SHORT_G1",
        "holding": "STAGGERED_5D",
        "buffer": 0,
        "known_limitation": "decile_monotonicity_0.43",
        "representation": "RAW",
    },
    "depth_recovery_5m": {
        "factor_status": "FEATURE_GRADE",
        "strategy_status": "NON_STANDALONE",
        "representation": "MA10",
        "grade": "FEATURE_GRADE",
    },
    "signed_amount_impact": {
        "factor_status": "FREEZE",
        "reason": "execution_degradation",
        "representation": "MA3",
    },
    "signed_sqrt_amount_impact": {
        "factor_status": "FREEZE",
        "reason": "redundant_r3_representative",
        "representation": "RAW",
    },
    "permanent_impact_1m": {
        "factor_status": "BACKLOG",
        "reason": "medium_parked_not_this_tree",
    },
    "mid_trade_impact": {
        "factor_status": "BACKLOG",
        "reason": "medium_buffer_first_not_this_tree",
    },
    "impact_asymmetry": {
        "factor_status": "BACKLOG",
        "reason": "medium_buffer_first_not_this_tree",
    },
}

STOPPED_SEARCH = (
    "further impact_per_trade optimization",
    "short-only buffer",
    "long-only buffer",
    "finer buffer grid",
    "G7:G10 or wider long breadth",
    "Phase E cvxpy on this factor",
    "depth_recovery tail/buffer/20D/optimizer",
    "re-opening the SLOW research tree",
)

FEATURE_GATE = {
    "abs_rank_ic_min": 0.02,
    "abs_icir_min": 3.0,
    "monotonicity_min": 0.70,
    "yearly_sign_min": 0.75,
}

FORBIDDEN = (
    "impact minute threshold",
    "size bucket",
    "5m recovery definition",
    "signed amount definition",
    "permanent impact formula",
    "factor direction",
    "primitive rebuild",
)

NET_GATE = {
    "net_annu_min": 0.10,
    "net_sharpe_min": 1.5,
    "turnover_l1_max": 1.0,
    "ic_retention_min": 0.60,
    "monotonicity_min": 0.70,
}

POOL_FACTOR_ROOT = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "liquidity_impact_family" / "factors"
)
OUT_ROOT = Path(RESULT_ROOT) / SPRINT_ID

PLOT_DPI = 150


@dataclass(frozen=True)
class HorizonRow:
    factor: str
    window: str
    horizon_days: int
    rank_ic_raw: float
    icir_raw: float
    positive_ic_fraction: float
    n_ic_dates: int
    n_names_avg: float
    ic_retention_vs_h1: float


def contract_payload() -> Dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "sprint_id": SPRINT_ID,
        "universe": UNIVERSE,
        "signal_shift_convention": "RankIC(h) = Spearman(factor.shift(h), excess_c2c)",
        "h1_equals_candidate_pool_tplus1": True,
        "factors": list(FACTORS),
        "tier1": list(TIER1),
        "tier2": list(TIER2),
        "r3_near_aliases": list(R3_NEAR_ALIASES),
        "r3_max_abs_corr": {
            f"{a}|{b}": corr for (a, b), corr in R3_MAX_ABS_CORR.items()
        },
        "phase_a_horizons": list(HORIZONS),
        "phase_b_rebalance_days": list(REBALANCE_DAYS),
        "phase_b_staggered_not_weekday_snapshot": True,
        "phase_c_smoothing": ["raw", "MA3", "MA5", "MA10"],
        "phase_c_slow_factors": list(SLOW_FACTORS),
        "phase_c_medium_parked": list(MEDIUM_PARKED),
        "phase_c_smooth_layer": "daily_signal_exposure_not_primitive",
        "phase_c_no_production_to_gate": True,
        "phase_c_ic_retention_min": PHASE_C_IC_RETENTION_MIN,
        "phase_c_mono_min": PHASE_C_MONO_MIN,
        "phase_c_promote_max": PHASE_C_PROMOTE_MAX,
        "phase_b_slow_grid": list(PHASE_B_SLOW_GRID),
        "phase_b_holds": list(PHASE_B_HOLDS),
        "phase_b_candidates": [
            {"factor": name, "ma_window": window}
            for name, window in PHASE_B_CANDIDATES
        ],
        "phase_b_sleeve_definition": (
            "rank S_t into a new decile/H-L sleeve P_t; hold P_t for H days; "
            "book = equal-weight average of active sleeves; "
            "DO NOT average past signals then re-rank"
        ),
        "phase_b_turnover": "L1 on netted final weights, not sum of sleeve TO",
        "phase_b_r3_pick_after": True,
        "phase_d_factor": PHASE_D_FACTOR,
        "phase_d_ma_window": PHASE_D_MA_WINDOW,
        "phase_d_hold": PHASE_D_HOLD,
        "phase_d_buffer_widths": list(PHASE_D_BUFFER_WIDTHS),
        "phase_d1_ic_retention_min": PHASE_D1_IC_RETENTION_MIN,
        "phase_d_near_netsr": PHASE_D_NEAR_NETSR,
        "phase_d2_tails": list(PHASE_D2_TAILS),
        "phase_d_asymmetric_tail": "dollar_neutral_long_breadth_not_short_overweight",
        "phase_d_no_hold_grid": True,
        "sprint_status": SPRINT_STATUS,
        "phase_d_no_4x3_grid": True,
        "phase_e_optional": PHASE_E_OPTIONAL,
        "phase_e_status": PHASE_E_STATUS,
        "phase_e_l1_penalty": "CANCELLED for this sprint; only at multifactor portfolio layer later",
        "net_sharpe_gate_not_relaxed": 1.5,
        "monotonicity_gate_not_relaxed": 0.7,
        "impact_per_trade_not_claimed_monotonic_strategy_grade": True,
        "inventory": {k: dict(v) for k, v in INVENTORY.items()},
        "stopped_search": list(STOPPED_SEARCH),
        "strategy_gate": dict(NET_GATE),
        "feature_gate": dict(FEATURE_GATE),
        "half_life_fast_max": FAST_HALF_LIFE_MAX,
        "half_life_medium_max": MEDIUM_HALF_LIFE_MAX,
        "net_gate": dict(NET_GATE),
        "forbidden": list(FORBIDDEN),
        "allowed": ["horizon", "smoothing", "rebalance", "buffer"],
        "phase_a_only_this_run": False,
        "do_not_keep_drop": True,
        "do_not_touch_primitive_formulas": True,
        "rankic_h_definition": (
            "Spearman(factor.shift(h), one_day_excess_c2c_t); "
            "NOT cumulative r_{t->t+h}"
        ),
    }


def factor_narrow_path(name: str) -> Path:
    return POOL_FACTOR_ROOT / name / "factor_narrow.parquet"


def load_factor_wide(name: str) -> pd.DataFrame:
    path = factor_narrow_path(name)
    if not path.exists():
        raise FileNotFoundError(f"missing frozen factor_narrow: {path}")
    return narrow_to_wide(pd.read_parquet(path))


def ic_stats(rank_ic: pd.Series) -> Tuple[float, float, float, int]:
    series = pd.to_numeric(rank_ic, errors="coerce").dropna()
    if series.empty:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(series.mean())
    std = float(series.std())
    icir = mean / std * (250 ** 0.5) if std > 0 else float("nan")
    pos = float((series > 0).mean())
    return mean, icir, pos, int(len(series))


def interpolate_half_life(
    horizons: Sequence[int],
    abs_ic: Sequence[float],
) -> float:
    """Smallest h where |IC(h)| ≈ 0.5 |IC(1)|, linear in h between knots."""
    if len(horizons) < 2 or len(horizons) != len(abs_ic):
        raise ValueError("horizons and abs_ic must be same length >= 2")
    base = float(abs_ic[0])
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    target = 0.5 * base
    if float(abs_ic[-1]) > target:
        return float("inf")
    for i in range(1, len(horizons)):
        y0 = float(abs_ic[i - 1])
        y1 = float(abs_ic[i])
        if y1 <= target <= y0 or y1 <= target:
            h0 = float(horizons[i - 1])
            h1 = float(horizons[i])
            if abs(y0 - y1) < 1e-15:
                return h0
            frac = (target - y0) / (y1 - y0)
            return h0 + frac * (h1 - h0)
    return float("inf")


def classify_half_life(half_life: float) -> str:
    if not np.isfinite(half_life):
        return "UNRESOLVED"
    if half_life <= FAST_HALF_LIFE_MAX:
        return "FAST"
    if half_life <= MEDIUM_HALF_LIFE_MAX:
        return "MEDIUM"
    return "SLOW"


def recommended_next_phase(decay_class: str) -> str:
    if decay_class == "FAST":
        return "buffer_hysteresis_not_lower_refresh"
    if decay_class == "MEDIUM":
        return "smoothing_and_3d_5d_rebalance"
    if decay_class == "SLOW":
        return "smoothing_ma5_ma10_ok_for_5d_10d"
    return "inspect_sign_flip_or_weak_h1"


def rank_ic_at_horizon(
    factor_wide: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
    horizon: int,
    *,
    start,
    end,
) -> Tuple[pd.Series, float]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    signal, aligned_ret = prepare_factor_signal(
        factor_wide,
        start=start,
        end=end,
        mask=mask,
        signal_shift=horizon,
        ret_matrix=ret,
    )
    n_names = float(signal.notna().sum(axis=1).mean()) if len(signal) else float("nan")
    return compute_rank_ic(signal, aligned_ret), n_names


def evaluate_factor_horizons(
    name: str,
    factor_wide: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    window: str,
    start,
    end,
    horizons: Sequence[int] = HORIZONS,
) -> List[HorizonRow]:
    rows: List[HorizonRow] = []
    h1_abs = float("nan")
    for horizon in horizons:
        rank_ic, n_names = rank_ic_at_horizon(
            factor_wide, mask, ret, horizon, start=start, end=end
        )
        mean, icir, pos, n_dates = ic_stats(rank_ic)
        if horizon == 1:
            h1_abs = abs(mean)
        retention = (
            abs(mean) / h1_abs if h1_abs and np.isfinite(h1_abs) and h1_abs > 0 else float("nan")
        )
        rows.append(
            HorizonRow(
                factor=name,
                window=window,
                horizon_days=int(horizon),
                rank_ic_raw=mean,
                icir_raw=icir,
                positive_ic_fraction=pos,
                n_ic_dates=n_dates,
                n_names_avg=n_names,
                ic_retention_vs_h1=float(retention),
            )
        )
    return rows


def summarize_decay(rows: Sequence[HorizonRow]) -> Dict[str, object]:
    ordered = sorted(rows, key=lambda row: row.horizon_days)
    if not ordered:
        raise ValueError("no horizon rows")
    horizons = [row.horizon_days for row in ordered]
    abs_ic = [abs(row.rank_ic_raw) for row in ordered]
    half = interpolate_half_life(horizons, abs_ic)
    decay_class = classify_half_life(half)
    h1 = ordered[0]
    sign_flip = False
    for row in ordered[1:]:
        if np.sign(row.rank_ic_raw) != np.sign(h1.rank_ic_raw) and abs(row.rank_ic_raw) > 1e-6:
            sign_flip = True
            break
    return {
        "factor": h1.factor,
        "window": h1.window,
        "rank_ic_h1": h1.rank_ic_raw,
        "icir_h1": h1.icir_raw,
        "positive_ic_fraction_h1": h1.positive_ic_fraction,
        "half_life_days": None if not np.isfinite(half) else float(half),
        "half_life_unbounded": not np.isfinite(half),
        "decay_class": decay_class,
        "sign_flip_before_h20": sign_flip,
        "recommended_next_phase": recommended_next_phase(decay_class),
        "ic_h5_retention": next(
            (row.ic_retention_vs_h1 for row in ordered if row.horizon_days == 5),
            float("nan"),
        ),
        "ic_h10_retention": next(
            (row.ic_retention_vs_h1 for row in ordered if row.horizon_days == 10),
            float("nan"),
        ),
    }


def hac_variance(values: np.ndarray, lags: int) -> float:
    """Newey-West / Bartlett HAC variance of a 1-d series (not of the mean)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < max(10, lags + 2):
        return float("nan")
    demean = x - float(x.mean())
    gamma0 = float(np.dot(demean, demean) / n)
    var = gamma0
    max_lag = min(int(lags), n - 2)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demean[lag:], demean[:-lag]) / n)
        var += 2.0 * weight * gamma
    return float(var)


def icir_newey_west(rank_ic: pd.Series, lags: int) -> float:
    series = pd.to_numeric(rank_ic, errors="coerce").dropna()
    if series.empty:
        return float("nan")
    var = hac_variance(series.to_numpy(), lags)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(series.mean() / np.sqrt(var) * (250 ** 0.5))


def icir_nonoverlapping(rank_ic: pd.Series, stride: int) -> Tuple[float, int]:
    series = pd.to_numeric(rank_ic, errors="coerce").dropna().sort_index()
    if stride < 1:
        raise ValueError("stride must be >= 1")
    sampled = series.iloc[::stride]
    _mean, icir, _pos, n_dates = ic_stats(sampled)
    return icir, n_dates


def run_horizon_icir_audit(
    *,
    output_root: Optional[Path] = None,
    factors: Optional[Sequence[str]] = None,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """HAC / non-overlapping ICIR on Phase A lag-h daily RankIC series.

    Phase A labels are one-day excess returns, so adjacent h-day labels do
    not overlap. IC_t can still be serially correlated because the factor
    is persistent; naive ICIR(h) can look too large. This audit does not
    change half-life rankings.
    """
    out = Path(output_root) if output_root is not None else OUT_ROOT / "phase_a"
    out.mkdir(parents=True, exist_ok=True)
    names = list(factors) if factors is not None else list(FACTORS)
    start, end = FULL_START, FULL_END
    mask, ret = load_fast_context("full", verify_hash=verify_hash)
    rows: List[Dict[str, object]] = []
    for name in names:
        wide = load_factor_wide(name)
        for horizon in HORIZONS:
            rank_ic, _n_names = rank_ic_at_horizon(
                wide, mask, ret, horizon, start=start, end=end
            )
            naive_mean, naive_icir, pos, n_dates = ic_stats(rank_ic)
            nw = icir_newey_west(rank_ic, lags=horizon)
            skip_icir, skip_n = icir_nonoverlapping(rank_ic, stride=horizon)
            rows.append(
                {
                    "factor": name,
                    "horizon_days": int(horizon),
                    "rank_ic_raw": naive_mean,
                    "icir_naive": naive_icir,
                    "icir_newey_west": nw,
                    "icir_nonoverlap_stride_h": skip_icir,
                    "n_ic_dates": n_dates,
                    "n_nonoverlap": skip_n,
                    "positive_ic_fraction": pos,
                    "nw_lags": int(horizon),
                    "label_definition": "one_day_excess_at_lag_h",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "icir_audit.csv", index=False)
    return frame


def load_baseline_h1(name: str) -> float:
    path = POOL_FACTOR_ROOT / name / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["rank_ic_mean_raw"])


def parity_ok(observed_h1: float, baseline_h1: float, atol: float = H1_PARITY_ATOL) -> bool:
    return abs(observed_h1 - baseline_h1) <= atol


def plot_ic_decay(detail: pd.DataFrame, path: Path, *, window: str) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    subset = detail.loc[detail["window"] == window].copy()
    for name in FACTORS:
        block = subset.loc[subset["factor"] == name].sort_values("horizon_days")
        if block.empty:
            continue
        ax.plot(
            block["horizon_days"],
            block["rank_ic_raw"] * 100.0,
            marker="o",
            linewidth=1.8,
            label=name,
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xticks(list(HORIZONS))
    ax.set_xlabel("Forward horizon h (trading days)")
    ax.set_ylabel("RankIC raw (%)")
    ax.set_title(f"Liquidity-impact RankIC decay — {window} ({UNIVERSE})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_phase_a_report(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    parity: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Sprint 16 Phase A — Liquidity Impact Alpha Horizon",
        "",
        f"Contract: `{CONTRACT_VERSION}`",
        "Scope: RankIC(h) decay only. Frozen formulas were not modified.",
        f"Universe: `{UNIVERSE}`. Return: excess c2c vs benchmark, T+h.",
        "",
        "## Gate reminder",
        "",
        "Gross Sharpe is **not** a success criterion in this sprint.",
        "Later phases use net economics: NetAnnual>10%, NetSharpe>1.5, L1 TO<1.0, "
        "IC retention≥60%, mono>0.7.",
        "",
        "## Known redundancy (do not treat as independent alpha)",
        "",
        "`signed_amount_impact`, `signed_sqrt_amount_impact`, `impact_per_trade` "
        "are cluster R3 (`|ρ|` 0.92–0.95). After execution tests, keep one by "
        "cost-adjusted Sharpe.",
        "",
        "## H=1 parity vs candidate-pool `summary.json`",
        "",
        "```",
        parity.to_string(index=False),
        "```",
        "",
        "## Decay summary (FULL)",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## How to read the class",
        "",
        "- **FAST** (half-life ≤ 2.5d): do not weekly-rebalance; test buffer / hysteresis.",
        "- **MEDIUM** (2.5–6d): MA3/MA5 plus staggered 3D/5D holding.",
        "- **SLOW** (>6d): MA5/MA10 is economically plausible; 5D/10D holding is in-scope.",
        "- **UNRESOLVED**: |IC| never fell to half of H=1 by 20d, or H=1 is degenerate.",
        "",
        "Phase B/C/D are **not** run in this artifact. Next step follows "
        "`recommended_next_phase` per factor.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase_a(
    *,
    output_root: Optional[Path] = None,
    factors: Optional[Sequence[str]] = None,
    window: str = "full",
    verify_hash: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(output_root) if output_root is not None else OUT_ROOT / "phase_a"
    out.mkdir(parents=True, exist_ok=True)
    names = list(factors) if factors is not None else list(FACTORS)
    unknown = sorted(set(names).difference(FACTORS))
    if unknown:
        raise KeyError(f"factors not in frozen sprint list: {unknown}")

    (out / "contract.json").write_text(
        json.dumps(contract_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if window != "full":
        raise ValueError("Phase A contract window is full 2019-01-01..2026-07-31")
    start, end = FULL_START, FULL_END
    mask, ret = load_fast_context(window, verify_hash=verify_hash)

    detail_rows: List[HorizonRow] = []
    parity_rows: List[Dict[str, object]] = []
    for name in names:
        wide = load_factor_wide(name)
        rows = evaluate_factor_horizons(
            name, wide, mask, ret, window=window, start=start, end=end
        )
        detail_rows.extend(rows)
        h1 = next(row.rank_ic_raw for row in rows if row.horizon_days == 1)
        baseline = load_baseline_h1(name)
        parity_rows.append(
            {
                "factor": name,
                "rank_ic_h1": h1,
                "candidate_pool_rank_ic_raw": baseline,
                "abs_delta": abs(h1 - baseline),
                "parity_pass": parity_ok(h1, baseline),
            }
        )

    detail = pd.DataFrame([asdict(row) for row in detail_rows])
    parity = pd.DataFrame(parity_rows)
    summary_rows = []
    for name in names:
        block = [row for row in detail_rows if row.factor == name]
        summary_rows.append(summarize_decay(block))
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(out / "ic_decay_detail.csv", index=False)
    summary.to_csv(out / "ic_decay_summary.csv", index=False)
    parity.to_csv(out / "h1_parity.csv", index=False)
    plot_ic_decay(detail, out / "ic_decay_curve.png", window=window)
    write_phase_a_report(summary, detail, parity, out / "report.md")
    return summary, detail, parity


__all__ = [
    "BUFFER_WIDTHS",
    "CONTRACT_VERSION",
    "FACTORS",
    "FEATURE_GATE",
    "HORIZONS",
    "NET_GATE",
    "OUT_ROOT",
    "PHASE_B_CANDIDATES",
    "PHASE_B_HOLDS",
    "PHASE_D_BUFFER_WIDTHS",
    "PHASE_D_FACTOR",
    "PHASE_D_HOLD",
    "PHASE_D1_IC_RETENTION_MIN",
    "PHASE_D2_TAILS",
    "PHASE_E_OPTIONAL",
    "PHASE_E_STATUS",
    "SPRINT_STATUS",
    "INVENTORY",
    "REBALANCE_DAYS",
    "SMOOTHING_WINDOWS",
    "PHASE_B_SLOW_GRID",
    "PHASE_C_IC_RETENTION_MIN",
    "PHASE_C_MONO_MIN",
    "SLOW_FACTORS",
    "MEDIUM_PARKED",
    "TIER1",
    "TIER2",
    "classify_half_life",
    "contract_payload",
    "icir_newey_west",
    "interpolate_half_life",
    "run_horizon_icir_audit",
    "run_phase_a",
]
