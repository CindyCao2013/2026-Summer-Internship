"""Frozen AI-v1 production execution contract: EXEC_V2V_TPLUS1_V1.

Primary is frozen before looking at performance. Do not switch to O2O/C2C
because they look better.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRIMARY_EXECUTION_CONTRACT = "EXEC_V2V_TPLUS1_V1"
LEGACY_C2C_DIAGNOSTIC = "LEGACY_C2C_DIAGNOSTIC"
ROBUSTNESS_O2O = "EXEC_O2O_TPLUS1_V1"
ROBUSTNESS_C2C_DELAYED = "EXEC_C2C_TPLUS1_DELAYED_V1"

ALLOWED_EXECUTION_CONTRACTS = (
    PRIMARY_EXECUTION_CONTRACT,
    LEGACY_C2C_DIAGNOSTIC,
    ROBUSTNESS_O2O,
    ROBUSTNESS_C2C_DELAYED,
)

# Classification thresholds frozen BEFORE inspecting factor names.
# Same |IC| floor as Phase B0 nonlinear_should_review.
IC_ABS_FLOOR = 0.008
EXEC_NONEMPTY_ABS = 0.004
PRESERVATION_ROBUST = 0.50

HORIZONS: Tuple[int, ...] = (1, 3, 5, 10, 20)
AUDIT_WINDOW_START = pd.Timestamp("2023-01-01")
AUDIT_WINDOW_END = pd.Timestamp("2024-12-31")
MIN_LISTING_DAYS = 60


def resolve_execution_contract(
    execution_contract: Optional[str] = None,
) -> str:
    """Default is executable V2V. Legacy C2C is allowed only as an explicit flag."""
    if execution_contract is None or str(execution_contract).strip() == "":
        return PRIMARY_EXECUTION_CONTRACT
    name = str(execution_contract).strip()
    if name not in ALLOWED_EXECUTION_CONTRACTS:
        raise ValueError(
            "unknown execution_contract {!r}; allowed={}".format(
                name, ALLOWED_EXECUTION_CONTRACTS
            )
        )
    return name


def assert_not_legacy_default(execution_contract: Optional[str] = None) -> str:
    resolved = resolve_execution_contract(execution_contract)
    return resolved


def is_legacy_c2c(execution_contract: Optional[str] = None) -> bool:
    return resolve_execution_contract(execution_contract) == LEGACY_C2C_DIAGNOSTIC


def map_feature_to_holding(
    dates: Sequence,
    feature_date,
    horizon: int,
) -> Dict[str, object]:
    """Explicit mapping. Does not use DataFrame.shift.

    feature T, entry = next trading day T+1, exit = T+1+h.
    1D exit = T+2, 3D exit = T+4, 5D exit = T+6, 10D = T+11, 20D = T+21.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    t = pd.Timestamp(feature_date).normalize()
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    if t not in dates:
        raise KeyError("feature_date {} not in calendar".format(t.date()))
    pos = int(dates.get_loc(t))
    entry_pos = pos + 1
    exit_pos = pos + 1 + h
    out = {
        "feature_date": t,
        "horizon": h,
        "entry_date": pd.NaT,
        "exit_date": pd.NaT,
        "entry_offset_trading_days": 1,
        "exit_offset_trading_days": 1 + h,
        "valid": False,
    }
    if exit_pos >= len(dates):
        return out
    out["entry_date"] = pd.Timestamp(dates[entry_pos])
    out["exit_date"] = pd.Timestamp(dates[exit_pos])
    out["valid"] = True
    return out


def holding_return_from_prices(
    price: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    horizon: int,
    start_lag: int = 1,
) -> pd.DataFrame:
    """price[T+start_lag+h] / price[T+start_lag] - 1. No truncation: tail is NaN.

    start_lag=1 is the executable contract (enter next session).
    start_lag=0 would start at T and is forbidden for production V2V.
    """
    if int(start_lag) < 1:
        raise ValueError(
            "executable labels must not start before T+1 (start_lag={})".format(start_lag)
        )
    dates = pd.DatetimeIndex(dates).normalize()
    px = price.reindex(index=dates)
    n_dates, n_sym = px.shape
    arr = px.to_numpy(dtype=float)
    h = int(horizon)
    lab = np.full((n_dates, n_sym), np.nan, dtype=float)
    for i in range(n_dates):
        e = i + int(start_lag)
        x = e + h
        if x >= n_dates:
            break
        entry = arr[e]
        exitp = arr[x]
        ok = np.isfinite(entry) & np.isfinite(exitp) & (entry > 0) & (exitp > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = exitp / entry - 1.0
        r = np.where(ok, r, np.nan)
        lab[i] = r
    return pd.DataFrame(lab, index=dates, columns=px.columns)


def daily_ratio_return(price: pd.DataFrame) -> pd.DataFrame:
    """ret[D] = price[D] / price[D-1] - 1. Matches DolphinDB ratios() / get_Ret_Matrix v2v."""
    prev = price.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = price.astype(float) / prev.astype(float) - 1.0
    r = r.where(np.isfinite(r.to_numpy()) & (prev > 0) & (price > 0))
    return r


def compound_daily_from_lag(
    daily_ret: pd.DataFrame,
    *,
    horizon: int,
    first_return_lag: int,
) -> pd.DataFrame:
    """Compound daily_ret[T+first_return_lag] .. daily_ret[T+first_return_lag+horizon-1].

    Executable V2V 1D uses first_return_lag=2 (skip VWAP[T+1]/VWAP[T]).
    Legacy C2C 1D uses first_return_lag=1 (Close[T+1]/Close[T]).
    """
    dates = pd.DatetimeIndex(daily_ret.index).normalize()
    arr = daily_ret.reindex(index=dates).to_numpy(dtype=float)
    n_dates, n_sym = arr.shape
    h = int(horizon)
    lag = int(first_return_lag)
    lab = np.full((n_dates, n_sym), np.nan, dtype=float)
    for i in range(n_dates):
        j0 = i + lag
        j1 = j0 + h
        if j1 > n_dates:
            break
        block = arr[j0:j1]
        bad = ~np.isfinite(block).all(axis=0)
        cum = np.prod(1.0 + block, axis=0) - 1.0
        cum[bad] = np.nan
        lab[i] = cum
    return pd.DataFrame(lab, index=dates, columns=daily_ret.columns)


def classify_degradation(legacy_ic: float, exec_ic: float) -> str:
    """Transparent classes. Thresholds are module constants, not fit to names."""
    leg = float(legacy_ic) if np.isfinite(legacy_ic) else 0.0
    ex = float(exec_ic) if np.isfinite(exec_ic) else 0.0
    if abs(leg) < IC_ABS_FLOOR:
        return "INCONCLUSIVE"
    sign_ok = np.sign(leg) == np.sign(ex) and abs(ex) >= 1e-15
    if (not sign_ok) or abs(ex) < EXEC_NONEMPTY_ABS:
        return "TIMING_SENSITIVE"
    preservation = abs(ex) / abs(leg)
    if abs(ex) >= IC_ABS_FLOOR and sign_ok and preservation >= PRESERVATION_ROBUST:
        return "ROBUST_EXECUTABLE"
    return "DECAY_SENSITIVE"


def ic_preservation(legacy_ic: float, exec_ic: float) -> float:
    if not (np.isfinite(legacy_ic) and np.isfinite(exec_ic)):
        return float("nan")
    if abs(float(legacy_ic)) < IC_ABS_FLOOR:
        return float("nan")
    return abs(float(exec_ic)) / abs(float(legacy_ic))


def production_execution_contract_dict() -> Dict[str, object]:
    return {
        "primary": PRIMARY_EXECUTION_CONTRACT,
        "status": "FROZEN_BEFORE_PERFORMANCE",
        "feature": "L2 factor formed on date T, fully known after market close T",
        "execution": "enter VWAP on trading day T+1; exit VWAP on T+h+1",
        "stock_return_h": "VWAP_stock[T+h+1] / VWAP_stock[T+1] - 1",
        "benchmark_return_h": "VWAP_benchmark[T+h+1] / VWAP_benchmark[T+1] - 1",
        "label_h": "stock_return_h - benchmark_return_h",
        "horizons": list(HORIZONS),
        "get_Ret_Matrix_v2v": (
            "ratios(S_DQ_AVGPRICE*S_DQ_ADJFACTOR)-1 contextby symbol => "
            "ret_v2v[D] = adjVWAP[D]/adjVWAP[D-1]-1. "
            "factor.shift(1) would pair factor T with VWAP[T+1]/VWAP[T] and is NOT executable."
        ),
        "executable_daily_compound": (
            "1D = ret_v2v[T+2]; hD = prod_{k=2..h+1}(1+ret_v2v[T+k])-1"
        ),
        "benchmark": {
            "id": "000852.SH",
            "official_index_eod_has_avgprice": False,
            "amount_over_volume_is_index_vwap": False,
            "method": (
                "CSI1000 WEIGHT (AINDEXCSI1000WEIGHT) x constituent adj VWAP "
                "holding returns; same entry/exit dates as stocks. Not mixed with C2C."
            ),
            "weight_pit": "WEIGHT for entry date T+1; Wind OPDATE is prior evening",
        },
        "investability": {
            "signal_tradable_T": "fast_context universe_mask on feature date T (not_limit x not_st x trade_status)",
            "entry_tradable_T1": (
                "T+1 not suspended AND T+1 adj VWAP>0 AND T+1 close not at limit "
                "AND listing-age proxy (cumsum finite VWAP >= 60) on T+1. "
                "No CSI1000 membership filter."
            ),
            "min_listing_days": MIN_LISTING_DAYS,
        },
        "robustness": {
            ROBUSTNESS_O2O: "Open[T+1] -> Open[T+h+1]",
            ROBUSTNESS_C2C_DELAYED: "Close[T+1] -> Close[T+h+1]",
            "not_optimized": True,
        },
        "legacy": {
            "name": LEGACY_C2C_DIAGNOSTIC,
            "mapping": "factor T -> Close[T+1]/Close[T]-1 (FS-3 / shift(1))",
            "status": "LEGACY_RESEARCH_BENCHMARK",
            "requires_explicit_flag": True,
        },
        "classification_thresholds": {
            "IC_ABS_FLOOR": IC_ABS_FLOOR,
            "EXEC_NONEMPTY_ABS": EXEC_NONEMPTY_ABS,
            "PRESERVATION_ROBUST": PRESERVATION_ROBUST,
            "frozen_before_names": True,
        },
        "candidate_pool_v1": "FROZEN_FEATURE_HYPOTHESIS_LIBRARY",
        "fs3_fs4_f_kbest_60_xgb_y5": "LEGACY_RESEARCH_BENCHMARK",
        "do_not_switch_primary_on_sharpe": True,
    }


def three_date_v2v_proof(
    dates: Sequence,
    adj_vwap: pd.Series,
    *,
    symbol: str,
) -> List[Dict[str, object]]:
    """Prove ret_v2v[D] = VWAP[D]/VWAP[D-1]-1 on three consecutive dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    if len(dates) < 3:
        raise ValueError("need 3 dates")
    px = adj_vwap.reindex(dates).astype(float)
    rows = []
    for i, d in enumerate(dates[:3]):
        if i == 0:
            formula = None
            value = float("nan")
        else:
            prev = dates[i - 1]
            formula = "adjVWAP[{}]/adjVWAP[{}]-1".format(d.date(), prev.date())
            if px.loc[d] > 0 and px.loc[prev] > 0:
                value = float(px.loc[d] / px.loc[prev] - 1.0)
            else:
                value = float("nan")
        rec = map_feature_to_holding(dates, d, 1)
        rows.append(
            {
                "symbol": symbol,
                "date": str(pd.Timestamp(d).date()),
                "adj_vwap": float(px.loc[d]) if np.isfinite(px.loc[d]) else float("nan"),
                "ret_v2v_formula": formula,
                "ret_v2v_value": value,
                "if_shift1_would_pair_factor_prev_with": formula,
                "shift1_executable": False,
                "factor_this_date_1d_entry": str(pd.Timestamp(rec["entry_date"]).date())
                if pd.notna(rec["entry_date"])
                else None,
                "factor_this_date_1d_exit": str(pd.Timestamp(rec["exit_date"]).date())
                if pd.notna(rec["exit_date"])
                else None,
                "executable_1d_return": (
                    "adjVWAP[{}]/adjVWAP[{}]-1".format(
                        pd.Timestamp(rec["exit_date"]).date(),
                        pd.Timestamp(rec["entry_date"]).date(),
                    )
                    if rec["valid"]
                    else None
                ),
            }
        )
    return rows
