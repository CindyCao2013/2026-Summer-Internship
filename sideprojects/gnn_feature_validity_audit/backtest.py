"""Frozen-orientation, point-in-time single-factor decile evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_adapter import DataBundle
from features import preprocess_factor


DECILE_LABELS = ["Q{}".format(index) for index in range(1, 11)]

SUMMARY_COLUMNS = [
    "factor_id",
    "family",
    "factor_type",
    "data_status",
    "test_status",
    "raw_direction",
    "frozen_direction",
    "orientation_source",
    "calibration_start",
    "calibration_end",
    "evaluation_start",
    "evaluation_end",
    "n_valid_days",
    "valid_day_ratio",
    "median_stocks_per_day",
    "median_unique_values",
    "coverage",
    "composite_member_eligible",
    "composite_member_reason",
    "skipped_insufficient_stocks",
    "skipped_insufficient_unique_values",
    "skipped_qcut_failed",
    "skipped_missing_group_return",
    "hl_annual_return",
    "hl_sharpe",
    "q1_annual_return",
    "q10_annual_return",
    "decile_monotonicity",
    "pass_return",
    "pass_sharpe",
    "pass_monotonicity",
    "overall_pass",
    "failure_reason",
]


@dataclass
class EvaluationResult:
    factor_id: str
    summary: Dict[str, object]
    decile_daily: pd.DataFrame
    hl_daily: pd.Series
    processed_oriented: Optional[pd.DataFrame]
    composite_usable: bool
    skipped_day_reasons: Dict[str, int]


def annual_return(returns: pd.Series, annualization_days: int) -> float:
    values = pd.Series(returns, dtype=float).dropna()
    if values.empty:
        return np.nan
    return float(values.mean() * int(annualization_days))


def annualized_sharpe(returns: pd.Series, annualization_days: int) -> float:
    values = pd.Series(returns, dtype=float).dropna()
    if len(values) < 2:
        return np.nan
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(
        values.mean() / standard_deviation * np.sqrt(int(annualization_days))
    )


def decile_monotonicity(decile_annual_returns: Sequence[float]) -> float:
    values = np.asarray(decile_annual_returns, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        return np.nan
    correlation = spearmanr(np.arange(1, 11), values).correlation
    return float(correlation)


def strict_pass(
    hl_annual_return: float,
    hl_sharpe: float,
    monotonicity: float,
    thresholds: Mapping[str, float],
) -> Tuple[bool, bool, bool, bool]:
    pass_return = bool(
        np.isfinite(hl_annual_return)
        and hl_annual_return >= float(thresholds["hl_annual_return"])
    )
    pass_sharpe = bool(
        np.isfinite(hl_sharpe)
        and hl_sharpe >= float(thresholds["hl_sharpe"])
    )
    pass_monotonicity = bool(
        np.isfinite(monotonicity)
        and monotonicity >= float(thresholds["decile_monotonicity"])
    )
    return (
        pass_return,
        pass_sharpe,
        pass_monotonicity,
        bool(pass_return and pass_sharpe and pass_monotonicity),
    )


def failure_status(
    pass_return: bool,
    pass_sharpe: bool,
    pass_monotonicity: bool,
) -> str:
    failed = [
        name
        for name, passed in (
            ("RETURN", pass_return),
            ("SHARPE", pass_sharpe),
            ("MONOTONICITY", pass_monotonicity),
        )
        if not passed
    ]
    if not failed:
        return "PASS"
    if len(failed) > 1:
        return "FAIL_MULTIPLE"
    return "FAIL_{}".format(failed[0])


def split_calibration_evaluation(
    dates: pd.DatetimeIndex,
    calibration_fraction: float,
    embargo_rows: int = 0,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    ordered = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if len(ordered) < 2:
        return ordered, pd.DatetimeIndex([])
    cut = int(np.floor(len(ordered) * float(calibration_fraction)))
    cut = max(1, min(cut, len(ordered) - 1))
    calibration_end = max(0, cut - max(0, int(embargo_rows)))
    return ordered[:calibration_end], ordered[cut:]


def build_execution_eligible_mask(
    bundle: DataBundle,
    lag: int,
) -> pd.DataFrame:
    """Align signal eligibility with T+1 entry and T+lag exit tradability."""
    if int(lag) < 2:
        raise ValueError("Post-close c2c evaluation requires lag >= 2")
    signal_eligible = bundle.eligible_mask.reindex(
        index=bundle.sample_dates, columns=bundle.symbols
    ).fillna(False)
    tradable = bundle.tradable_mask.reindex_like(signal_eligible).fillna(False)
    entry_tradable = tradable.shift(-(int(lag) - 1)).fillna(False)
    exit_tradable = tradable.shift(-int(lag)).fillna(False)
    return signal_eligible & entry_tradable & exit_tradable


def mean_daily_rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    eligible: pd.DataFrame,
) -> float:
    daily: List[float] = []
    for date in pd.DatetimeIndex(dates):
        values = factor.loc[date]
        returns = forward_returns.loc[date]
        valid = (
            eligible.loc[date].fillna(False)
            & values.notna()
            & returns.notna()
        )
        if int(valid.sum()) < 2:
            continue
        if values.loc[valid].nunique(dropna=True) < 2:
            continue
        correlation = values.loc[valid].corr(
            returns.loc[valid], method="spearman"
        )
        if np.isfinite(correlation):
            daily.append(float(correlation))
    return float(np.mean(daily)) if daily else np.nan


def freeze_direction(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligible: pd.DataFrame,
    calibration_dates: pd.DatetimeIndex,
    default_orientation,
    orientation_method: str,
    zero_fallback: int,
) -> Tuple[int, str, float]:
    if orientation_method == "economic":
        direction = int(float(default_orientation))
        if direction not in (-1, 1):
            raise ValueError("Economic orientation must be -1 or 1")
        return direction, "economic", np.nan
    if orientation_method == "composite":
        return 1, "component_frozen_directions", np.nan
    if orientation_method != "calibration_30pct":
        raise ValueError(
            "Unsupported orientation method: {}".format(orientation_method)
        )
    calibration_ic = mean_daily_rank_ic(
        factor, forward_returns, calibration_dates, eligible
    )
    if not np.isfinite(calibration_ic) or calibration_ic == 0:
        direction = int(zero_fallback)
        source = "calibration_30pct_zero_fallback"
    else:
        direction = 1 if calibration_ic > 0 else -1
        source = "calibration_30pct_rank_ic"
    return direction, source, calibration_ic


def assign_deciles(values: pd.Series) -> Optional[pd.Series]:
    """Assign Q1..Q10 without ticker-order tie breaking."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 10 or clean.nunique(dropna=True) < 10:
        return None
    try:
        labels = pd.qcut(
            clean,
            q=10,
            labels=DECILE_LABELS,
            duplicates="raise",
        )
    except ValueError:
        return None
    result = pd.Series(labels.astype(str), index=clean.index, dtype=object)
    if set(result.unique()) != set(DECILE_LABELS):
        return None
    return result


def build_decile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    min_stocks: int,
    min_unique_values: int,
    raw_factor: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    stats: List[Dict[str, object]] = []
    skipped = {
        "insufficient_stocks": 0,
        "insufficient_unique_values": 0,
        "qcut_failed": 0,
        "missing_group_return": 0,
    }
    for date in dates:
        values = factor.loc[date]
        raw_values = raw_factor.loc[date] if raw_factor is not None else values
        returns = forward_returns.loc[date]
        valid = (
            eligible.loc[date].fillna(False)
            & values.notna()
            & raw_values.notna()
            & returns.notna()
        )
        count = int(valid.sum())
        unique = int(raw_values.loc[valid].nunique(dropna=True))
        stats.append(
            {"date": date, "n_stocks": count, "n_unique_values": unique}
        )
        if count < int(min_stocks):
            skipped["insufficient_stocks"] += 1
            continue
        if unique < int(min_unique_values):
            skipped["insufficient_unique_values"] += 1
            continue
        deciles = assign_deciles(values.loc[valid])
        if deciles is None:
            skipped["qcut_failed"] += 1
            continue
        row: Dict[str, object] = {"date": date}
        complete = True
        for label in DECILE_LABELS:
            members = deciles.index[deciles == label]
            group_return = returns.loc[members].mean()
            if not np.isfinite(group_return):
                complete = False
                break
            row[label] = float(group_return)
        if not complete:
            skipped["missing_group_return"] += 1
            continue
        row["H-L"] = float(row["Q10"]) - float(row["Q1"])
        rows.append(row)
    decile = (
        pd.DataFrame(rows).set_index("date")
        if rows
        else pd.DataFrame(columns=DECILE_LABELS + ["H-L"])
    )
    if not decile.empty:
        decile = decile.reindex(columns=DECILE_LABELS + ["H-L"])
        expected = decile["Q10"] - decile["Q1"]
        if not np.allclose(
            decile["H-L"].to_numpy(),
            expected.to_numpy(),
            equal_nan=True,
        ):
            raise AssertionError("H-L is not Q10 minus Q1")
    daily_stats = (
        pd.DataFrame(stats).set_index("date")
        if stats
        else pd.DataFrame(columns=["n_stocks", "n_unique_values"])
    )
    return decile, daily_stats, skipped


def calibration_composite_eligibility(
    raw_factor: pd.DataFrame,
    processed_factor: pd.DataFrame,
    eligible: pd.DataFrame,
    dates: pd.DatetimeIndex,
    decile_config: Mapping[str, object],
) -> Tuple[bool, str]:
    """Freeze member usability using calibration data only."""
    if len(dates) == 0:
        return False, "No calibration dates after embargo"
    valid_day_count = 0
    counts: List[int] = []
    unique_counts: List[int] = []
    valid_observations = 0
    eligible_observations = 0
    for date in dates:
        eligible_today = eligible.loc[date].fillna(False)
        valid = (
            eligible_today
            & raw_factor.loc[date].notna()
            & processed_factor.loc[date].notna()
        )
        count = int(valid.sum())
        unique = int(raw_factor.loc[date, valid].nunique(dropna=True))
        counts.append(count)
        unique_counts.append(unique)
        valid_observations += count
        eligible_observations += int(eligible_today.sum())
        if (
            count >= int(decile_config["min_stocks_per_day"])
            and unique >= int(decile_config["min_unique_values_per_day"])
        ):
            valid_day_count += 1
    coverage = (
        float(valid_observations / eligible_observations)
        if eligible_observations
        else 0.0
    )
    valid_day_ratio = float(valid_day_count / len(dates))
    median_stocks = float(np.median(counts)) if counts else 0.0
    median_unique = float(np.median(unique_counts)) if unique_counts else 0.0
    failures = []
    if coverage < float(decile_config["min_overall_stock_day_coverage"]):
        failures.append("calibration coverage {:.3f}".format(coverage))
    if valid_day_ratio < float(decile_config["min_valid_day_ratio"]):
        failures.append(
            "calibration valid-day ratio {:.3f}".format(valid_day_ratio)
        )
    if median_stocks < int(decile_config["min_stocks_per_day"]):
        failures.append(
            "calibration median stocks {:.1f}".format(median_stocks)
        )
    if median_unique < int(decile_config["min_unique_values_per_day"]):
        failures.append(
            "calibration median unique values {:.1f}".format(median_unique)
        )
    return not failures, "; ".join(failures)


def _empty_summary(
    registry_row: Mapping[str, object],
    *,
    data_status: str,
    test_status: str,
    reason: str,
    calibration_dates: pd.DatetimeIndex,
    evaluation_dates: pd.DatetimeIndex,
) -> Dict[str, object]:
    row: Dict[str, object] = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "factor_id": registry_row["factor_id"],
            "family": registry_row["family"],
            "factor_type": registry_row["factor_type"],
            "data_status": data_status,
            "test_status": test_status,
            "raw_direction": 1,
            "orientation_source": registry_row.get(
                "orientation_method", "not_applicable"
            ),
            "calibration_start": (
                calibration_dates.min() if len(calibration_dates) else pd.NaT
            ),
            "calibration_end": (
                calibration_dates.max() if len(calibration_dates) else pd.NaT
            ),
            "evaluation_start": (
                evaluation_dates.min() if len(evaluation_dates) else pd.NaT
            ),
            "evaluation_end": (
                evaluation_dates.max() if len(evaluation_dates) else pd.NaT
            ),
            "failure_reason": reason,
        }
    )
    return row


def _dates_with_forward_returns(
    forward_returns: pd.DataFrame,
) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(forward_returns.index[forward_returns.notna().any(axis=1)])


def evaluate_factor(
    factor_id: str,
    raw_panel: Optional[pd.DataFrame],
    registry_row: Mapping[str, object],
    bundle: DataBundle,
    config: Mapping[str, object],
    *,
    data_status: str = "AVAILABLE",
    unavailable_reason: str = "",
    already_preprocessed: bool = False,
) -> EvaluationResult:
    lag = int(config["timing"]["signal_lag_trading_rows"])
    returns = bundle.stock_returns.reindex(
        index=bundle.sample_dates, columns=bundle.symbols
    )
    forward_returns = returns.shift(-lag)
    valid_dates = _dates_with_forward_returns(forward_returns)
    calibration_dates, evaluation_dates = split_calibration_evaluation(
        valid_dates,
        float(config["orientation"]["calibration_fraction"]),
        embargo_rows=int(config["timing"]["calibration_embargo_rows"]),
    )
    if data_status != "AVAILABLE" or raw_panel is None:
        summary = _empty_summary(
            registry_row,
            data_status=data_status,
            test_status=data_status,
            reason=unavailable_reason,
            calibration_dates=calibration_dates,
            evaluation_dates=evaluation_dates,
        )
        return EvaluationResult(
            factor_id=factor_id,
            summary=summary,
            decile_daily=pd.DataFrame(columns=DECILE_LABELS + ["H-L"]),
            hl_daily=pd.Series(dtype=float, name="H-L"),
            processed_oriented=None,
            composite_usable=False,
            skipped_day_reasons={},
        )

    signal_eligible = bundle.eligible_mask.reindex(
        index=bundle.sample_dates, columns=bundle.symbols
    ).fillna(False)
    execution_eligible = build_execution_eligible_mask(bundle, lag)
    panel = raw_panel.reindex(
        index=bundle.sample_dates, columns=bundle.symbols
    ).where(signal_eligible)
    if already_preprocessed:
        processed = panel.astype(float).replace([np.inf, -np.inf], np.nan)
    else:
        processed = preprocess_factor(
            panel,
            neutralization=str(registry_row["neutralization"]),
            industry=bundle.industry.reindex(
                index=bundle.sample_dates, columns=bundle.symbols
            ),
            market_cap=bundle.market_cap.reindex(
                index=bundle.sample_dates, columns=bundle.symbols
            ).where(signal_eligible),
            config=config,
        )

    decile_cfg = config["deciles"]
    composite_usable, composite_reason = calibration_composite_eligibility(
        panel,
        processed,
        execution_eligible,
        calibration_dates,
        decile_cfg,
    )
    direction, orientation_source, calibration_ic = freeze_direction(
        processed,
        forward_returns,
        execution_eligible,
        calibration_dates,
        registry_row.get("default_orientation"),
        str(registry_row["orientation_method"]),
        int(config["orientation"]["zero_rank_ic_fallback_direction"]),
    )
    oriented = processed * direction
    decile, daily_stats, skipped = build_decile_returns(
        oriented,
        forward_returns,
        execution_eligible,
        evaluation_dates,
        min_stocks=int(decile_cfg["min_stocks_per_day"]),
        min_unique_values=int(decile_cfg["min_unique_values_per_day"]),
        raw_factor=panel,
    )

    eligible_eval = execution_eligible.reindex(
        index=evaluation_dates, columns=bundle.symbols
    )
    factor_valid = oriented.reindex(index=evaluation_dates).notna() & eligible_eval
    denominator = float(eligible_eval.sum().sum())
    coverage = (
        float(factor_valid.sum().sum()) / denominator
        if denominator > 0
        else np.nan
    )
    median_stocks = (
        float(daily_stats["n_stocks"].median())
        if not daily_stats.empty
        else 0.0
    )
    median_unique = (
        float(daily_stats["n_unique_values"].median())
        if not daily_stats.empty
        else 0.0
    )
    n_valid = int(len(decile))
    valid_ratio = (
        float(n_valid / len(evaluation_dates))
        if len(evaluation_dates)
        else 0.0
    )

    base_summary = _empty_summary(
        registry_row,
        data_status="AVAILABLE",
        test_status="UNTESTABLE",
        reason="",
        calibration_dates=calibration_dates,
        evaluation_dates=evaluation_dates,
    )
    base_summary.update(
        {
            "frozen_direction": int(direction),
            "orientation_source": orientation_source,
            "calibration_mean_rank_ic": calibration_ic,
            "n_valid_days": n_valid,
            "valid_day_ratio": valid_ratio,
            "median_stocks_per_day": median_stocks,
            "median_unique_values": median_unique,
            "coverage": coverage,
            "composite_member_eligible": composite_usable,
            "composite_member_reason": composite_reason,
            "skipped_insufficient_stocks": skipped["insufficient_stocks"],
            "skipped_insufficient_unique_values": skipped[
                "insufficient_unique_values"
            ],
            "skipped_qcut_failed": skipped["qcut_failed"],
            "skipped_missing_group_return": skipped["missing_group_return"],
        }
    )

    validity_failures: List[str] = []
    status = "UNTESTABLE"
    if not np.isfinite(coverage) or coverage < float(
        decile_cfg["min_overall_stock_day_coverage"]
    ):
        status = "INSUFFICIENT_COVERAGE"
        validity_failures.append(
            "coverage {:.3f} < {:.3f}".format(
                coverage,
                float(decile_cfg["min_overall_stock_day_coverage"]),
            )
        )
    elif median_unique < int(decile_cfg["min_unique_values_per_day"]):
        status = "INSUFFICIENT_CROSS_SECTIONAL_VARIATION"
        validity_failures.append(
            "median unique values {:.1f} < {}".format(
                median_unique, int(decile_cfg["min_unique_values_per_day"])
            )
        )
    else:
        if valid_ratio < float(decile_cfg["min_valid_day_ratio"]):
            validity_failures.append(
                "valid day ratio {:.3f} < {:.3f}".format(
                    valid_ratio, float(decile_cfg["min_valid_day_ratio"])
                )
            )
        if n_valid < int(decile_cfg["min_evaluation_days"]):
            validity_failures.append(
                "valid days {} < {}".format(
                    n_valid, int(decile_cfg["min_evaluation_days"])
                )
            )
        if median_stocks < int(decile_cfg["min_stocks_per_day"]):
            validity_failures.append(
                "median stocks {:.1f} < {}".format(
                    median_stocks, int(decile_cfg["min_stocks_per_day"])
                )
            )

    if validity_failures:
        base_summary["test_status"] = status
        base_summary["failure_reason"] = "; ".join(validity_failures)
        return EvaluationResult(
            factor_id=factor_id,
            summary=base_summary,
            decile_daily=decile,
            hl_daily=(
                decile["H-L"]
                if "H-L" in decile
                else pd.Series(dtype=float, name="H-L")
            ),
            processed_oriented=oriented,
            composite_usable=composite_usable,
            skipped_day_reasons=skipped,
        )

    annualization = int(config["metrics"]["annualization_days"])
    decile_annual = [
        annual_return(decile[label], annualization) for label in DECILE_LABELS
    ]
    hl_annual = annual_return(decile["H-L"], annualization)
    hl_sharpe = annualized_sharpe(decile["H-L"], annualization)
    monotonicity = decile_monotonicity(decile_annual)
    thresholds = config["metrics"]["pass_thresholds"]
    (
        pass_return,
        pass_sharpe,
        pass_monotonicity,
        overall,
    ) = strict_pass(hl_annual, hl_sharpe, monotonicity, thresholds)
    test_status = failure_status(
        pass_return, pass_sharpe, pass_monotonicity
    )
    failed_names = [
        name
        for name, passed in (
            ("return", pass_return),
            ("sharpe", pass_sharpe),
            ("monotonicity", pass_monotonicity),
        )
        if not passed
    ]
    base_summary.update(
        {
            "test_status": test_status,
            "hl_annual_return": hl_annual,
            "hl_sharpe": hl_sharpe,
            "q1_annual_return": decile_annual[0],
            "q10_annual_return": decile_annual[-1],
            "decile_monotonicity": monotonicity,
            "pass_return": pass_return,
            "pass_sharpe": pass_sharpe,
            "pass_monotonicity": pass_monotonicity,
            "overall_pass": overall,
            "failure_reason": (
                "" if overall else "Failed " + ", ".join(failed_names)
            ),
        }
    )
    return EvaluationResult(
        factor_id=factor_id,
        summary=base_summary,
        decile_daily=decile,
        hl_daily=decile["H-L"].copy(),
        processed_oriented=oriented,
        composite_usable=composite_usable,
        skipped_day_reasons=skipped,
    )


def summary_frame(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    rows = [result.summary for result in results]
    frame = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    extras = [column for column in frame.columns if column not in SUMMARY_COLUMNS]
    return frame[SUMMARY_COLUMNS + extras]
