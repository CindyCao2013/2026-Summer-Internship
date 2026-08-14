"""Pure pandas/numpy helpers for mid-trade-amount normalization.

This module deliberately contains no database access.  It freezes the
calculation rules shared by the cache builder and the report layer:

* scales are calculated on the complete market trading calendar;
* every scale uses exactly ``shift(1)``, ``window=20`` and
  ``min_periods=20``;
* a missing source value on any of the preceding 20 market days invalidates
  that scale (older observations are never pulled forward to fill the gap);
* amount intervals are lower-exclusive and upper-inclusive;
* A1 selection uses distribution coverage only, never return statistics;
* factor direction is supplied as frozen configuration and is never inferred
  from the evaluation sample.

The canonical key is ``Symbol + TradeDate``.  For compatibility with existing
factor-reproduction frames, helpers also accept a lower-case ``symbol`` column
when ``Symbol`` is absent, while still enforcing a one-to-one symbol-day key.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


KEY_DATE_COLUMN = "TradeDate"
SYMBOL_COLUMN_CANDIDATES = ("Symbol", "symbol")
ROLLING_WINDOW = 20

A0_FACTOR_ID = "mid_trade_amount_share_abs_4w20w"
A1_FACTOR_ID = "mid_trade_amount_share_adv20"
A2_FACTOR_ID = "mid_trade_amount_share_ats20"
A3_FACTOR_ID = "mid_trade_amount_share_rollq"

A0_LOWER_RMB = 40_000.0
A0_UPPER_RMB = 200_000.0

ADV_LOWER_BPS_GRID = (0.5, 1.0, 2.0)
ADV_UPPER_BPS_GRID = (5.0, 10.0, 20.0)
ATS_LOWER_MULTIPLE_GRID = (0.25, 0.5, 0.75)
ATS_UPPER_MULTIPLE_GRID = (1.5, 2.0, 3.0)
FROZEN_A2_LOWER_MULTIPLE = 0.5
FROZEN_A2_UPPER_MULTIPLE = 2.0

FROZEN_CONFIG_HASH_FIELD = "config_sha256"
FROZEN_CONFIG_SCHEMA_VERSION = "mid_trade_amount_normalization_v1"
FROZEN_DIRECTION_POLICY = "frozen_before_evaluation_never_inferred"


def _resolve_symbol_column(
    frame: pd.DataFrame,
    symbol_col: Optional[str] = None,
) -> str:
    """Resolve the symbol column without silently accepting ambiguity."""
    if symbol_col is not None:
        if symbol_col not in frame.columns:
            raise ValueError(f"missing symbol column {symbol_col!r}")
        return symbol_col
    present = [
        column for column in SYMBOL_COLUMN_CANDIDATES if column in frame.columns
    ]
    if len(present) != 1:
        raise ValueError(
            "frame must contain exactly one of 'Symbol' or 'symbol'; "
            f"found={present}"
        )
    return present[0]


def _normalize_trade_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def assert_unique_symbol_trade_date(
    frame: pd.DataFrame,
    *,
    symbol_col: Optional[str] = None,
    date_col: str = KEY_DATE_COLUMN,
    frame_name: str = "frame",
) -> None:
    """Hard-check non-null, unique ``Symbol + TradeDate`` keys.

    Dates are normalized before checking, so two timestamps on the same
    trading date cannot evade the uniqueness gate.
    """
    resolved_symbol = _resolve_symbol_column(frame, symbol_col)
    if date_col not in frame.columns:
        raise ValueError(f"{frame_name} missing date column {date_col!r}")
    if frame[[resolved_symbol, date_col]].isna().any().any():
        raise ValueError(
            f"{frame_name} contains null Symbol/TradeDate key values"
        )

    keys = pd.DataFrame(
        {
            "Symbol": frame[resolved_symbol].astype(str),
            "TradeDate": _normalize_trade_dates(frame[date_col]),
        },
        index=frame.index,
    )
    duplicated = keys.duplicated(["Symbol", "TradeDate"], keep=False)
    if duplicated.any():
        examples = (
            keys.loc[duplicated, ["Symbol", "TradeDate"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{frame_name} must be unique at Symbol+TradeDate; "
            f"examples={examples}"
        )


def merge_symbol_trade_date_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    how: str = "left",
    symbol_col: Optional[str] = None,
    date_col: str = KEY_DATE_COLUMN,
    suffixes: Tuple[str, str] = ("", "_right"),
    require_match: bool = False,
    left_name: str = "left",
    right_name: str = "right",
) -> pd.DataFrame:
    """Merge two symbol-day frames after hard one-to-one key validation."""
    if how not in {"left", "right", "inner", "outer"}:
        raise ValueError(f"unsupported merge type: {how!r}")
    left_symbol = _resolve_symbol_column(left, symbol_col)
    right_symbol = _resolve_symbol_column(
        right,
        symbol_col if symbol_col in right.columns else None,
    )
    assert_unique_symbol_trade_date(
        left,
        symbol_col=left_symbol,
        date_col=date_col,
        frame_name=left_name,
    )
    assert_unique_symbol_trade_date(
        right,
        symbol_col=right_symbol,
        date_col=date_col,
        frame_name=right_name,
    )

    left_copy = left.copy()
    right_copy = right.copy()
    left_copy[left_symbol] = left_copy[left_symbol].astype(str)
    right_copy[right_symbol] = right_copy[right_symbol].astype(str)
    left_copy[date_col] = _normalize_trade_dates(left_copy[date_col])
    right_copy[date_col] = _normalize_trade_dates(right_copy[date_col])
    if right_symbol != left_symbol:
        if left_symbol in right_copy.columns:
            raise ValueError(
                f"{right_name} contains ambiguous symbol columns "
                f"{right_symbol!r} and {left_symbol!r}"
            )
        right_copy = right_copy.rename(columns={right_symbol: left_symbol})

    marker = "__symbol_date_merge_status__"
    if marker in left_copy.columns or marker in right_copy.columns:
        raise ValueError(f"reserved merge marker column already exists: {marker}")
    merged = left_copy.merge(
        right_copy,
        on=[left_symbol, date_col],
        how=how,
        suffixes=suffixes,
        validate="one_to_one",
        indicator=marker,
        sort=False,
    )
    if require_match and not merged[marker].eq("both").all():
        counts = merged[marker].value_counts().to_dict()
        raise ValueError(
            "Symbol+TradeDate join has unmatched keys; "
            f"merge_counts={counts}"
        )
    return merged.drop(columns=marker)


def _normalize_market_calendar(
    market_calendar: Iterable[Any],
    *,
    date_col: str = KEY_DATE_COLUMN,
) -> pd.DatetimeIndex:
    if isinstance(market_calendar, pd.DataFrame):
        if date_col not in market_calendar.columns:
            raise ValueError(
                f"market calendar missing date column {date_col!r}"
            )
        raw_dates = market_calendar[date_col]
    else:
        raw_dates = list(market_calendar)
    dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="raise"))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    if dates.hasnans:
        raise ValueError("market calendar contains null dates")
    if dates.duplicated().any():
        raise ValueError("market calendar contains duplicate dates")
    if len(dates) == 0:
        raise ValueError("market calendar is empty")
    return dates.sort_values()


def _prepare_positive_scale_source(
    values: pd.Series,
    *,
    column: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype("float64")
    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    if (~finite & numeric.notna().to_numpy()).any():
        raise ValueError(f"{column} contains infinite values")
    if (numeric.dropna() < 0).any():
        raise ValueError(f"{column} contains negative values")
    # Zero cannot be a denominator and therefore counts as missing history.
    return numeric.where(numeric > 0)


def _rolling_scale_and_evidence(
    source: pd.Series,
    symbols: pd.Series,
    dates: pd.Series,
    *,
    aggregation: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return lagged scale, valid-history count, and latest source date."""
    lagged = source.groupby(symbols, sort=False).shift(1)
    grouped_lagged = lagged.groupby(symbols, sort=False)
    if aggregation == "mean":
        scale = grouped_lagged.transform(
            lambda values: values.rolling(
                ROLLING_WINDOW,
                min_periods=ROLLING_WINDOW,
            ).mean()
        )
    elif aggregation == "median":
        scale = grouped_lagged.transform(
            lambda values: values.rolling(
                ROLLING_WINDOW,
                min_periods=ROLLING_WINDOW,
            ).median()
        )
    else:
        raise ValueError(f"unsupported rolling aggregation: {aggregation!r}")

    lagged_valid = (
        source.notna()
        .astype("int64")
        .groupby(symbols, sort=False)
        .shift(1)
        .fillna(0)
    )
    history_count = lagged_valid.groupby(symbols, sort=False).transform(
        lambda values: values.rolling(
            ROLLING_WINDOW,
            min_periods=1,
        ).sum()
    )
    history_count = history_count.astype("int64")

    epoch = pd.Timestamp("1970-01-01")
    source_day_number = (dates - epoch).dt.days.astype("float64")
    source_day_number = source_day_number.where(source.notna())
    lagged_source_day = source_day_number.groupby(
        symbols,
        sort=False,
    ).shift(1)
    source_max_number = lagged_source_day.groupby(
        symbols,
        sort=False,
    ).transform(
        lambda values: values.rolling(
            ROLLING_WINDOW,
            min_periods=1,
        ).max()
    )
    source_max_date = pd.to_datetime(
        source_max_number,
        unit="D",
        origin="unix",
        errors="coerce",
    )
    return scale.astype("float64"), history_count, source_max_date


def build_lagged_trade_size_scales(
    daily_scale: pd.DataFrame,
    market_calendar: Iterable[Any],
    *,
    symbol_col: Optional[str] = None,
    date_col: str = KEY_DATE_COLUMN,
    total_amount_col: str = "total_amount",
    daily_median_col: str = "daily_median_trade_amount",
) -> pd.DataFrame:
    """Build strict ADV20/ATS20 scales on a complete market calendar.

    The output contains every input symbol crossed with every market-calendar
    date.  This is intentional: rolling over the observed rows alone would
    silently replace a missing market day with an older observation.

    ``history_count`` is conservative joint evidence (the minimum of ADV and
    ATS source counts).  Scale-specific counts and source dates are retained as
    well, so a missing median does not obscure otherwise valid ADV evidence.
    """
    resolved_symbol = _resolve_symbol_column(daily_scale, symbol_col)
    required = {
        resolved_symbol,
        date_col,
        total_amount_col,
        daily_median_col,
    }
    missing = sorted(required.difference(daily_scale.columns))
    if missing:
        raise ValueError(f"daily scale primitive missing columns: {missing}")

    frame = daily_scale.loc[
        :,
        [resolved_symbol, date_col, total_amount_col, daily_median_col],
    ].copy()
    frame[resolved_symbol] = frame[resolved_symbol].astype(str)
    frame[date_col] = _normalize_trade_dates(frame[date_col])
    assert_unique_symbol_trade_date(
        frame,
        symbol_col=resolved_symbol,
        date_col=date_col,
        frame_name="daily scale primitive",
    )
    frame[total_amount_col] = _prepare_positive_scale_source(
        frame[total_amount_col],
        column=total_amount_col,
    )
    frame[daily_median_col] = _prepare_positive_scale_source(
        frame[daily_median_col],
        column=daily_median_col,
    )

    calendar = _normalize_market_calendar(
        market_calendar,
        date_col=date_col,
    )
    outside_calendar = ~frame[date_col].isin(calendar)
    if outside_calendar.any():
        examples = (
            frame.loc[outside_calendar, date_col]
            .drop_duplicates()
            .sort_values()
            .head(5)
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "daily scale primitive contains dates outside market calendar; "
            f"examples={examples}"
        )

    symbols = sorted(frame[resolved_symbol].unique().tolist())
    if not symbols:
        columns = [
            resolved_symbol,
            date_col,
            total_amount_col,
            daily_median_col,
            "ADV20_lag1",
            "ADV20_median_lag1",
            "ATS20_lag1",
            "ADV20_history_count",
            "ATS20_history_count",
            "history_count",
            "ADV20_source_max_date",
            "ATS20_source_max_date",
            "source_max_date",
        ]
        return pd.DataFrame(columns=columns)

    full_index = pd.MultiIndex.from_product(
        [symbols, calendar],
        names=[resolved_symbol, date_col],
    )
    full = (
        frame.set_index([resolved_symbol, date_col])[
            [total_amount_col, daily_median_col]
        ]
        .reindex(full_index)
        .reset_index()
    )
    symbols_series = full[resolved_symbol]
    dates_series = full[date_col]

    adv_mean, adv_count, adv_source_max = _rolling_scale_and_evidence(
        full[total_amount_col],
        symbols_series,
        dates_series,
        aggregation="mean",
    )
    adv_median, adv_median_count, adv_median_source_max = (
        _rolling_scale_and_evidence(
            full[total_amount_col],
            symbols_series,
            dates_series,
            aggregation="median",
        )
    )
    if not adv_count.equals(adv_median_count) or not adv_source_max.equals(
        adv_median_source_max
    ):
        raise AssertionError("ADV mean/median evidence unexpectedly differs")
    ats_median, ats_count, ats_source_max = _rolling_scale_and_evidence(
        full[daily_median_col],
        symbols_series,
        dates_series,
        aggregation="median",
    )

    full["ADV20_lag1"] = adv_mean
    full["ADV20_median_lag1"] = adv_median
    full["ATS20_lag1"] = ats_median
    full["ADV20_history_count"] = adv_count
    full["ATS20_history_count"] = ats_count
    full["history_count"] = np.minimum(adv_count, ats_count).astype("int64")
    full["ADV20_source_max_date"] = adv_source_max
    full["ATS20_source_max_date"] = ats_source_max
    full["source_max_date"] = pd.concat(
        [adv_source_max, ats_source_max],
        axis=1,
    ).max(axis=1)

    validate_lagged_scale_evidence(
        full,
        symbol_col=resolved_symbol,
        date_col=date_col,
    )
    return full


def validate_lagged_scale_evidence(
    scales: pd.DataFrame,
    *,
    symbol_col: Optional[str] = None,
    date_col: str = KEY_DATE_COLUMN,
) -> None:
    """Validate no-look-ahead and complete-history scale invariants."""
    resolved_symbol = _resolve_symbol_column(scales, symbol_col)
    required = {
        resolved_symbol,
        date_col,
        "ADV20_lag1",
        "ADV20_median_lag1",
        "ATS20_lag1",
        "ADV20_history_count",
        "ATS20_history_count",
        "history_count",
        "ADV20_source_max_date",
        "ATS20_source_max_date",
        "source_max_date",
    }
    missing = sorted(required.difference(scales.columns))
    if missing:
        raise ValueError(f"lagged scale evidence missing columns: {missing}")
    assert_unique_symbol_trade_date(
        scales,
        symbol_col=resolved_symbol,
        date_col=date_col,
        frame_name="lagged scales",
    )
    dates = _normalize_trade_dates(scales[date_col])
    for count_column in (
        "ADV20_history_count",
        "ATS20_history_count",
        "history_count",
    ):
        count = pd.to_numeric(scales[count_column], errors="raise")
        if not count.between(0, ROLLING_WINDOW).all():
            raise ValueError(
                f"{count_column} outside [0, {ROLLING_WINDOW}]"
            )

    evidence_specs = (
        (
            "ADV20_lag1",
            "ADV20_history_count",
            "ADV20_source_max_date",
        ),
        (
            "ADV20_median_lag1",
            "ADV20_history_count",
            "ADV20_source_max_date",
        ),
        (
            "ATS20_lag1",
            "ATS20_history_count",
            "ATS20_source_max_date",
        ),
    )
    for scale_column, count_column, source_column in evidence_specs:
        scale = pd.to_numeric(scales[scale_column], errors="raise")
        count = pd.to_numeric(scales[count_column], errors="raise")
        source_date = pd.to_datetime(scales[source_column], errors="coerce")
        expected_valid = count.eq(ROLLING_WINDOW)
        if not scale.notna().equals(expected_valid):
            raise ValueError(
                f"{scale_column} validity is inconsistent with "
                f"{count_column}"
            )
        if (scale.dropna() <= 0).any():
            raise ValueError(f"{scale_column} must be positive when valid")
        if source_date.notna().any():
            future_or_current = source_date.notna() & source_date.ge(dates)
            if future_or_current.any():
                raise ValueError(
                    f"{source_column} violates strict shift(1)"
                )

    generic_source = pd.to_datetime(
        scales["source_max_date"],
        errors="coerce",
    )
    if (generic_source.notna() & generic_source.ge(dates)).any():
        raise ValueError("source_max_date violates strict shift(1)")


def _safe_relative_ratio(
    numerator: Any,
    denominator: Any,
    *,
    multiplier: float,
) -> Any:
    left, right = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    out = np.full(left.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(left)
        & np.isfinite(right)
        & (left >= 0)
        & (right > 0)
    )
    np.divide(left, right, out=out, where=valid)
    out *= multiplier
    return float(out) if out.ndim == 0 else out


def relative_trade_size_adv_bps(amount: Any, adv20_lag1: Any) -> Any:
    """Return trade amount as basis points of lagged ADV20."""
    return _safe_relative_ratio(
        amount,
        adv20_lag1,
        multiplier=10_000.0,
    )


def relative_trade_size_ats(amount: Any, ats20_lag1: Any) -> Any:
    """Return trade amount as a multiple of lagged ATS20."""
    return _safe_relative_ratio(
        amount,
        ats20_lag1,
        multiplier=1.0,
    )


def validate_threshold_pair(
    lower: float,
    upper: float,
    *,
    label: str = "threshold",
) -> Tuple[float, float]:
    """Validate a positive lower-exclusive, upper-inclusive interval."""
    lower_value = float(lower)
    upper_value = float(upper)
    if not np.isfinite(lower_value) or not np.isfinite(upper_value):
        raise ValueError(f"{label} bounds must be finite")
    if lower_value < 0 or upper_value <= 0:
        raise ValueError(f"{label} bounds must be non-negative/positive")
    if lower_value >= upper_value:
        raise ValueError(
            f"{label} requires lower < upper; "
            f"got {lower_value} >= {upper_value}"
        )
    return lower_value, upper_value


def _validated_trade_amounts(amounts: Sequence[float]) -> np.ndarray:
    values = np.asarray(amounts, dtype=float)
    if values.ndim != 1:
        raise ValueError("trade amounts must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("trade amounts contain infinite values")
    if (values[np.isfinite(values)] < 0).any():
        raise ValueError("trade amounts contain negative values")
    return values[np.isfinite(values) & (values > 0)]


def amount_share_in_interval(
    amounts: Sequence[float],
    lower_exclusive: float,
    upper_inclusive: float,
    *,
    total_amount: Optional[float] = None,
) -> float:
    """Amount share in ``(lower_exclusive, upper_inclusive]``."""
    lower, upper = validate_threshold_pair(
        lower_exclusive,
        upper_inclusive,
        label="amount interval",
    )
    values = _validated_trade_amounts(amounts)
    if total_amount is None:
        denominator = float(values.sum())
    else:
        denominator = float(total_amount)
        if not np.isfinite(denominator) or denominator <= 0:
            return np.nan
    if denominator <= 0:
        return np.nan
    selected = float(values[(values > lower) & (values <= upper)].sum())
    tolerance = max(1e-10, abs(denominator) * 1e-12)
    if selected > denominator + tolerance:
        raise ValueError("selected amount exceeds total amount")
    return selected / denominator


def compute_a0_share(
    amounts: Sequence[float],
    *,
    lower_rmb: float = A0_LOWER_RMB,
    upper_rmb: float = A0_UPPER_RMB,
    total_amount: Optional[float] = None,
) -> float:
    """A0: fixed-RMB trade-amount share."""
    return amount_share_in_interval(
        amounts,
        lower_rmb,
        upper_rmb,
        total_amount=total_amount,
    )


def compute_a1_share(
    amounts: Sequence[float],
    adv20_lag1: float,
    lower_bps: float,
    upper_bps: float,
    *,
    total_amount: Optional[float] = None,
) -> float:
    """A1: amount share inside a lagged-ADV20 relative-size interval."""
    lower, upper = validate_threshold_pair(
        lower_bps,
        upper_bps,
        label="A1 ADV-bps interval",
    )
    scale = float(adv20_lag1)
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return amount_share_in_interval(
        amounts,
        scale * lower / 10_000.0,
        scale * upper / 10_000.0,
        total_amount=total_amount,
    )


def compute_a2_share(
    amounts: Sequence[float],
    ats20_lag1: float,
    lower_multiple: float = FROZEN_A2_LOWER_MULTIPLE,
    upper_multiple: float = FROZEN_A2_UPPER_MULTIPLE,
    *,
    total_amount: Optional[float] = None,
) -> float:
    """A2: amount share inside a lagged-ATS20 multiple interval."""
    lower, upper = validate_threshold_pair(
        lower_multiple,
        upper_multiple,
        label="A2 ATS-multiple interval",
    )
    scale = float(ats20_lag1)
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return amount_share_in_interval(
        amounts,
        scale * lower,
        scale * upper,
        total_amount=total_amount,
    )


def compute_a3_share(
    amounts: Sequence[float],
    daily_q20: float,
    daily_q80: float,
    *,
    total_amount: Optional[float] = None,
) -> float:
    """A3: same-day Q20-Q80 amount share, formed after that day's close."""
    lower = float(daily_q20)
    upper = float(daily_q80)
    if np.isnan(lower) or np.isnan(upper):
        return np.nan
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("A3 daily quantiles must be finite or NaN")
    if lower < 0 or upper < 0:
        raise ValueError("A3 daily quantiles must be non-negative")
    if lower > upper:
        raise ValueError("A3 requires daily_q20 <= daily_q80")
    if lower == upper:
        # A constant same-day trade-size distribution has a valid but empty
        # lower-exclusive/upper-inclusive middle interval.
        values = _validated_trade_amounts(amounts)
        denominator = (
            float(values.sum())
            if total_amount is None
            else float(total_amount)
        )
        return 0.0 if np.isfinite(denominator) and denominator > 0 else np.nan
    return amount_share_in_interval(
        amounts,
        lower,
        upper,
        total_amount=total_amount,
    )


def compute_dynamic_factor_shares(
    amounts: Sequence[float],
    *,
    adv20_lag1: float,
    ats20_lag1: float,
    daily_q20: float,
    daily_q80: float,
    a1_lower_bps: float,
    a1_upper_bps: float,
    a2_lower_multiple: float = FROZEN_A2_LOWER_MULTIPLE,
    a2_upper_multiple: float = FROZEN_A2_UPPER_MULTIPLE,
    total_amount: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate A0/A1/A2/A3 with one shared amount denominator."""
    return {
        A0_FACTOR_ID: compute_a0_share(
            amounts,
            total_amount=total_amount,
        ),
        A1_FACTOR_ID: compute_a1_share(
            amounts,
            adv20_lag1,
            a1_lower_bps,
            a1_upper_bps,
            total_amount=total_amount,
        ),
        A2_FACTOR_ID: compute_a2_share(
            amounts,
            ats20_lag1,
            a2_lower_multiple,
            a2_upper_multiple,
            total_amount=total_amount,
        ),
        A3_FACTOR_ID: compute_a3_share(
            amounts,
            daily_q20,
            daily_q80,
            total_amount=total_amount,
        ),
    }


def amount_share_from_aggregates(
    selected_amount: Any,
    total_amount: Any,
) -> Any:
    """Convert server-side selected/total amount aggregates into a share."""
    selected, total = np.broadcast_arrays(
        np.asarray(selected_amount, dtype=float),
        np.asarray(total_amount, dtype=float),
    )
    finite_selected = selected[np.isfinite(selected)]
    if (finite_selected < 0).any():
        raise ValueError("selected amount contains negative values")
    finite_total = total[np.isfinite(total)]
    if (finite_total < 0).any():
        raise ValueError("total amount contains negative values")
    tolerance = np.maximum(1e-10, np.abs(total) * 1e-12)
    impossible = (
        np.isfinite(selected)
        & np.isfinite(total)
        & (total >= 0)
        & (selected > total + tolerance)
    )
    if impossible.any():
        raise ValueError("selected amount exceeds total amount")
    out = np.full(selected.shape, np.nan, dtype=float)
    valid = np.isfinite(selected) & np.isfinite(total) & (total > 0)
    np.divide(selected, total, out=out, where=valid)
    return float(out) if out.ndim == 0 else out


def _number_token(value: float) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("candidate threshold must be finite")
    text = format(number, ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def candidate_grid_name(
    variant: str,
    lower: float,
    upper: float,
) -> str:
    """Return a deterministic, column-safe A1/A2 candidate name."""
    normalized = str(variant).upper()
    lower_value, upper_value = validate_threshold_pair(
        lower,
        upper,
        label=f"{normalized} candidate",
    )
    lower_token = _number_token(lower_value)
    upper_token = _number_token(upper_value)
    if normalized == "A1":
        return f"a1_adv20_l{lower_token}_h{upper_token}_bps"
    if normalized == "A2":
        return f"a2_ats20_l{lower_token}_h{upper_token}_x"
    raise ValueError("candidate grid variant must be 'A1' or 'A2'")


@dataclass(frozen=True)
class CandidateSpec:
    """One threshold pair in a frozen stability grid."""

    variant: str
    name: str
    lower: float
    upper: float
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_candidate_grid(
    variant: str,
    *,
    lower_values: Optional[Sequence[float]] = None,
    upper_values: Optional[Sequence[float]] = None,
) -> Tuple[CandidateSpec, ...]:
    """Build the small, economically interpretable A1 or A2 grid."""
    normalized = str(variant).upper()
    if normalized == "A1":
        lowers = ADV_LOWER_BPS_GRID if lower_values is None else lower_values
        uppers = ADV_UPPER_BPS_GRID if upper_values is None else upper_values
        unit = "bps_of_adv20_lag1"
    elif normalized == "A2":
        lowers = (
            ATS_LOWER_MULTIPLE_GRID
            if lower_values is None
            else lower_values
        )
        uppers = (
            ATS_UPPER_MULTIPLE_GRID
            if upper_values is None
            else upper_values
        )
        unit = "multiple_of_ats20_lag1"
    else:
        raise ValueError("candidate grid variant must be 'A1' or 'A2'")

    candidates: List[CandidateSpec] = []
    for lower, upper in product(lowers, uppers):
        lower_value, upper_value = validate_threshold_pair(
            lower,
            upper,
            label=f"{normalized} candidate",
        )
        candidates.append(
            CandidateSpec(
                variant=normalized,
                name=candidate_grid_name(
                    normalized,
                    lower_value,
                    upper_value,
                ),
                lower=lower_value,
                upper=upper_value,
                unit=unit,
            )
        )
    validate_candidate_grid(candidates)
    return tuple(candidates)


def validate_candidate_grid(candidates: Sequence[CandidateSpec]) -> None:
    """Hard-check threshold, naming, and uniqueness invariants for a grid."""
    if not candidates:
        raise ValueError("candidate grid is empty")
    variants = set()
    names = set()
    pairs = set()
    for candidate in candidates:
        if not isinstance(candidate, CandidateSpec):
            raise TypeError("candidate grid must contain CandidateSpec values")
        variant = candidate.variant.upper()
        variants.add(variant)
        lower, upper = validate_threshold_pair(
            candidate.lower,
            candidate.upper,
            label=f"{variant} candidate",
        )
        expected_name = candidate_grid_name(variant, lower, upper)
        if candidate.name != expected_name:
            raise ValueError(
                f"candidate name mismatch: {candidate.name!r} != "
                f"{expected_name!r}"
            )
        if candidate.name in names:
            raise ValueError(f"duplicate candidate name: {candidate.name}")
        pair = (lower, upper)
        if pair in pairs:
            raise ValueError(f"duplicate candidate thresholds: {pair}")
        names.add(candidate.name)
        pairs.add(pair)
    if len(variants) != 1:
        raise ValueError(
            f"candidate grid mixes variants: {sorted(variants)}"
        )


def candidate_grid_frame(
    variant: str,
    *,
    lower_values: Optional[Sequence[float]] = None,
    upper_values: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Return a candidate grid in a serialization-friendly table."""
    return pd.DataFrame(
        [
            candidate.to_dict()
            for candidate in build_candidate_grid(
                variant,
                lower_values=lower_values,
                upper_values=upper_values,
            )
        ]
    )


def _forbid_return_selection_columns(columns: Iterable[Any]) -> None:
    forbidden_exact = {"return", "returns", "ret", "ic", "pnl", "alpha"}
    forbidden_fragments = (
        "future_return",
        "forward_return",
        "rank_ic",
        "icir",
        "sharpe",
        "h_l_return",
        "hl_return",
        "pnl_",
        "alpha_",
    )
    found = []
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in forbidden_exact or any(
            fragment in normalized for fragment in forbidden_fragments
        ):
            found.append(str(column))
    if found:
        raise ValueError(
            "A1 calibration must be distribution-only; "
            f"return/outcome columns are forbidden: {found}"
        )


def freeze_a1_distribution_candidate(
    distribution: pd.DataFrame,
    a0_overall_coverage: float,
    *,
    candidate_col: str = "candidate",
    quintile_col: str = "market_cap_quintile",
    selected_amount_col: str = "selected_amount",
    total_amount_col: str = "total_amount",
    lower_col: str = "lower_bps",
    upper_col: str = "upper_bps",
    a0_quintile_coverage: Optional[Mapping[Any, float]] = None,
    minimum_quintile_coverage: float = 0.10,
    maximum_quintile_coverage: float = 0.80,
    expected_quintiles: int = 5,
) -> Dict[str, Any]:
    """Freeze A1 using amount coverage only.

    Candidates are eligible only when every market-cap quintile has selected
    amount coverage in the inclusive ``[10%, 80%]`` band.  Ranking is:

    1. absolute overall coverage difference versus A0;
    2. mean absolute quintile difference versus A0 when an A0 quintile map is
       supplied, otherwise the candidate's max-minus-min quintile spread;
    3. lower threshold;
    4. upper threshold;
    5. candidate name (a final deterministic guard).

    The function rejects return/outcome columns and never imports an evaluation
    module, making the Stage-A/Stage-B boundary auditable.
    """
    _forbid_return_selection_columns(distribution.columns)
    required = {
        candidate_col,
        quintile_col,
        selected_amount_col,
        total_amount_col,
        lower_col,
        upper_col,
    }
    missing = sorted(required.difference(distribution.columns))
    if missing:
        raise ValueError(f"A1 distribution summary missing columns: {missing}")
    if distribution.empty:
        raise ValueError("A1 distribution summary is empty")

    target = float(a0_overall_coverage)
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("A0 overall coverage must be within [0, 1]")
    lower_coverage = float(minimum_quintile_coverage)
    upper_coverage = float(maximum_quintile_coverage)
    if (
        not np.isfinite(lower_coverage)
        or not np.isfinite(upper_coverage)
        or lower_coverage < 0
        or upper_coverage > 1
        or lower_coverage >= upper_coverage
    ):
        raise ValueError("invalid A1 quintile coverage bounds")
    if int(expected_quintiles) <= 0:
        raise ValueError("expected_quintiles must be positive")

    frame = distribution.loc[:, list(required)].copy()
    if frame[[candidate_col, quintile_col]].isna().any().any():
        raise ValueError("A1 distribution contains null candidate/quintile")
    frame[candidate_col] = frame[candidate_col].astype(str)
    for column in (
        selected_amount_col,
        total_amount_col,
        lower_col,
        upper_col,
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
            "float64"
        )
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"A1 distribution {column} must be finite")
    if (frame[selected_amount_col] < 0).any():
        raise ValueError("A1 selected amount must be non-negative")
    if (frame[total_amount_col] <= 0).any():
        raise ValueError("A1 total amount must be positive")
    tolerance = np.maximum(
        1e-10,
        frame[total_amount_col].abs() * 1e-12,
    )
    if (
        frame[selected_amount_col]
        > frame[total_amount_col] + tolerance
    ).any():
        raise ValueError("A1 selected amount exceeds total amount")

    quintiles = list(pd.unique(frame[quintile_col]))
    if len(quintiles) != int(expected_quintiles):
        raise ValueError(
            "A1 distribution must contain exactly "
            f"{expected_quintiles} market-cap quintiles; found={quintiles}"
        )
    a0_by_quintile: Optional[Dict[Any, float]] = None
    if a0_quintile_coverage is not None:
        a0_by_quintile = {}
        for quintile in quintiles:
            if quintile not in a0_quintile_coverage:
                raise ValueError(
                    f"A0 quintile coverage missing {quintile!r}"
                )
            value = float(a0_quintile_coverage[quintile])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    "A0 quintile coverage must be within [0, 1]"
                )
            a0_by_quintile[quintile] = value

    records: List[Dict[str, Any]] = []
    for candidate, block in frame.groupby(candidate_col, sort=True):
        if block[lower_col].nunique(dropna=False) != 1 or block[
            upper_col
        ].nunique(dropna=False) != 1:
            raise ValueError(
                f"candidate {candidate!r} has inconsistent thresholds"
            )
        lower_bps, upper_bps = validate_threshold_pair(
            block[lower_col].iloc[0],
            block[upper_col].iloc[0],
            label=f"A1 candidate {candidate}",
        )
        grouped = (
            block.groupby(quintile_col, sort=False)[
                [selected_amount_col, total_amount_col]
            ]
            .sum()
            .reindex(quintiles)
        )
        if grouped.isna().any().any():
            raise ValueError(
                f"candidate {candidate!r} is missing a market-cap quintile"
            )
        quintile_coverage = (
            grouped[selected_amount_col] / grouped[total_amount_col]
        )
        eligible = quintile_coverage.between(
            lower_coverage - 1e-15,
            upper_coverage + 1e-15,
        ).all()
        overall_coverage = float(
            block[selected_amount_col].sum()
            / block[total_amount_col].sum()
        )
        overall_gap = abs(overall_coverage - target)
        if a0_by_quintile is None:
            quintile_gap = float(
                quintile_coverage.max() - quintile_coverage.min()
            )
            quintile_gap_definition = "max_minus_min_candidate_coverage"
        else:
            quintile_gap = float(
                np.mean(
                    [
                        abs(
                            float(quintile_coverage.loc[quintile])
                            - a0_by_quintile[quintile]
                        )
                        for quintile in quintiles
                    ]
                )
            )
            quintile_gap_definition = (
                "mean_absolute_coverage_gap_vs_a0_quintiles"
            )
        records.append(
            {
                "candidate": candidate,
                "lower_bps": lower_bps,
                "upper_bps": upper_bps,
                "eligible": bool(eligible),
                "overall_coverage": overall_coverage,
                "overall_coverage_gap_vs_a0": overall_gap,
                "quintile_coverage_gap": quintile_gap,
                "quintile_coverage_gap_definition": (
                    quintile_gap_definition
                ),
                "quintile_coverage": {
                    str(quintile): float(quintile_coverage.loc[quintile])
                    for quintile in quintiles
                },
            }
        )

    eligible_records = [record for record in records if record["eligible"]]
    if not eligible_records:
        raise ValueError(
            "no A1 candidate satisfies 10%-80% coverage in every "
            "market-cap quintile"
        )
    chosen = sorted(
        eligible_records,
        key=lambda record: (
            round(float(record["overall_coverage_gap_vs_a0"]), 15),
            round(float(record["quintile_coverage_gap"]), 15),
            float(record["lower_bps"]),
            float(record["upper_bps"]),
            str(record["candidate"]),
        ),
    )[0]
    return {
        "variant": "A1",
        "factor_id": A1_FACTOR_ID,
        "candidate": chosen["candidate"],
        "lower_bps": chosen["lower_bps"],
        "upper_bps": chosen["upper_bps"],
        "selection_basis": "distribution_only_no_returns",
        "target_a0_overall_coverage": target,
        "overall_coverage": chosen["overall_coverage"],
        "overall_coverage_gap_vs_a0": chosen[
            "overall_coverage_gap_vs_a0"
        ],
        "quintile_coverage": chosen["quintile_coverage"],
        "quintile_coverage_gap": chosen["quintile_coverage_gap"],
        "quintile_coverage_gap_definition": chosen[
            "quintile_coverage_gap_definition"
        ],
        "quintile_coverage_bounds": [
            lower_coverage,
            upper_coverage,
        ],
        "tie_break_order": [
            "overall_coverage_gap_vs_a0",
            "quintile_coverage_gap",
            "lower_bps",
            "upper_bps",
            "candidate",
        ],
        "eligible_candidate_count": len(eligible_records),
        "candidate_count": len(records),
    }


# A concise alias for callers that frame the operation as selection.
select_a1_frozen_candidate = freeze_a1_distribution_candidate


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            string_key = str(key)
            if string_key in normalized:
                raise ValueError(
                    f"config keys collide after string conversion: {key!r}"
                )
            normalized[string_key] = _jsonable(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NA:
        raise ValueError("frozen config cannot contain pd.NA")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("frozen config cannot contain NaN/Infinity")
        return number
    if value is None or isinstance(value, str):
        return value
    raise TypeError(
        f"unsupported frozen-config value type: {type(value).__name__}"
    )


def canonical_frozen_config_json(
    config: Mapping[str, Any],
    *,
    hash_field: str = FROZEN_CONFIG_HASH_FIELD,
) -> str:
    """Serialize config deterministically, excluding its self hash."""
    if not isinstance(config, Mapping):
        raise TypeError("frozen config must be a mapping")
    payload = copy.deepcopy(dict(config))
    payload.pop(hash_field, None)
    return json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def frozen_config_sha256(
    config: Mapping[str, Any],
    *,
    hash_field: str = FROZEN_CONFIG_HASH_FIELD,
) -> str:
    """SHA256 of canonical config content, excluding the hash field itself."""
    canonical = canonical_frozen_config_json(
        config,
        hash_field=hash_field,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_config(
    config: Mapping[str, Any],
    *,
    hash_field: str = FROZEN_CONFIG_HASH_FIELD,
) -> Dict[str, Any]:
    """Return a JSON-safe deep copy carrying its canonical content hash."""
    frozen = _jsonable(copy.deepcopy(dict(config)))
    if "schema_version" not in frozen:
        frozen["schema_version"] = FROZEN_CONFIG_SCHEMA_VERSION
    frozen.pop(hash_field, None)
    frozen[hash_field] = frozen_config_sha256(
        frozen,
        hash_field=hash_field,
    )
    return frozen


def _validate_direction(direction: Any) -> int:
    if isinstance(direction, (bool, np.bool_)):
        raise ValueError("effective direction must be -1 or 1, not bool")
    try:
        value = int(direction)
    except (TypeError, ValueError):
        raise ValueError("effective direction must be -1 or 1")
    if value not in (-1, 1) or float(direction) != value:
        raise ValueError("effective direction must be exactly -1 or 1")
    return value


def validate_frozen_config(
    config: Mapping[str, Any],
    *,
    expected_sha256: Optional[str] = None,
    required_keys: Sequence[str] = (),
    hash_field: str = FROZEN_CONFIG_HASH_FIELD,
) -> str:
    """Validate frozen-config integrity and known semantic invariants.

    Returns the verified SHA256.  ``expected_sha256`` is intended for a Stage-B
    manifest pin; a mismatch fails before any return evaluation is run.
    """
    if not isinstance(config, Mapping):
        raise TypeError("frozen config must be a mapping")
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"frozen config missing required keys: {missing}")
    if hash_field not in config:
        raise ValueError(f"frozen config missing {hash_field!r}")
    stored = str(config[hash_field])
    actual = frozen_config_sha256(config, hash_field=hash_field)
    if stored != actual:
        raise ValueError(
            f"frozen config SHA256 mismatch: stored={stored}, actual={actual}"
        )
    if expected_sha256 is not None and actual != str(expected_sha256):
        raise ValueError(
            "frozen config does not match expected Stage-B SHA256: "
            f"{actual} != {expected_sha256}"
        )

    a1 = config.get("a1", config.get("A1"))
    if isinstance(a1, Mapping):
        if "selection_basis" in a1 and a1["selection_basis"] != (
            "distribution_only_no_returns"
        ):
            raise ValueError(
                "A1 frozen config must use distribution-only selection"
            )
        if "lower_bps" in a1 or "upper_bps" in a1:
            if "lower_bps" not in a1 or "upper_bps" not in a1:
                raise ValueError("A1 frozen config has incomplete thresholds")
            validate_threshold_pair(
                a1["lower_bps"],
                a1["upper_bps"],
                label="frozen A1",
            )
    a2 = config.get("a2", config.get("A2"))
    if isinstance(a2, Mapping) and (
        "lower_multiple" in a2 or "upper_multiple" in a2
    ):
        if "lower_multiple" not in a2 or "upper_multiple" not in a2:
            raise ValueError("A2 frozen config has incomplete thresholds")
        validate_threshold_pair(
            a2["lower_multiple"],
            a2["upper_multiple"],
            label="frozen A2",
        )

    directions = config.get("effective_direction")
    if isinstance(directions, Mapping):
        for direction in directions.values():
            _validate_direction(direction)
    elif directions is not None:
        _validate_direction(directions)
    return actual


def _parity_key_frame(
    frame: pd.DataFrame,
    value_col: str,
    *,
    symbol_col: Optional[str],
    date_col: str,
    output_value_col: str,
    frame_name: str,
) -> pd.DataFrame:
    resolved_symbol = _resolve_symbol_column(frame, symbol_col)
    if value_col not in frame.columns:
        raise ValueError(f"{frame_name} missing value column {value_col!r}")
    assert_unique_symbol_trade_date(
        frame,
        symbol_col=resolved_symbol,
        date_col=date_col,
        frame_name=frame_name,
    )
    return pd.DataFrame(
        {
            "Symbol": frame[resolved_symbol].astype(str),
            "TradeDate": _normalize_trade_dates(frame[date_col]),
            output_value_col: pd.to_numeric(
                frame[value_col],
                errors="raise",
            ).astype("float64"),
        }
    )


def _spearman_without_optional_dependencies(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if len(left) == 0:
        return np.nan
    if np.array_equal(left, right):
        return 1.0
    if len(left) == 1:
        return 1.0 if left[0] == right[0] else np.nan
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def compare_a0_parity(
    rebuilt: pd.DataFrame,
    authoritative: pd.DataFrame,
    *,
    rebuilt_value_col: str = "value",
    authoritative_value_col: str = "value",
    symbol_col: Optional[str] = None,
    date_col: str = KEY_DATE_COLUMN,
    minimum_spearman: float = 1.0 - 1e-12,
    maximum_abs_error: float = 1e-10,
) -> Dict[str, Any]:
    """Compare rebuilt A0 values with the authoritative strict panel."""
    if not 0 <= float(minimum_spearman) <= 1:
        raise ValueError("minimum_spearman must be within [0, 1]")
    if float(maximum_abs_error) < 0:
        raise ValueError("maximum_abs_error must be non-negative")

    left = _parity_key_frame(
        rebuilt,
        rebuilt_value_col,
        symbol_col=symbol_col,
        date_col=date_col,
        output_value_col="rebuilt_value",
        frame_name="rebuilt A0",
    )
    right = _parity_key_frame(
        authoritative,
        authoritative_value_col,
        symbol_col=symbol_col,
        date_col=date_col,
        output_value_col="authoritative_value",
        frame_name="authoritative A0",
    )
    merged = left.merge(
        right,
        on=["Symbol", "TradeDate"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=True,
    )
    key_match = bool(merged["_merge"].eq("both").all())
    matched = merged.loc[merged["_merge"].eq("both")].copy()
    rebuilt_values = matched["rebuilt_value"]
    authoritative_values = matched["authoritative_value"]
    nan_pattern_match = bool(
        rebuilt_values.isna().equals(authoritative_values.isna())
    )
    paired = rebuilt_values.notna() & authoritative_values.notna()
    finite = (
        paired
        & np.isfinite(rebuilt_values)
        & np.isfinite(authoritative_values)
    )
    nonfinite_pair_count = int((paired & ~finite).sum())
    left_numeric = rebuilt_values.loc[finite].to_numpy(dtype=float)
    right_numeric = authoritative_values.loc[finite].to_numpy(dtype=float)
    if len(left_numeric):
        differences = np.abs(left_numeric - right_numeric)
        observed_max_abs_error = float(differences.max())
        spearman = _spearman_without_optional_dependencies(
            left_numeric,
            right_numeric,
        )
    else:
        observed_max_abs_error = np.nan
        spearman = np.nan

    error_pass = bool(
        np.isfinite(observed_max_abs_error)
        and observed_max_abs_error <= float(maximum_abs_error)
    )
    spearman_pass = bool(
        np.isfinite(spearman)
        and spearman >= float(minimum_spearman)
    )
    passed = bool(
        key_match
        and nan_pattern_match
        and nonfinite_pair_count == 0
        and error_pass
        and spearman_pass
    )
    return {
        "passed": passed,
        "key_match": key_match,
        "nan_pattern_match": nan_pattern_match,
        "rebuilt_row_count": int(len(left)),
        "authoritative_row_count": int(len(right)),
        "matched_key_count": int(len(matched)),
        "unmatched_key_count": int((merged["_merge"] != "both").sum()),
        "compared_value_count": int(len(left_numeric)),
        "nonfinite_pair_count": nonfinite_pair_count,
        "spearman": spearman,
        "minimum_spearman": float(minimum_spearman),
        "max_abs_error": observed_max_abs_error,
        "maximum_abs_error": float(maximum_abs_error),
    }


def assert_a0_parity(
    rebuilt: pd.DataFrame,
    authoritative: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the A0 parity comparison and raise when the hard gate fails."""
    result = compare_a0_parity(rebuilt, authoritative, **kwargs)
    if not result["passed"]:
        raise ValueError(f"A0 parity gate failed: {result}")
    return result


def apply_frozen_direction(values: Any, effective_direction: int) -> Any:
    """Apply a supplied direction without inspecting values to choose its sign."""
    direction = _validate_direction(effective_direction)
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values * direction
    array = np.asarray(values, dtype=float)
    directed = array * direction
    return float(directed) if directed.ndim == 0 else directed


def evaluate_frozen_direction(
    raw_hl_returns: Sequence[float],
    *,
    effective_direction: int,
) -> Dict[str, Any]:
    """Summarize raw/effective H-L returns under one immutable direction."""
    direction = _validate_direction(effective_direction)
    raw = pd.Series(raw_hl_returns, dtype="float64")
    if np.isinf(raw.to_numpy(dtype=float)).any():
        raise ValueError("raw H-L returns contain infinite values")
    effective = apply_frozen_direction(raw, direction)
    return {
        "effective_direction": direction,
        "direction_policy": FROZEN_DIRECTION_POLICY,
        "direction_was_inferred": False,
        "raw_hl_mean": float(raw.mean()) if raw.notna().any() else np.nan,
        "effective_hl_mean": (
            float(effective.mean()) if effective.notna().any() else np.nan
        ),
        "effective_hl_mean_positive": bool(
            effective.notna().any() and effective.mean() > 0
        ),
        "effective_hl_returns": effective,
    }


def implied_annual_fee(
    average_daily_turnover: float,
    *,
    fee_bps: float = 7.5,
    annualization: int = 250,
) -> float:
    """Annual implied fee: turnover × fee-bps/10,000 × trading days."""
    turnover = float(average_daily_turnover)
    fee = float(fee_bps)
    periods = int(annualization)
    if np.isnan(turnover):
        return np.nan
    if not np.isfinite(turnover) or turnover < 0:
        raise ValueError("average daily turnover must be non-negative")
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("fee_bps must be non-negative")
    if periods <= 0:
        raise ValueError("annualization must be positive")
    return turnover * fee / 10_000.0 * periods


def format_fee_bps_label(fee_bps: float = 7.5) -> str:
    """Return an unambiguous bps label (never a percent-rate label)."""
    fee = float(fee_bps)
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("fee_bps must be non-negative")
    return f"fee={fee:g} bps"


fee_bps_label = format_fee_bps_label


__all__ = [
    "A0_FACTOR_ID",
    "A0_LOWER_RMB",
    "A0_UPPER_RMB",
    "A1_FACTOR_ID",
    "A2_FACTOR_ID",
    "A3_FACTOR_ID",
    "ADV_LOWER_BPS_GRID",
    "ADV_UPPER_BPS_GRID",
    "ATS_LOWER_MULTIPLE_GRID",
    "ATS_UPPER_MULTIPLE_GRID",
    "CandidateSpec",
    "FROZEN_A2_LOWER_MULTIPLE",
    "FROZEN_A2_UPPER_MULTIPLE",
    "FROZEN_CONFIG_HASH_FIELD",
    "FROZEN_CONFIG_SCHEMA_VERSION",
    "FROZEN_DIRECTION_POLICY",
    "ROLLING_WINDOW",
    "amount_share_from_aggregates",
    "amount_share_in_interval",
    "apply_frozen_direction",
    "assert_a0_parity",
    "assert_unique_symbol_trade_date",
    "build_candidate_grid",
    "build_lagged_trade_size_scales",
    "candidate_grid_frame",
    "candidate_grid_name",
    "canonical_frozen_config_json",
    "compare_a0_parity",
    "compute_a0_share",
    "compute_a1_share",
    "compute_a2_share",
    "compute_a3_share",
    "compute_dynamic_factor_shares",
    "evaluate_frozen_direction",
    "fee_bps_label",
    "format_fee_bps_label",
    "freeze_a1_distribution_candidate",
    "freeze_config",
    "frozen_config_sha256",
    "implied_annual_fee",
    "merge_symbol_trade_date_one_to_one",
    "relative_trade_size_adv_bps",
    "relative_trade_size_ats",
    "select_a1_frozen_candidate",
    "validate_candidate_grid",
    "validate_frozen_config",
    "validate_lagged_scale_evidence",
    "validate_threshold_pair",
]
