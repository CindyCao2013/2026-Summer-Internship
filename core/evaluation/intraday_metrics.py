"""Unified metric definitions for intraday cross-sectional research.

This module has no DolphinDB dependency.  Data adapters must provide exact
constituent-EW market returns and raw decile returns.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

ANNUALIZATION_DAYS = 250
GROUP_KEYS = ["Date", "Bartime", "return_window"]
GROUP_PANEL_COLUMNS = [
    "Date",
    "Bartime",
    "return_window",
    "group",
    "n_assets",
    "n_market_assets",
    "group_return_raw",
    "market_return",
    "group_return_excess",
]


def _numeric(series: Iterable[float]) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").dropna()


def annualized_return(
    returns: Iterable[float],
    annualization_days: int = ANNUALIZATION_DAYS,
) -> float:
    values = _numeric(returns)
    if values.empty:
        return np.nan
    return float(values.mean() * annualization_days)


def annualized_sharpe(
    returns: Iterable[float],
    annualization_days: int = ANNUALIZATION_DAYS,
) -> float:
    values = _numeric(returns)
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.nan
    return float(values.mean() / std * np.sqrt(annualization_days))


def max_drawdown(returns: Iterable[float]) -> float:
    values = _numeric(returns)
    if values.empty:
        return np.nan
    wealth = (1.0 + values).cumprod()
    wealth = pd.concat([pd.Series([1.0]), wealth], ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def benchmark_beta(
    portfolio_return: Iterable[float],
    market_return: Iterable[float],
) -> float:
    aligned = pd.concat(
        [
            pd.to_numeric(
                pd.Series(portfolio_return), errors="coerce"
            ).rename("portfolio"),
            pd.to_numeric(
                pd.Series(market_return), errors="coerce"
            ).rename("market"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return np.nan
    variance = aligned["market"].var(ddof=1)
    if not np.isfinite(variance) or variance == 0:
        return np.nan
    return float(aligned["portfolio"].cov(aligned["market"]) / variance)


def benchmark_correlation(
    portfolio_return: Iterable[float],
    market_return: Iterable[float],
) -> float:
    aligned = pd.concat(
        [
            pd.to_numeric(
                pd.Series(portfolio_return), errors="coerce"
            ).rename("portfolio"),
            pd.to_numeric(
                pd.Series(market_return), errors="coerce"
            ).rename("market"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return np.nan
    return float(aligned["portfolio"].corr(aligned["market"]))


def _normalize_group(group: object) -> str:
    value = str(group)
    if value.startswith("group_"):
        suffix = value[len("group_") :]
        if suffix.isdigit():
            return f"G{int(suffix) + 1}"
    return value


def build_group_excess_panel(
    group_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Join exact constituent-EW benchmark and calculate decile excess."""
    required_group = {
        *GROUP_KEYS,
        "group",
        "n_assets",
        "group_return_raw",
    }
    required_market = {
        *GROUP_KEYS,
        "n_market_assets",
        "market_return",
    }
    missing_group = required_group - set(group_returns.columns)
    missing_market = required_market - set(market_returns.columns)
    if missing_group or missing_market:
        raise KeyError(
            {
                "missing_group_columns": sorted(missing_group),
                "missing_market_columns": sorted(missing_market),
            }
        )
    group = group_returns.copy()
    market = market_returns.copy()
    group["Date"] = pd.to_datetime(group["Date"])
    market["Date"] = pd.to_datetime(market["Date"])
    if market.duplicated(GROUP_KEYS).any():
        raise ValueError("Market return must be unique by Date/Bartime/horizon")
    panel = group.merge(
        market[list(required_market)],
        on=GROUP_KEYS,
        how="left",
        validate="many_to_one",
    )
    if panel["market_return"].isna().any():
        raise ValueError("Missing exact market return for group observations")
    panel["group_return_raw"] = pd.to_numeric(
        panel["group_return_raw"], errors="coerce"
    )
    panel["market_return"] = pd.to_numeric(
        panel["market_return"], errors="coerce"
    )
    panel["group_return_excess"] = (
        panel["group_return_raw"] - panel["market_return"]
    )
    panel["group"] = panel["group"].map(_normalize_group)
    return panel[GROUP_PANEL_COLUMNS].sort_values(
        ["return_window", "Bartime", "Date", "group"]
    )


def build_hl_panel(
    group_panel: pd.DataFrame,
    *,
    direction: int,
    low_group: str = "G1",
    high_group: str = "G10",
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Build raw G10-G1 and frozen-direction H-L daily returns."""
    if direction not in (-1, 1):
        raise ValueError(f"Direction must be +/-1, got {direction}")
    required = set(GROUP_PANEL_COLUMNS)
    missing = required - set(group_panel.columns)
    if missing:
        raise KeyError(f"Missing group panel columns: {sorted(missing)}")
    key_columns = GROUP_KEYS
    value_columns = [
        "group_return_raw",
        "group_return_excess",
    ]
    low = group_panel[group_panel["group"] == low_group][
        [*key_columns, *value_columns]
    ].rename(
        columns={
            "group_return_raw": "low_return_raw",
            "group_return_excess": "low_return_excess",
        }
    )
    high = group_panel[group_panel["group"] == high_group][
        [*key_columns, "market_return", *value_columns]
    ].rename(
        columns={
            "group_return_raw": "high_return_raw",
            "group_return_excess": "high_return_excess",
        }
    )
    hl = high.merge(
        low,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    hl["raw_hl_return"] = hl["high_return_raw"] - hl["low_return_raw"]
    hl["excess_hl_return"] = (
        hl["high_return_excess"] - hl["low_return_excess"]
    )
    hl["hl_equivalence_error"] = (
        hl["raw_hl_return"] - hl["excess_hl_return"]
    ).abs()
    if (
        not hl.empty
        and float(hl["hl_equivalence_error"].max()) > tolerance
    ):
        raise AssertionError(
            "H-L excess equivalence failed: "
            f"{hl['hl_equivalence_error'].max()} > {tolerance}"
        )
    hl["direction"] = direction
    hl["hl_return"] = direction * hl["raw_hl_return"]
    return hl[
        [
            *key_columns,
            "market_return",
            "low_return_raw",
            "high_return_raw",
            "low_return_excess",
            "high_return_excess",
            "raw_hl_return",
            "excess_hl_return",
            "direction",
            "hl_return",
            "hl_equivalence_error",
        ]
    ].sort_values(["return_window", "Bartime", "Date"])


def summarize_ic_series(
    rank_ic: Iterable[float],
    *,
    direction: int = 1,
    annualization_days: int = ANNUALIZATION_DAYS,
) -> dict:
    if direction not in (-1, 1):
        raise ValueError(f"Direction must be +/-1, got {direction}")
    values = _numeric(rank_ic)
    mean_ic = float(values.mean()) if len(values) else np.nan
    std_ic = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    icir = (
        mean_ic / std_ic * np.sqrt(annualization_days)
        if np.isfinite(std_ic) and std_ic != 0
        else np.nan
    )
    return {
        "rank_ic": mean_ic,
        "annualized_icir": float(icir),
        "ic_win_rate": (
            float((direction * values > 0).mean())
            if len(values)
            else np.nan
        ),
        "n_dates": int(len(values)),
    }


def summarize_cross_sectional_metrics(
    group_panel: pd.DataFrame,
    hl_panel: pd.DataFrame,
    *,
    factor_name: str,
    annualization_days: int = ANNUALIZATION_DAYS,
) -> pd.DataFrame:
    """Return versioned group and H-L metric rows."""
    rows = []
    keys = ["Bartime", "return_window"]
    for (bartime, window, group_name), frame in group_panel.groupby(
        [*keys, "group"], sort=True
    ):
        raw = frame["group_return_raw"]
        excess = frame["group_return_excess"]
        rows.append(
            {
                "factor": factor_name,
                "metric_scope": "cross_sectional_group",
                "bartime": str(bartime),
                "return_window": str(window),
                "group": str(group_name),
                "group_return_raw": float(raw.mean()),
                "group_return_excess": float(excess.mean()),
                "group_excess_annualized_return": annualized_return(
                    excess, annualization_days
                ),
                "group_excess_sharpe": annualized_sharpe(
                    excess, annualization_days
                ),
                "group_excess_max_drawdown": max_drawdown(excess),
                "raw_hl_return": np.nan,
                "hl_return": np.nan,
                "hl_annualized_return": np.nan,
                "hl_sharpe": np.nan,
                "hl_max_drawdown": np.nan,
                "hl_market_beta": np.nan,
                "hl_market_corr": np.nan,
                "direction": np.nan,
                "direction_consistent": np.nan,
                "n_dates": int(frame["Date"].nunique()),
                "avg_group_assets": float(frame["n_assets"].mean()),
                "avg_market_assets": float(
                    frame["n_market_assets"].mean()
                ),
            }
        )
    for (bartime, window), frame in hl_panel.groupby(keys, sort=True):
        adjusted = frame["hl_return"]
        rows.append(
            {
                "factor": factor_name,
                "metric_scope": "cross_sectional_hl",
                "bartime": str(bartime),
                "return_window": str(window),
                "group": "H-L",
                "group_return_raw": np.nan,
                "group_return_excess": np.nan,
                "group_excess_annualized_return": np.nan,
                "group_excess_sharpe": np.nan,
                "group_excess_max_drawdown": np.nan,
                "raw_hl_return": float(frame["raw_hl_return"].mean()),
                "hl_return": float(adjusted.mean()),
                "hl_annualized_return": annualized_return(
                    adjusted, annualization_days
                ),
                "hl_sharpe": annualized_sharpe(
                    adjusted, annualization_days
                ),
                "hl_max_drawdown": max_drawdown(adjusted),
                "hl_market_beta": benchmark_beta(
                    adjusted, frame["market_return"]
                ),
                "hl_market_corr": benchmark_correlation(
                    adjusted, frame["market_return"]
                ),
                "direction": int(frame["direction"].iloc[0]),
                "direction_consistent": bool(adjusted.mean() > 0),
                "n_dates": int(frame["Date"].nunique()),
                "avg_group_assets": np.nan,
                "avg_market_assets": np.nan,
            }
        )
    return pd.DataFrame(rows)
