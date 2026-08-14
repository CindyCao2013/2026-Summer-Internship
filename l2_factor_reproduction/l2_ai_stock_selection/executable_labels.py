"""Explicit executable V2V (and robustness) labels. No DataFrame.shift for ML y."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    HORIZONS,
    compound_daily_from_lag,
    daily_ratio_return,
    holding_return_from_prices,
    map_feature_to_holding,
)


def assert_no_c2c_mix(stock_method: str, bench_method: str) -> None:
    if stock_method != bench_method:
        raise ValueError(
            "refusing mixed execution: stock={} benchmark={}".format(
                stock_method, bench_method
            )
        )


def date_mapping_table(
    dates: Sequence,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    rows = []
    for t in dates:
        for h in horizons:
            rec = map_feature_to_holding(dates, t, h)
            rows.append(
                {
                    "feature_date": rec["feature_date"],
                    "horizon": h,
                    "entry_date": rec["entry_date"],
                    "exit_date": rec["exit_date"],
                    "valid": rec["valid"],
                    "entry_offset_trading_days": rec["entry_offset_trading_days"],
                    "exit_offset_trading_days": rec["exit_offset_trading_days"],
                }
            )
    return pd.DataFrame(rows)


def excess_from_prices(
    stock_price: pd.DataFrame,
    bench_price: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    horizon: int,
    method: str,
) -> pd.DataFrame:
    """stock and benchmark use the same price field and the same entry/exit dates."""
    assert_no_c2c_mix(method, method)
    stock_h = holding_return_from_prices(stock_price, dates, horizon=horizon, start_lag=1)
    bench_wide = pd.DataFrame(
        {c: bench_price for c in stock_h.columns}, index=bench_price.index
    )
    bench_h = holding_return_from_prices(bench_wide, dates, horizon=horizon, start_lag=1)
    # bench_h is identical across columns; take first
    b = bench_h.iloc[:, 0]
    return stock_h.sub(b, axis=0)


def excess_from_reconstructed_index(
    stock_holding: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    mapping: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Benchmark = entry-date CSI1000 WEIGHT × constituent holding returns.

    weights index = calendar date (entry), columns = symbols, values sum ~ 100.
    Missing constituent prices: drop and renormalize remaining weights.
    """
    hmap = mapping.loc[mapping["horizon"] == int(horizon)].copy()
    hmap["feature_date"] = pd.to_datetime(hmap["feature_date"]).dt.normalize()
    hmap["entry_date"] = pd.to_datetime(hmap["entry_date"]).dt.normalize()
    hmap = hmap.set_index("feature_date")
    w = weights.copy()
    w.index = pd.to_datetime(w.index).normalize()
    common = stock_holding.columns.intersection(w.columns)
    y = stock_holding.reindex(columns=common)
    feat_idx = y.index.intersection(hmap.index)
    entry = hmap.loc[feat_idx, "entry_date"]
    valid = hmap.loc[feat_idx, "valid"].astype(bool) & entry.notna()
    w_ent = w.reindex(index=entry.values, columns=common)
    w_ent.index = feat_idx
    w_ent = w_ent.astype(float).where(valid, np.nan)
    yv = y.reindex(index=feat_idx).astype(float)
    w_ok = w_ent.where(w_ent > 0).where(yv.notna())
    den = w_ok.sum(axis=1).replace(0, np.nan)
    bench = (w_ok * yv).sum(axis=1) / den
    out = stock_holding.astype(float).sub(bench.reindex(stock_holding.index), axis=0)
    out.loc[~out.index.isin(feat_idx[valid.to_numpy()])] = np.nan
    # also NaN where bench missing
    out = out.where(bench.reindex(out.index).notna(), np.nan)
    return out


def build_all_horizons_from_prices(
    stock_price: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    horizons: Sequence[int] = HORIZONS,
    bench_price: Optional[pd.Series] = None,
    weights: Optional[pd.DataFrame] = None,
    method: str = "v2v",
) -> Dict[int, pd.DataFrame]:
    mapping = date_mapping_table(dates, horizons)
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        stock_h = holding_return_from_prices(
            stock_price, dates, horizon=int(h), start_lag=1
        )
        if weights is not None:
            y = excess_from_reconstructed_index(
                stock_h, weights, mapping=mapping, horizon=int(h)
            )
        elif bench_price is not None:
            y = excess_from_prices(
                stock_price, bench_price, dates, horizon=int(h), method=method
            )
        else:
            raise ValueError("need bench_price or weights")
        out[int(h)] = y
    out["_mapping"] = mapping  # type: ignore
    return out


def tail_invalid_ok(y: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int) -> bool:
    """Last h+1 feature dates must be all-NaN (need T+1+h exit)."""
    n = int(horizon) + 1
    tail = y.reindex(index=dates).iloc[-n:]
    return bool(tail.isna().all().all())


def load_production_labels(
    *,
    execution_contract: Optional[str] = None,
    horizons: Sequence[int] = HORIZONS,
) -> Dict[int, pd.DataFrame]:
    """Load discovery labels. Default is EXEC_V2V_TPLUS1_V1. Legacy C2C is not implicit."""
    from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
        LEGACY_C2C_DIAGNOSTIC,
        resolve_execution_contract,
    )
    from l2_factor_reproduction.l2_ai_stock_selection.paths import discovery_label_dir

    contract = resolve_execution_contract(execution_contract)
    if contract == LEGACY_C2C_DIAGNOSTIC:
        raise ValueError(
            "legacy C2C labels require an explicit loader; "
            "execution_contract=LEGACY_C2C_DIAGNOSTIC is not an implicit default"
        )
    root = discovery_label_dir(contract)
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        path = root / "forward_return_{}d.parquet".format(int(h))
        if not path.exists():
            raise FileNotFoundError(
                "production labels missing for {} h={} ({})".format(contract, h, path)
            )
        y = pd.read_parquet(path)
        y.index = pd.to_datetime(y.index).normalize()
        out[int(h)] = y
    return out
