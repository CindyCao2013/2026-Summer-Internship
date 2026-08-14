"""Shared context loader for D2/D3 dimension density runners (full sample + OOS split)."""

from __future__ import annotations

import datetime as dt
from typing import Dict

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from alpha_frozen_stack_v1 import FROZEN_OHLCV_REPS
from factor_attribution import OHLCV_FROZEN_REPS
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_liquidity_norm import build_liquidity_norm_cache
from liquidity_normalization import effective_turnover
from run_l2_validation import build_any_eod


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dimension_density_context() -> dict:
    """Load full 2020–2025 sample for chronological discovery/confirmation split."""
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )

    close = enriched.close.loc[start:end]
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    frozen_panels: Dict[str, pd.DataFrame] = {}
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        try:
            frozen_panels[spec.representative] = build_any_eod(
                spec.representative, pv_cache, norm_cache
            ).loc[start:end]
            log(f"  frozen OK {spec.representative}")
        except Exception as exc:
            log(f"  frozen SKIP {spec.representative}: {exc}")

    turnover = effective_turnover(enriched.turnover, enriched.amount, enriched.float_mktcap).loc[start:end]
    ret_1d = pv_cache.get("ret_1d").loc[start:end]

    exposure_panels = {
        "size": np.log(enriched.float_mktcap.replace(0, np.nan)).loc[start:end],
        "liquidity": frozen_panels.get(
            "low_vol_liquidity_quality_60d",
            pv_cache.get("amount_mean_20d").loc[start:end],
        ),
        "volatility": frozen_panels.get(
            "volatility_60d", pv_cache.get("volatility_60d").loc[start:end]
        ),
        "turnover": turnover,
        "ret_1d": ret_1d,
    }

    frozen_list = [frozen_panels[r] for r in OHLCV_FROZEN_REPS if r in frozen_panels]

    cross_dim_anchors = {
        "D1_rep": frozen_panels["low_vol_liquidity_quality_60d"],
        "D4_rep": frozen_panels["winner_sentiment_reversal_5d"],
        "D5_rep": frozen_panels["upside_fragility_20d"],
        "size": exposure_panels["size"],
        "turnover": turnover,
        "ret_1d": ret_1d,
    }

    def get_ret_matrix(s, e, idx):
        return Factor_Dev_Lib.get_Ret_Matrix(s, e, method="c2c", base_index=idx)

    return {
        "start": start,
        "end": end,
        "session": session,
        "close": close,
        "ret": ret,
        "pv_cache": pv_cache,
        "norm_cache": norm_cache,
        "frozen_panels": frozen_panels,
        "frozen_list": frozen_list,
        "exposure_panels": exposure_panels,
        "cross_dim_anchors": cross_dim_anchors,
        "universes": cfg.UNIVERSE_LIST,
        "get_ret_matrix": get_ret_matrix,
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
    }


def build_candidate(name: str, source: str, ctx: dict) -> pd.DataFrame:
    from factor_formulas import build_factor
    from factor_formulas_eod_engine import build_eod_engine_factor

    if name in ctx["frozen_panels"]:
        return ctx["frozen_panels"][name]
    if source == "pv":
        return build_factor(name, ctx["pv_cache"])
    if source == "eod_engine":
        return build_eod_engine_factor(name, ctx["pv_cache"])
    raise ValueError(f"Unknown source: {source}")
