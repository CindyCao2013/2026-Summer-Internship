"""Shared event table → 24 daily LR candidates. No-event is NA, not 0."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.liquidity_resilience.contracts import (
    DENOM_FLOOR_DEPTH,
    DENOM_FLOOR_FLOW,
    DENOM_FLOOR_OBI,
    DENOM_FLOOR_SPREAD,
    FROZEN_CANDIDATE_NAMES,
    SHOCK_ACTIVE_BUY,
    SHOCK_ACTIVE_SELL,
    SHOCK_DEPTH,
    SHOCK_SPREAD,
)
from l2_factor_reproduction.liquidity_resilience.primitives import (
    enrich_minutes,
    finite_book_valid,
    session_matrix,
    shift_session,
)
from l2_factor_reproduction.liquidity_resilience.recovery import (
    event_median,
    obi_persistence,
    obi_restoration,
    recovery_fraction,
    replenishment_efficiency,
    shock_size_weighted_mean,
    spread_recovery_fraction,
    spread_residual_width,
)
from l2_factor_reproduction.liquidity_resilience.shocks import (
    active_buy_shock,
    active_sell_shock,
    depth_depletion_shock,
    spread_widening_shock,
    trailing_median_2d,
)

_SESSION_COLS = (
    "bid_depth_5",
    "ask_depth_5",
    "depth",
    "spread",
    "obi",
    "active_buy_amount",
    "active_sell_amount",
)


def _horizon_valid(n_keys: int) -> Dict[int, np.ndarray]:
    """Boolean (n_keys,) — column t has a complete +h slot in this session grid."""
    idx = np.arange(n_keys)
    return {h: (idx + h) < n_keys for h in (1, 3, 5)}


def _pre_valid(n_keys: int) -> np.ndarray:
    idx = np.arange(n_keys)
    return idx > 0


def collect_session_events(
    symbols: np.ndarray,
    keys: np.ndarray,
    matrices: Mapping[str, np.ndarray],
    *,
    session: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Detect shocks and pack recovery trajectories for one AM/PM session."""
    n_sym, n_key = matrices["depth"].shape
    stats: Dict[str, object] = {
        "session": session,
        "n_symbol_minutes": int(n_sym * n_key),
        "n_symbols": int(n_sym),
    }
    if n_sym == 0:
        return pd.DataFrame(), stats

    valid_t0 = finite_book_valid(matrices)
    bid = matrices["bid_depth_5"]
    ask = matrices["ask_depth_5"]
    depth = matrices["depth"]
    spread = matrices["spread"]
    obi = matrices["obi"]
    buy = matrices["active_buy_amount"]
    sell = matrices["active_sell_amount"]

    bid_pre = shift_session(bid, 1)
    ask_pre = shift_session(ask, 1)
    depth_pre = shift_session(depth, 1)
    spread_pre = shift_session(spread, 1)
    obi_pre = shift_session(obi, 1)
    valid_pre_book = (
        np.isfinite(bid_pre)
        & np.isfinite(ask_pre)
        & np.isfinite(depth_pre)
        & (depth_pre > 0)
        & np.isfinite(spread_pre)
        & np.isfinite(obi_pre)
    )
    pre_ok = valid_t0 & valid_pre_book & _pre_valid(n_key)[None, :]

    trail_buy = trailing_median_2d(buy)
    trail_sell = trailing_median_2d(sell)

    shocks = {
        SHOCK_ACTIVE_BUY: active_buy_shock(buy, sell, valid=pre_ok, trail_buy=trail_buy),
        SHOCK_ACTIVE_SELL: active_sell_shock(buy, sell, valid=pre_ok, trail_sell=trail_sell),
        SHOCK_DEPTH: depth_depletion_shock(depth_pre, depth, valid=pre_ok),
        SHOCK_SPREAD: spread_widening_shock(spread_pre, spread, valid=pre_ok),
    }

    fwd = {h: {c: shift_session(matrices[c], -h) for c in _SESSION_COLS} for h in (1, 3, 5)}
    h_ok = _horizon_valid(n_key)

    frames: List[pd.DataFrame] = []
    for shock_type, mask in shocks.items():
        ii, jj = np.where(mask)
        stats[f"n_raw_{shock_type}"] = int(mask.sum())
        stats[f"n_raw_{shock_type}_zero_depth"] = int(
            np.sum(mask & (~np.isfinite(depth) | (depth <= 0)))
        )
        if ii.size == 0:
            continue
        rec = {
            "Symbol": symbols[ii],
            "mkey": keys[jj],
            "session": session,
            "shock_type": shock_type,
            "bid_pre": bid_pre[ii, jj],
            "bid_t0": bid[ii, jj],
            "ask_pre": ask_pre[ii, jj],
            "ask_t0": ask[ii, jj],
            "depth_pre": depth_pre[ii, jj],
            "depth_t0": depth[ii, jj],
            "spread_pre": spread_pre[ii, jj],
            "spread_t0": spread[ii, jj],
            "obi_pre": obi_pre[ii, jj],
            "obi_t0": obi[ii, jj],
            "active_buy_t0": buy[ii, jj],
            "active_sell_t0": sell[ii, jj],
        }
        for h in (1, 3, 5):
            rec[f"valid_{h}"] = h_ok[h][jj] & np.isfinite(fwd[h]["depth"][ii, jj])
            rec[f"bid_t{h}"] = fwd[h]["bid_depth_5"][ii, jj]
            rec[f"ask_t{h}"] = fwd[h]["ask_depth_5"][ii, jj]
            rec[f"depth_t{h}"] = fwd[h]["depth"][ii, jj]
            rec[f"spread_t{h}"] = fwd[h]["spread"][ii, jj]
            rec[f"obi_t{h}"] = fwd[h]["obi"][ii, jj]
            rec[f"valid_{h}"] = (
                rec[f"valid_{h}"]
                & np.isfinite(rec[f"bid_t{h}"])
                & np.isfinite(rec[f"ask_t{h}"])
                & np.isfinite(rec[f"spread_t{h}"])
                & np.isfinite(rec[f"obi_t{h}"])
            )
        frames.append(pd.DataFrame(rec))

    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not events.empty:
        events = _attach_recovery_metrics(events)
    return events, stats


def _attach_recovery_metrics(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    for h in (1, 3, 5):
        vh = out[f"valid_{h}"].to_numpy(dtype=bool)
        out[f"ask_recovery_{h}"] = np.where(
            vh,
            recovery_fraction(out["ask_pre"], out["ask_t0"], out[f"ask_t{h}"], denom_floor=DENOM_FLOOR_DEPTH),
            np.nan,
        )
        out[f"bid_recovery_{h}"] = np.where(
            vh,
            recovery_fraction(out["bid_pre"], out["bid_t0"], out[f"bid_t{h}"], denom_floor=DENOM_FLOOR_DEPTH),
            np.nan,
        )
        out[f"ask_efficiency_{h}"] = np.where(
            vh,
            replenishment_efficiency(
                out["ask_t0"], out[f"ask_t{h}"], out["active_buy_t0"], denom_floor=DENOM_FLOOR_FLOW
            ),
            np.nan,
        )
        out[f"bid_efficiency_{h}"] = np.where(
            vh,
            replenishment_efficiency(
                out["bid_t0"], out[f"bid_t{h}"], out["active_sell_t0"], denom_floor=DENOM_FLOOR_FLOW
            ),
            np.nan,
        )
        out[f"spread_recovery_{h}"] = np.where(
            vh,
            spread_recovery_fraction(
                out["spread_pre"], out["spread_t0"], out[f"spread_t{h}"], denom_floor=DENOM_FLOOR_SPREAD
            ),
            np.nan,
        )
        out[f"obi_restoration_{h}"] = np.where(
            vh,
            obi_restoration(out["obi_pre"], out["obi_t0"], out[f"obi_t{h}"], denom_floor=DENOM_FLOOR_OBI),
            np.nan,
        )
    out["spread_residual_5"] = np.where(
        out["valid_5"].to_numpy(dtype=bool),
        spread_residual_width(out["spread_pre"], out["spread_t5"], denom_floor=DENOM_FLOOR_SPREAD),
        np.nan,
    )
    out["obi_persistence_5"] = np.where(
        out["valid_5"].to_numpy(dtype=bool),
        obi_persistence(out["obi_pre"], out["obi_t0"], out["obi_t5"], denom_floor=DENOM_FLOOR_OBI),
        np.nan,
    )
    return out


def events_from_minutes(minute: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build the shared event table for one TradeDate (AM+PM, all symbols)."""
    enriched = enrich_minutes(minute)
    parts: List[pd.DataFrame] = []
    stats: Dict[str, object] = {}
    for session in ("AM", "PM"):
        symbols, keys, matrices = session_matrix(enriched, session, _SESSION_COLS)
        ev, st = collect_session_events(symbols, keys, matrices, session=session)
        parts.append(ev)
        for k, v in st.items():
            if k == "session":
                continue
            stats[f"{session}_{k}"] = v
    events = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
    if "TradeDate" in enriched.columns and not events.empty:
        events["TradeDate"] = pd.to_datetime(enriched["TradeDate"].iloc[0]).normalize()
    return events, stats


def _subset(events: pd.DataFrame, shock: str) -> pd.DataFrame:
    if events.empty:
        return events
    return events.loc[events["shock_type"] == shock]


def aggregate_daily(events: pd.DataFrame, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """TradeDate × Symbol daily exposures. Missing events → NA, never 0-filled."""
    names = list(FROZEN_CANDIDATE_NAMES)
    if events is None or events.empty:
        idx = pd.Index([] if symbols is None else list(symbols), name="Symbol")
        out = pd.DataFrame(index=idx, columns=names, dtype=float)
        return out.reset_index()

    buy = _subset(events, SHOCK_ACTIVE_BUY)
    sell = _subset(events, SHOCK_ACTIVE_SELL)
    depth = _subset(events, SHOCK_DEPTH)
    spread = _subset(events, SHOCK_SPREAD)

    if symbols is None:
        syms = pd.Index(events["Symbol"].astype(str).unique())
    else:
        syms = pd.Index(list(symbols))
    out = pd.DataFrame(index=syms, columns=names, dtype=float)
    out.index.name = "Symbol"

    def _col_median(frame: pd.DataFrame, col: str) -> pd.Series:
        if frame.empty or col not in frame.columns:
            return pd.Series(dtype=float)
        return frame.groupby("Symbol", sort=False)[col].median()

    def _col_wmean(frame: pd.DataFrame, col: str, wcol: str) -> pd.Series:
        if frame.empty or col not in frame.columns:
            return pd.Series(dtype=float)
        values = {}
        for sym, g in frame.groupby("Symbol", sort=False):
            values[str(sym)] = shock_size_weighted_mean(g[col], g[wcol])
        return pd.Series(values, dtype=float)

    for h in (1, 3, 5):
        out[f"ask_depth_recovery_{h}m"] = _col_median(buy, f"ask_recovery_{h}")
        out[f"bid_depth_recovery_{h}m"] = _col_median(sell, f"bid_recovery_{h}")
        out[f"ask_replenishment_efficiency_{h}m"] = _col_wmean(buy, f"ask_efficiency_{h}", "active_buy_t0")
        out[f"bid_replenishment_efficiency_{h}m"] = _col_wmean(sell, f"bid_efficiency_{h}", "active_sell_t0")
        out[f"spread_recovery_{h}m"] = _col_median(spread, f"spread_recovery_{h}")

    out["spread_residual_width_5m"] = _col_median(spread, "spread_residual_5")
    out["obi_restoration_buy_3m"] = _col_median(buy, "obi_restoration_3")
    out["obi_restoration_buy_5m"] = _col_median(buy, "obi_restoration_5")
    out["obi_restoration_sell_3m"] = _col_median(sell, "obi_restoration_3")
    out["obi_restoration_sell_5m"] = _col_median(sell, "obi_restoration_5")
    out["obi_shock_persistence_5m"] = _col_median(depth, "obi_persistence_5")

    bid3 = _col_median(sell, "bid_recovery_3")
    ask3 = _col_median(buy, "ask_recovery_3")
    bid5 = _col_median(sell, "bid_recovery_5")
    ask5 = _col_median(buy, "ask_recovery_5")
    out["depth_resilience_asymmetry_3m"] = bid3.reindex(out.index) - ask3.reindex(out.index)
    out["depth_resilience_asymmetry_5m"] = bid5.reindex(out.index) - ask5.reindex(out.index)
    spr_sell = _col_median(sell, "spread_recovery_5")
    spr_buy = _col_median(buy, "spread_recovery_5")
    out["spread_resilience_asymmetry_5m"] = spr_sell.reindex(out.index) - spr_buy.reindex(out.index)

    out = out.reindex(columns=names)
    if "TradeDate" in events.columns:
        out.insert(0, "TradeDate", pd.to_datetime(events["TradeDate"].iloc[0]).normalize())
    return out.reset_index()


def daily_from_minutes(minute: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    events, stats = events_from_minutes(minute)
    symbols = (
        enrich_minutes(minute)["Symbol"].astype(str).unique()
        if not minute.empty
        else []
    )
    daily = aggregate_daily(events, symbols=symbols)
    return daily, events, stats
