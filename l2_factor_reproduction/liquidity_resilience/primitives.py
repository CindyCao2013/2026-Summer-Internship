"""Minute-level book primitives. Reuse L5 depth and lid spread/OBI definitions."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.liquidity_resilience.session import AM_KEYS, PM_KEYS, session_of_mkey
from l2_factor_reproduction.python import liquidity_impact_daily as lid

PRIMITIVE_COLUMNS = (
    "Symbol",
    "TradeDate",
    "mkey",
    "session",
    "exchange",
    "has_book",
    "bid1",
    "ask1",
    "bid_depth_5",
    "ask_depth_5",
    "depth",
    "spread",
    "obi",
    "active_buy_amount",
    "active_sell_amount",
)


def enrich_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add spread / OBI / depth / session using liquidity-impact conventions."""
    out = frame.copy()
    if "Symbol" not in out.columns:
        suffix = out["exchange"] if "exchange" in out.columns else ""
        out["Symbol"] = out["symbol_raw"].astype(str) + suffix.astype(str)
    out["mkey"] = out["mkey"].astype(np.int32)
    has_book = out["has_book"]
    if has_book.dtype != bool:
        has_book = pd.to_numeric(has_book, errors="coerce").fillna(0) > 0
    else:
        has_book = has_book.fillna(False)
    bid1 = pd.to_numeric(out["bid1"], errors="coerce")
    ask1 = pd.to_numeric(out["ask1"], errors="coerce")
    mid = (bid1 + ask1) / 2.0
    spread = np.where(
        has_book.to_numpy() & (mid.to_numpy() > 0) & (ask1.to_numpy() >= bid1.to_numpy()),
        (ask1 - bid1) / mid,
        np.nan,
    )
    bid5 = pd.to_numeric(out["bid_depth_5"], errors="coerce")
    ask5 = pd.to_numeric(out["ask_depth_5"], errors="coerce")
    depth = bid5 + ask5
    obi = np.where(depth.to_numpy() > 0, (bid5 - ask5) / depth, np.nan)
    out["has_book"] = has_book.to_numpy()
    out["spread"] = spread
    out["depth"] = depth.to_numpy()
    out["obi"] = obi
    out["session"] = session_of_mkey(out["mkey"])
    out["active_buy_amount"] = pd.to_numeric(out["active_buy_amount"], errors="coerce")
    out["active_sell_amount"] = pd.to_numeric(out["active_sell_amount"], errors="coerce")
    keep = [c for c in PRIMITIVE_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in keep]
    return out[keep + extra]


def session_matrix(
    minute: pd.DataFrame,
    session: str,
    columns: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Reindex one session to the canonical mkey grid. Returns (symbols, keys, col->2d).

    AM and PM are built separately so shifts never cross lunch.
    """
    keys = AM_KEYS if session == "AM" else PM_KEYS
    cols = list(columns)
    block = minute.loc[minute["session"] == session, ["Symbol", "mkey", *cols]].copy()
    symbols = (
        pd.Index(block["Symbol"].astype(str).unique()).sort_values()
        if not block.empty
        else pd.Index([], dtype=object)
    )
    n_sym = int(len(symbols))
    n_key = int(len(keys))
    matrices: Dict[str, np.ndarray] = {
        c: np.full((n_sym, n_key), np.nan, dtype=float) for c in cols
    }
    if n_sym == 0 or block.empty:
        return symbols.to_numpy(), keys, matrices
    key_pos = {int(k): i for i, k in enumerate(keys)}
    sym_pos = {str(s): i for i, s in enumerate(symbols)}
    mkeys = block["mkey"].astype(int).to_numpy()
    syms = block["Symbol"].astype(str).to_numpy()
    rows = np.fromiter((sym_pos[s] for s in syms), dtype=np.int32, count=len(syms))
    cols_i = np.fromiter((key_pos.get(int(k), -1) for k in mkeys), dtype=np.int32, count=len(mkeys))
    valid = cols_i >= 0
    rows = rows[valid]
    cols_i = cols_i[valid]
    for c in cols:
        vals = pd.to_numeric(block[c], errors="coerce").to_numpy(dtype=float)[valid]
        matrices[c][rows, cols_i] = vals
    return symbols.to_numpy(), keys, matrices


def shift_session(arr: np.ndarray, steps: int) -> np.ndarray:
    """Shift along the minute axis. Positive steps look backward (pre)."""
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError("shift_session expects (n_symbols, n_minutes)")
    out = np.full_like(x, np.nan)
    if steps == 0:
        return x.copy()
    if steps > 0:
        out[:, steps:] = x[:, :-steps]
    else:
        h = -steps
        out[:, :-h] = x[:, h:]
    return out


def finite_book_valid(matrices: Dict[str, np.ndarray]) -> np.ndarray:
    """Minute is usable iff L5 book, spread, and OBI are finite."""
    depth = matrices["depth"]
    spread = matrices["spread"]
    bid = matrices["bid_depth_5"]
    ask = matrices["ask_depth_5"]
    return (
        np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(spread)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & np.isfinite(matrices["obi"])
    )


def attach_exchange_suffix(frame: pd.DataFrame, exchange: str) -> pd.DataFrame:
    suffix = lid.EXCHANGES[exchange]["suffix"]
    out = frame.copy()
    out["exchange"] = suffix
    out["Symbol"] = out["symbol_raw"].astype(str) + suffix
    return out
