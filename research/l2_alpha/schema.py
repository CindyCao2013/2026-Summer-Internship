"""Locked ClickHouse SSL2 schema for Sprint 4.4 Phase 1."""

from __future__ import annotations

DATABASE = "cmds"

SSE_SSL2_TABLE = "SSE_AL_SSL2_EXG"
SZSE_SSL2_TABLE = "SZSE_AL_SSL2_EXG"

# (table, exchange_suffix, has_withdraw_stats)
SNAPSHOT_TABLES = (
    (SSE_SSL2_TABLE, ".SH", True),
    (SZSE_SSL2_TABLE, ".SZ", False),  # SZSE schema has no Bid/AskWithdraw*
)

IDENTITY_COLUMNS = ("Symbol", "ExchTime")

BOOK_ARRAY_COLUMNS = (
    "BidPrices",
    "BidVolumes",
    "BidNums",
    "AskPrices",
    "AskVolumes",
    "AskNums",
)

BOOK_SCALAR_COLUMNS = (
    "TotalBidVolume",
    "TotalAskVolume",
    "BidVWAP",
    "AskVWAP",
    "BidWithdrawNum",
    "BidWithdrawVolume",
    "BidWithdrawAmount",
    "AskWithdrawNum",
    "AskWithdrawVolume",
    "AskWithdrawAmount",
)

# ClickHouse arrays are 1-indexed; level 1 = top of book.
N_DEPTH_LEVELS = 10
DEFAULT_WOI_LAMBDA = 0.5

FACTOR_NAMES = (
    "l2_top_book_imbalance",
    "l2_depth_imbalance",
    "l2_weighted_oi",
    "l2_microprice_bias",
    "l2_relative_spread",
    "l2_cancel_pressure",
    "l2_liquidity_skew",
    "l2_liquidity_wall",
)

NARROW_COLUMNS = ("tradetime", "symbol", "factorname", "value")
