#!/usr/bin/env python
"""Sprint 6B Phase 1 golden tests — cancellation / order lifecycle.

A_synthetic_handcalc   Hand-computed SZSE grouped-scan rollup + ASOF fill on
                       a tiny synthetic frame (no ClickHouse).
B_synthetic_sse        Hand-computed SSE valid-D aggregation logic mirrored
                       in pandas on a tiny synthetic frame (no ClickHouse).
C_real_day_crosscheck  Real 2024-06-28 single-stock ClickHouse cross-check:
                       module SQL vs an independent pandas lifecycle oracle
                       for 600000.SH (SSE) and 000001.SZ (SZSE).

Run: python l2_factor_reproduction/tests/golden_cancel_lifecycle.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_cancel_lifecycle import (  # noqa: E402
    fill_rollup,
    szse_assemble,
    sse_daily_sql,
    szse_rollup_sql,
    szse_trade_totals_sql,
    szse_zero_price_side_sql,
    build_candidates,
    shock_20d,
)

TOL = 1e-9
results = []


def check(tag: str, got, want, tol: float = TOL) -> None:
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    results.append((tag, ok))
    print(f"  [{'OK ' if ok else 'FAIL'}] {tag}: got={got} want={want}")


# ---------------------------------------------------------------------------
# A. SZSE synthetic hand calculation
# ---------------------------------------------------------------------------

def test_szse_synthetic() -> None:
    print("[A] SZSE synthetic hand calculation")
    # order groups after the grouped scan (one row per order key)
    groups = pd.DataFrame([
        # Symbol, order_price, n_order_rows, cancel_qty, cancel_events, buy?
        ("000001", 10.0, 1, 300.0, 1, 1),   # limit buy cancel 300 @ 10
        ("000001", 10.5, 1, 200.0, 1, 0),   # limit sell cancel 200 @ 10.5
        ("000001", 0.0, 1, 500.0, 1, 1),    # market buy cancel 500 -> ASOF
        ("000001", 0.0, 0, 100.0, 1, 0),    # unmatched cancel (no order row)
    ], columns=["Symbol", "order_price", "n_order_rows", "cancel_qty",
                "cancel_events", "cancel_is_buy"])
    rollup = pd.DataFrame({
        "Symbol": ["000001"],
        # limit-value legs: buy 300*10=3000 ; sell 200*10.5=2100
        "buy_cancel_value_limit": [3000.0],
        "sell_cancel_value_limit": [2100.0],
        "buy_cancel_qty": [800.0],   # 300 + 500 (qty is price-independent)
        "sell_cancel_qty": [300.0],  # 200 + 100
        "buy_cancel_event_count": [2],
        "sell_cancel_event_count": [2],
        "buy_cancelled_unique_order_count": [2],
        "sell_cancelled_unique_order_count": [2],
        "zero_price_cancel_count": [2],
        "unmatched_cancel_order_count": [1],
        "total_cancel_events": [4],
        "matched_cancel_events": [3],
    })
    # ASOF side table: market buy cancel 500 filled at last trade 10.2;
    # (unmatched sell cancel has order_price=0 too but is excluded from the
    # side table by construction of zero_price side query? no — the side
    # query takes order_price=0 groups regardless of n_order_rows; the
    # unmatched one also gets a fill attempt. Model both.)
    fills = pd.DataFrame([
        ("000001", 1, 500.0, 10.2),   # filled: +5100 buy value
        ("000001", 0, 100.0, np.nan),  # fill failure: 0 value + failure
    ], columns=["Symbol", "cancel_is_buy", "cancel_qty", "fill_price"])
    totals = pd.DataFrame({
        "Symbol": ["000001"],
        "total_trade_value": [100000.0],
        "total_trade_qty": [10000.0],
        "total_trade_count": [50],
    })

    out = szse_assemble(rollup, fills, totals).iloc[0]
    check("A buy_cancel_value", out["buy_cancel_value"], 3000.0 + 5100.0)
    check("A sell_cancel_value", out["sell_cancel_value"], 2100.0)
    check("A join_coverage", out["join_coverage"], 0.75)
    check("A invalid_cancel_count", out["invalid_cancel_count"], 2)
    check("A market_order_price_fill_count",
          out["market_order_price_fill_count"], 1)
    check("A zero_price_cancel_count", out["zero_price_cancel_count"], 2)

    c = build_candidates(out.to_frame().T.assign(
        symbol="000001.SZ", TradeDate=pd.Timestamp("2024-06-28"))).iloc[0]
    # value pressure = (8100-2100)/(8100+2100)
    check("A cancel_value_pressure", c["cancel_value_pressure"],
          6000.0 / 10200.0)
    # count pressure = (2-2)/4 = 0
    check("A cancel_count_pressure", c["cancel_count_pressure"], 0.0)
    # value intensity = 10200/100000
    check("A cancel_value_intensity", c["cancel_value_intensity"], 0.102)
    # qty intensity = 1100/10000
    check("A cancel_qty_intensity", c["cancel_qty_intensity"], 0.11)
    # relative size = (10200/4)/(100000/50) = 2550/2000
    check("A relative_cancel_order_size",
          c["relative_cancel_order_size"], 1.275)


# ---------------------------------------------------------------------------
# B. SSE synthetic hand calculation (pandas oracle mirroring the SQL)
# ---------------------------------------------------------------------------

def sse_pandas_oracle(rows: pd.DataFrame) -> dict:
    """Independent oracle for sse_daily_sql: valid-D filter + aggregation."""
    d = rows[rows["Type"] == "D"]
    valid = d[
        (d["Price"] > 0) & (d["Volume"] > 0)
        & (d["BSFlag"].isin(["B", "S"]))
        & (d[["BidOrderNo", "AskOrderNo"]].notna().any(axis=1))
    ]
    t = rows[rows["Type"] == "T"]
    return {
        "buy_cancel_value": float(
            (valid.loc[valid.BSFlag == "B", "Price"]
             * valid.loc[valid.BSFlag == "B", "Volume"]).sum()),
        "sell_cancel_value": float(
            (valid.loc[valid.BSFlag == "S", "Price"]
             * valid.loc[valid.BSFlag == "S", "Volume"]).sum()),
        "buy_cancel_event_count": int((valid.BSFlag == "B").sum()),
        "sell_cancel_event_count": int((valid.BSFlag == "S").sum()),
        "invalid_cancel_count": int(len(d) - len(valid)),
        "total_trade_value": float((t["Price"] * t["Volume"]).sum()),
        "total_trade_count": int(len(t)),
    }


def test_sse_synthetic() -> None:
    print("[B] SSE synthetic hand calculation")
    rows = pd.DataFrame([
        # Type, BSFlag, BidOrderNo, AskOrderNo, Price, Volume
        ("D", "B", 101, None, 10.0, 300),    # valid buy cancel -> 3000
        ("D", "S", None, 202, 10.5, 200),    # valid sell cancel -> 2100
        ("D", "B", 103, None, 0.0, 100),     # invalid: price 0
        ("D", "N", 104, None, 10.0, 100),    # invalid: bad BSFlag
        ("T", "B", 101, 202, 10.1, 1000),    # trade -> 10100
        ("T", "B", 105, 203, 10.2, 500),     # trade -> 5100
    ], columns=["Type", "BSFlag", "BidOrderNo", "AskOrderNo",
                "Price", "Volume"])
    got = sse_pandas_oracle(rows)
    check("B buy_cancel_value", got["buy_cancel_value"], 3000.0)
    check("B sell_cancel_value", got["sell_cancel_value"], 2100.0)
    check("B buy_cancel_event_count", got["buy_cancel_event_count"], 1)
    check("B invalid_cancel_count", got["invalid_cancel_count"], 2)
    check("B total_trade_value", got["total_trade_value"], 15200.0)
    check("B total_trade_count", got["total_trade_count"], 2)


# ---------------------------------------------------------------------------
# C. real-day ClickHouse cross-check vs pandas lifecycle oracle
# ---------------------------------------------------------------------------

def test_real_day_crosscheck() -> None:
    print("[C] real-day CH cross-check 2024-06-28 "
          "(600000.SH / 000001.SZ)")
    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client

    client = connect_hf_client()
    day = pd.Timestamp("2024-06-28")

    # --- SSE 600000: module SQL vs pandas oracle on raw rows
    sql = sse_daily_sql(day, symbols=["600000"])
    sql_result = client.query_df(sql).iloc[0]
    raw = client.query_df(
        "SELECT Type, BSFlag, BidOrderNo, AskOrderNo,"
        " toFloat64(Price) AS Price, toFloat64(Volume) AS Volume"
        " FROM cmds.SSE_AL_TICK_EXG"
        " WHERE ExchTime >= toDateTime64('2024-06-28 00:00:00',6,"
        " 'Asia/Shanghai') AND ExchTime < toDateTime64('2024-06-28"
        " 00:00:00',6,'Asia/Shanghai') + toIntervalDay(1)"
        " AND Symbol = '600000'"
        " AND (((toHour(ExchTime)*60+toMinute(ExchTime)) >= 570"
        " AND (toHour(ExchTime)*60+toMinute(ExchTime)) < 690)"
        " OR ((toHour(ExchTime)*60+toMinute(ExchTime)) >= 780"
        " AND (toHour(ExchTime)*60+toMinute(ExchTime)) < 900))"
        " AND Type IN ('D','T')"
    )
    oracle = sse_pandas_oracle(raw)
    for key in ["buy_cancel_value", "sell_cancel_value",
                "buy_cancel_event_count", "sell_cancel_event_count",
                "invalid_cancel_count", "total_trade_value",
                "total_trade_count"]:
        want = oracle[key]
        got = float(sql_result[key])
        tol = max(abs(want) * 1e-9, 1e-6)
        check(f"C SSE {key}", got, want, tol)

    # --- SZSE 000001: module full path vs independent lifecycle oracle
    rollup = client.query_df(szse_rollup_sql(day, symbols=["000001"]))
    fills = client.query_df(szse_zero_price_side_sql(day, symbols=["000001"]))
    totals = client.query_df(szse_trade_totals_sql(day, symbols=["000001"]))
    got_row = szse_assemble(rollup, fills, totals).iloc[0]

    raw = client.query_df(
        "SELECT Category, SeqNo, BidOrderNo, AskOrderNo,"
        " toFloat64(Price) AS Price, toFloat64(Volume) AS Volume"
        " FROM cmds.SZSE_AL_TICK_EXG"
        " WHERE ExchTime >= toDateTime64('2024-06-28 00:00:00',6,"
        " 'Asia/Shanghai') AND ExchTime < toDateTime64('2024-06-28"
        " 00:00:00',6,'Asia/Shanghai') + toIntervalDay(1)"
        " AND Symbol = '000001' AND Type = '011'"
        " AND (((toHour(ExchTime)*60+toMinute(ExchTime)) >= 570"
        " AND (toHour(ExchTime)*60+toMinute(ExchTime)) < 690)"
        " OR ((toHour(ExchTime)*60+toMinute(ExchTime)) >= 780"
        " AND (toHour(ExchTime)*60+toMinute(ExchTime)) < 900))"
        " AND Category IN ('1','2','4','F')"
    )
    orders = raw[raw.Category.isin(["1", "2"])]
    cancels = raw[raw.Category == "4"]
    trades = raw[raw.Category == "F"]
    order_px = orders.groupby("SeqNo")["Price"].max()
    buy_c = cancels[cancels.BidOrderNo > 0]
    sell_c = cancels[cancels.AskOrderNo > 0]
    buy_val = sum(
        r.Volume * order_px.get(r.BidOrderNo, 0.0)
        for r in buy_c.itertuples()
        if not pd.isna(r.BidOrderNo)
    )
    sell_val = sum(
        r.Volume * order_px.get(r.AskOrderNo, 0.0)
        for r in sell_c.itertuples()
        if not pd.isna(r.AskOrderNo)
    )
    # oracle treats zero-price (market) orders as 0-value; the module adds
    # ASOF-filled value on top, so module value >= oracle value.
    check("C SZSE buy_cancel_value >= oracle",
          float(got_row["buy_cancel_value"] >= buy_val - 1e-6), True)
    check("C SZSE sell_cancel_value >= oracle",
          float(got_row["sell_cancel_value"] >= sell_val - 1e-6), True)
    check("C SZSE buy_cancel_event_count",
          int(got_row["buy_cancel_event_count"]), int(len(buy_c)))
    check("C SZSE sell_cancel_event_count",
          int(got_row["sell_cancel_event_count"]), int(len(sell_c)))
    trade_value = float((trades["Price"] * trades["Volume"]).sum())
    check("C SZSE total_trade_value", float(got_row["total_trade_value"]),
          trade_value, max(trade_value * 1e-9, 1e-6))
    check("C SZSE join_coverage == 1",
          float(got_row["join_coverage"]), 1.0)


def test_shock_semantics() -> None:
    print("[D] shock_20d excludes current day")
    x = pd.Series(np.arange(1.0, 31.0))
    got = shock_20d(x)
    # at t=25: mean of x[4..23] (shifted window of 20 ending at t-1)
    idx = 25
    hist = x.iloc[idx - 20:idx]
    want = (x.iloc[idx] - hist.mean()) / hist.std(ddof=0)
    check("D shock_20d value", float(got.iloc[idx]), float(want))
    check("D shock_20d warm-up NaN", bool(got.iloc[:20].isna().all()), True)


def main() -> int:
    test_szse_synthetic()
    test_sse_synthetic()
    test_shock_semantics()
    test_real_day_crosscheck()
    failed = [t for t, ok in results if not ok]
    print(f"[golden {'PASS' if not failed else 'FAIL'}] "
          f"{len(results) - len(failed)}/{len(results)} checks")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
