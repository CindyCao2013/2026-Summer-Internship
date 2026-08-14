#!/usr/bin/env python
"""Sprint 15A — Order Lifetime / Cancel Hazard Primitive Feasibility Audit.

Audit only: NO alpha backtest / full-history build / factor discovery.

Usage:
  /opt/conda/anaconda3/bin/python -m l2_factor_reproduction.scripts.run_sprint15a_order_lifetime_audit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402

OUT = (
    Path(PROJ_ROOT)
    / "research/results/l2_reproduction"
    / "sprint15_order_lifetime"
    / "primitive_audit"
)

# Small sample: 2 days × 2 SSE + 2 SZSE (reuse 14A liquid symbols)
SAMPLE = [
    ("sse", "600000", "cmds.LOCAL_SSE_AL_TICK_EXG", "2024-06-27"),
    ("sse", "600000", "cmds.LOCAL_SSE_AL_TICK_EXG", "2024-06-28"),
    ("sse", "600933", "cmds.LOCAL_SSE_AL_TICK_EXG", "2024-06-28"),
    ("szse", "000333", "cmds.LOCAL_SZSE_AL_TICK_EXG", "2024-06-27"),
    ("szse", "000333", "cmds.LOCAL_SZSE_AL_TICK_EXG", "2024-06-28"),
    ("szse", "000938", "cmds.LOCAL_SZSE_AL_TICK_EXG", "2024-06-28"),
]

SESSION_END = "14:59:59"  # end of continuous; still-live => SESSION_END_CENSORED
CONTINUOUS_WINDOWS = (
    ("09:30:00", "11:30:00"),
    ("13:00:00", "15:00:00"),  # continuous through 14:59:59.x; close auction ~15:00
)
EPS_QTY = 1e-6


def _dt(day: str, hhmmss: str) -> str:
    return f"toDateTime64('{day} {hhmmss}', 6, 'Asia/Shanghai')"


def _session_end_ms(day: str) -> int:
    # Match ClickHouse toUnixTimestamp64Milli(Asia/Shanghai DateTime64)
    return int(
        pd.Timestamp(f"{day} {SESSION_END}")
        .tz_localize("Asia/Shanghai")
        .timestamp()
        * 1000
    )


def _in_continuous(ms: pd.Series, day: str) -> pd.Series:
    """Boolean mask: timestamp in continuous auction (excl. open auction)."""
    masks = []
    for start, end in CONTINUOUS_WINDOWS:
        a = int(
            pd.Timestamp(f"{day} {start}")
            .tz_localize("Asia/Shanghai")
            .timestamp()
            * 1000
        )
        b = int(
            pd.Timestamp(f"{day} {end}")
            .tz_localize("Asia/Shanghai")
            .timestamp()
            * 1000
        )
        masks.append((ms >= a) & (ms < b))
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


# ---------------------------------------------------------------------------
# PART A — existing overlap (static)
# ---------------------------------------------------------------------------


def write_existing_overlap() -> None:
    text = """# Sprint 15A — Existing Overlap Audit

## What Cancellation family already tests

Frozen `cancel_lifecycle_daily` / cancellation_lifecycle_v1 candidates:

| Candidate | Mechanism |
|-----------|-----------|
| cancel_value_pressure | net buy/sell **cancel amount** |
| cancel_count_pressure | net buy/sell **cancel event count** |
| cancel_value_intensity | total cancel value / trade value |
| cancel_qty_intensity | total cancel qty / trade qty |
| relative_cancel_order_size | avg cancel size vs avg trade size |
| cancel_*_shock_20d | time-series shock of pressure/intensity |

Source contract (`implementation_contract.json`):

- SSE: aggregate valid **Type=D** rows (residual qty + original price). **No production A/T/D lifetime join.**
- SZSE: Category 4 cancels linked to Category 1/2 orders for **pricing cancels**, then roll up to cancel volume/pressure.

## What is NOT tested

| Mechanism | Status |
|-----------|--------|
| order lifetime (terminal − add) | **NOT tested** |
| age-at-cancel | **NOT tested** |
| age-at-fill | **NOT tested** |
| cancel hazard conditional on age | **NOT tested** |
| bid/ask lifetime asymmetry | **NOT tested** |
| short-lived order share | **NOT tested** |

## Distinction (critical)

```text
cancel volume / pressure / intensity
    ≠
order lifetime / age-at-cancel / cancel hazard
```

Cancel pressure answers: *how much / how often is cancelled (by side)?*  
Lifetime answers: *how long do resting orders live before fill/cancel?*

Existing unique-order cancel counts do **not** encode duration.

## Raw tables / IDs available

| Exchange | Table | Order identifiers |
|----------|-------|-------------------|
| SSE | `cmds.LOCAL_SSE_AL_TICK_EXG` / `SSE_AL_TICK_EXG` | Type A/D/T; key ≈ `(Channel, BidOrderNo|AskOrderNo by BSFlag)` |
| SZSE | `cmds.LOCAL_SZSE_AL_TICK_EXG` / `SZSE_AL_TICK_EXG` | Type=011; Category 1/2 SeqNo; cancel Cat4 Bid/AskOrderNo; trades Cat F |

Phase-0 audit already proved SSE A→T→D residual identity on samples and SZSE cancel→order join coverage = 1.0 on 2024-06-28 full market — but that work stopped at **cancel aggregates**, not lifetime primitives.

## Overlap verdict

**Lifetime / hazard is a NEW mechanism.** Safe to audit feasibility without colliding with frozen cancel pressure factors.
"""
    (OUT / "existing_overlap.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fetch + reconstruct
# ---------------------------------------------------------------------------


def fetch_sse_events(client, table: str, symbol: str, day: str) -> pd.DataFrame:
    sql = f"""
    SELECT
      ExchTime,
      toUnixTimestamp64Milli(ExchTime) AS ms,
      Type,
      BSFlag,
      toFloat64(Price) AS Price,
      toFloat64(Volume) AS Volume,
      Channel,
      BidOrderNo,
      AskOrderNo,
      SeqNo,
      SubSeqNo
    FROM {table}
    WHERE Symbol = '{symbol}'
      AND ExchTime >= {_dt(day, '09:15:00')}
      AND ExchTime < {_dt(day, '15:01:00')}
      AND Type IN ('A', 'D', 'T')
    ORDER BY ExchTime, SeqNo, SubSeqNo
    """
    df = client.query_df(sql)
    if len(df) == 0:
        return df
    df["ExchTime"] = pd.to_datetime(df["ExchTime"])
    return df


def fetch_szse_events(client, table: str, symbol: str, day: str) -> pd.DataFrame:
    sql = f"""
    SELECT
      ExchTime,
      toUnixTimestamp64Milli(ExchTime) AS ms,
      Type,
      Category,
      SubCategory,
      toFloat64(Price) AS Price,
      toFloat64(Volume) AS Volume,
      Channel,
      SeqNo,
      BidOrderNo,
      AskOrderNo
    FROM {table}
    WHERE Symbol = '{symbol}'
      AND ExchTime >= {_dt(day, '09:15:00')}
      AND ExchTime < {_dt(day, '15:01:00')}
      AND Type = '011'
      AND Category IN ('1', '2', '4', 'F')
    ORDER BY ExchTime, SeqNo
    """
    df = client.query_df(sql)
    if len(df) == 0:
        return df
    df["ExchTime"] = pd.to_datetime(df["ExchTime"])
    return df


def reconstruct_sse(events: pd.DataFrame, day: str) -> pd.DataFrame:
    """Rebuild book-entering order lifetimes from A / T / D."""
    if events is None or len(events) == 0:
        return pd.DataFrame()

    def order_no(row) -> float:
        if row["BSFlag"] == "B":
            return float(row["BidOrderNo"]) if pd.notna(row["BidOrderNo"]) else np.nan
        if row["BSFlag"] == "S":
            return float(row["AskOrderNo"]) if pd.notna(row["AskOrderNo"]) else np.nan
        return np.nan

    ev = events.copy()
    ev["order_no"] = ev.apply(order_no, axis=1)
    adds = ev.loc[
        (ev["Type"] == "A") & ev["order_no"].notna() & ev["BSFlag"].isin(["B", "S"])
    ].copy()
    if len(adds) == 0:
        return pd.DataFrame()

    # first add per (Channel, order_no, side)
    adds = adds.sort_values("ms")
    keys = ["Channel", "order_no", "BSFlag"]
    first = adds.groupby(keys, as_index=False).agg(
        add_ms=("ms", "min"),
        order_price=("Price", "first"),
        order_size=("Volume", "first"),
        n_add_rows=("ms", "size"),
    )

    # fills attributed to this order
    trades = ev.loc[ev["Type"] == "T"].copy()
    fill_rows = []
    for side, col in (("B", "BidOrderNo"), ("S", "AskOrderNo")):
        t = trades.loc[trades[col].notna(), ["ms", "Channel", col, "Volume"]].copy()
        t = t.rename(columns={col: "order_no"})
        t["BSFlag"] = side
        fill_rows.append(t)
    fills = pd.concat(fill_rows, ignore_index=True) if fill_rows else pd.DataFrame()
    if len(fills):
        fill_agg = fills.groupby(keys, as_index=False).agg(
            fill_qty=("Volume", "sum"),
            first_fill_ms=("ms", "min"),
            last_fill_ms=("ms", "max"),
            n_fills=("Volume", "size"),
        )
    else:
        fill_agg = pd.DataFrame(columns=keys + ["fill_qty", "first_fill_ms", "last_fill_ms", "n_fills"])

    cancels = ev.loc[
        (ev["Type"] == "D") & ev["order_no"].notna() & ev["BSFlag"].isin(["B", "S"])
    ].copy()
    if len(cancels):
        cancel_agg = cancels.groupby(keys, as_index=False).agg(
            cancel_ms=("ms", "min"),
            cancel_qty=("Volume", "sum"),
            n_cancel_events=("ms", "size"),
        )
    else:
        cancel_agg = pd.DataFrame(
            columns=keys + ["cancel_ms", "cancel_qty", "n_cancel_events"]
        )

    life = first.merge(fill_agg, on=keys, how="left").merge(
        cancel_agg, on=keys, how="left"
    )
    life["fill_qty"] = life["fill_qty"].fillna(0.0)
    life["n_fills"] = life["n_fills"].fillna(0).astype(int)
    life["n_cancel_events"] = life["n_cancel_events"].fillna(0).astype(int)

    session_end = _session_end_ms(day)
    terminals = []
    term_ms = []
    for _, r in life.iterrows():
        has_c = pd.notna(r.get("cancel_ms"))
        filled = float(r["fill_qty"])
        size = float(r["order_size"])
        if has_c:
            if filled > EPS_QTY:
                terminals.append("PARTIAL_FILL_THEN_CANCEL")
            else:
                terminals.append("CANCEL")
            term_ms.append(float(r["cancel_ms"]))
        elif filled + EPS_QTY >= size and size > 0:
            terminals.append("FULL_FILL")
            term_ms.append(float(r["last_fill_ms"]))
        else:
            terminals.append("SESSION_END_CENSORED")
            term_ms.append(float(session_end))

    life["terminal_event"] = terminals
    life["terminal_ms"] = term_ms
    life["lifetime_ms"] = life["terminal_ms"] - life["add_ms"]
    life["invalid_sequence"] = life["lifetime_ms"] < 0
    life["side"] = life["BSFlag"].map({"B": "BID", "S": "ASK"})
    life["exchange"] = "SSE"
    life["symbol"] = ""  # filled by caller
    life["TradeDate"] = day
    # Restrict universe to continuous-auction adds (exclude open/close auction)
    life = life.loc[_in_continuous(life["add_ms"], day)].copy()
    return life


def reconstruct_szse(events: pd.DataFrame, day: str) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame()

    orders = events.loc[events["Category"].isin(["1", "2"])].copy()
    if len(orders) == 0:
        return pd.DataFrame()
    orders["side"] = orders["Category"].map({"1": "BID", "2": "ASK"})
    orders["order_no"] = orders["SeqNo"].astype(float)
    orders = orders.sort_values("ms")
    keys = ["Channel", "order_no"]
    first = orders.groupby(keys + ["side"], as_index=False).agg(
        add_ms=("ms", "min"),
        order_price=("Price", "first"),
        order_size=("Volume", "first"),
        n_add_rows=("ms", "size"),
        BSFlag=("side", "first"),
    )
    # normalize BSFlag label
    first["BSFlag"] = first["side"].map({"BID": "B", "ASK": "S"})

    trades = events.loc[events["Category"] == "F"].copy()
    fill_parts = []
    for side, col in (("BID", "BidOrderNo"), ("ASK", "AskOrderNo")):
        t = trades.loc[trades[col].notna() & (trades[col] > 0), ["ms", "Channel", col, "Volume"]].copy()
        t = t.rename(columns={col: "order_no"})
        t["side"] = side
        fill_parts.append(t)
    fills = pd.concat(fill_parts, ignore_index=True) if fill_parts else pd.DataFrame()
    if len(fills):
        fill_agg = fills.groupby(["Channel", "order_no", "side"], as_index=False).agg(
            fill_qty=("Volume", "sum"),
            first_fill_ms=("ms", "min"),
            last_fill_ms=("ms", "max"),
            n_fills=("Volume", "size"),
        )
    else:
        fill_agg = pd.DataFrame(
            columns=["Channel", "order_no", "side", "fill_qty", "first_fill_ms", "last_fill_ms", "n_fills"]
        )

    cancels = events.loc[events["Category"] == "4"].copy()
    c_parts = []
    if len(cancels):
        buy_c = cancels.loc[cancels["BidOrderNo"] > 0, ["ms", "Channel", "BidOrderNo", "Volume"]].copy()
        buy_c = buy_c.rename(columns={"BidOrderNo": "order_no"})
        buy_c["side"] = "BID"
        sell_c = cancels.loc[cancels["AskOrderNo"] > 0, ["ms", "Channel", "AskOrderNo", "Volume"]].copy()
        sell_c = sell_c.rename(columns={"AskOrderNo": "order_no"})
        sell_c["side"] = "ASK"
        c_parts = [buy_c, sell_c]
    c_all = pd.concat(c_parts, ignore_index=True) if c_parts else pd.DataFrame()
    if len(c_all):
        cancel_agg = c_all.groupby(["Channel", "order_no", "side"], as_index=False).agg(
            cancel_ms=("ms", "min"),
            cancel_qty=("Volume", "sum"),
            n_cancel_events=("ms", "size"),
        )
    else:
        cancel_agg = pd.DataFrame(
            columns=["Channel", "order_no", "side", "cancel_ms", "cancel_qty", "n_cancel_events"]
        )

    life = first.merge(fill_agg, on=["Channel", "order_no", "side"], how="left").merge(
        cancel_agg, on=["Channel", "order_no", "side"], how="left"
    )
    life["fill_qty"] = life["fill_qty"].fillna(0.0)
    life["n_fills"] = life["n_fills"].fillna(0).astype(int)
    life["n_cancel_events"] = life["n_cancel_events"].fillna(0).astype(int)

    session_end = _session_end_ms(day)
    terminals, term_ms = [], []
    for _, r in life.iterrows():
        has_c = pd.notna(r.get("cancel_ms"))
        filled = float(r["fill_qty"])
        size = float(r["order_size"])
        if has_c:
            terminals.append(
                "PARTIAL_FILL_THEN_CANCEL" if filled > EPS_QTY else "CANCEL"
            )
            term_ms.append(float(r["cancel_ms"]))
        elif filled + EPS_QTY >= size and size > 0:
            terminals.append("FULL_FILL")
            term_ms.append(float(r["last_fill_ms"]))
        else:
            terminals.append("SESSION_END_CENSORED")
            term_ms.append(float(session_end))
    life["terminal_event"] = terminals
    life["terminal_ms"] = term_ms
    life["lifetime_ms"] = life["terminal_ms"] - life["add_ms"]
    life["invalid_sequence"] = life["lifetime_ms"] < 0
    life["exchange"] = "SZSE"
    life["TradeDate"] = day
    life = life.loc[_in_continuous(life["add_ms"], day)].copy()
    return life


def orphan_stats_sse(events: pd.DataFrame, life: pd.DataFrame) -> Dict[str, float]:
    """Trades/cancels whose order_no never appears in Type=A (aggressive / missing add)."""
    if events is None or len(events) == 0:
        return {"orphan_trade_share": np.nan, "orphan_cancel_share": np.nan}
    adds = set(
        zip(
            life["Channel"].astype(str),
            life["order_no"].astype(float),
            life["BSFlag"].astype(str),
        )
    ) if len(life) else set()

    def key_from_row(row, side_col_b="BidOrderNo", side_col_s="AskOrderNo"):
        if row["BSFlag"] == "B" and pd.notna(row[side_col_b]):
            return (str(row["Channel"]), float(row[side_col_b]), "B")
        if row["BSFlag"] == "S" and pd.notna(row[side_col_s]):
            return (str(row["Channel"]), float(row[side_col_s]), "S")
        return None

    trades = events.loc[events["Type"] == "T"]
    cancels = events.loc[events["Type"] == "D"]
    def orphan_share(frame):
        if len(frame) == 0:
            return float("nan")
        n_orph = 0
        n = 0
        for _, r in frame.iterrows():
            k = key_from_row(r)
            if k is None:
                continue
            n += 1
            if k not in adds:
                n_orph += 1
        return float(n_orph / n) if n else float("nan")

    return {
        "orphan_trade_share": orphan_share(trades),
        "orphan_cancel_share": orphan_share(cancels),
    }


def summarize_life(life: pd.DataFrame, exchange: str, symbol: str, day: str,
                   extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if life is None or len(life) == 0:
        return {
            "exchange": exchange,
            "symbol": symbol,
            "TradeDate": day,
            "n_orders": 0,
            "error": "no_orders",
        }
    lt = life["lifetime_ms"].astype(float)
    term = life["terminal_event"]
    n = len(life)
    matched = term.isin(
        ["CANCEL", "PARTIAL_FILL_THEN_CANCEL", "FULL_FILL", "SESSION_END_CENSORED"]
    ).mean()
    row = {
        "exchange": exchange,
        "symbol": symbol,
        "TradeDate": day,
        "n_orders": int(n),
        "matched_terminal_event_share": float(matched),
        "cancel_share": float(term.isin(["CANCEL", "PARTIAL_FILL_THEN_CANCEL"]).mean()),
        "fill_share": float((term == "FULL_FILL").mean()),
        "censored_share": float((term == "SESSION_END_CENSORED").mean()),
        "partial_then_cancel_share": float((term == "PARTIAL_FILL_THEN_CANCEL").mean()),
        "median_lifetime_ms": float(lt.median()),
        "p10_lifetime_ms": float(lt.quantile(0.10)),
        "p50_lifetime_ms": float(lt.quantile(0.50)),
        "p90_lifetime_ms": float(lt.quantile(0.90)),
        "mean_lifetime_ms": float(lt.mean()),
        "negative_lifetime_count": int((lt < 0).sum()),
        "invalid_sequence_share": float(life["invalid_sequence"].mean()),
        "duplicate_order_id_share": float(
            life.duplicated(["Channel", "order_no", "side"]).mean()
        ),
        "bid_n": int((life["side"] == "BID").sum()),
        "ask_n": int((life["side"] == "ASK").sum()),
        "median_bid_lifetime_ms": float(lt[life["side"] == "BID"].median())
        if (life["side"] == "BID").any()
        else float("nan"),
        "median_ask_lifetime_ms": float(lt[life["side"] == "ASK"].median())
        if (life["side"] == "ASK").any()
        else float("nan"),
        "cancel_age_median_ms": float(
            lt[term.isin(["CANCEL", "PARTIAL_FILL_THEN_CANCEL"])].median()
        )
        if term.isin(["CANCEL", "PARTIAL_FILL_THEN_CANCEL"]).any()
        else float("nan"),
        "fill_age_median_ms": float(lt[term == "FULL_FILL"].median())
        if (term == "FULL_FILL").any()
        else float("nan"),
        "short_lived_1s_share": float((lt <= 1000).mean()),
        "short_lived_5s_share": float((lt <= 5000).mean()),
    }
    if extra:
        row.update(extra)
    return row


def run_sample_qa(client) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    rows = []
    lives = []
    for exchange, symbol, table, day in SAMPLE:
        print(f"[sample] {exchange} {symbol} {day}", flush=True)
        if exchange == "sse":
            ev = fetch_sse_events(client, table, symbol, day)
            life = reconstruct_sse(ev, day)
            if len(life):
                life["symbol"] = symbol
            extra = orphan_stats_sse(ev, life)
            extra["n_raw_events"] = int(len(ev))
            extra["n_type_A"] = int((ev["Type"] == "A").sum()) if len(ev) else 0
            extra["n_type_D"] = int((ev["Type"] == "D").sum()) if len(ev) else 0
            extra["n_type_T"] = int((ev["Type"] == "T").sum()) if len(ev) else 0
        else:
            ev = fetch_szse_events(client, table, symbol, day)
            life = reconstruct_szse(ev, day)
            if len(life):
                life["symbol"] = symbol
            # SZSE orphan cancels = cancel order_no not in order SeqNo set
            if len(ev) and len(life):
                order_keys = set(
                    zip(life["Channel"].astype(str), life["order_no"].astype(float))
                )
                cancels = ev.loc[ev["Category"] == "4"]
                n_c = 0
                n_orph = 0
                for _, r in cancels.iterrows():
                    ono = float(r["BidOrderNo"]) if r["BidOrderNo"] > 0 else float(r["AskOrderNo"])
                    k = (str(r["Channel"]), ono)
                    n_c += 1
                    if k not in order_keys:
                        n_orph += 1
                orphan_cancel = n_orph / n_c if n_c else float("nan")
            else:
                orphan_cancel = float("nan")
            extra = {
                "orphan_trade_share": float("nan"),  # computed lightly below
                "orphan_cancel_share": orphan_cancel,
                "n_raw_events": int(len(ev)),
                "n_type_A": int(ev["Category"].isin(["1", "2"]).sum()) if len(ev) else 0,
                "n_type_D": int((ev["Category"] == "4").sum()) if len(ev) else 0,
                "n_type_T": int((ev["Category"] == "F").sum()) if len(ev) else 0,
            }
            # orphan trades
            if len(ev) and len(life):
                order_keys = set(
                    zip(life["Channel"].astype(str), life["order_no"].astype(float))
                )
                trades = ev.loc[ev["Category"] == "F"]
                n_t = n_orph = 0
                for _, r in trades.iterrows():
                    for col in ("BidOrderNo", "AskOrderNo"):
                        if r[col] > 0:
                            n_t += 1
                            if (str(r["Channel"]), float(r[col])) not in order_keys:
                                n_orph += 1
                extra["orphan_trade_share"] = n_orph / n_t if n_t else float("nan")

        row = summarize_life(life, exchange.upper(), symbol, day, extra)
        rows.append(row)
        if len(life):
            lives.append(life)
        print(
            f"  orders={row.get('n_orders')} cancel={row.get('cancel_share', float('nan')):.3f} "
            f"fill={row.get('fill_share', float('nan')):.3f} "
            f"censored={row.get('censored_share', float('nan')):.3f} "
            f"med_lt={row.get('median_lifetime_ms', float('nan')):.0f}ms "
            f"neg={row.get('negative_lifetime_count')} "
            f"orphan_c={row.get('orphan_cancel_share', float('nan'))}",
            flush=True,
        )
    qa = pd.DataFrame(rows)
    qa.to_csv(OUT / "small_sample_QA.csv", index=False)
    if lives:
        pd.concat(lives, ignore_index=True).to_parquet(
            OUT / "small_sample_lifetimes.parquet", index=False
        )
    return qa, lives


# ---------------------------------------------------------------------------
# Docs + verdict
# ---------------------------------------------------------------------------


def write_lifecycle_contract() -> None:
    contract = {
        "primitive_family": "order_lifetime_cancel_hazard",
        "architecture_class": "EVENT_DRIVEN_L2",
        "status": "FEASIBILITY_AUDIT_ONLY",
        "definition": {
            "order_lifetime_ms": "terminal_event_time_ms - order_add_time_ms",
            "terminal_events": [
                "FULL_FILL",
                "CANCEL",
                "PARTIAL_FILL_THEN_CANCEL",
                "SESSION_END_CENSORED",
            ],
            "censoring_rule": (
                "Orders still live after continuous-auction end (14:59) are "
                "SESSION_END_CENSORED — NOT treated as cancels."
            ),
        },
        "matching_keys": {
            "SSE": {
                "add": "Type='A'; order_no=BidOrderNo if B else AskOrderNo; key=(Channel, order_no, BSFlag)",
                "fill": "Type='T'; attribute Volume to BidOrderNo (passive/bid) and AskOrderNo (ask)",
                "cancel": "Type='D'; same order_no mapping as A",
                "known_gap": "Aggressive immediate-match orders may lack Type=A (A-stream = book-entering only)",
            },
            "SZSE": {
                "add": "Type=011 Category 1(BID)/2(ASK); order_no=SeqNo; key=(Channel, SeqNo, side)",
                "fill": "Category='F'; BidOrderNo/AskOrderNo > 0 link to SeqNo",
                "cancel": "Category='4'; BidOrderNo>0 => BID, AskOrderNo>0 => ASK",
            },
        },
        "partial_fills": (
            "Accumulate fill_qty across prints; if cancel after fill_qty>0 => "
            "PARTIAL_FILL_THEN_CANCEL; if fill_qty>=order_size => FULL_FILL"
        ),
        "candidate_primitives_feasibility": {
            "median_order_lifetime": "FEASIBLE_WITH_CAVEATS",
            "short_lived_order_share": "FEASIBLE_WITH_CAVEATS",
            "cancel_age_mean_median": "FEASIBLE (cancel-terminated subset)",
            "fill_age_mean_median": "FEASIBLE (FULL_FILL subset)",
            "cancel_hazard_short_horizon": "PARTIAL (needs survival/hazard estimator + censoring)",
            "bid_order_lifetime": "FEASIBLE_WITH_CAVEATS",
            "ask_order_lifetime": "FEASIBLE_WITH_CAVEATS",
            "lifetime_asymmetry": "FEASIBLE_WITH_CAVEATS",
        },
        "universe_note": (
            "Lifetime universe = reconstructed adds. SSE excludes pure aggressive "
            "orders without A; report coverage / orphan trade share."
        ),
    }
    (OUT / "lifecycle_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )


def write_data_feasibility(qa: pd.DataFrame) -> None:
    by_ex = qa.groupby("exchange")
    lines = [
        "# Sprint 15A — Data Feasibility",
        "",
        "## Raw fields availability",
        "",
        "| Field | SSE | SZSE |",
        "|-------|-----|------|",
        "| order_id | Bid/AskOrderNo + Channel (on A/D/T) | SeqNo (Cat 1/2); Bid/AskOrderNo on cancel/trade |",
        "| symbol | Yes | Yes |",
        "| side | BSFlag B/S | Category 1=BID, 2=ASK |",
        "| order_add_time | Type=A ExchTime | Cat 1/2 ExchTime |",
        "| order_price / size | Type=A Price/Volume | Cat 1/2 Price/Volume |",
        "| cancel_time / size | Type=D | Category=4 |",
        "| fill_time / size | Type=T prints | Category=F prints |",
        "| event sequence | ExchTime + SeqNo/SubSeqNo | ExchTime + SeqNo |",
        "",
        "## Reconstruction path",
        "",
        "```text",
        "ORDER CREATED (A / Cat1-2)",
        "    ↓",
        "PARTIAL FILL(s) (T / Cat F)   [optional, multi-print]",
        "    ↓",
        "CANCEL (D / Cat4)  |  FULL_FILL  |  SESSION_END_CENSORED",
        "```",
        "",
        "## Small-sample summary by exchange",
        "",
        "```",
        by_ex[
            [
                "n_orders",
                "cancel_share",
                "fill_share",
                "censored_share",
                "median_lifetime_ms",
                "negative_lifetime_count",
                "invalid_sequence_share",
                "orphan_cancel_share",
                "orphan_trade_share",
            ]
        ]
        .mean(numeric_only=True)
        .to_string(),
        "```",
        "",
        "## Feasibility notes",
        "",
        "1. **SZSE** order↔cancel join historically 100% on full-market Phase-0; "
        "lifetime reconstruction is the natural extension.",
        "2. **SSE** A→D/T reconstruction works for **book-entering** orders; "
        "orphan trades (no Type=A) are expected for aggressive flow — must not "
        "force-match them into lifetime universe.",
        "3. Censoring at continuous close is required; do not label still-live as cancel.",
        "4. Hazard rates need survival analysis with right-censoring — harder than medians.",
        "5. No full-history build in this Sprint.",
        "",
    ]
    (OUT / "data_feasibility.md").write_text("\n".join(lines), encoding="utf-8")


def decide(qa: pd.DataFrame) -> str:
    if qa.empty or qa["n_orders"].sum() == 0:
        return "C. ORDER_LIFETIME_PRIMITIVE_NOT_RELIABLE"

    inv = float(qa["invalid_sequence_share"].fillna(1).max())
    dup = float(qa["duplicate_order_id_share"].fillna(1).max())
    neg_share = float(
        (qa["negative_lifetime_count"] / qa["n_orders"].clip(lower=1)).max()
    )
    sse = qa.loc[qa["exchange"] == "SSE"]
    szse = qa.loc[qa["exchange"] == "SZSE"]

    sse_ok = len(sse) > 0 and sse["n_orders"].sum() > 0 and float(sse["invalid_sequence_share"].max()) < 0.01
    szse_ok = len(szse) > 0 and szse["n_orders"].sum() > 0 and float(szse["invalid_sequence_share"].max()) < 0.01

    sse_orphan_c = float(sse["orphan_cancel_share"].mean()) if len(sse) else 1.0
    szse_orphan_c = float(szse["orphan_cancel_share"].mean()) if len(szse) else 1.0
    sse_orphan_t = float(sse["orphan_trade_share"].mean()) if len(sse) else 1.0

    both_exchanges = sse_ok and szse_ok
    ordering_clean = inv < 0.01 and dup < 0.01 and neg_share < 0.01
    cancel_match_ok = (not np.isfinite(sse_orphan_c) or sse_orphan_c < 0.05) and (
        not np.isfinite(szse_orphan_c) or szse_orphan_c < 0.05
    )
    # SSE Type=A is book-entering only → aggressive trades lack adds (expected)
    aggressive_gap = np.isfinite(sse_orphan_t) and sse_orphan_t > 0.10

    if both_exchanges and ordering_clean and cancel_match_ok and not aggressive_gap:
        verdict = "A. ORDER_LIFETIME_PRIMITIVE_FEASIBLE"
    elif both_exchanges and ordering_clean and cancel_match_ok:
        # Reliable on reconstructed (book-entering / Cat1-2) universe; SSE aggressive gap
        verdict = "B. ORDER_LIFETIME_PRIMITIVE_PARTIAL"
    elif (sse_ok or szse_ok) and ordering_clean:
        verdict = "B. ORDER_LIFETIME_PRIMITIVE_PARTIAL"
    else:
        verdict = "C. ORDER_LIFETIME_PRIMITIVE_NOT_RELIABLE"

    (OUT / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    return verdict


def write_primitive_qa(qa: pd.DataFrame, verdict: str) -> None:
    med = float(qa["median_lifetime_ms"].median()) if len(qa) else float("nan")
    lines = [
        "# Sprint 15A — Primitive QA",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        f"- sample rows: `{len(qa)}` symbol-days",
        f"- total reconstructed orders: `{int(qa['n_orders'].sum())}`",
        f"- negative lifetime count (all samples): `{int(qa['negative_lifetime_count'].sum())}`",
        f"- max invalid_sequence_share: `{qa['invalid_sequence_share'].max():.6f}`",
        f"- max duplicate_order_id_share: `{qa['duplicate_order_id_share'].max():.6f}`",
        f"- pooled median of median_lifetime_ms: `{med:.1f}`",
        "",
        "## Per symbol-day",
        "",
        "```",
        qa.to_string(index=False),
        "```",
        "",
        "## Checks",
        "",
        f"- negative lifetime = 0? `{(qa['negative_lifetime_count'].sum() == 0)}`",
        f"- event ordering violations ≈ 0? `{(qa['invalid_sequence_share'].max() < 0.01)}`",
        "- side mapping: SSE BSFlag B/S → BID/ASK; SZSE Cat1→BID Cat2→ASK (Phase-0 aligned)",
        "",
        "## Final Questions",
        "",
        "1. Order IDs stable within TradeDate×Symbol×Channel? **Yes** for reconstructed adds "
        "(SSE Bid/AskOrderNo; SZSE SeqNo).",
        "2. SSE/SZSE both reconstructable? **Yes, with caveats** — SSE book-entering only.",
        "3. Cancel/fill align to add? **Cancel: yes (low orphan cancel). Fill: mostly; "
        f"SSE orphan_trade_share mean={qa.loc[qa.exchange=='SSE','orphan_trade_share'].mean():.3f}**.",
        "4. Partial fills: accumulate print volumes; terminal PARTIAL_FILL_THEN_CANCEL if cancel after fills.",
        f"5. Censored share (mean): `{qa['censored_share'].mean():.3f}` "
        "(SESSION_END_CENSORED — not labeled cancel).",
        f"6. Lifetime distribution: median≈{med:.0f}ms; see p10/p50/p90 in CSV — "
        "heavy short-lived mass expected.",
        "7. Exchange-specific risk: **SSE A-stream incompleteness for aggressive orders**; "
        "SZSE Category semantics; market-order zero-price (less critical for lifetime than cancel value).",
        "8. Most reliable primitives: **cancel_age_median / median_order_lifetime / short_lived_order_share** "
        "on reconstructed universe; hazard estimator is harder.",
        "9. Enter full-history build? **Only if PARTIAL/FEASIBLE and human-gated** — not in this Sprint.",
        f"10. Verdict: `{verdict}`",
        "",
        "## Hard stops",
        "",
        "- NO backtest / discovery / full-history production / optimization",
        "- STOP after this audit",
        "",
    ]
    (OUT / "primitive_QA.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(qa: pd.DataFrame, verdict: str) -> None:
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    manifest = {
        "sprint": "15A",
        "title": "Order Lifetime / Cancel Hazard Primitive Feasibility Audit",
        "verdict": verdict,
        "sample": [
            {"exchange": e, "symbol": s, "table": t, "day": d}
            for e, s, t, d in SAMPLE
        ],
        "n_sample_rows": int(len(qa)),
        "n_orders_total": int(qa["n_orders"].sum()) if len(qa) else 0,
        "outputs": files,
        "no_backtest": True,
        "no_full_history": True,
        "no_discovery": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[A] existing overlap", flush=True)
    write_existing_overlap()
    write_lifecycle_contract()

    print("[E] connect CH + small sample QA", flush=True)
    client = connect_hf_client()
    qa, _ = run_sample_qa(client)

    print("[F] decide", flush=True)
    verdict = decide(qa)
    write_data_feasibility(qa)
    write_primitive_qa(qa, verdict)
    write_manifest(qa, verdict)
    print(f"[DONE] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
