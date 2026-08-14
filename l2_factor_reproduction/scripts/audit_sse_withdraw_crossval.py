#!/usr/bin/env python
"""Sprint 6B Phase 0 gate 6 — SSE dual-source cancel cross-validation.

Compares, per symbol-day over 2024-06 (SSE universe):

  A) Tick Type='D' cancel aggregation (full calendar day, by BSFlag)
  B) SSE AL_SSL2 native cumulative withdraw counters
     (BidWithdrawNum/Volume/Amount, AskWithdrawNum/Volume/Amount;
     daily value = last snapshot of the day)

Grouped by board (main 600/601/603/605 vs STAR 688) x activity tercile
(same-month trade count), reporting per group:
  daily Spearman (mean over days), median relative difference,
  p95 absolute relative difference, missing share, coverage.

Purpose: decide whether the STAR residual-identity deviation comes from
A/T lifecycle reconstruction or from the D cancel records themselves.

Read-only audit; no production writes.

Usage:
    python audit_sse_withdraw_crossval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402

OUT_DIR = (
    PROJ_ROOT / "research" / "results" / "l2_reproduction" / "primitives"
    / "cancel_lifecycle_daily"
)
MONTH_START, MONTH_END = "2024-06-01", "2024-06-30"
MAIN_PREFIXES = ("600", "601", "603", "605")
STAR_PREFIXES = ("688",)
A_SHARE_FILTER = (
    "substring(Symbol,1,3) IN ('600','601','603','605','688')"
)


def _week_chunks():
    days = pd.date_range(MONTH_START, MONTH_END, freq="D")
    for i in range(0, len(days), 7):
        block = days[i:i + 7]
        yield str(block[0].date()), str(block[-1].date())


def fetch_tick_cancels(client) -> pd.DataFrame:
    frames = []
    for start, end in _week_chunks():
        df = client.query_df(f"""
          SELECT Symbol, toDate(ExchTime) AS TradeDate, BSFlag,
                 count() AS d_num,
                 sum(Volume) AS d_qty,
                 sum(Volume * toFloat64(Price)) AS d_value
          FROM cmds.SSE_AL_TICK_EXG
          WHERE ExchTime >= toDateTime64('{start} 00:00:00',6,'Asia/Shanghai')
            AND ExchTime < toDateTime64('{end} 00:00:00',6,'Asia/Shanghai')
                          + toIntervalDay(1)
            AND Type = 'D' AND {A_SHARE_FILTER}
          GROUP BY Symbol, TradeDate, BSFlag""")
        frames.append(df)
        print(f"[tick D] {start}..{end}: {len(df):,} rows", flush=True)
    out = pd.concat(frames, ignore_index=True)
    buy = out.loc[out["BSFlag"] == "B"].rename(
        columns={"d_num": "buy_d_num", "d_qty": "buy_d_qty",
                 "d_value": "buy_d_value"}
    ).drop(columns="BSFlag")
    sell = out.loc[out["BSFlag"] == "S"].rename(
        columns={"d_num": "sell_d_num", "d_qty": "sell_d_qty",
                 "d_value": "sell_d_value"}
    ).drop(columns="BSFlag")
    return buy.merge(
        sell, on=["Symbol", "TradeDate"], how="outer"
    ).fillna(0)


def fetch_ssl2_withdraw(client) -> pd.DataFrame:
    frames = []
    for start, end in _week_chunks():
        df = client.query_df(f"""
          SELECT Symbol, toDate(ExchTime) AS TradeDate,
                 argMax(BidWithdrawNum, ExchTime) AS bid_wd_num,
                 argMax(BidWithdrawVolume, ExchTime) AS bid_wd_qty,
                 argMax(toFloat64(BidWithdrawAmount), ExchTime) AS bid_wd_value,
                 argMax(AskWithdrawNum, ExchTime) AS ask_wd_num,
                 argMax(AskWithdrawVolume, ExchTime) AS ask_wd_qty,
                 argMax(toFloat64(AskWithdrawAmount), ExchTime) AS ask_wd_value,
                 count() AS n_snapshots
          FROM cmds.SSE_AL_SSL2_EXG
          WHERE ExchTime >= toDateTime64('{start} 00:00:00',6,'Asia/Shanghai')
            AND ExchTime < toDateTime64('{end} 00:00:00',6,'Asia/Shanghai')
                          + toIntervalDay(1)
            AND {A_SHARE_FILTER}
          GROUP BY Symbol, TradeDate""")
        frames.append(df)
        print(f"[ssl2 wd] {start}..{end}: {len(df):,} rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def fetch_activity(client) -> pd.DataFrame:
    return client.query_df(f"""
      SELECT Symbol, countIf(Type='T') AS n_trades
      FROM cmds.SSE_AL_TICK_EXG
      WHERE ExchTime >= toDateTime64('{MONTH_START} 00:00:00',6,'Asia/Shanghai')
        AND ExchTime < toDateTime64('{MONTH_END} 00:00:00',6,'Asia/Shanghai')
                      + toIntervalDay(1)
        AND {A_SHARE_FILTER}
      GROUP BY Symbol""")


def main() -> int:
    client = connect_hf_client()
    tick = fetch_tick_cancels(client)
    ssl2 = fetch_ssl2_withdraw(client)
    activity = fetch_activity(client)

    merged = ssl2.merge(
        tick, on=["Symbol", "TradeDate"], how="outer", indicator=True
    )
    merged["TradeDate"] = pd.to_datetime(merged["TradeDate"])
    merged["board"] = np.where(
        merged["Symbol"].str[:3].isin(STAR_PREFIXES), "STAR", "main"
    )
    act = activity.set_index("Symbol")["n_trades"]
    merged["n_trades"] = merged["Symbol"].map(act)
    liquid = merged.dropna(subset=["n_trades"])
    merged["activity_tercile"] = pd.qcut(
        liquid["n_trades"].rank(method="first"), 3,
        labels=["low", "mid", "high"],
    ).reindex(merged.index)

    # per symbol-day relative differences (buy side value as primary)
    for side, d_col, s_col in [
        ("buy", "buy_d_value", "bid_wd_value"),
        ("sell", "sell_d_value", "ask_wd_value"),
    ]:
        denom = merged[s_col].abs()
        merged[f"{side}_rel_diff"] = np.where(
            denom > 0, (merged[d_col] - merged[s_col]).abs() / denom, np.nan
        )

    merged.to_csv(OUT_DIR / "sse_withdraw_crossval_2024-06.csv", index=False)

    rows = []
    for (board, tercile), block in merged.groupby(
        ["board", "activity_tercile"], observed=True
    ):
        for side, d_col, s_col in [
            ("buy", "buy_d_value", "bid_wd_value"),
            ("sell", "sell_d_value", "ask_wd_value"),
        ]:
            valid = block.dropna(subset=[d_col, s_col])
            valid = valid.loc[valid[s_col] > 0]
            spearman_by_day = valid.groupby("TradeDate").apply(
                lambda g: g[d_col].corr(g[s_col], method="spearman")
                if len(g) >= 30 else np.nan
            )
            rel = valid[f"{side}_rel_diff"]
            rows.append(
                {
                    "board": board,
                    "activity_tercile": str(tercile),
                    "side": side,
                    "n_symbol_days": len(block),
                    "coverage_ssl2": float(
                        block[s_col].notna().mean()
                    ),
                    "missing_share_either": float(
                        (block["_merge"] != "both").mean()
                    ),
                    "daily_spearman_mean": float(
                        spearman_by_day.dropna().mean()
                    ),
                    "median_rel_diff": float(rel.median()),
                    "p95_rel_diff": float(rel.quantile(0.95)),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUT_DIR / "sse_withdraw_crossval_2024-06_summary.csv", index=False
    )
    print("\n== cross-validation summary ==")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
