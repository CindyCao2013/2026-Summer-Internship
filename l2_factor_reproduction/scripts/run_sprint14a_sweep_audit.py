#!/usr/bin/env python
"""Sprint 14A — Sweep / Book Penetration primitive feasibility audit.

Audit only: no alpha backtest / discovery / FV.
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
    / "research/results/l2_reproduction/sprint14_sweep_penetration/primitive_audit"
)

SAMPLE = [
    # exchange, symbol, tick_table, book_table, day
    ("sse", "600000", "cmds.LOCAL_SSE_AL_TICK_EXG", "cmds.LOCAL_SSE_AL_SSL2_EXG", "2024-06-28"),
    ("sse", "600933", "cmds.LOCAL_SSE_AL_TICK_EXG", "cmds.LOCAL_SSE_AL_SSL2_EXG", "2024-06-28"),
    ("szse", "000333", "cmds.LOCAL_SZSE_AL_TICK_EXG", "cmds.LOCAL_SZSE_AL_SSL2_EXG", "2024-06-28"),
    ("szse", "000938", "cmds.LOCAL_SZSE_AL_TICK_EXG", "cmds.LOCAL_SZSE_AL_SSL2_EXG", "2024-06-28"),
]

# Continuous session only; skip auction endpoints.
SESSION_START = "09:30:00"
SESSION_END_AM = "11:30:00"
SESSION_START_PM = "13:00:00"
SESSION_END_PM = "14:57:00"  # exclude close auction vicinity


def _dt(day: str, hhmmss: str) -> str:
    return f"toDateTime64('{day} {hhmmss}', 6, 'Asia/Shanghai')"


def fetch_books(client, book: str, symbol: str, day: str) -> pd.DataFrame:
    sql = f"""
    SELECT
      ExchTime AS book_time,
      toUnixTimestamp64Milli(ExchTime) AS book_ms,
      BidPrices, BidVolumes, AskPrices, AskVolumes
    FROM {book}
    WHERE Symbol = '{symbol}'
      AND (
        (ExchTime >= {_dt(day, SESSION_START)} AND ExchTime < {_dt(day, SESSION_END_AM)})
        OR (ExchTime >= {_dt(day, SESSION_START_PM)} AND ExchTime < {_dt(day, SESSION_END_PM)})
      )
      AND length(BidPrices) > 0 AND length(AskPrices) > 0
      AND toFloat64(AskPrices[1]) > 0 AND toFloat64(BidPrices[1]) > 0
      AND toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])
    ORDER BY ExchTime
    """
    df = client.query_df(sql)
    df["book_time"] = pd.to_datetime(df["book_time"])
    return df


def fetch_trades(client, tick: str, exchange: str, symbol: str, day: str) -> pd.DataFrame:
    if exchange == "sse":
        sql = f"""
        SELECT
          ExchTime AS trade_time,
          toUnixTimestamp64Milli(ExchTime) AS trade_ms,
          toFloat64(Price) AS trade_price,
          toFloat64(Volume) AS trade_volume,
          toFloat64(Amount) AS trade_amount,
          BSFlag,
          BidOrderNo, AskOrderNo, Channel, SeqNo, SubSeqNo,
          if(BSFlag = 'B', 1, if(BSFlag = 'S', -1, 0)) AS trade_direction,
          if(BSFlag = 'B', BidOrderNo, if(BSFlag = 'S', AskOrderNo, 0)) AS agg_order_no
        FROM {tick}
        WHERE Symbol = '{symbol}' AND Type = 'T' AND BSFlag IN ('B', 'S')
          AND (
            (ExchTime >= {_dt(day, SESSION_START)} AND ExchTime < {_dt(day, SESSION_END_AM)})
            OR (ExchTime >= {_dt(day, SESSION_START_PM)} AND ExchTime < {_dt(day, SESSION_END_PM)})
          )
        ORDER BY ExchTime, SeqNo, SubSeqNo
        """
    else:
        sql = f"""
        SELECT
          ExchTime AS trade_time,
          toUnixTimestamp64Milli(ExchTime) AS trade_ms,
          toFloat64(Price) AS trade_price,
          toFloat64(Volume) AS trade_volume,
          toFloat64(Price) * toFloat64(Volume) AS trade_amount,
          BidOrderNo, AskOrderNo, Channel, SeqNo,
          if(BidOrderNo > AskOrderNo, 1, if(BidOrderNo < AskOrderNo, -1, 0)) AS trade_direction,
          if(BidOrderNo > AskOrderNo, BidOrderNo,
             if(BidOrderNo < AskOrderNo, AskOrderNo, 0)) AS agg_order_no
        FROM {tick}
        WHERE Symbol = '{symbol}' AND Type = '011' AND Category = 'F'
          AND BidOrderNo != AskOrderNo
          AND (
            (ExchTime >= {_dt(day, SESSION_START)} AND ExchTime < {_dt(day, SESSION_END_AM)})
            OR (ExchTime >= {_dt(day, SESSION_START_PM)} AND ExchTime < {_dt(day, SESSION_END_PM)})
          )
        ORDER BY ExchTime, SeqNo
        """
    df = client.query_df(sql)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    return df


def _arr_f64(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    return np.asarray([float(v) if v is not None else np.nan for v in x], dtype=float)


def estimate_penetration(
    direction: int,
    trade_price: float,
    trade_volume: float,
    bid_px: np.ndarray,
    bid_vol: np.ndarray,
    ask_px: np.ndarray,
    ask_vol: np.ndarray,
) -> Dict[str, Any]:
    """Conservative trade-level estimates vs pre-trade book."""
    out = {
        "best_price_before": np.nan,
        "displayed_depth_before": np.nan,
        "estimated_levels_penetrated": np.nan,
        "estimated_depth_consumed_ratio": np.nan,
        "penetration_price_distance": np.nan,
        "size_implied_levels": np.nan,
        "quality_flag": "ok",
    }
    if direction == 1:
        px, vol = ask_px, ask_vol
    elif direction == -1:
        px, vol = bid_px, bid_vol
    else:
        out["quality_flag"] = "neutral_side"
        return out

    mask = np.isfinite(px) & (px > 0) & np.isfinite(vol) & (vol >= 0)
    px, vol = px[mask], vol[mask]
    if len(px) == 0:
        out["quality_flag"] = "empty_side_book"
        return out

    best = float(px[0])
    depth = float(vol.sum())
    out["best_price_before"] = best
    out["displayed_depth_before"] = depth

    # Price-based levels: how many displayed levels at or through trade price.
    if direction == 1:
        hit = np.where(px <= trade_price + 1e-9)[0]
        dist = (trade_price - best) / best if best > 0 else np.nan
        # sanity: buy should not print below best ask (crossing) much; allow tiny float
        if trade_price + 1e-6 < best:
            out["quality_flag"] = "buy_below_ask1"
    else:
        hit = np.where(px >= trade_price - 1e-9)[0]
        dist = (best - trade_price) / best if best > 0 else np.nan
        if trade_price - 1e-6 > best:
            out["quality_flag"] = "sell_above_bid1"

    levels = int(len(hit)) if len(hit) else 0
    # If trade price does not match any level (hidden / stale), mark ambiguous.
    if levels == 0:
        out["quality_flag"] = "price_not_on_ladder"
        levels = np.nan
    out["estimated_levels_penetrated"] = levels
    out["penetration_price_distance"] = dist

    # Depth consumed ratio vs total displayed same-side depth L1..N
    out["estimated_depth_consumed_ratio"] = (
        float(trade_volume / depth) if depth > 0 else np.nan
    )

    # Size-implied walk from best (ignores actual print price): how many levels
    # needed to absorb trade_volume. Diagnostic only — near size/depth proxy.
    cum = 0.0
    need = 0
    for v in vol:
        cum += float(v)
        need += 1
        if cum + 1e-12 >= trade_volume:
            break
    if cum + 1e-12 < trade_volume:
        need = len(vol)  # exhausted book
        if out["quality_flag"] == "ok":
            out["quality_flag"] = "size_exceeds_displayed"
    out["size_implied_levels"] = need
    return out


def align_and_estimate(trades: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or books.empty:
        return pd.DataFrame()
    book_ms = books["book_ms"].to_numpy(dtype=np.int64)
    # arrays as object columns
    bid_pxs = [_arr_f64(x) for x in books["BidPrices"]]
    bid_vols = [_arr_f64(x) for x in books["BidVolumes"]]
    ask_pxs = [_arr_f64(x) for x in books["AskPrices"]]
    ask_vols = [_arr_f64(x) for x in books["AskVolumes"]]
    book_times = books["book_time"].to_numpy()

    rows = []
    for i, tr in trades.iterrows():
        tms = int(tr["trade_ms"])
        # latest book strictly before trade
        j = np.searchsorted(book_ms, tms, side="left") - 1
        if j < 0:
            rows.append(
                {
                    "trade_time": tr["trade_time"],
                    "trade_ms": tms,
                    "trade_direction": int(tr["trade_direction"]),
                    "trade_price": float(tr["trade_price"]),
                    "trade_volume": float(tr["trade_volume"]),
                    "trade_amount": float(tr["trade_amount"]),
                    "agg_order_no": int(tr["agg_order_no"]),
                    "channel": int(tr["Channel"]) if "Channel" in tr and pd.notna(tr["Channel"]) else -1,
                    "reference_snapshot_time": pd.NaT,
                    "alignment_lag_ms": np.nan,
                    "best_price_before": np.nan,
                    "displayed_depth_before": np.nan,
                    "estimated_levels_penetrated": np.nan,
                    "estimated_depth_consumed_ratio": np.nan,
                    "penetration_price_distance": np.nan,
                    "size_implied_levels": np.nan,
                    "quality_flag": "missing_reference_book",
                }
            )
            continue
        est = estimate_penetration(
            int(tr["trade_direction"]),
            float(tr["trade_price"]),
            float(tr["trade_volume"]),
            bid_pxs[j],
            bid_vols[j],
            ask_pxs[j],
            ask_vols[j],
        )
        rows.append(
            {
                "trade_time": tr["trade_time"],
                "trade_ms": tms,
                "trade_direction": int(tr["trade_direction"]),
                "trade_price": float(tr["trade_price"]),
                "trade_volume": float(tr["trade_volume"]),
                "trade_amount": float(tr["trade_amount"]),
                "agg_order_no": int(tr["agg_order_no"]),
                "channel": int(tr["Channel"]) if "Channel" in tr and pd.notna(tr["Channel"]) else -1,
                "reference_snapshot_time": book_times[j],
                "alignment_lag_ms": float(tms - int(book_ms[j])),
                **est,
            }
        )
    return pd.DataFrame(rows)


def parent_order_stats(events: pd.DataFrame) -> Dict[str, float]:
    if events.empty:
        return {}
    g = (
        events.groupby(["channel", "agg_order_no", "trade_direction"], sort=False)
        .agg(
            n_fills=("trade_price", "size"),
            n_prices=("trade_price", "nunique"),
            tot_vol=("trade_volume", "sum"),
            max_levels=("estimated_levels_penetrated", "max"),
            t0=("trade_ms", "min"),
            t1=("trade_ms", "max"),
        )
        .reset_index()
    )
    return {
        "n_parent_orders": float(len(g)),
        "mean_fills": float(g["n_fills"].mean()),
        "share_multi_fill": float((g["n_fills"] >= 2).mean()),
        "share_multi_price": float((g["n_prices"] >= 2).mean()),
        "share_ge3_prices": float((g["n_prices"] >= 3).mean()),
        "share_parent_levels_ge2": float(
            (g["max_levels"].fillna(0) >= 2).mean()
        ),
        "mean_span_ms": float((g["t1"] - g["t0"]).mean()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = connect_hf_client()
    all_events: List[pd.DataFrame] = []
    freq_rows: List[Dict[str, Any]] = []
    parent_rows: List[Dict[str, Any]] = []

    for exchange, symbol, tick, book, day in SAMPLE:
        print(f"[sample] {exchange} {symbol} {day}", flush=True)
        books = fetch_books(client, book, symbol, day)
        trades = fetch_trades(client, tick, exchange, symbol, day)
        print(f"  books={len(books)} trades={len(trades)}", flush=True)

        # book gap stats
        bms = books["book_ms"].to_numpy(dtype=np.int64)
        gaps = np.diff(bms)
        gaps = gaps[(gaps > 0) & (gaps < 60_000)]
        freq_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "day": day,
                "n_books": len(books),
                "n_trades": len(trades),
                "med_book_gap_ms": float(np.median(gaps)) if len(gaps) else np.nan,
                "p90_book_gap_ms": float(np.quantile(gaps, 0.9)) if len(gaps) else np.nan,
                "mean_ask_levels": float(
                    np.mean([len(_arr_f64(x)) for x in books["AskPrices"]])
                )
                if len(books)
                else np.nan,
            }
        )

        events = align_and_estimate(trades, books)
        if events.empty:
            continue
        events.insert(0, "TradeDate", pd.Timestamp(day))
        events.insert(0, "symbol", f"{symbol}.{'SH' if exchange=='sse' else 'SZ'}")
        events.insert(0, "exchange", exchange)
        all_events.append(events)

        ps = parent_order_stats(events)
        ps.update(exchange=exchange, symbol=symbol, day=day)
        parent_rows.append(ps)
        print(f"  parent_stats={ps}", flush=True)

    events = pd.concat(all_events, ignore_index=True)
    events_path = OUT / "small_sample_events.parquet"
    events.to_parquet(events_path, index=False)
    print(f"[write] {events_path} rows={len(events)}", flush=True)

    # ---- distributions / QA ----
    ok = events["quality_flag"].isin(["ok", "size_exceeds_displayed"])
    lev = events.loc[ok, "estimated_levels_penetrated"].astype(float)
    dist_rows = []
    for name, mask in [
        ("all_ok", ok),
        ("buy_ok", ok & (events["trade_direction"] == 1)),
        ("sell_ok", ok & (events["trade_direction"] == -1)),
    ]:
        sub = events.loc[mask]
        lv = sub["estimated_levels_penetrated"].astype(float)
        dist_rows.append(
            {
                "slice": name,
                "n": int(len(sub)),
                "frac_levels_1": float((lv == 1).mean()) if len(sub) else np.nan,
                "frac_levels_ge2": float((lv >= 2).mean()) if len(sub) else np.nan,
                "frac_levels_ge3": float((lv >= 3).mean()) if len(sub) else np.nan,
                "mean_levels": float(lv.mean()) if len(sub) else np.nan,
                "mean_depth_consumed_ratio": float(
                    sub["estimated_depth_consumed_ratio"].mean()
                )
                if len(sub)
                else np.nan,
                "mean_align_lag_ms": float(sub["alignment_lag_ms"].mean())
                if len(sub)
                else np.nan,
                "med_align_lag_ms": float(sub["alignment_lag_ms"].median())
                if len(sub)
                else np.nan,
                "p90_align_lag_ms": float(sub["alignment_lag_ms"].quantile(0.9))
                if len(sub)
                else np.nan,
            }
        )

    # by size bucket (CNY)
    am = events.loc[ok, "trade_amount"]
    buckets = pd.cut(
        am,
        bins=[-np.inf, 1e4, 4e4, 2e5, 1e6, np.inf],
        labels=["<=1e4", "1e4-4e4", "4e4-2e5", "2e5-1e6", ">1e6"],
    )
    for b, idx in events.loc[ok].groupby(buckets).groups.items():
        sub = events.loc[list(idx)]
        lv = sub["estimated_levels_penetrated"].astype(float)
        dist_rows.append(
            {
                "slice": f"size_{b}",
                "n": int(len(sub)),
                "frac_levels_1": float((lv == 1).mean()),
                "frac_levels_ge2": float((lv >= 2).mean()),
                "frac_levels_ge3": float((lv >= 3).mean()),
                "mean_levels": float(lv.mean()),
                "mean_depth_consumed_ratio": float(
                    sub["estimated_depth_consumed_ratio"].mean()
                ),
                "mean_align_lag_ms": float(sub["alignment_lag_ms"].mean()),
                "med_align_lag_ms": float(sub["alignment_lag_ms"].median()),
                "p90_align_lag_ms": float(sub["alignment_lag_ms"].quantile(0.9)),
            }
        )

    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUT / "small_sample_distribution.csv", index=False)

    # quality / alignment diagnostics
    qcounts = events["quality_flag"].value_counts(dropna=False)
    align = {
        "n_events": int(len(events)),
        "missing_reference_book_rate": float(
            (events["quality_flag"] == "missing_reference_book").mean()
        ),
        "price_not_on_ladder_rate": float(
            (events["quality_flag"] == "price_not_on_ladder").mean()
        ),
        "side_violation_rate": float(
            events["quality_flag"]
            .isin(["buy_below_ask1", "sell_above_bid1"])
            .mean()
        ),
        "ambiguous_or_bad_rate": float((~ok).mean()),
        "ok_rate": float(ok.mean()),
        "med_align_lag_ms": float(events.loc[ok, "alignment_lag_ms"].median()),
        "p90_align_lag_ms": float(events.loc[ok, "alignment_lag_ms"].quantile(0.9)),
        "p99_align_lag_ms": float(events.loc[ok, "alignment_lag_ms"].quantile(0.99)),
        "quality_flag_counts": qcounts.to_dict(),
    }
    pd.DataFrame(
        [
            {"metric": k, "value": json.dumps(v) if isinstance(v, dict) else v}
            for k, v in align.items()
        ]
    ).to_csv(OUT / "alignment_diagnostics.csv", index=False)
    pd.DataFrame(freq_rows).to_csv(OUT / "book_frequency_sample.csv", index=False)
    pd.DataFrame(parent_rows).to_csv(OUT / "parent_order_stats.csv", index=False)

    # ---- overlap diagnostics (small sample) ----
    sub = events.loc[ok].copy()
    sub["large_trade_indicator"] = (sub["trade_amount"] > 2e5).astype(float)
    # amount_to_depth proxy at event: trade_amount / (depth_shares * mid_approx)
    # use displayed_depth (shares) * best_price as notional depth proxy
    sub["depth_notional_proxy"] = (
        sub["displayed_depth_before"] * sub["best_price_before"]
    )
    sub["amount_to_depth_proxy"] = sub["trade_amount"] / sub["depth_notional_proxy"].clip(
        lower=1.0
    )
    # OBI proxy from not available at event without both sides packed — skip exact OBI;
    # use signed direction * depth_consumed as weak proxy placeholder not for conclusion.
    corr_pairs = [
        ("estimated_levels_penetrated", "trade_amount"),
        ("estimated_levels_penetrated", "trade_volume"),
        ("estimated_levels_penetrated", "large_trade_indicator"),
        ("estimated_levels_penetrated", "amount_to_depth_proxy"),
        ("estimated_levels_penetrated", "size_implied_levels"),
        ("estimated_depth_consumed_ratio", "trade_amount"),
        ("estimated_depth_consumed_ratio", "amount_to_depth_proxy"),
        ("penetration_price_distance", "trade_amount"),
        ("size_implied_levels", "trade_amount"),
    ]
    corr_rows = []
    for a, b in corr_pairs:
        x = sub[a].astype(float)
        y = sub[b].astype(float)
        m = x.notna() & y.notna()
        rho = float(x[m].corr(y[m], method="spearman")) if m.sum() > 50 else np.nan
        corr_rows.append({"left": a, "right": b, "spearman": rho, "n": int(m.sum())})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "overlap_diagnostics.csv", index=False)

    # side contract check
    buy = sub[sub["trade_direction"] == 1]
    sell = sub[sub["trade_direction"] == -1]
    side_qa = {
        "buy_mean_price_minus_best": float(
            (buy["trade_price"] - buy["best_price_before"]).mean()
        )
        if len(buy)
        else np.nan,
        "sell_mean_best_minus_price": float(
            (sell["best_price_before"] - sell["trade_price"]).mean()
        )
        if len(sell)
        else np.nan,
        "buy_frac_price_ge_best": float(
            (buy["trade_price"] + 1e-9 >= buy["best_price_before"]).mean()
        )
        if len(buy)
        else np.nan,
        "sell_frac_price_le_best": float(
            (sell["trade_price"] - 1e-9 <= sell["best_price_before"]).mean()
        )
        if len(sell)
        else np.nan,
    }

    summary = {
        "n_events": int(len(events)),
        "ok_events": int(ok.sum()),
        "align": align,
        "parent_order": parent_rows,
        "book_freq": freq_rows,
        "overlap": corr_rows,
        "side_qa": side_qa,
        "dist_head": dist.head(10).to_dict(orient="records"),
    }
    (OUT / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
