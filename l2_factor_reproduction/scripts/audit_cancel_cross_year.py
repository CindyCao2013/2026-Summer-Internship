#!/usr/bin/env python
"""Sprint 6B Phase 0 — cross-year / cross-segment cancellation audit.

Two modes:

  sse_redo   Redo of the SSE 4-year lifecycle audit with the high-turnover
             and median picks restricted to the frozen A-share universe
             (SSE prefixes 600/601/603/605/688). Excludes 204001-style
             non-stock securities. Overwrites sse_cross_year_audit.csv.

  segment_expand  Expanded audit: 2019/2022/2024/2026 x {SSE main board,
             STAR(688), ChiNext(300 via SZSE)} x 10 stocks each, evenly
             spaced by same-day trade-count rank. Metrics per stock-day:
             residual-exact share, over-trade event rate, over-trade qty
             rate, abnormal cancel value share. Output:
             segment_cross_year_audit.csv. Result gates whether the 2022
             STAR 688007 anomaly is systematic or idiosyncratic.

Usage:
    python audit_cancel_cross_year.py --mode sse_redo
    python audit_cancel_cross_year.py --mode segment_expand
"""

from __future__ import annotations

import argparse
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

YEARS = {"2019": "2019-06-28", "2022": "2022-06-28", "2024": "2024-06-28",
         "2026": "2026-08-05"}
# STAR board opened 2019-07-22; sample it on a later 2019 date
STAR_2019_DATE = "2019-09-02"

SSE_MAIN_PREFIXES = ("600", "601", "603", "605")
STAR_PREFIXES = ("688",)
CHINEXT_PREFIXES = ("300",)


def day_pred(d: str) -> str:
    return (
        f"ExchTime >= toDateTime64('{d} 00:00:00',6,'Asia/Shanghai')"
        f" AND ExchTime < toDateTime64('{d} 00:00:00',6,'Asia/Shanghai')"
        f" + toIntervalDay(1)"
    )


def _prefix_filter(prefixes) -> str:
    inner = ",".join(f"'{p}'" for p in prefixes)
    return f"substring(Symbol,1,3) IN ({inner})"


def sse_lifecycle(client, symbol: str, d: str) -> dict:
    df = client.query_df(f"""
      SELECT Type, ExchTime t, BSFlag side,
             if(Type='T', 0, if(BSFlag='B', BidOrderNo, AskOrderNo)) ord,
             BidOrderNo bno, AskOrderNo ano, Price px, Volume qty
      FROM cmds.SSE_AL_TICK_EXG WHERE {day_pred(d)} AND Symbol='{symbol}'""")
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    a = df[df.Type == "A"]
    t = df[df.Type == "T"]
    dd = df[df.Type == "D"]
    if len(a) == 0:
        return None
    tb = t.groupby("bno")["qty"].sum()
    ta = t.groupby("ano")["qty"].sum()
    traded = pd.concat([tb, ta]).groupby(level=0).sum()
    life = a.set_index("ord").groupby(level=0).agg(
        a_qty=("qty", "sum"), px=("px", "first"))
    life["traded"] = traded.reindex(life.index).fillna(0)
    dg = dd.groupby("ord").agg(
        d_qty=("qty", "sum"), d_px=("px", "first"), d_n=("qty", "size"))
    life = life.join(dg, how="left")
    has_d = life["d_qty"].notna()
    resid_ok = (
        life.loc[has_d, "d_qty"]
        == life.loc[has_d, "a_qty"] - life.loc[has_d, "traded"]
    )
    cancel_value = (life.loc[has_d, "d_qty"] * life.loc[has_d, "d_px"])
    abnormal_cancel_value = float(cancel_value.loc[~resid_ok].sum())
    total_cancel_value = float(cancel_value.sum())
    over = life["traded"] > life["a_qty"]
    trade_value = float((life["traded"] * life["px"]).sum())
    over_qty = float((life.loc[over, "traded"] - life.loc[over, "a_qty"]).sum())
    return {
        "n_a": len(a), "n_t": len(t), "n_d": len(dd),
        "resid_exact": float(resid_ok.mean()) if has_d.any() else np.nan,
        "px_same": float(
            (life.loc[has_d, "d_px"] == life.loc[has_d, "px"]).mean()
        ) if has_d.any() else np.nan,
        "d_after_full_fill": float(
            (life["traded"] >= life["a_qty"]) .__and__(has_d).mean()
        ) if has_d.any() else 0.0,
        "anomaly_over_trade": float(over.mean()),
        "over_trade_qty_rate": (
            over_qty / float(life["traded"].sum())
            if life["traded"].sum() > 0 else np.nan
        ),
        "abnormal_cancel_value_share": (
            abnormal_cancel_value / total_cancel_value
            if total_cancel_value > 0 else np.nan
        ),
        "trade_value": trade_value,
    }


def szse_lifecycle(client, symbol: str, d: str) -> dict:
    df = client.query_df(f"""
      SELECT Category cat, SubCategory subcat, SeqNo seq,
             BidOrderNo bno, AskOrderNo ano, Price px, Volume qty
      FROM cmds.SZSE_AL_TICK_EXG WHERE {day_pred(d)} AND Symbol='{symbol}'""")
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    orders = df[df.cat.isin(["1", "2"])]
    trades = df[df.cat == "F"]
    cancels = df[df.cat == "4"]
    if len(orders) == 0:
        return None
    tb = trades.groupby("bno")["qty"].sum()
    ta = trades.groupby("ano")["qty"].sum()
    traded = pd.concat([tb, ta]).groupby(level=0).sum()
    life = orders.set_index("seq").groupby(level=0).agg(
        a_qty=("qty", "sum"), px=("px", "first"))
    life["traded"] = traded.reindex(life.index).fillna(0)
    cb = cancels.groupby("bno")["qty"].sum()
    ca = cancels.groupby("ano")["qty"].sum()
    cancelled = pd.concat([cb, ca]).groupby(level=0).sum()
    life["d_qty"] = cancelled.reindex(life.index)
    has_d = life["d_qty"].notna()
    resid_ok = (
        life.loc[has_d, "d_qty"]
        == life.loc[has_d, "a_qty"] - life.loc[has_d, "traded"]
    )
    # SZSE cancel rows carry no usable price; use the order's own price
    # (market orders px=0 contribute zero value and are excluded from the
    # abnormal-value share denominator is left as-is: conservative)
    cancel_value = life.loc[has_d, "d_qty"] * life.loc[has_d, "px"]
    abnormal_cancel_value = float(cancel_value.loc[~resid_ok].sum())
    total_cancel_value = float(cancel_value.sum())
    over = life["traded"] > life["a_qty"]
    over_qty = float((life.loc[over, "traded"] - life.loc[over, "a_qty"]).sum())
    return {
        "n_a": len(orders), "n_t": len(trades), "n_d": len(cancels),
        "resid_exact": float(resid_ok.mean()) if has_d.any() else np.nan,
        "px_same": np.nan,  # SZSE cancel rows do not carry original price
        "d_after_full_fill": float(
            (life["traded"] >= life["a_qty"]).__and__(has_d).mean()
        ) if has_d.any() else 0.0,
        "anomaly_over_trade": float(over.mean()),
        "over_trade_qty_rate": (
            over_qty / float(life["traded"].sum())
            if life["traded"].sum() > 0 else np.nan
        ),
        "abnormal_cancel_value_share": (
            abnormal_cancel_value / total_cancel_value
            if total_cancel_value > 0 else np.nan
        ),
        "trade_value": float((trades["px"] * trades["qty"]).sum()),
    }


def pick_symbols(client, table: str, d: str, prefixes, n: int) -> list:
    # trade-row predicate differs by feed: SSE trades are Type='T',
    # SZSE trades are Category='F'
    trade_pred = "Type='T'" if "SSE_" in table else "Category='F'"
    act = client.query_df(f"""
      SELECT Symbol, countIf({trade_pred}) AS n_trd
      FROM {table}
      WHERE {day_pred(d)} AND {_prefix_filter(prefixes)}
      GROUP BY Symbol HAVING n_trd > 100
      ORDER BY n_trd DESC""")
    if len(act) == 0:
        return []
    k = min(n, len(act))
    idx = np.linspace(0, len(act) - 1, k).round().astype(int)
    return [act.iloc[i]["Symbol"] for i in sorted(set(idx))]


def run_sse_redo(client) -> None:
    rows = []
    a_share = SSE_MAIN_PREFIXES + STAR_PREFIXES
    for yr, d in YEARS.items():
        act = client.query_df(f"""
          SELECT Symbol, countIf(Type='T') AS n_trd
          FROM cmds.SSE_AL_TICK_EXG
          WHERE {day_pred(d)} AND {_prefix_filter(a_share)}
          GROUP BY Symbol HAVING n_trd > 100
          ORDER BY n_trd DESC""")
        hi = act.iloc[0]["Symbol"]
        med = act.iloc[len(act) // 2]["Symbol"]
        for tag, sym in [("large_600000", "600000"),
                         (f"high_turnover_{hi}", hi),
                         (f"median_{med}", med)]:
            r = sse_lifecycle(client, sym, d)
            r.update(year=yr, pick=tag, date=d)
            rows.append(r)
            print(yr, tag, sym, flush=True)
    out = pd.DataFrame(rows)[[
        "year", "date", "pick", "n_a", "n_t", "n_d", "resid_exact",
        "px_same", "d_after_full_fill", "anomaly_over_trade",
        "over_trade_qty_rate", "abnormal_cancel_value_share",
    ]]
    out.to_csv(OUT_DIR / "sse_cross_year_audit.csv", index=False)
    print(out.to_string(index=False))


ALL_SEGMENTS = [
    ("sse_main", "cmds.SSE_AL_TICK_EXG", SSE_MAIN_PREFIXES, sse_lifecycle),
    ("star_688", "cmds.SSE_AL_TICK_EXG", STAR_PREFIXES, sse_lifecycle),
    ("chinext_300", "cmds.SZSE_AL_TICK_EXG", CHINEXT_PREFIXES,
     szse_lifecycle),
]


def run_segment_expand(client, only: str = None) -> None:
    segments = [s for s in ALL_SEGMENTS if only is None or s[0] == only]
    rows = []
    for yr, d in YEARS.items():
        for seg, table, prefixes, fn in segments:
            seg_date = STAR_2019_DATE if (
                seg == "star_688" and yr == "2019"
            ) else d
            symbols = pick_symbols(client, table, seg_date, prefixes, 10)
            print(f"{yr} {seg}: {len(symbols)} symbols @ {seg_date}",
                  flush=True)
            for sym in symbols:
                r = fn(client, sym, seg_date)
                if r is None:
                    print(f"  {sym}: no orders, skipped", flush=True)
                    continue
                r.update(year=yr, segment=seg, date=seg_date, symbol=sym)
                rows.append(r)
                print(f"  {sym} done", flush=True)
    out = pd.DataFrame(rows)[[
        "year", "segment", "date", "symbol", "n_a", "n_t", "n_d",
        "resid_exact", "px_same", "d_after_full_fill", "anomaly_over_trade",
        "over_trade_qty_rate", "abnormal_cancel_value_share", "trade_value",
    ]]
    out_path = OUT_DIR / "segment_cross_year_audit.csv"
    if only is not None and out_path.exists():
        previous = pd.read_csv(out_path)
        previous = previous.loc[previous["segment"] != only]
        out = pd.concat([previous, out], ignore_index=True)
    out.to_csv(out_path, index=False)
    print("\nsegment summary:")
    print(out.groupby(["year", "segment"]).agg(
        n=("symbol", "count"),
        resid_exact=("resid_exact", "median"),
        over_trade=("anomaly_over_trade", "median"),
        over_trade_qty_rate=("over_trade_qty_rate", "median"),
        abnormal_cancel_value_share=(
            "abnormal_cancel_value_share", "median"),
    ).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sse_redo", "segment_expand"],
                        required=True)
    parser.add_argument("--segments", default=None,
                        help="segment_expand only: restrict to one segment "
                             "(sse_main|star_688|chinext_300)")
    args = parser.parse_args()
    client = connect_hf_client()
    if args.mode == "sse_redo":
        run_sse_redo(client)
    else:
        run_segment_expand(client, only=args.segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
