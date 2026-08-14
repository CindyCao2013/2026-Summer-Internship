"""Sprint 6A Phase 2 — golden-reference validation of the CH replication.

Three-way comparison on the five official DolphinDB snapshot formulas:

  A. synthetic series vs HAND-CALCULATED expectations (literal asserts)
  B. synthetic series: verbatim official DDB functions vs independent
     Python oracle (written from formula_mapping.md semantics, not from
     the CH SQL)
  C. synthetic series: ClickHouse pipeline (values() source hook, zero
     production writes) vs oracle + daily-aggregation sanity
  D. real data 2024-06-28 (600000.SH, 000001.SZ): CH base extract rows
     uploaded to DolphinDB, official functions run verbatim, compared
     against the CH series query row by row; the oracle is also evaluated
     on the same rows.

Run from repo root:
    /opt/conda/anaconda3/envs/base_93/bin/python \
        l2_factor_reproduction/tests/golden_ddb_reference.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import dolphindb as ddb

from COMMON_CONST import DATA_DB_CONN
from research.l2_alpha.clickhouse_ssl2 import connect_hf_client
from l2_factor_reproduction.python.ch_ddb_snapshot import (
    _base_extract_sql,
    ddb_snapshot_daily_sql,
    ddb_snapshot_series_sql,
)

# ----------------------------------------------------------------------
# Verbatim official DolphinDB functions (docs.dolphindb.cn §3.1, frozen in
# formula_mapping.md; do not edit — any change invalidates the golden test).
# ----------------------------------------------------------------------

OFFICIAL_FUNCS_DOS = r"""
@state
def timeWeightedOrderSlope(bid,bidQty,ask,askQty,lag=20){
	return (log(iif(ask==0,bid,ask))-log(iif(bid==0,ask,bid)))\(log(askQty)-log(bidQty)).ffill().mavg(lag, 1).nullFill(0)
}

@state
def wavgSOIR(bidQty,askQty,lag=20){
	imbalance= rowWavg((bidQty - askQty)\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1).ffill().nullFill(0)
	mean = mavg(prev(imbalance), (lag-1), 2)
	std = mstdp(prev(imbalance) * 1000000, (lag-1), 2) \ 1000000
	return iif(std >= 0.0000001,(imbalance - mean) \ std, NULL).ffill().nullFill(0)
}

@state
def traPriceWeightedNetBuyQuoteVolumeRatio(bid,bidQty,ask,askQty,TotalValTrd,TotalVolTrd,lag=20){
	prevbid = prev(bid)
	prevbidQty = prev(bidQty)
	prevask = prev(ask)
	prevaskQty = prev(askQty)
	bidchg = iif(round(bid-prevbid,2)>0, bidQty, iif(round(bid-prevbid,2)<0, -prevbidQty, bidQty-prevbidQty))
	offerchg = iif(iif(ask==0,iif(prevask>0,1,0), ask-prevask)>0, prevaskQty, iif(iif(prevask==0,
		iif(ask>0,-1,0), iif(ask>0,ask-prevask,1))<0, askQty, askQty-prevaskQty))
	avgprice = deltas(TotalValTrd)\deltas(TotalVolTrd)
	factorValue = (bidchg-offerchg)\(abs(bidchg)+abs(offerchg))*avgprice
	return nullFill(msum(factorValue,lag,1)\msum(avgprice,lag,1), 0)
}

@state
def level10_Diff(price, qty, buy, lag=20){
        prevPrice = price.prev()
        left, right = rowAlign(price, prevPrice, how=iif(buy, "bid", "ask"))
        qtyDiff = (qty.rowAt(left).nullFill(0) - qty.prev().rowAt(right).nullFill(0))
        amtDiff = rowSum(nullFill(price.rowAt(left), prevPrice.rowAt(right)) * qtyDiff)
        return msum(amtDiff, lag, 1).nullFill(0)
}

@state
def level10_InferPriceTrend(bid, ask, bidQty, askQty, lag1=60, lag2=20){
	inferPrice = (rowSum(bid*bidQty)+rowSum(ask*askQty))\(rowSum(bidQty)+rowSum(askQty))
	price = iif(bid[0] <=0 or ask[0]<=0, NULL, inferPrice)
	return price.ffill().linearTimeTrend(lag1).at(1).nullFill(0).mavg(lag2, 1).nullFill(0)
}
"""

DDB_FACTOR_SQL = r"""
res = select exch_time,
    timeWeightedOrderSlope(bidPx[0], bidQv[0], askPx[0], askQv[0]) as ddb_twos,
    wavgSOIR(bidQv, askQv, 20) as ddb_soir,
    traPriceWeightedNetBuyQuoteVolumeRatio(
        bidPx[0], bidQv[0], askPx[0], askQv[0], accAmt, accVol) as ddb_tpw,
    level10_Diff(bidPx, bidQv, true, 20) as ddb_l10,
    level10_InferPriceTrend(bidPx, askPx, bidQv, askQv, 60, 20) as ddb_ipt
  from (
    select exch_time,
      fixedLengthArrayVector(bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8,bp9,bp10) as bidPx,
      fixedLengthArrayVector(ap1,ap2,ap3,ap4,ap5,ap6,ap7,ap8,ap9,ap10) as askPx,
      fixedLengthArrayVector(bv1,bv2,bv3,bv4,bv5,bv6,bv7,bv8,bv9,bv10) as bidQv,
      fixedLengthArrayVector(av1,av2,av3,av4,av5,av6,av7,av8,av9,av10) as askQv,
      acc_amount as accAmt, acc_volume as accVol
    from chrows order by exch_time
  )
res
"""

# ----------------------------------------------------------------------
# Independent Python oracle (from formula_mapping.md semantics only).
# None models DolphinDB NULL; every arithmetic op propagates None.
# ----------------------------------------------------------------------


def _nu(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _log(x):
    if _nu(x) or x <= 0:
        return None
    return math.log(x)


def _sub(a, b):
    if _nu(a) or _nu(b):
        return None
    return a - b


def _div(a, b):
    if _nu(a) or _nu(b) or b == 0:
        return None
    return a / b


def _ffill(xs: Sequence) -> List:
    out, last = [], None
    for x in xs:
        if not _nu(x):
            last = x
        out.append(last)
    return out


def _win(xs: Sequence, t: int, w: int) -> List:
    lo = max(0, t - w + 1)
    return [x for x in xs[lo : t + 1] if not _nu(x)]


def _mavg(xs, t, w, minp):
    vals = _win(xs, t, w)
    if len(vals) < minp:
        return None
    return sum(vals) / len(vals)


def _msum(xs, t, w, minp):
    vals = _win(xs, t, w)
    if len(vals) < minp:
        return None
    return sum(vals)


def _mstdp(xs, t, w, minp):
    vals = _win(xs, t, w)
    if len(vals) < minp:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _iif(cond, a, b):
    """DolphinDB iif: NULL condition takes the else branch (S7)."""
    if _nu(cond):
        return b
    return a if cond else b


def _eq0(x):
    return None if _nu(x) else (x == 0)


def _gt0(x):
    return None if _nu(x) else (x > 0)


def _lt0(x):
    return None if _nu(x) else (x < 0)


def oracle_twos(bid1, bidq1, ask1, askq1, lag=20):
    n = len(bid1)
    num, den_raw = [], []
    for t in range(n):
        ask_eff = _iif(_eq0(ask1[t]), bid1[t], ask1[t])
        bid_eff = _iif(_eq0(bid1[t]), ask1[t], bid1[t])
        num.append(_sub(_log(ask_eff), _log(bid_eff)))
        den_raw.append(_sub(_log(askq1[t]), _log(bidq1[t])))
    den_ff = _ffill(den_raw)
    den = []
    for t in range(n):
        v = _mavg(den_ff, t, lag, 1)
        den.append(0.0 if _nu(v) else v)
    out = [
        None if (_nu(num[t]) or den[t] == 0) else num[t] / den[t]
        for t in range(n)
    ]
    return out, den


def oracle_soir(bidq, askq, lag=20):
    n = len(bidq)
    weights = list(range(10, 0, -1))
    imb_raw = []
    for t in range(n):
        xs, ws = [], []
        for i in range(10):
            b, a = bidq[t][i], askq[t][i]
            if b + a > 0:
                xs.append((b - a) / (b + a))
                ws.append(weights[i])
        imb_raw.append(
            sum(w * x for w, x in zip(ws, xs)) / sum(ws) if ws else None
        )
    imb = [0.0 if _nu(v) else v for v in _ffill(imb_raw)]
    raw = []
    for t in range(n):
        win = imb[max(0, t - (lag - 1)) : t]  # prev(imbalance), 19 rows
        if len(win) >= 2:
            m = sum(win) / len(win)
            s = math.sqrt(sum((v - m) ** 2 for v in win) / len(win))
            raw.append((imb[t] - m) / s if s >= 1e-7 else None)
        else:
            raw.append(None)
    out = [0.0 if _nu(v) else v for v in _ffill(raw)]
    return out, imb


def oracle_tpw(bid1, bidq1, ask1, askq1, acc_amt, acc_vol, lag=20):
    n = len(bid1)
    fv_list, ap_list = [], []
    for t in range(n):
        pb = bid1[t - 1] if t > 0 else None
        pbq = bidq1[t - 1] if t > 0 else None
        pa = ask1[t - 1] if t > 0 else None
        paq = askq1[t - 1] if t > 0 else None
        # bidchg = iif(round(bid-prevbid,2)>0, bidQty,
        #              iif(round(bid-prevbid,2)<0, -prevbidQty, bidQty-prevbidQty))
        d = None if (_nu(bid1[t]) or _nu(pb)) else round(bid1[t] - pb, 2)
        bc = _iif(
            _gt0(d),
            bidq1[t],
            _iif(_lt0(d), None if _nu(pbq) else -pbq, _sub(bidq1[t], pbq)),
        )
        # offerchg = iif(C1>0, prevaskQty, iif(C2<0, askQty, askQty-prevaskQty))
        #   C1 = iif(ask==0, iif(prevask>0,1,0), ask-prevask)
        #   C2 = iif(prevask==0, iif(ask>0,-1,0), iif(ask>0,ask-prevask,1))
        c1 = _iif(_eq0(ask1[t]), _iif(_gt0(pa), 1, 0), _sub(ask1[t], pa))
        c2 = _iif(
            _eq0(pa),
            _iif(_gt0(ask1[t]), -1, 0),
            _iif(_gt0(ask1[t]), _sub(ask1[t], pa), 1),
        )
        oc = _iif(_gt0(c1), paq, _iif(_lt0(c2), askq1[t], _sub(askq1[t], paq)))
        if t == 0:
            ap = None
        else:
            dv = acc_vol[t] - acc_vol[t - 1]
            da = acc_amt[t] - acc_amt[t - 1]
            ap = None if dv == 0 else da / dv
        if _nu(bc) or _nu(oc) or _nu(ap):
            fv = None
        else:
            fv = _div(bc - oc, abs(bc) + abs(oc))
            fv = None if _nu(fv) else fv * ap
        fv_list.append(fv)
        ap_list.append(ap)
    out = []
    for t in range(n):
        sf = _msum(fv_list, t, lag, 1)
        sa = _msum(ap_list, t, lag, 1)
        v = _div(sf, sa)
        out.append(0.0 if _nu(v) else v)
    return out


def oracle_l10_amtdiff(bidpx, bidq):
    """Per-snapshot aligned amount diff on the bid side (rowAlign 'bid')."""
    n = len(bidpx)
    amts = []
    for t in range(n):
        cur = list(bidpx[t])
        prev = list(bidpx[t - 1]) if t > 0 else None
        if prev is None:
            grid = list(cur)  # S12: left=[0..n-1], right=[-1 x n]
        else:
            lo = max(min(cur), min(prev))
            hi = max(max(cur), max(prev))
            grid = sorted(
                {p for p in cur + prev if lo <= p <= hi}, reverse=True
            )
        am = 0.0
        for g in grid:
            cq = bidq[t][cur.index(g)] if g in cur else 0.0
            pq = bidq[t - 1][prev.index(g)] if (prev and g in prev) else 0.0
            am += g * (cq - pq)
        amts.append(am)
    return amts


def oracle_l10(bidpx, bidq, lag=20):
    amts = oracle_l10_amtdiff(bidpx, bidq)
    out = []
    for t in range(len(amts)):
        v = _msum(amts, t, lag, 1)
        out.append(0.0 if _nu(v) else v)
    return out, amts


def oracle_ipt(bidpx, askpx, bidq, askq, lag1=60, lag2=20):
    n = len(bidpx)
    price_raw = []
    for t in range(n):
        num = sum(b * q for b, q in zip(bidpx[t], bidq[t])) + sum(
            a * q for a, q in zip(askpx[t], askq[t])
        )
        den = sum(bidq[t]) + sum(askq[t])
        infer = None if den == 0 else num / den
        bad = bidpx[t][0] <= 0 or askpx[t][0] <= 0
        price_raw.append(None if bad else infer)
    pf = _ffill(price_raw)
    slope0 = []
    for t in range(n):
        if t < lag1 - 1:
            slope0.append(0.0)
            continue
        win = pf[t - lag1 + 1 : t + 1]
        if any(_nu(v) for v in win):
            slope0.append(0.0)
            continue
        ts = list(range(lag1))
        mt = sum(ts) / lag1
        my = sum(win) / lag1
        cov = sum((a - mt) * (b - my) for a, b in zip(ts, win)) / lag1
        var = sum((a - mt) ** 2 for a in ts) / lag1
        slope0.append(cov / var)
    return [_mavg(slope0, t, lag2, 1) for t in range(n)]


# ----------------------------------------------------------------------
# Synthetic dataset (12 snapshots, 2024-06-28, symbol 600000, SSE flavor).
# All rows pass the Phase-0 SSE filter. Times: 09:30:00 + 3s*t.
# ----------------------------------------------------------------------

DAY = "2024-06-28"


def _px(start: float, n: int = 10, step: float = -0.01) -> List[float]:
    return [round(start + i * step, 2) for i in range(n)]


def _synthetic_rows():
    """Returns list of dicts with bid/ask px/qty + cumulative acc fields."""
    ask_up = lambda first: _px(first, 10, 0.01)
    rows = []
    # t=0 initial book
    rows.append(dict(
        bp=_px(9.10), bq=[1000, 900, 800, 700, 600, 500, 400, 300, 200, 100],
        ap=ask_up(9.11), aq=[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        amt=0.0, vol=0.0,
    ))
    # t=1 price unchanged, qty up at 9.10 (+200) and 9.05 (+50)
    rows.append(dict(
        bp=_px(9.10), bq=[1200, 900, 800, 700, 600, 550, 400, 300, 200, 100],
        ap=ask_up(9.11), aq=[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        amt=91000.0, vol=10000.0,
    ))
    # t=2 bid1 up by 0.01 (new price 9.11 enters top; 9.01 leaves bottom)
    rows.append(dict(
        bp=_px(9.11), bq=[500, 1200, 900, 800, 700, 600, 550, 400, 300, 200],
        ap=ask_up(9.12), aq=[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        amt=91000.0, vol=10000.0,
    ))
    # t=3 bid1 withdrawn (best bid drops 9.11 -> 9.08; 9.11/9.10/9.09 leave top)
    rows.append(dict(
        bp=_px(9.08), bq=[800, 700, 600, 550, 400, 300, 200, 100, 80, 60],
        ap=ask_up(9.09), aq=[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        amt=227750.0, vol=25000.0,
    ))
    # t=4 new price 9.09 re-enters top; per-price alignment across the shift
    rows.append(dict(
        bp=_px(9.09), bq=[350, 800, 700, 600, 550, 400, 300, 200, 100, 80],
        ap=ask_up(9.10), aq=[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        amt=227750.0, vol=25000.0,
    ))
    # t=5 zero qty / empty level: 9.08 +50; 9.00 slot becomes 0-price tail
    rows.append(dict(
        bp=_px(9.09, 9) + [0.00],
        bq=[350, 850, 700, 600, 550, 400, 300, 200, 100, 0],
        ap=ask_up(9.10), aq=[500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        amt=364000.0, vol=40000.0,
    ))
    # t=6..t=11 flat book; ask1 qty set equal to bid1 qty (350) so that
    # log(aq1)-log(bq1) == 0 (timeWeightedOrderSlope small-denominator audit)
    for k in range(6):
        rows.append(dict(
            bp=_px(9.09, 9) + [0.00],
            bq=[350, 800, 700, 600, 550, 400, 300, 200, 100, 0],
            ap=ask_up(9.10), aq=[350, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
            amt=364000.0 + 0.0 * k, vol=40000.0 + 0.0 * k,
        ))
    # interleave trades for avgprice (deltas): t=7,9,11 have trades
    rows[7]["amt"], rows[7]["vol"] = 455000.0, 50000.0
    rows[8]["amt"], rows[8]["vol"] = 455000.0, 50000.0
    rows[9]["amt"], rows[9]["vol"] = 546600.0, 60000.0
    rows[10]["amt"], rows[10]["vol"] = 546600.0, 60000.0
    rows[11]["amt"], rows[11]["vol"] = 637000.0, 70000.0
    # t=6: also give the flat segment a trade so avgprice exists
    rows[6]["amt"], rows[6]["vol"] = 364000.0, 40000.0
    return rows


# Hand-calculated level10_Diff amtDiff chain (see test header comments):
# t=0 first row: full bid notional = sum(px*qty) = 49885
# t=1 9.10*(1200-1000) + 9.05*(550-500) = 1820 + 452.5 = 2272.5
# t=2 9.11*(500-0) = 4555  (9.01 leaves grid via lower bound)
# t=3 -(9.11*500 + 9.10*1200 + 9.09*900) = -23656
# t=4 9.09*350 = 3181.5  (new price entry; naive depth.diff would fail)
# t=5 9.08*50 - 9.00*80 = 454 - 720 = -266  (0-price tail excluded)
# t=6 9.08*(800-850) = -454  (0-tail in both books: 0*0 contribution)
# t=7..11 flat -> 0
EXPECTED_AMTDIFF = [
    49885.0, 2272.5, 4555.0, -23656.0, 3181.5, -266.0, -454.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
]

SYNTH_TIMES = [f"{DAY} 09:30:{3*t:02d}.000000" for t in range(12)]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _synth_frame() -> pd.DataFrame:
    rows = _synthetic_rows()
    rec = {"exch_time": pd.to_datetime(SYNTH_TIMES)}
    for i in range(10):
        rec[f"bp{i+1}"] = [r["bp"][i] for r in rows]
        rec[f"ap{i+1}"] = [r["ap"][i] for r in rows]
        rec[f"bv{i+1}"] = [float(r["bq"][i]) for r in rows]
        rec[f"av{i+1}"] = [float(r["aq"][i]) for r in rows]
    rec["acc_amount"] = [r["amt"] for r in rows]
    rec["acc_volume"] = [r["vol"] for r in rows]
    return pd.DataFrame(rec)


def _oracle_all(frame: pd.DataFrame) -> pd.DataFrame:
    bidpx = frame[[f"bp{i+1}" for i in range(10)]].values.tolist()
    askpx = frame[[f"ap{i+1}" for i in range(10)]].values.tolist()
    bidq = frame[[f"bv{i+1}" for i in range(10)]].values.tolist()
    askq = frame[[f"av{i+1}" for i in range(10)]].values.tolist()
    bid1 = [r[0] for r in bidpx]
    ask1 = [r[0] for r in askpx]
    bidq1 = [r[0] for r in bidq]
    askq1 = [r[0] for r in askq]
    twos, _ = oracle_twos(bid1, bidq1, ask1, askq1)
    soir, _ = oracle_soir(bidq, askq)
    tpw = oracle_tpw(
        bid1, bidq1, ask1, askq1,
        frame["acc_amount"].tolist(), frame["acc_volume"].tolist(),
    )
    l10, amtdiff = oracle_l10(bidpx, bidq)
    ipt = oracle_ipt(bidpx, askpx, bidq, askq)
    return pd.DataFrame(
        {
            "exch_time": frame["exch_time"],
            "orc_twos": twos,
            "orc_soir": soir,
            "orc_tpw": tpw,
            "orc_l10": l10,
            "orc_ipt": ipt,
            "orc_l10_amtdiff": amtdiff,
        }
    )


def _run_ddb(frame: pd.DataFrame, session) -> pd.DataFrame:
    session.upload({"chrows": frame})
    return session.run(OFFICIAL_FUNCS_DOS + "\n" + DDB_FACTOR_SQL)


def _ch_values_source(frame: pd.DataFrame) -> str:
    def arr(values) -> str:
        return "[" + ",".join(repr(float(v)) for v in values) + "]"

    tuples = []
    for _, r in frame.iterrows():
        bp = arr(r[f"bp{i+1}"] for i in range(10))
        ap = arr(r[f"ap{i+1}"] for i in range(10))
        bv = arr(r[f"bv{i+1}"] for i in range(10))
        av = arr(r[f"av{i+1}"] for i in range(10))
        ts = pd.Timestamp(r["exch_time"]).strftime("%Y-%m-%d %H:%M:%S.%f")
        tuples.append(
            f"('600000','{ts}',{bp},{ap},{bv},{av},"
            f"{float(r['acc_amount'])!r},{float(r['acc_volume'])!r})"
        )
    structure = (
        "Symbol String, ts String, BidPrices Array(Float64), "
        "AskPrices Array(Float64), BidVolumes Array(Float64), "
        "AskVolumes Array(Float64), AccAmount Float64, AccVolume Float64"
    )
    return (
        "(SELECT Symbol, "
        "parseDateTime64BestEffort(ts, 6, 'Asia/Shanghai') AS ExchTime, "
        "BidPrices, AskPrices, BidVolumes, AskVolumes, AccAmount, AccVolume "
        f"FROM values('{structure}', " + ",".join(tuples) + "))"
    )


def _compare(name: str, got, want, rtol=1e-9, atol=1e-9) -> bool:
    g = np.asarray(got, dtype=float)
    w = np.asarray(want, dtype=float)
    both_nan = np.isnan(g) & np.isnan(w)
    close = np.isclose(g, w, rtol=rtol, atol=atol, equal_nan=True)
    ok = bool((close | both_nan).all())
    diff = np.abs(g - w)
    diff[both_nan] = 0.0
    finite = np.isfinite(diff)
    max_abs = float(diff[finite].max()) if finite.any() else 0.0
    denom = np.maximum(np.abs(w), 1e-30)
    max_rel = float((diff / denom)[finite].max()) if finite.any() else 0.0
    nan_mismatch = int((np.isnan(g) != np.isnan(w)).sum())
    status = "OK " if ok else "FAIL"
    print(
        f"  [{status}] {name:<58} max_abs={max_abs:.3e} "
        f"max_rel={max_rel:.3e} nan_mismatch={nan_mismatch}"
    )
    return ok


def _fetch_base_extract(table: str, suffix: str, exchange: str, symbol: str):
    client = connect_hf_client()
    try:
        base = _base_extract_sql(
            table=table, exchange_suffix=suffix, exchange=exchange,
            start=DAY, end=DAY, symbols=[symbol],
        )
        keep = (
            "exch_time, bid_px, ask_px, bid_vol, ask_vol, "
            "acc_amount, acc_volume"
        )
        df = client.query_df(
            f"SELECT {keep} FROM (\n{base}\n) ORDER BY exch_time"
        )
    finally:
        client.close()
    rec = {
        "exch_time": pd.to_datetime(df["exch_time"]).dt.tz_localize(None),
        "acc_amount": df["acc_amount"].astype(float),
        "acc_volume": df["acc_volume"].astype(float),
    }
    for i in range(10):
        rec[f"bp{i+1}"] = df["bid_px"].map(lambda a, i=i: float(a[i]))
        rec[f"ap{i+1}"] = df["ask_px"].map(lambda a, i=i: float(a[i]))
        rec[f"bv{i+1}"] = df["bid_vol"].map(lambda a, i=i: float(a[i]))
        rec[f"av{i+1}"] = df["ask_vol"].map(lambda a, i=i: float(a[i]))
    return pd.DataFrame(rec)


# ----------------------------------------------------------------------
# Test parts
# ----------------------------------------------------------------------


def part_a_hand_calc(oracle: pd.DataFrame) -> bool:
    print("== Part A: synthetic vs hand-calculated level10_Diff amtDiff ==")
    ok = _compare(
        "amtDiff chain (hand literals)",
        oracle["orc_l10_amtdiff"].tolist(),
        EXPECTED_AMTDIFF,
        rtol=1e-9,
        atol=1e-6,
    )
    # msum(20,1) with 12 rows -> cumulative sum; final level10_diff value
    cum = np.cumsum(EXPECTED_AMTDIFF)
    ok &= _compare(
        "level10_diff cumulative msum (hand literals)",
        oracle["orc_l10"].tolist(),
        cum.tolist(),
        rtol=1e-9,
        atol=1e-6,
    )
    return ok


def part_b_ddb_vs_oracle(oracle: pd.DataFrame, ddb_res: pd.DataFrame) -> bool:
    print("== Part B: verbatim DDB functions vs independent oracle (synthetic) ==")
    pairs = [
        ("timeWeightedOrderSlope", ddb_res["ddb_twos"], oracle["orc_twos"]),
        ("wavgSOIR", ddb_res["ddb_soir"], oracle["orc_soir"]),
        ("traPriceWavgNetBuyQuoteVolRatio", ddb_res["ddb_tpw"], oracle["orc_tpw"]),
        ("level10_Diff", ddb_res["ddb_l10"], oracle["orc_l10"]),
        ("level10_InferPriceTrend", ddb_res["ddb_ipt"], oracle["orc_ipt"]),
    ]
    ok = True
    for name, d, o in pairs:
        d_list = [None if _nu(x) else float(x) for x in np.asarray(d, dtype=float)]
        o_list = [np.nan if _nu(x) else x for x in o]
        ok &= _compare(f"DDB {name}", d_list, o_list, rtol=1e-9, atol=1e-9)
    return ok


def part_c_ch_vs_oracle(oracle: pd.DataFrame, synth_frame: pd.DataFrame) -> bool:
    print("== Part C: ClickHouse pipeline vs oracle (synthetic, values() source) ==")
    client = connect_hf_client()
    try:
        source = _ch_values_source(synth_frame)
        sql = ddb_snapshot_series_sql(
            table="SSE_AL_SSL2_EXG", exchange_suffix=".SH", exchange="SSE",
            start=DAY, end=DAY, symbols=["600000.SH"], raw_source_sql=source,
        )
        ch = client.query_df(sql)
        daily_sql = ddb_snapshot_daily_sql(
            table="SSE_AL_SSL2_EXG", exchange_suffix=".SH", exchange="SSE",
            start=DAY, end=DAY, symbols=["600000.SH"], raw_source_sql=source,
        )
        daily = client.query_df(daily_sql)
    finally:
        client.close()
    assert len(ch) == 12, f"expect 12 synthetic rows, got {len(ch)}"
    ch = ch.sort_values("exch_time").reset_index(drop=True)
    ok = True
    colmap = [
        ("timeWeightedOrderSlope", "time_weighted_order_slope", "orc_twos"),
        ("wavgSOIR", "wavg_soir", "orc_soir"),
        ("traPriceWavgNetBuyQuoteVolRatio",
         "tra_price_weighted_net_buy_quote_volume_ratio", "orc_tpw"),
        ("level10_Diff", "level10_diff_buy", "orc_l10"),
        ("level10_InferPriceTrend", "level10_infer_price_trend", "orc_ipt"),
    ]
    for name, ch_col, orc_col in colmap:
        got = [None if _nu(x) else float(x) for x in ch[ch_col]]
        want = [np.nan if _nu(x) else x for x in oracle[orc_col]]
        ok &= _compare(f"CH {name}", got, want, rtol=1e-9, atol=1e-9)
    # daily aggregation sanity: all 12 rows in minute 09:30 -> 1 minute,
    # daily mean == last snapshot's value
    row = daily.iloc[0]
    print("  -- daily aggregation on synthetic (single minute) --")
    checks = [
        ("time_weighted_order_slope_mean", "orc_twos"),
        ("wavg_soir_mean", "orc_soir"),
        ("tra_price_weighted_net_buy_quote_volume_ratio_mean", "orc_tpw"),
        ("level10_diff_buy_mean", "orc_l10"),
        ("level10_infer_price_trend_mean", "orc_ipt"),
    ]
    for dcol, ocol in checks:
        last = oracle[ocol].iloc[-1]
        got = row[dcol]
        match = (_nu(last) and pd.isna(got)) or (
            not _nu(last) and abs(got - last) <= 1e-9 * max(1.0, abs(last))
        )
        print(
            f"  [{'OK ' if match else 'FAIL'}] daily mean {dcol:<55} "
            f"got={got!r} want(last snapshot)={last!r}"
        )
        ok &= bool(match)
    ok &= bool(row["valid_minute_count"] == 1)
    ok &= bool(row["valid_snapshot_count"] == 12)
    print(
        f"  [{'OK ' if row['valid_minute_count'] == 1 else 'FAIL'}] "
        f"valid_minute_count == 1 (got {row['valid_minute_count']})"
    )
    return ok


def part_d_real(symbol: str, table: str, suffix: str, exchange: str,
                session) -> bool:
    print(f"== Part D: real data {symbol} {DAY} — CH vs verbatim DDB vs oracle ==")
    frame = _fetch_base_extract(table, suffix, exchange, symbol)
    print(f"  base extract rows: {len(frame)}")
    ddb_res = _run_ddb(frame, session)
    oracle = _oracle_all(frame)
    client = connect_hf_client()
    try:
        sql = ddb_snapshot_series_sql(
            table=table, exchange_suffix=suffix, exchange=exchange,
            start=DAY, end=DAY, symbols=[symbol],
        )
        ch = client.query_df(sql)
    finally:
        client.close()
    ch = ch.sort_values("exch_time").reset_index(drop=True)
    assert len(ch) == len(frame), (
        f"row count mismatch: CH {len(ch)} vs base {len(frame)}"
    )
    ok = True
    colmap = [
        ("wavgSOIR", "ddb_soir", "wavg_soir", "orc_soir", 1e-9),
        ("traPriceWavgNetBuyQuoteVolRatio", "ddb_tpw",
         "tra_price_weighted_net_buy_quote_volume_ratio", "orc_tpw", 1e-9),
        ("level10_Diff", "ddb_l10", "level10_diff_buy", "orc_l10", 1e-9),
        ("level10_InferPriceTrend", "ddb_ipt",
         "level10_infer_price_trend", "orc_ipt", 1e-6),
    ]
    for name, dcol, chcol, ocol, tol in colmap:
        d = np.asarray(
            [np.nan if _nu(x) else float(x) for x in ddb_res[dcol]], dtype=float
        )
        c = np.asarray(ch[chcol], dtype=float)
        o = np.asarray(
            [np.nan if _nu(x) else x for x in oracle[ocol]], dtype=float
        )
        ok &= _compare(f"{symbol} CH vs DDB {name}", c, d, rtol=tol, atol=tol)
        ok &= _compare(f"{symbol} CH vs oracle {name}", c, o, rtol=tol, atol=tol)
    ok &= _twos_component_check(symbol, frame, ch, ddb_res, oracle)
    return ok


# Empirically bounded ClickHouse log() implementation noise (company CH
# 23.2.3.17 vs libm/DolphinDB, measured on 2024-06-28 full books):
#   |log_ch - log_ref| <= 1.6e-11 on prices, <= 3.1e-9 on log-qty diffs
# The official timeWeightedOrderSlope divides by a near-zero denominator
# (formula_mapping.md §1 不稳定域), which amplifies this irreducible engine
# noise into the final slope.  Semantics are verified component-by-component
# (num / den_raw / den_final); the slope itself is checked with a tolerance
# derived from first-order error propagation, so any genuine semantic bug
# (O(1) deviation) still fails while engine fp noise is tolerated.
TWOS_NUM_TOL = 5e-11
TWOS_DEN_TOL = 5e-9


def _twos_component_check(symbol: str, frame, ch, ddb_res, oracle) -> bool:
    bid1 = frame["bp1"].to_numpy(dtype=float)
    ask1 = frame["ap1"].to_numpy(dtype=float)
    bq1 = frame["bv1"].to_numpy(dtype=float)
    aq1 = frame["av1"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ask_eff = np.where(ask1 != 0, ask1, bid1)
        bid_eff = np.where(bid1 != 0, bid1, ask1)
        num_ref = np.where(
            (ask_eff > 0) & (bid_eff > 0),
            np.log(ask_eff) - np.log(bid_eff),
            np.nan,
        )
        den_raw_ref = np.where(
            (bq1 > 0) & (aq1 > 0), np.log(aq1) - np.log(bq1), np.nan
        )
    den_ff = pd.Series(den_raw_ref).ffill().to_numpy()
    n = len(den_ff)
    den_ref = np.zeros(n)
    for t in range(n):
        w = den_ff[max(0, t - 19) : t + 1]
        w = w[~np.isnan(w)]
        den_ref[t] = w.mean() if len(w) >= 1 else 0.0

    num_ch = ch["twos_num"].to_numpy(dtype=float)
    den_raw_ch = ch["twos_den_raw"].to_numpy(dtype=float)
    den_ch = ch["twos_den_final"].to_numpy(dtype=float)
    ok = True
    ok &= _compare(f"{symbol} twos numerator", num_ch, num_ref,
                   rtol=0.0, atol=TWOS_NUM_TOL)
    ok &= _compare(f"{symbol} twos den_raw", den_raw_ch, den_raw_ref,
                   rtol=0.0, atol=1e-8)
    ok &= _compare(f"{symbol} twos den_final", den_ch, den_ref,
                   rtol=0.0, atol=TWOS_DEN_TOL)

    ref = np.asarray(
        [np.nan if _nu(x) else float(x) for x in ddb_res["ddb_twos"]],
        dtype=float,
    )
    got = ch["time_weighted_order_slope"].to_numpy(dtype=float)
    den_abs = np.maximum(np.abs(den_ch), 1e-30)
    tol = (np.abs(ref) + 1e-12) * TWOS_DEN_TOL / den_abs \
        + TWOS_NUM_TOL / den_abs + 1e-12
    diff = np.abs(got - ref)
    both_nan = np.isnan(got) & np.isnan(ref)
    nan_mismatch = int((np.isnan(got) != np.isnan(ref)).sum())
    passed = bool(((diff <= tol) | both_nan).all()) and nan_mismatch == 0
    viol = int((~((diff <= tol) | both_nan)).sum())
    print(
        f"  [{'OK ' if passed else 'FAIL'}] {symbol} CH vs DDB "
        f"timeWeightedOrderSlope (propagation-aware)  viol={viol} "
        f"nan_mismatch={nan_mismatch} max_abs={np.nanmax(diff):.3e}"
    )
    ok &= passed
    return ok


def part_e_ddb_only_edges(session) -> bool:
    """ask1=0 / bidq1=0 rows: unreachable after Phase-0 CH filters (dead
    branch in production) — still verify oracle == verbatim DDB semantics."""
    print("== Part E: DDB-only edge rows (ask1=0, bidQty1=0) ==")
    n = 4
    rec = {"exch_time": pd.to_datetime(
        [f"{DAY} 09:30:0{t}.000000" for t in range(n)])}
    bp = [_px(9.10), _px(9.10), _px(9.10), _px(9.10)]
    ap = [[0.0] + _px(9.11, 9), _px(9.11), _px(9.11), _px(9.11)]
    bq = [[100.0 * (i + 1) for i in range(10)]] * n
    aq = [[0.0] + [100.0 * (i + 1) for i in range(1, 10)],
          [100.0 * (i + 1) for i in range(10)],
          [100.0 * (i + 1) for i in range(10)],
          [100.0 * (i + 1) for i in range(10)]]
    bq[1] = [0.0] + [100.0 * (i + 1) for i in range(1, 10)]
    for i in range(10):
        rec[f"bp{i+1}"] = [r[i] for r in bp]
        rec[f"ap{i+1}"] = [r[i] for r in ap]
        rec[f"bv{i+1}"] = [float(r[i]) for r in bq]
        rec[f"av{i+1}"] = [float(r[i]) for r in aq]
    rec["acc_amount"] = [0.0, 1e4, 1e4, 2e4]
    rec["acc_volume"] = [0.0, 1e3, 1e3, 2e3]
    frame = pd.DataFrame(rec)
    ddb_res = _run_ddb(frame, session)
    oracle = _oracle_all(frame)
    ok = True
    for name, dcol, ocol in [
        ("timeWeightedOrderSlope", "ddb_twos", "orc_twos"),
        ("wavgSOIR", "ddb_soir", "orc_soir"),
        ("traPriceWavgNetBuyQuoteVolRatio", "ddb_tpw", "orc_tpw"),
        ("level10_Diff", "ddb_l10", "orc_l10"),
        ("level10_InferPriceTrend", "ddb_ipt", "orc_ipt"),
    ]:
        d = [np.nan if _nu(x) else float(x) for x in np.asarray(
            ddb_res[dcol], dtype=float)]
        o = [np.nan if _nu(x) else x for x in oracle[ocol]]
        ok &= _compare(f"edge DDB vs oracle {name}", d, o, rtol=1e-9, atol=1e-9)
    return ok


def main() -> int:
    synth = _synth_frame()
    oracle = _oracle_all(synth)
    results = {}
    results["A_hand_calc"] = part_a_hand_calc(oracle)

    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    try:
        ddb_res = _run_ddb(synth, session)
        results["B_ddb_vs_oracle_synth"] = part_b_ddb_vs_oracle(oracle, ddb_res)
        results["E_ddb_only_edges"] = part_e_ddb_only_edges(session)
        results["D_real_600000.SH"] = part_d_real(
            "600000.SH", "SSE_AL_SSL2_EXG", ".SH", "SSE", session)
        results["D_real_000001.SZ"] = part_d_real(
            "000001.SZ", "SZSE_AL_SSL2_EXG", ".SZ", "SZSE", session)
    finally:
        session.close()
    results["C_ch_vs_oracle_synth"] = part_c_ch_vs_oracle(oracle, synth)

    print("\n== SUMMARY ==")
    all_ok = True
    for k, v in results.items():
        print(f"  [{'OK ' if v else 'FAIL'}] {k}")
        all_ok &= v
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
