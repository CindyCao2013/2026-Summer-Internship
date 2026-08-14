"""Round-4 probes: rowWavg/rowSum NULL semantics via explicit null assignment,
rowAlign first-row behavior via direct assignment.
"""
from __future__ import annotations

import os
import sys

import dolphindb as ddb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from COMMON_CONST import DATA_DB_CONN


PROBES = {
    "rowwavg_null_elem": (
        "v=[0.1,0.2,-0.4,0.1,0,0.3,-0.1,0.05,0.02,0.6]; v[0]=00F; v[8]=00F;"
        "x=array(DOUBLE[],0,10).append!([v]);"
        "t=table(x as x);select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as w from t"
    ),
    "rowwavg_allnull_row": (
        "v=0.1+0*1..10; v[0:10]=00F;"
        "x=array(DOUBLE[],0,10).append!([v]);"
        "t=table(x as x);select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as w from t"
    ),
    "rowsum_with_nulls": (
        "v=[1.,2.,3.]; v[0]=00F; v[2]=00F;"
        "x=array(DOUBLE[],0,3).append!([v]);"
        "t=table(x as x);select rowSum(x) as s from t"
    ),
    "rowsum_all_null": (
        "v=[1.,2.,3.]; v[0:3]=00F;"
        "x=array(DOUBLE[],0,3).append!([v]);"
        "t=table(x as x);select rowSum(x) as s from t"
    ),
    "rowalign_firstrow": (
        "p=array(DOUBLE[],0,3).append!([9.01 9.00 8.99, 9.02 9.01 9.00]);"
        "pp=prev(p);"
        "l,r=rowAlign(p, pp, 'bid');"
        "[l, r, pp]"
    ),
    "rowalign_doc_bid": (
        "left=array(DOUBLE[],0,5).append!([9.01 9.00 8.99 8.98 8.97, 9.00 8.98 8.97 8.96 8.95, 8.99 8.97 8.95 8.93 8.91]);"
        "right=array(DOUBLE[],0,5).append!([9.02 9.01 9.00 8.99 8.98, 9.01 9.00 8.99 8.98 8.97, 9.00 8.98 8.97 8.96 8.95]);"
        "l,r=rowAlign(left,right,'bid'); [l,r]"
    ),
    "rowalign_ask_zero_tail": (
        "left=array(DOUBLE[],0,5).append!([9.02 9.03 9.04 0 0]);"
        "right=array(DOUBLE[],0,5).append!([9.01 9.02 9.03 9.04 9.05]);"
        "l,r=rowAlign(left,right,'ask'); [l,r]"
    ),
    "rowwavg_with_0div0_level": (
        "b=array(DOUBLE[],0,3).append!([10., 0., 5.]);"
        "a=array(DOUBLE[],0,3).append!([20., 0., 5.]);"
        "x=(b-a)\\(b+a);"
        "t=table(x as x); select x, rowWavg(x, 1 1 1) as w from t"
    ),
    # mavg over an expression that includes prev(): check NULL head length
    "wavgsoir_full_small": (
        "def wavgSOIR(bidQty,askQty,lag=20){\n"
        "  imbalance= rowWavg((bidQty - askQty)\\(bidQty + askQty), 10 9 8 7 6 5 4 3 2 1).ffill().nullFill(0)\n"
        "  mean = mavg(prev(imbalance), (lag-1), 2)\n"
        "  std = mstdp(prev(imbalance) * 1000000, (lag-1), 2) \\ 1000000\n"
        "  return iif(std >= 0.0000001,(imbalance - mean) \\ std, NULL).ffill().nullFill(0)\n"
        "}\n"
        "bq=array(DOUBLE[],0,10).append!([100 90 80 70 60 50 40 30 20 10, 110 90 80 70 60 50 40 30 20 10, 90 90 80 70 60 50 40 30 20 10, 100 100 80 70 60 50 40 30 20 10, 100 90 90 70 60 50 40 30 20 10]);"
        "aq=array(DOUBLE[],0,10).append!([100 90 80 70 60 50 40 30 20 10, 100 90 80 70 60 50 40 30 20 10, 100 90 80 70 60 50 40 30 20 10, 100 90 80 70 60 50 40 30 20 10, 100 90 80 70 60 50 40 30 20 10]);"
        "t=table(bq as bq, aq as aq);"
        "select wavgSOIR(bq, aq, 5) as soir from t"
    ),
}


def main() -> int:
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    try:
        for name, script in PROBES.items():
            try:
                result = s.run(script)
                print(f"== {name} ==\n{result!r}\n")
            except Exception as exc:  # noqa: BLE001
                print(f"== {name} == ERROR: {exc}\n")
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
