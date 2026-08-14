"""Round-3 probes: rowWavg/rowSum NULL semantics (00F literals) and
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
        "x=array(DOUBLE[],0,10).append!([00F,0.2,-0.4,0.1,0,0.3,-0.1,0.05,00F,0.6]);"
        "t=table(x as x);select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as v from t"
    ),
    "rowwavg_null_elem_renorm43": "(9*0.2+8*(-0.4)+7*0.1+6*0+5*0.3+4*(-0.1)+3*0.05+1*0.6)\\43.",
    "rowwavg_null_elem_keep55": "(9*0.2+8*(-0.4)+7*0.1+6*0+5*0.3+4*(-0.1)+3*0.05+1*0.6)\\55.",
    "rowwavg_allnull_row": (
        "x=array(DOUBLE[],0,10).append!([00F,00F,00F,00F,00F,00F,00F,00F,00F,00F]);"
        "t=table(x as x);select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as v from t"
    ),
    "rowsum_with_nulls": (
        "x=array(DOUBLE[],0,3).append!([00F, 3., 00F]);"
        "t=table(x as x);select rowSum(x) as s from t"
    ),
    "rowsum_all_null": (
        "x=array(DOUBLE[],0,3).append!([00F, 00F, 00F]);"
        "t=table(x as x);select rowSum(x) as s from t"
    ),
    # rowAlign on first row (prev is NULL vector) — direct assignment, print both
    "rowalign_firstrow": (
        "p=array(DOUBLE[],0,3).append!([9.01 9.00 8.99, 9.02 9.01 9.00]);"
        "pp=prev(p);"
        "l,r=rowAlign(p, pp, 'bid');"
        "['left'=l, 'right'=r, 'prev'=pp]"
    ),
    # full doc example cross-check (bid): expected from documentation
    "rowalign_doc_bid": (
        "left=array(DOUBLE[],0,5).append!([9.01 9.00 8.99 8.98 8.97, 9.00 8.98 8.97 8.96 8.95, 8.99 8.97 8.95 8.93 8.91]);"
        "right=array(DOUBLE[],0,5).append!([9.02 9.01 9.00 8.99 8.98, 9.01 9.00 8.99 8.98 8.97, 9.00 8.98 8.97 8.96 8.95]);"
        "l,r=rowAlign(left,right,'bid'); ['l'=l,'r'=r]"
    ),
    # qtyDiff doc example: expected [[-8,-2,-5,3,12],[-10,7,-15,-3,7],[-12,7,-15,-12,-21,-10]]
    "rowat_qtydiff_doc": (
        "left=array(DOUBLE[],0,5).append!([9.01 9.00 8.99 8.98 8.97, 9.00 8.98 8.97 8.96 8.95, 8.99 8.97 8.95 8.93 8.91]);"
        "right=array(DOUBLE[],0,5).append!([9.02 9.01 9.00 8.99 8.98, 9.01 9.00 8.99 8.98 8.97, 9.00 8.98 8.97 8.96 8.95]);"
        "lq=array(INT[],0,5).append!([10 5 15 20 13, 12 15 20 21 18, 7 8 9 9 10]);"
        "rq=array(INT[],0,5).append!([8 12 10 12 8, 10 5 15 18 13, 12 15 20 21 19]);"
        "l,r=rowAlign(left,right,'bid');"
        "lq.rowAt(l).nullFill(0) - rq.rowAt(r).nullFill(0)"
    ),
    # ask-side sanity (documenting why zeros are risky for ask alignment)
    "rowalign_ask_zero_tail": (
        "left=array(DOUBLE[],0,5).append!([9.02 9.03 9.04 0 0]);"
        "right=array(DOUBLE[],0,5).append!([9.01 9.02 9.03 9.04 9.05]);"
        "l,r=rowAlign(left,right,'ask'); ['l'=l,'r'=r]"
    ),
    # iif with array-vector condition containing NULL (0/0 case in imbalance)
    "iif_arr_null_cond": (
        "b=array(DOUBLE[],0,3).append!([10., 0., 5.]); a=array(DOUBLE[],0,3).append!([20., 0., 5.]);"
        "x=(b-a)\\(b+a);"
        "t=table(x as x); select rowWavg(x, 1 1 1) as w from t"
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
