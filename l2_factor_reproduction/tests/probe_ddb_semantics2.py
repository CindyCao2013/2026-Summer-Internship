"""Round-2 probes: rowWavg NULL handling, rowAt array-vector form, prev on
array vector columns, rowSum NULL handling, nullFill pairwise, round-sign.
"""
from __future__ import annotations

import os
import sys

import dolphindb as ddb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from COMMON_CONST import DATA_DB_CONN


PROBES = {
    # rowWavg with one NULL element: renormalize weights or propagate NULL?
    "rowwavg_null_elem": (
        "x=array(DOUBLE[],0,10).append!([NULL,0.2,-0.4,0.1,0,0.3,-0.1,0.05,NULL,0.6]);"
        "t=table(x as x);select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as v from t"
    ),
    "rowwavg_null_elem_byhand_drop": (
        "(9*0.2+8*(-0.4)+7*0.1+6*0+5*0.3+4*(-0.1)+3*0.05+1*0.6)\\(9+8+7+6+5+4+3+1)."
    ),
    "rowwavg_null_elem_byhand_keep55": (
        "(9*0.2+8*(-0.4)+7*0.1+6*0+5*0.3+4*(-0.1)+3*0.05+1*0.6)\\55."
    ),
    "rowwavg_all_null": (
        "x=array(DOUBLE[],0,10).append!([NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL]);"
        "t=table(x as x);select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as v from t"
    ),
    # rowAt with array-vector index containing -1
    "rowat_neg1": (
        "x=array(DOUBLE[],0,5).append!([9.01 9.00 8.99 8.98 8.97]);"
        "idx=array(INT[],0,5).append!([[-1,0,2,-1,4]]);"
        "x.rowAt(idx)"
    ),
    # prev() on array vector column (first row NULL?)
    "prev_arrayvec_first": (
        "p=array(DOUBLE[],0,3).append!([1. 2 3, 4. 5 6, 7. 8 9]);"
        "t=table(p as p);select p, prev(p) as pp from t"
    ),
    # rowAlign when prev row is NULL
    "rowalign_null_prev": (
        "p=array(DOUBLE[],0,3).append!([9.01 9.00 8.99, 9.02 9.01 9.00]);"
        "t=table(p as p);"
        "select rowAlign(p, prev(p), 'bid') as aligned from t"
    ),
    # rowSum null handling
    "rowsum_nulls": (
        "x=array(DOUBLE[],0,3).append!([NULL 3 NULL, NULL NULL NULL, 1. 2 3]);"
        "t=table(x as x);select rowSum(x) as s from t"
    ),
    # nullFill pairwise (A nulls filled from B)
    "nullfill_pairwise": "nullFill([1.,NULL,3.], [10.,20.,30.])",
    # round sign comparisons used by traPriceWeighted
    "round_sign": "round([0.0049, -0.0049, 0.0, NULL], 2)",
    # mavg over prev() window: replicate wavgSOIR window alignment
    # imbalance = [0.1,0.2,0.3,0.4,0.5]; lag=5 -> mavg(prev(x),4,2)
    "wavgsoir_window_align": (
        "x=[0.1,0.2,0.3,0.4,0.5]; [mavg(prev(x), 4, 2), mstdp(prev(x)*1000000, 4, 2)\\1000000]"
    ),
    "wavgsoir_window_byhand": "avg([0.1,0.2,0.3,0.4])",
    # linearTimeTrend slope by hand: x=[10,11,13,14,15], window 5
    "ltt_byhand": (
        "x=[10.,11.,13.,14.,15.]; t=0..4; "
        "[linearTimeTrend(x,5).at(1)[4], covar(t,x)\\var(t)]"
    ),
    # mavg on scalar series produced by iif(...).ffill().nullFill(0) chain (wavgSOIR tail)
    "wavgsoir_tail": (
        "x=[NULL,1.,NULL,2.]; x.ffill().nullFill(0)"
    ),
    # deltas on cumulative trade value/volume
    "deltas_div": "deltas([100.,150.,150.,400.])\\deltas([10.,20.,20.,50.])",
    # cumsum-based state function sanity for delayedTrade (phase 4 reference only)
    "int_div": "7\\2",
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
