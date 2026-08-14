"""Probe DolphinDB engine semantics needed for faithful CH replication.

Run from repo root:
    /opt/conda/anaconda3/envs/base_93/bin/python \\
        l2_factor_reproduction/tests/probe_ddb_semantics.py
"""
from __future__ import annotations

import os
import sys

import dolphindb as ddb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from COMMON_CONST import DATA_DB_CONN


PROBES = {
    # --- moving window null / minPeriods semantics ---
    "mavg_null_window": "mavg([1.,NULL,3,NULL,5], 3, 1)",
    "mavg_minp2": "mavg([1.,NULL,3,NULL,5], 3, 2)",
    "msum_null": "msum([1.,NULL,3,NULL,5], 3, 1)",
    "mstdp_1elem": "mstdp([5.,NULL,7], 3, 1)",
    "mstdp_scaled": "mstdp([0.1,0.2,0.3]*1000000, 3, 2)\\1000000",
    # --- prev / deltas on scalars and array vectors ---
    "prev_scalar_first": "prev([1.,2.,3.])[0]",
    "deltas_first": "deltas([100.,110.,121.])",
    # --- rowWavg with NULL elements inside the array vector ---
    "rowwavg_null_elem": (
        "t=table([array(DOUBLE[],0,10).append!([NULL,0.2,-0.4,0.1,0,0.3,-0.1,0.05,NULL,0.6])] as x);"
        "select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as v from t"
    ),
    "rowwavg_plain": (
        "t=table([array(DOUBLE[],0,10).append!([0.1,0.2,-0.4,0.1,0,0.3,-0.1,0.05,0.02,0.6])] as x);"
        "select rowWavg(x, 10 9 8 7 6 5 4 3 2 1) as v from t"
    ),
    "rowwavg_byhand": "(10*0.1+9*0.2+8*(-0.4)+7*0.1+6*0+5*0.3+4*(-0.1)+3*0.05+2*0.02+1*0.6)\\55.",
    # --- rowAt with -1 index ---
    "rowat_neg1": (
        "x=array(DOUBLE[],0,5).append!([9.01 9.00 8.99 8.98 8.97]); x.rowAt([-1,0,2])"
    ),
    # --- prev() on array vector column: first row ---
    "prev_arrayvec_first": (
        "t=table([array(DOUBLE[],0,3).append!([1. 2 3, 4. 5 6, 7. 8 9])] as p);"
        "select prev(p) as pp from t"
    ),
    # --- linearTimeTrend warm-up ---
    "ltt_warmup": "linearTimeTrend([1.,2.,3.,4.,5.,6.,7.], 5)",
    "ltt_value_check": (
        "x=[1.,2.,3.,4.,5.,6.,7.]; linearTimeTrend(x,5).at(1)"
    ),
    # --- round() semantics ---
    "round2": "round([1.005, 2.675, 9.9999, -0.0049, 0.005], 2)",
    # --- iif with NULL condition ---
    "iif_null_cond": "iif(NULL>0, 1., 2.)",
    # --- ffill then mavg chain on all-NULL head ---
    "ffill_mavg_head": "ffill([NULL,NULL,2.,4.]).mavg(2,1).nullFill(0)",
    # --- msum minPeriods=1 with NULL head ---
    "msum_null_head": "msum([NULL,NULL,3.,1.], 3, 1).nullFill(0)",
    # --- elementwise div on array vectors producing NULL (0/0) ---
    "arr_div_zero": (
        "b=array(DOUBLE[],0,3).append!([10. 0 5]); a=array(DOUBLE[],0,3).append!([20. 0 5]);"
        "(b-a)\\(b+a)"
    ),
    # --- rowAlign with trailing zeros (empty levels) ---
    "rowalign_zero_tail": (
        "left=array(DOUBLE[],0,5).append!([9.01 9.00 8.99 0 0]);"
        "right=array(DOUBLE[],0,5).append!([9.02 9.01 9.00 8.99 8.98]);"
        "l,r=rowAlign(left,right,'bid'); [l,r]"
    ),
    # --- rowAlign when prev row is NULL vector ---
    "rowalign_null_prev": (
        "t=table([array(DOUBLE[],0,3).append!([9.01 9.00 8.99, 9.02 9.01 9.00])] as p);"
        "l,r=rowAlign(p, prev(p), 'bid'); select l,r from t"
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
