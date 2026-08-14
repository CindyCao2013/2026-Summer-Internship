#!/usr/bin/env python
"""Rolling train + predict. Default variant is paper V1 (slow, needs GPU)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphanet.data import load_eod_from_ddb
from alphanet.paths import FACTORS, ensure_result_dirs
from alphanet.rolling import rolling_predict
from alphanet.variants import get_config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v1")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--max-folds", type=int, default=None)
    args = p.parse_args()
    cfg = get_config(args.variant)
    if args.start:
        cfg = replace(cfg, start=args.start)
    if args.end:
        cfg = replace(cfg, end=args.end)
    ensure_result_dirs()
    panel = load_eod_from_ddb(cfg.start, cfg.end)
    factor = rolling_predict(panel, cfg, n_seeds=args.n_seeds, max_folds=args.max_folds)
    path = FACTORS / "{}_factor.parquet".format(cfg.variant)
    print("wrote", path, "shape", factor.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
