#!/usr/bin/env python
"""CSI500 enhancement grid over active-weight caps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from alphanet.data import load_eod_from_ddb
from alphanet.enhance import run_enhance_grid
from alphanet.paths import ENHANCE, FACTORS
from alphanet.variants import get_config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v1")
    p.add_argument("--factor", default=None)
    args = p.parse_args()
    cfg = get_config(args.variant)
    path = Path(args.factor) if args.factor else FACTORS / "{}_factor_neutral.parquet".format(cfg.variant)
    if not path.exists():
        path = FACTORS / "{}_factor.parquet".format(cfg.variant)
    factor = pd.read_parquet(path)
    panel = load_eod_from_ddb(cfg.start, cfg.end)
    members = panel.index_members.get(cfg.enhance.benchmark)
    if members is None:
        members = pd.DataFrame(1.0, index=factor.index, columns=factor.columns)
        print("WARNING: no index members on panel; using factor universe as benchmark")
    table = run_enhance_grid(
        factor,
        panel.ret_1d,
        members,
        industry=panel.industry,
        log_mcap=panel.log_mcap,
        cfg=cfg.enhance,
    )
    print(table)
    print("wrote", ENHANCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
