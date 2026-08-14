#!/usr/bin/env python
"""Neutralize the factor and write RankIC + 10-layer reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from alphanet.data import load_eod_from_ddb
from alphanet.evaluate import decile_backtest, ic_test, write_eval_artifacts
from alphanet.neutralize import neutralize_panel
from alphanet.paths import FACTORS
from alphanet.variants import get_config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v1")
    p.add_argument("--factor", default=None)
    args = p.parse_args()
    cfg = get_config(args.variant)
    path = Path(args.factor) if args.factor else FACTORS / "{}_factor.parquet".format(cfg.variant)
    factor = pd.read_parquet(path)
    panel = load_eod_from_ddb(cfg.start, cfg.end)
    neut = neutralize_panel(
        factor,
        industry=panel.industry,
        log_mcap=panel.log_mcap,
        ret_1d=panel.ret_1d,
        turn=panel.features["turn"],
        horizon=cfg.train.horizon,
        min_obs=cfg.eval.min_cs_obs,
    )
    ic_result = ic_test(neut, panel.ret_1d, horizon=cfg.eval.rebalance_every, mask=panel.tradable)
    decile = decile_backtest(neut, panel.ret_1d, eval_cfg=cfg.eval, mask=panel.tradable)
    paths = write_eval_artifacts(cfg.variant, ic_result, decile)
    print(ic_result["summary"])
    print(decile["table"])
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
