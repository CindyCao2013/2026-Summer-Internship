#!/usr/bin/env python
"""Compare AlphaNet synthetic factors with explicit daily factors.

Usage:
  PYTHONPATH=. /opt/conda/anaconda3/bin/python \\
      alphanet/scripts/run_compare_factors.py --variant v1

  PYTHONPATH=. /opt/conda/anaconda3/bin/python \\
      alphanet/scripts/run_compare_factors.py --synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphanet.compare import (
    build_classic_style_factors,
    load_alphanet_factor,
    load_candidate_summary,
    load_pool_factors,
    make_synthetic_alphanet,
    run_comparison,
    select_pool_representatives,
)
from alphanet.config import EvalConfig
from alphanet.data import load_eod_from_ddb, panel_from_synthetic
from alphanet.paths import COMPARE_ROOT, ensure_result_dirs
from alphanet.synthetic import make_synthetic_panel
from alphanet.variants import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_pool(max_pool_factors: int, max_per_family: int):
    summary = load_candidate_summary()
    reps = select_pool_representatives(
        summary, max_per_family=max_per_family, max_total=max_pool_factors
    )
    pool, status = load_pool_factors(reps)
    return pool, status, reps


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaNet vs explicit daily factors")
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--factor", default=None, help="optional AlphaNet parquet path")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run on a synthetic panel (no DDB / no trained AlphaNet required)",
    )
    parser.add_argument("--max-pool-factors", type=int, default=12)
    parser.add_argument("--max-per-family", type=int, default=1)
    parser.add_argument("--style-window", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ensure_result_dirs()
    out_dir = Path(args.output_dir) if args.output_dir else COMPARE_ROOT
    pool, status, reps = _load_pool(args.max_pool_factors, args.max_per_family)
    if reps is not None and not reps.empty:
        logger.info("selected %s pool representatives", len(reps))
    if status is not None and not status.empty:
        n_ok = int((status["status"] == "ok").sum())
        logger.info("pool narrow files loaded=%s / %s", n_ok, len(status))

    if args.synthetic:
        cfg = get_config("smoke")
        synth = make_synthetic_panel(n_days=90, n_stocks=48, seed=11)
        panel = panel_from_synthetic(synth)
        styles = build_classic_style_factors(panel, window=args.style_window)
        alpha = make_synthetic_alphanet(styles, seed=11)
        note = (
            "synthetic demo: AlphaNet = 0.75*cs_z(momentum) + 0.25*noise; "
            "not a trained model. Classic styles are real formulas on synthetic prices."
        )
        result = run_comparison(
            alpha,
            styles,
            pool,
            variant="synthetic",
            ret_1d=panel.ret_1d,
            mask=panel.tradable,
            eval_cfg=EvalConfig(
                n_groups=10,
                rebalance_every=cfg.eval.rebalance_every,
                min_cs_obs=cfg.eval.min_cs_obs,
                fee_one_way=cfg.eval.fee_one_way,
            ),
            output_dir=out_dir,
            pool_status=status,
            data_note=note,
            min_obs=cfg.eval.min_cs_obs,
        )
    else:
        cfg = get_config(args.variant)
        try:
            alpha = load_alphanet_factor(
                cfg.variant, Path(args.factor) if args.factor else None
            )
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            logger.info("先训练/评估生成因子，或使用 --synthetic 跑通对比框架。")
            return 1
        panel = load_eod_from_ddb(cfg.start, cfg.end)
        styles = build_classic_style_factors(panel, window=args.style_window)
        note = "live AlphaNet-{} vs classic EOD styles + candidate_pool_v1 representatives".format(
            cfg.variant
        )
        result = run_comparison(
            alpha,
            styles,
            pool,
            variant=cfg.variant,
            ret_1d=panel.ret_1d,
            mask=panel.tradable,
            eval_cfg=cfg.eval,
            output_dir=out_dir,
            pool_status=status,
            data_note=note,
            min_obs=cfg.eval.min_cs_obs,
        )

    logger.info("report: %s", result["report"])
    logger.info("verdict: %s — %s", result["verdict"][0], result["verdict"][1])
    print(result["corr_summary"].head(10).to_string())
    if result["residual"] is not None:
        print(result["residual"]["comparison"].to_string(index=False))
    print("wrote", result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
