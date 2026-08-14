"""CLI helpers. Run from the repo root with ``PYTHONPATH=.``."""

from __future__ import annotations

import argparse
from typing import Optional

from alphanet.pipeline import run_synthetic_pipeline, write_run_readme
from alphanet.variants import get_config, list_variants


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaNet reproduction / optimization")
    p.add_argument("--variant", default="smoke", help="v1 / v1_adam / smoke / ...")
    p.add_argument("--list-variants", action="store_true")
    p.add_argument("--n-days", type=int, default=80)
    p.add_argument("--n-stocks", type=int, default=36)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--max-folds", type=int, default=1)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_variants:
        for k, v in list_variants().items():
            print("{:18s} {}".format(k, v))
        return 0
    cfg = get_config(args.variant)
    write_run_readme(cfg)
    if args.variant in ("smoke", "ci") or args.n_days <= 120:
        result = run_synthetic_pipeline(
            cfg,
            n_days=args.n_days,
            n_stocks=args.n_stocks,
            n_seeds=args.n_seeds,
            max_folds=args.max_folds,
        )
        print("variant:", cfg.variant)
        print("factor shape:", result["factor"].shape)
        print("RankIC:", result["ic"]["summary"])
        print("artifacts:", result["paths"])
        return 0
    print("Live DDB run: use alphanet/scripts/run_train.py after run_prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
