#!/usr/bin/env python
"""Run Alpha Selection Engine on completed batch summaries."""

import argparse

import factor_config as cfg
from factor_selection import run_selection_report


def main():
    parser = argparse.ArgumentParser(description="Alpha Selection Engine v1")
    parser.add_argument("--track", default=cfg.TRACK, help="result subfolder track name")
    parser.add_argument("--batch-tag", default=None, help="batch_summary_{tag}.csv suffix")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    batch_tag = args.batch_tag or cfg.resolve_batch_tag(
        args.track, cfg.resolve_factor_list(args.track)
    )
    result_root = cfg.result_root_for(args.track)
    run_selection_report(result_root, batch_tag=batch_tag, top_k=args.top_k)


if __name__ == "__main__":
    main()
