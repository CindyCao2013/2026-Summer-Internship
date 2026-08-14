#!/usr/bin/env python
"""Load EOD from DolphinDB and cache under research/results/alphanet_v1/cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphanet.config import PAPER_END, PAPER_START
from alphanet.data import load_eod_from_ddb
from alphanet.paths import ensure_result_dirs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=PAPER_START)
    p.add_argument("--end", default=PAPER_END)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()
    ensure_result_dirs()
    panel = load_eod_from_ddb(args.start, args.end, cache=not args.no_cache)
    print("calendar", panel.calendar[0].date(), "->", panel.calendar[-1].date(), "n=", len(panel.calendar))
    print("symbols", len(panel.symbols))
    print("meta", panel.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
