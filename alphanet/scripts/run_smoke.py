#!/usr/bin/env python
"""Synthetic smoke: train 1 fold / 1 seed, write IC + 10-layer artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphanet.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--variant", "smoke", "--n-seeds", "1", "--max-folds", "2"]))
