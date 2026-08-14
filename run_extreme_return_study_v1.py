#!/usr/bin/env python
"""Launcher for CSI300 Extreme Return Effect Study v1.

Delegates to research/extreme_return_study/run_study.py

Usage:
  OMP_NUM_THREADS=1 python run_extreme_return_study_v1.py
  OMP_NUM_THREADS=1 python run_extreme_return_study_v1.py --smoke
"""

from __future__ import annotations

import runpy
from pathlib import Path

STUDY = Path(__file__).resolve().parent / "research" / "extreme_return_study" / "run_study.py"

if __name__ == "__main__":
    runpy.run_path(str(STUDY), run_name="__main__")
