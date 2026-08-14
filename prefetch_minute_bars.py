#!/usr/bin/env python3
"""DEPRECATED — local minute-bar prefetch is no longer used.

MinuteBarStore now queries DolphinDB on-demand. Use factor scripts directly;
no prefetch step is required.

This script is kept only to print a clear message if invoked by old workflows.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "prefetch_minute_bars.py is deprecated.\n"
        "MinuteBarStore fetches minute bars directly from DolphinDB on each "
        "get_data() call (with in-process memory cache).\n"
        "Remove prefetch from your pipeline and run factor/backtest scripts directly.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
