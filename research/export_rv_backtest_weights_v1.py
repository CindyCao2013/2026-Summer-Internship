#!/usr/bin/env python3
"""Export realized_volatility backtest weights CSV.

Frozen tuple: 14:29 / Ret_30 / direction=-1.
Portfolio: equal-weight extreme deciles, long gross +0.5 / short gross -0.5.

Output columns: date, strategyname, symbol, weight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.freeze_intraday_alpha_v1 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_FREEZE,
    verify_spec,
)
from research.intraday_portfolio_simulator_v1 import (  # noqa: E402
    _build_positions,
    _fetch_extreme_constituents,
    _periods,
)
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    _connect,
    _evaluate,
)
from research.run_intraday_alpha_oos_v1 import (  # noqa: E402
    _build_factor_chunked,
    _filter_slots,
)

FACTOR_NAME = "realized_volatility"
DEFAULT_OUTPUT = (
    ROOT
    / "research/results/intraday_portfolio_simulator_v1"
    / "realized_volatility_backtest_weights.csv"
)


def _export_period_weights(
    session,
    *,
    factor_name: str,
    period_name: str,
    period: dict,
    spec: dict,
    chunk_months: int,
) -> pd.DataFrame:
    print(
        f"[WEIGHTS] {factor_name}/{period_name} "
        f"{spec['bartime']}/{spec['horizon']} "
        f"direction={spec['direction']}",
        flush=True,
    )
    signal = _filter_slots(
        _build_factor_chunked(
            factor_name,
            period["start"],
            period["end"],
            chunk_months,
        ),
        {str(spec["bartime"])},
    )
    filtered, _, _, _ = _evaluate(
        session,
        f"weights_{factor_name}_{period_name}",
        signal,
        apply_limit_filter=True,
    )
    constituents = _fetch_extreme_constituents(
        session,
        f"{factor_name}_{period_name}",
        filtered,
        str(spec["horizon"]),
    )
    rows = []
    for _, day in constituents.groupby("tradetime", sort=True):
        positions = _build_positions(day, int(spec["direction"]))
        part = positions[["Date", "Symbol", "entry_weight"]].copy()
        part["Date"] = pd.to_datetime(part["Date"]).dt.strftime("%Y-%m-%d")
        part = part.rename(
            columns={
                "Date": "date",
                "Symbol": "symbol",
                "entry_weight": "weight",
            }
        )
        part["strategyname"] = factor_name
        rows.append(part[["date", "strategyname", "symbol", "weight"]])
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--period", action="append")
    parser.add_argument("--chunk-months", type=int, default=6)
    args = parser.parse_args()

    freeze = verify_spec(args.freeze)
    if FACTOR_NAME not in freeze["factors"]:
        raise ValueError(f"{FACTOR_NAME} is not in the freeze file")
    spec = freeze["factors"][FACTOR_NAME]
    available = _periods(freeze)
    period_order = args.period or list(available)
    unknown = [name for name in period_order if name not in available]
    if unknown:
        raise ValueError(f"Unknown periods: {unknown}")
    if args.chunk_months < 1:
        raise ValueError("Chunk months must be positive")

    session = _connect()
    frames = [
        _export_period_weights(
            session,
            factor_name=FACTOR_NAME,
            period_name=period_name,
            period=available[period_name],
            spec=spec,
            chunk_months=args.chunk_months,
        )
        for period_name in period_order
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(
        f"Wrote {len(out):,} rows / {out['date'].nunique()} dates → {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
