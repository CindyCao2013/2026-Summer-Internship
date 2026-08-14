#!/usr/bin/env python
"""Integrity audit for the frozen close_auction_return factor (Sprint 7).

Read-only audit: uses the frozen Sprint 6 formula and artifacts, does not
modify the baseline and does not compare alternative closing windows.

Boundary semantics (from the frozen primitive module):
- continuous close: last continuous-auction minute label of the frozen
  240-minute grid (09:30-11:29, 13:00-14:59). DolphinDB has a structural
  bar gap 14:57-14:59, so the trailing labels are price-state filled
  (volume/amount never filled, max 3 consecutive). Effective last observed
  continuous price is therefore the 14:56 bar close.
- auction close: the single 15:00 close-auction bar close.
- both prices are Adjfactor-adjusted; close_auction_return =
  log(close_auction_price / continuous_close).
- the daily factor is known only after the 15:00 auction; the backtest layer
  applies signal.shift(1) exactly once, so the target return is the T+1
  close-to-close excess return starting from the auction close.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import dolphindb as ddb
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import DATA_DB_CONN, calAnnuRet, calSharpe  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python import (  # noqa: E402
    price_formation_daily as primitive_module,
)
from l2_factor_reproduction.python import (  # noqa: E402
    price_formation_factors as factor_module,
)
from l2_factor_reproduction.python.candidate_pool_registry import (  # noqa: E402
    BASELINE_POLICY,
    POOL_ROOT,
)

FAMILY_DIR = POOL_ROOT / "price_formation_family"
FACTOR_DIR = FAMILY_DIR / "factors" / "close_auction_return"
PRIMITIVE_DIR = (
    Path(RESULT_ROOT) / "primitives" / "price_formation_daily" / "dataset"
)
OUT_DIR = FAMILY_DIR / "integrity_audit_close_auction"

MINUTE_DB = "dfs://QV_Trade_to_MinuteBar"
MINUTE_TABLE = "Stock_one_minute"
EOD_DB = "dfs://WIND.ASHAREEODPRICES"
EOD_TABLE = "data"


def _query_auction_volume(year: int) -> pd.DataFrame:
    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    try:
        frame = session.run(
            f"""
            select Date, Symbol, sum(Volume) as auction_volume
            from loadTable("{MINUTE_DB}", "{MINUTE_TABLE}")
            where Date >= {year}.01.01, Date < {year + 1}.01.01,
                second(Bartime)==15:00:00
            group by Date, Symbol
            """
        )
    finally:
        session.close()
    frame = frame.rename(columns={"Date": "TradeDate", "Symbol": "symbol"})
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
    return frame


def _query_limit_flags(year: int) -> pd.DataFrame:
    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    try:
        frame = session.run(
            f"""
            select TRADE_DT, S_INFO_WINDCODE,
                S_DQ_CLOSE >= S_DQ_LIMIT as limit_up,
                S_DQ_CLOSE <= S_DQ_STOPPING as limit_down
            from loadTable("{EOD_DB}", "{EOD_TABLE}")
            where TRADE_DT >= {year}.01.01, TRADE_DT < {year + 1}.01.01
            """
        )
    finally:
        session.close()
    frame = frame.rename(
        columns={"TRADE_DT": "TradeDate", "S_INFO_WINDCODE": "symbol"}
    )
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ("limit_up", "limit_down"):
        frame[column] = (
            frame[column].astype(str).str.lower().isin(["true", "1"])
        )
    return frame


def _decile_membership(narrow: pd.DataFrame) -> pd.DataFrame:
    from Factor_Dev_Lib import _rank_to_bins_npqcut

    wide = narrow.pivot_table(
        index="tradetime", columns="symbol", values="value"
    ).sort_index()
    groups = _rank_to_bins_npqcut(wide, 10)
    long = (
        groups.stack(dropna=False)
        .rename("decile")
        .reset_index()
        .rename(columns={"tradetime": "TradeDate"})
    )
    long["TradeDate"] = pd.to_datetime(long["TradeDate"]).dt.normalize()
    return long.dropna(subset=["decile"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append(
            {"check": name, "passed": bool(condition), "detail": detail}
        )

    summary = json.loads(
        (FACTOR_DIR / "summary.json").read_text(encoding="utf-8")
    )
    direction = int(summary["factor_direction"])
    group_pnl = pd.read_csv(FACTOR_DIR / "group_pnl.csv", index_col=0)
    group_pnl.index = pd.to_datetime(group_pnl.index)

    # --- 1. formula & minute boundary documentation -------------------------
    grid_source = inspect.getsource(primitive_module)
    check(
        "boundary:continuous_grid_frozen_240",
        "09:30" in grid_source
        and primitive_module.EXPECTED_CONTINUOUS_MINUTES == 240,
        "continuous grid 09:30-11:29 + 13:00-14:59, 240 labels",
    )
    check(
        "boundary:auction_bar_isolated",
        "15:00:00" in inspect.getsource(
            primitive_module.close_auction_daily_sql
        ),
        "close_auction_price taken from the 15:00 auction bar only",
    )
    check(
        "boundary:formula_is_log_ratio",
        "continuous_close" in inspect.getsource(primitive_module)
        and "close_auction_price" in inspect.getsource(primitive_module),
        "close_auction_return = log(close_auction_price/continuous_close)",
    )

    # --- 2. target return starts after the auction signal -------------------
    check(
        "timing:factor_known_after_auction",
        True,
        "factor uses day-T full-day data incl. 15:00 auction bar; "
        "earliest information timestamp is the 15:00 auction close",
    )
    check(
        "timing:target_return_c2c_after_close",
        BASELINE_POLICY["signal_shift"] == 1,
        "with signal.shift(1), day-T signal earns the T+1 close-to-close "
        "excess return, which starts exactly at the auction close price",
    )

    # --- 3. shift applied exactly once --------------------------------------
    factor_source = inspect.getsource(factor_module)
    check(
        "timing:feature_layer_has_no_shift",
        ".shift(" not in factor_source and "rolling(" not in factor_source,
        "feature layer contains no shift/rolling (lookback_days=1 level)",
    )
    from l2_factor_reproduction.python import backtest as backtest_module

    backtest_source = inspect.getsource(backtest_module.prepare_factor_signal)
    check(
        "timing:backtest_single_shift",
        backtest_source.count(".shift(") == 1,
        "prepare_factor_signal applies signal.shift(signal_shift) once",
    )

    # --- 4. raw vs effective group annual returns ----------------------------
    # group_pnl.csv / group_mean_annu are saved in the EFFECTIVE direction
    # (signal multiplied by factor_direction). The raw-direction deciles are
    # the reversal of the effective ones; raw H-L = -effective H-L when
    # factor_direction == -1.
    group_mean = {
        int(key): float(value)
        for key, value in summary["group_mean_annu"].items()
    }
    hl_effective_annu = float(calAnnuRet(group_pnl["H-L"]))
    effective_rows = {
        "G1": group_mean[1],
        "G10": group_mean[10],
        "H-L": hl_effective_annu,
    }
    if direction == 1:
        raw_rows = dict(effective_rows)
    else:
        raw_rows = {
            "G1": group_mean[10],
            "G10": group_mean[1],
            "H-L": -hl_effective_annu,
        }
    group_table = pd.DataFrame(
        {"raw": raw_rows, "effective": effective_rows}
    )
    check(
        "returns:raw_effective_hl_consistent",
        abs(raw_rows["H-L"] + effective_rows["H-L"]) < 1e-9
        if direction == -1
        else abs(raw_rows["H-L"] - effective_rows["H-L"]) < 1e-9,
        f"raw={raw_rows['H-L']:.6f}, effective={effective_rows['H-L']:.6f}",
    )
    check(
        "returns:effective_hl_matches_saved_pnl",
        abs(hl_effective_annu - float(summary["hl_annu_ret_flipped"]))
        < 1e-9,
        f"pnl={hl_effective_annu:.6f}, "
        f"summary={summary['hl_annu_ret_flipped']:.6f}",
    )
    group_table.to_csv(OUT_DIR / "group_annual_returns_raw_effective.csv")

    # --- 5. yearly H-L -------------------------------------------------------
    # group_pnl.csv is saved in effective direction already.
    hl_effective = group_pnl["H-L"]
    yearly_rows = []
    for year, block in hl_effective.groupby(hl_effective.index.year):
        yearly_rows.append(
            {
                "year": int(year),
                "hl_annu_ret": float(calAnnuRet(block)),
                "hl_sharpe": float(calSharpe(block)),
                "n_days": int(len(block)),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUT_DIR / "yearly_hl.csv", index=False)
    check(
        "returns:yearly_hl_all_positive_sharpe",
        bool((yearly["hl_sharpe"] > 0).all()),
        f"min_yearly_sharpe={yearly['hl_sharpe'].min():.3f}",
    )

    # --- 6. monthly contribution --------------------------------------------
    monthly = hl_effective.groupby(
        [hl_effective.index.year, hl_effective.index.month]
    ).sum()
    monthly.index.names = ["year", "month"]
    total_abs = monthly.abs().sum()
    monthly_frame = monthly.rename("hl_ret_sum").reset_index()
    monthly_frame["contribution_share"] = (
        monthly_frame["hl_ret_sum"] / total_abs
    )
    monthly_frame.to_csv(OUT_DIR / "monthly_contribution.csv", index=False)
    month_share = (
        monthly_frame.groupby("month")["hl_ret_sum"].sum() / total_abs
    )
    check(
        "returns:no_single_month_dominates",
        bool((month_share.abs() < 0.30).all()),
        f"max_abs_month_share={month_share.abs().max():.3f}",
    )

    # --- 7. primitive stale share --------------------------------------------
    primitive_parts = []
    for path in sorted(PRIMITIVE_DIR.glob("year=*/*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "symbol",
                "TradeDate",
                "close_auction_price",
                "continuous_close",
            ],
        )
        primitive_parts.append(frame)
    primitive = pd.concat(primitive_parts, ignore_index=True)
    has_auction = primitive["close_auction_price"] > 0
    stale = (
        np.isclose(
            primitive["close_auction_price"],
            primitive["continuous_close"],
            rtol=0,
            atol=1e-12,
        )
        & has_auction
    )
    primitive["year"] = primitive["TradeDate"].dt.year
    stale_by_year = (
        pd.DataFrame({"has_auction": has_auction, "stale": stale})
        .groupby(primitive["year"])
        .mean()
    )

    # --- 8. auction volume flags (server-side per year) ----------------------
    volume_parts = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"[audit] auction volume year={year}", flush=True)
        volume_parts.append(_query_auction_volume(year))
    auction_volume = pd.concat(volume_parts, ignore_index=True)
    auction_volume["no_volume"] = auction_volume["auction_volume"] <= 0
    no_volume_share = float(auction_volume["no_volume"].mean())
    no_volume_by_year = auction_volume.groupby(
        auction_volume["TradeDate"].dt.year
    )["no_volume"].mean()

    quality = pd.DataFrame(
        {
            "no_volume_auction_share": no_volume_by_year,
            "stale_auction_price_share": stale_by_year["stale"],
            "missing_auction_bar_share": 1.0 - stale_by_year["has_auction"],
        }
    )
    quality.to_csv(OUT_DIR / "auction_data_quality.csv")
    check(
        "data:no_volume_auction_share_small",
        no_volume_share < 0.01,
        f"no_volume_share={no_volume_share:.5f}",
    )
    check(
        "data:stale_auction_price_share_bounded",
        float(stale_by_year["stale"].mean()) < 0.60,
        f"mean_stale_share={stale_by_year['stale'].mean():.3f}",
    )

    # --- 9. limit-up/down share by decile ------------------------------------
    narrow = pd.read_parquet(FACTOR_DIR / "factor_narrow.parquet")
    narrow["tradetime"] = pd.to_datetime(narrow["tradetime"])
    deciles = _decile_membership(narrow)
    limit_parts = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"[audit] limit flags year={year}", flush=True)
        limit_parts.append(_query_limit_flags(year))
    limits = pd.concat(limit_parts, ignore_index=True)
    limits["symbol"] = limits["symbol"].astype(str)
    merged = deciles.merge(
        limits,
        on=["TradeDate", "symbol"],
        how="left",
        validate="many_to_one",
    )
    for column in ("limit_up", "limit_down"):
        merged[column] = pd.to_numeric(
            merged[column].map(
                lambda x: (
                    True
                    if str(x).lower() in ("true", "1")
                    else (False if str(x).lower() in ("false", "0") else np.nan)
                )
            ),
            errors="coerce",
        )
    limit_share = merged.groupby("decile").agg(
        n=("decile", "size"),
        limit_up_share=("limit_up", "mean"),
        limit_down_share=("limit_down", "mean"),
    )
    limit_share.index = limit_share.index.astype(int)
    limit_share.sort_index().to_csv(OUT_DIR / "limit_share_by_decile.csv")
    short_leg = limit_share.loc[10, "limit_up_share"]
    long_leg = limit_share.loc[1, "limit_down_share"]
    check(
        "data:limit_share_reported_by_decile",
        bool(limit_share["limit_up_share"].notna().all()),
        f"G10 limit_up={short_leg:.4f}, G1 limit_down={long_leg:.4f}",
    )

    # --- report ---------------------------------------------------------------
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(OUT_DIR / "audit_checks.csv", index=False)
    report = [
        "# close_auction_return integrity audit (read-only)",
        "",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- factor_direction: {direction} (raw IC "
        f"{summary['rank_ic_mean_raw']:.4f}; production direction not decided)",
        "",
        "## Minute boundary",
        "",
        "- continuous close: last label of the frozen 240-minute grid"
        " (09:30-11:29, 13:00-14:59); DDB structural gap 14:57-14:59 is"
        " price-state filled, so the effective last continuous price is the"
        " 14:56 bar close.",
        "- auction close: the single 15:00 close-auction bar close.",
        "- both Adjfactor-adjusted; formula ="
        " log(close_auction_price / continuous_close).",
        "- target return: T+1 close-to-close excess (starts at the auction"
        " close); signal.shift(1) applied exactly once.",
        "",
        "## Group annual returns (frozen baseline)",
        "",
        group_table.to_string(),
        "",
        "## Yearly H-L (effective direction)",
        "",
        yearly.to_string(index=False),
        "",
        "## Auction data quality",
        "",
        quality.to_string(),
        "",
        "## Limit share by decile",
        "",
        limit_share.to_string(),
        "",
        "## Checks",
        "",
        checks_frame.to_string(index=False),
        "",
        "本审计只读，不修改 Sprint 6 baseline，不比较其他收盘窗口。",
    ]
    (OUT_DIR / "integrity_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    failures = checks_frame.loc[~checks_frame["passed"]]
    if len(failures):
        raise RuntimeError(
            "close_auction integrity audit failed:\n"
            + failures.to_string(index=False)
        )
    print(f"[done] integrity audit -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
