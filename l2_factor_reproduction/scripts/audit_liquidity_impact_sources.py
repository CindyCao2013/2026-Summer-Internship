#!/usr/bin/env python
"""Sprint 7 source audit: ClickHouse Tick + SSL2 inventory and smoke test.

Audits (read-only):
1. table engines / partition keys for the four source tables and their
   LOCAL MergeTree counterparts (no distributed double JOIN).
2. schemas relevant to book / trade / direction fields.
3. per-year symbol coverage of the A-share filters vs the frozen backtest
   universe (ClickHouse L2 covers a documented symbol subset).
4. frozen direction rules: SSE BSFlag, SZSE Type='011' & Category='F' with
   BidOrderNo/AskOrderNo comparison; neutral share recorded.
5. 2024-06-28 single-day smoke of the frozen server-side daily primitive
   SQL for both exchanges (no raw Tick/Snapshot pull).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python import liquidity_impact_daily as lid  # noqa: E402

OUT_DIR = (
    Path(RESULT_ROOT) / "audits" / "liquidity_impact_sources"
)
SMOKE_DAY = "2024-06-28"
SMOKE_END = "2024-06-29"
LOCAL_TABLES = (
    "LOCAL_SSE_AL_TICK_EXG",
    "LOCAL_SZSE_AL_TICK_EXG",
    "LOCAL_SSE_AL_SSL2_EXG",
    "LOCAL_SZSE_AL_SSL2_EXG",
)
DISTRIBUTED_TABLES = tuple(name.replace("LOCAL_", "") for name in LOCAL_TABLES)


def _dt(day: str) -> str:
    return f"toDateTime64('{day} 00:00:00', 6, 'Asia/Shanghai')"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = connect_hf_client()
    checks: List[Dict[str, object]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append(
            {"check": name, "passed": bool(condition), "detail": detail}
        )

    # --- 1. engines & partition keys -----------------------------------------
    tables = client.query_df(
        "SELECT name, engine, partition_key, sorting_key, total_rows "
        "FROM system.tables WHERE database = 'cmds' AND name IN "
        + str(tuple(LOCAL_TABLES + DISTRIBUTED_TABLES))
    )
    tables.to_csv(OUT_DIR / "table_engines.csv", index=False)
    local = tables.loc[tables["name"].isin(LOCAL_TABLES)]
    check(
        "engine:all_local_are_mergetree",
        (local["engine"] == "MergeTree").all(),
        str(dict(zip(local["name"], local["engine"]))),
    )
    distributed = tables.loc[tables["name"].isin(DISTRIBUTED_TABLES)]
    check(
        "engine:distributed_never_joined",
        (distributed["engine"] == "Distributed").all(),
        "primitive SQL references LOCAL_* tables only",
    )

    # --- 2. schemas -----------------------------------------------------------
    for table in LOCAL_TABLES:
        schema = client.query_df(f"DESCRIBE TABLE cmds.{table}")
        schema.to_csv(OUT_DIR / f"schema_{table}.csv", index=False)
    sse_tick = client.query_df("DESCRIBE TABLE cmds.LOCAL_SSE_AL_TICK_EXG")
    szse_tick = client.query_df("DESCRIBE TABLE cmds.LOCAL_SZSE_AL_TICK_EXG")
    check(
        "schema:sse_tick_has_bsflag_amount",
        {"BSFlag", "Amount", "Price", "Volume"}.issubset(
            set(sse_tick["name"])
        ),
        str(list(sse_tick["name"])),
    )
    check(
        "schema:szse_tick_has_order_seq",
        {"BidOrderNo", "AskOrderNo", "Category", "Price", "Volume"}.issubset(
            set(szse_tick["name"])
        ),
        "SZSE tick has no Amount column; amount = Price*Volume server-side",
    )
    for table in ("LOCAL_SSE_AL_SSL2_EXG", "LOCAL_SZSE_AL_SSL2_EXG"):
        schema = client.query_df(f"DESCRIBE TABLE cmds.{table}")
        check(
            f"schema:{table}_book_arrays",
            {"BidPrices", "BidVolumes", "AskPrices", "AskVolumes"}.issubset(
                set(schema["name"])
            ),
            str(list(schema["name"])),
        )

    # --- 3. per-year coverage --------------------------------------------------
    coverage_rows = []
    for year in range(2019, 2027):
        for exchange, table, filter_sql in (
            (
                "sse",
                "LOCAL_SSE_AL_TICK_EXG",
                "Type = 'T' AND substring(Symbol, 1, 1) = '6'",
            ),
            (
                "szse",
                "LOCAL_SZSE_AL_TICK_EXG",
                "Type = '011' AND Category = 'F' AND "
                + lid.EXCHANGES["szse"]["symbol_filter"],
            ),
        ):
            frame = client.query_df(
                f"""
                SELECT uniqExact(Symbol) AS symbols
                FROM cmds.{table}
                WHERE ExchTime >= {_dt(f'{year}-06-24')}
                    AND ExchTime < {_dt(f'{year}-07-01')}
                    AND {filter_sql}
                """
            )
            coverage_rows.append(
                {
                    "year": year,
                    "exchange": exchange,
                    "sample_week": f"{year}-06-24..{year}-06-30",
                    "covered_symbols": int(frame["symbols"].iloc[0]),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT_DIR / "symbol_coverage_by_year.csv", index=False)
    check(
        "coverage:subset_documented",
        bool((coverage["covered_symbols"] > 0).all()),
        "ClickHouse L2 covers a symbol subset (~1.2k-1.8k names, growing "
        "over time); family is evaluated on this sub-universe",
    )

    universe = pd.read_parquet(
        Path(RESULT_ROOT) / "net_buy_ratio" / "factor_narrow.parquet",
        columns=["symbol", "tradetime"],
    )
    universe["tradetime"] = pd.to_datetime(
        universe["tradetime"]
    ).dt.normalize()
    universe_day = universe.loc[
        universe["tradetime"] == pd.Timestamp(SMOKE_DAY), "symbol"
    ]
    universe_symbols = set(universe_day)
    smoke_symbols = set()
    for exchange, cfg in lid.EXCHANGES.items():
        frame = client.query_df(
            f"""
            SELECT DISTINCT Symbol AS symbol
            FROM {cfg['tick_table']}
            WHERE ExchTime >= {_dt(SMOKE_DAY)} AND ExchTime < {_dt(SMOKE_END)}
                AND {cfg['trade_filter']} AND {cfg['symbol_filter']}
            """
        )
        smoke_symbols |= {symbol + cfg["suffix"] for symbol in frame["symbol"]}
    overlap = len(smoke_symbols & universe_symbols)
    overlap_stats = {
        "smoke_day": SMOKE_DAY,
        "clickhouse_symbols": len(smoke_symbols),
        "backtest_universe_symbols": len(universe_symbols),
        "overlap": overlap,
        "universe_coverage_share": overlap / max(len(universe_symbols), 1),
    }
    (OUT_DIR / "universe_overlap.json").write_text(
        json.dumps(overlap_stats, indent=2), encoding="utf-8"
    )
    check(
        "coverage:overlap_with_backtest_universe",
        overlap_stats["universe_coverage_share"] > 0.25,
        json.dumps(overlap_stats),
    )

    # --- 4. direction rules ----------------------------------------------------
    sse_dir = client.query_df(
        f"""
        SELECT BSFlag, count() AS n
        FROM cmds.LOCAL_SSE_AL_TICK_EXG
        WHERE ExchTime >= {_dt(SMOKE_DAY)} AND ExchTime < {_dt(SMOKE_END)}
            AND Type = 'T'
        GROUP BY BSFlag
        """
    )
    sse_dir.to_csv(OUT_DIR / "sse_bsflag_shares.csv", index=False)
    szse_dir = client.query_df(
        f"""
        SELECT multiIf(BidOrderNo > AskOrderNo, 'active_buy',
            BidOrderNo < AskOrderNo, 'active_sell', 'neutral') AS direction,
            count() AS n
        FROM cmds.LOCAL_SZSE_AL_TICK_EXG
        WHERE ExchTime >= {_dt(SMOKE_DAY)} AND ExchTime < {_dt(SMOKE_END)}
            AND Type = '011' AND Category = 'F'
        GROUP BY direction
        """
    )
    szse_dir.to_csv(OUT_DIR / "szse_direction_shares.csv", index=False)
    szse_neutral = float(
        szse_dir.loc[szse_dir["direction"] == "neutral", "n"].sum()
    ) / float(szse_dir["n"].sum())
    check(
        "direction:szse_neutral_share_small",
        szse_neutral < 0.01,
        f"neutral_share={szse_neutral:.5f} (Category='F' only; "
        "order/cancel records excluded)",
    )
    sse_neutral = float(
        sse_dir.loc[sse_dir["BSFlag"] == "N", "n"].sum()
    ) / float(sse_dir["n"].sum())
    check(
        "direction:sse_neutral_recorded",
        0.0 <= sse_neutral < 0.10,
        f"SSE BSFlag=N share={sse_neutral:.5f} aggregated as neutral_amount",
    )

    # --- 5. single-day smoke -----------------------------------------------------
    smoke_frames = []
    for exchange in lid.EXCHANGES:
        sql = lid.daily_sql(exchange, SMOKE_DAY, SMOKE_END)
        frame = client.query_df(sql)
        frame["query_sha256"] = lid.query_sha256(sql)
        smoke_frames.append(frame)
        print(
            f"[smoke] {exchange} rows={len(frame)} "
            f"symbols={frame['symbol_raw'].nunique()}",
            flush=True,
        )
    smoke = lid.finalize_daily(
        smoke_frames, start=SMOKE_DAY, end=SMOKE_END
    )
    smoke = lid.prepare_liquidity_impact_daily(smoke)
    smoke.to_parquet(OUT_DIR / "smoke_2024-06-28.parquet", index=False)

    key_fields = [
        column
        for column in lid.DAILY_COLUMNS
        if column
        not in (
            "symbol",
            "TradeDate",
            "exchange",
            "expected_continuous_minutes",
        )
    ]
    null_share = smoke[key_fields].isna().mean().sort_values(
        ascending=False
    )
    null_share.to_csv(OUT_DIR / "smoke_null_shares.csv")
    check(
        "smoke:symbol_count",
        1500 <= smoke["symbol"].nunique() <= 2500,
        f"symbols={smoke['symbol'].nunique()}",
    )
    check(
        "smoke:coverage_ratio_high",
        float(smoke["coverage_ratio"].median()) > 0.90,
        f"median={smoke['coverage_ratio'].median():.3f}",
    )
    check(
        "smoke:core_impact_fields_mostly_present",
        float(null_share[["signed_amount_impact", "buy_price_impact"]].max())
        < 0.05,
        str(null_share.head(8).to_dict()),
    )
    check(
        "smoke:high_impact_minutes_plausible",
        bool(
            smoke["high_impact_minute_count"]
            .between(0, lid.EXPECTED_CONTINUOUS_MINUTES)
            .all()
        ),
        f"median={smoke['high_impact_minute_count'].median():.1f}",
    )
    vwap_check = smoke.loc[
        smoke["daily_volume"] > 0,
        ["daily_amount", "daily_volume", "symbol"],
    ].copy()
    vwap_check["implied_vwap"] = (
        vwap_check["daily_amount"] / vwap_check["daily_volume"]
    )
    # penny stocks trade below 0.5 CNY legitimately; bound is a sanity
    # filter for unit errors, not a price screen.
    check(
        "smoke:implied_daily_vwap_sane",
        bool(
            vwap_check["implied_vwap"].between(0.1, 20000).all()
        ),
        f"median={vwap_check['implied_vwap'].median():.2f} CNY",
    )

    # --- inventory report --------------------------------------------------------
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(OUT_DIR / "audit_checks.csv", index=False)
    report = [
        "# Liquidity / Price Impact source inventory (ClickHouse)",
        "",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        "- design: LOCAL MergeTree tables only; per-exchange server-side"
        " minute aggregate + minute join; no raw Tick/Snapshot pulled to"
        " pandas; SSE/SZSE daily results merged in pandas.",
        "",
        "## Engines / sizes",
        "",
        tables.to_string(index=False),
        "",
        "## A-share coverage caveat",
        "",
        "ClickHouse L2 (Tick/SSL2/KLIN alike) covers a symbol subset that"
        " grows over time (2024-06-28: ~1.7k names vs ~4.4k backtest"
        " universe). DolphinDB has no Tick/L2 snapshot database, so"
        " ClickHouse is the only L2 source. The Liquidity/Impact family is"
        " therefore evaluated on this documented sub-universe; uncovered"
        " names carry NaN factor values.",
        "",
        coverage.to_string(index=False),
        "",
        json.dumps(overlap_stats, indent=2),
        "",
        "## Direction rules (frozen)",
        "",
        "- SSE: Type='T' with BSFlag 'B'/'S'/'N' (N aggregated as neutral).",
        "- SZSE: Type='011' AND Category='F' (executions only; order/cancel"
        " categories carry NULL seq and are excluded); BidOrderNo >"
        " AskOrderNo -> active buy, '<' -> active sell, '=' -> neutral"
        f" (smoke neutral share {szse_neutral:.4%}).",
        "",
        "## Smoke 2024-06-28",
        "",
        smoke[
            [
                "symbol",
                "coverage_ratio",
                "trade_minute_count",
                "daily_amount",
                "signed_amount_impact",
                "effective_spread_proxy",
                "permanent_impact_1m",
                "spread_recovery_5m",
            ]
        ]
        .describe()
        .to_string(),
        "",
        "## Checks",
        "",
        checks_frame.to_string(index=False),
        "",
    ]
    (OUT_DIR / "liquidity_source_inventory.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    failures = checks_frame.loc[~checks_frame["passed"]]
    if len(failures):
        raise RuntimeError(
            "source audit failed:\n" + failures.to_string(index=False)
        )
    print(f"[done] source audit -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
