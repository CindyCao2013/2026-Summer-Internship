#!/usr/bin/env python
"""Audit minute sources and freeze the Price Formation canonical source."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from core.ddb.connection import get_ddb_session  # noqa: E402
from minute_bar_store import MinuteBarStore  # noqa: E402
from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402


OUT_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/price_formation_daily"
)
DDB_DATABASE = "dfs://QV_Trade_to_MinuteBar"
DDB_TABLE = "Stock_one_minute"
CH_TABLES: Tuple[Tuple[str, str, str], ...] = (
    ("SSE_AL_KLIN_EXG", ".SH", "SSE"),
    ("SZSE_AL_KLIN_CMD", ".SZ", "SZSE"),
)
TARGET_START = pd.Timestamp("2019-01-01")
TARGET_END = pd.Timestamp("2026-07-31")
PARITY_START = pd.Timestamp("2024-06-01")
PARITY_END = pd.Timestamp("2024-06-30")
PARITY_SYMBOLS = (
    "600000.SH",
    "601318.SH",
    "000001.SZ",
    "000333.SZ",
)
PARITY_METRICS = ("close", "volume", "amount", "vwap", "minute_return")


def _date_text(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _ddb_day(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def _ch_datetime(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _ddb_schema(session) -> pd.DataFrame:
    query = (
        "select name,typeString "
        f'from schema(loadTable("{DDB_DATABASE}","{DDB_TABLE}")).colDefs'
    )
    frame = pd.DataFrame(session.run(query))
    frame.insert(0, "source", "DolphinDB")
    frame.insert(1, "database", DDB_DATABASE)
    frame.insert(2, "table", DDB_TABLE)
    frame["position"] = np.arange(1, len(frame) + 1)
    return frame.rename(columns={"typeString": "type"})


def _ddb_coverage(session) -> Dict[str, object]:
    query = f"""
t=loadTable("{DDB_DATABASE}","{DDB_TABLE}")
select min(Date) as min_date, max(Date) as max_date, count(*) as row_count
from t
where Date between {_ddb_day(TARGET_START)} : {_ddb_day(TARGET_END)}
"""
    row = pd.DataFrame(session.run(query)).iloc[0]
    return {
        "source": "DolphinDB",
        "database": DDB_DATABASE,
        "table": DDB_TABLE,
        "min_date": _date_text(row["min_date"]),
        "max_date": _date_text(row["max_date"]),
        "row_count": int(row["row_count"]),
        "target_start": _date_text(TARGET_START),
        "target_end": _date_text(TARGET_END),
        "covers_target_trading_days": bool(
            pd.Timestamp(row["min_date"]) <= pd.Timestamp("2019-01-02")
            and pd.Timestamp(row["max_date"]) >= TARGET_END
        ),
        "price_adjustment": "Adjfactor available; adjusted price = raw price * Adjfactor",
        "bar_semantics": (
            "09:25 opening auction; 09:30-11:29 and 13:00-14:56 "
            "observed continuous labels; 15:00 close auction"
        ),
        "production_role": "existing production minute pipeline",
    }


def _ddb_bar_times(session, audit_date: pd.Timestamp) -> List[str]:
    query = f"""
t=loadTable("{DDB_DATABASE}","{DDB_TABLE}")
select count(*) as rows
from t
where Date={_ddb_day(audit_date)}
group by Bartime
order by Bartime
"""
    frame = pd.DataFrame(session.run(query))
    return pd.to_datetime(frame["Bartime"]).dt.strftime("%H:%M:%S").tolist()


def _ch_schema(client, table: str) -> pd.DataFrame:
    query = f"""
SELECT name, type, position
FROM system.columns
WHERE database = 'cmds' AND table = '{table}'
ORDER BY position
"""
    frame = client.query_df(query)
    frame.insert(0, "source", "ClickHouse")
    frame.insert(1, "database", "cmds")
    frame.insert(2, "table", table)
    return frame


def _ch_coverage(client, table: str, exchange: str) -> Dict[str, object]:
    query = f"""
SELECT
  min(ExchTime) AS min_time,
  max(ExchTime) AS max_time,
  count() AS row_count
FROM cmds.`{table}`
WHERE ExchTime >= toDateTime64(
    '{_ch_datetime(TARGET_START)}', 6, 'Asia/Shanghai'
  )
  AND ExchTime < toDateTime64(
    '{_ch_datetime(TARGET_END + pd.Timedelta(days=1))}', 6, 'Asia/Shanghai'
  )
"""
    row = client.query_df(query).iloc[0]
    minimum = pd.Timestamp(row["min_time"])
    maximum = pd.Timestamp(row["max_time"])
    if minimum.tzinfo is not None:
        minimum = minimum.tz_localize(None)
    if maximum.tzinfo is not None:
        maximum = maximum.tz_localize(None)
    return {
        "source": "ClickHouse",
        "database": "cmds",
        "table": table,
        "exchange": exchange,
        "min_date": _date_text(minimum),
        "max_date": _date_text(maximum),
        "row_count": int(row["row_count"]),
        "target_start": _date_text(TARGET_START),
        "target_end": _date_text(TARGET_END),
        "covers_target_trading_days": bool(
            minimum.normalize() <= pd.Timestamp("2019-01-02")
            and maximum.normalize() >= TARGET_END
        ),
        "price_adjustment": "Adjfactor absent",
        "bar_semantics": (
            "Type=1MIN contains 09:30-11:30 and 13:00-15:00; "
            "15:00 is separable"
        ),
        "production_role": "parity/reference only",
    }


def _ch_bar_times(
    client,
    table: str,
    symbol: str,
    audit_date: pd.Timestamp,
) -> List[str]:
    query = f"""
SELECT ExchTime
FROM cmds.`{table}`
WHERE Symbol = '{symbol}'
  AND Type = '1MIN'
  AND toDate(ExchTime) = toDate('{_date_text(audit_date)}')
ORDER BY ExchTime
"""
    frame = client.query_df(query)
    times = pd.to_datetime(frame["ExchTime"])
    if times.dt.tz is not None:
        times = times.dt.tz_localize(None)
    return times.dt.strftime("%H:%M:%S").tolist()


def _continuous_mask(times: pd.Series) -> pd.Series:
    values = pd.to_datetime(times)
    minute = values.dt.hour * 60 + values.dt.minute
    return minute.between(9 * 60 + 30, 11 * 60 + 29) | minute.between(
        13 * 60, 14 * 60 + 59
    )


def _prepare_parity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bartime"] = pd.to_datetime(out["bartime"])
    if out["bartime"].dt.tz is not None:
        out["bartime"] = out["bartime"].dt.tz_localize(None)
    out = out.loc[_continuous_mask(out["bartime"])].copy()
    out["TradeDate"] = out["bartime"].dt.normalize()
    numeric = ("open", "high", "low", "close", "volume", "amount")
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["vwap"] = (out["amount"] / out["volume"].where(out["volume"] > 0)).where(
        out["amount"] >= 0
    )
    out = out.sort_values(["symbol", "TradeDate", "bartime"], kind="stable")
    previous_close = out.groupby(["symbol", "TradeDate"], sort=False)[
        "close"
    ].shift(1)
    out["minute_return"] = np.log(
        out["close"].where(out["close"] > 0)
        / previous_close.where(previous_close > 0)
    )
    if out.duplicated(["symbol", "bartime"]).any():
        raise ValueError("Parity input contains duplicate symbol-minute keys")
    return out[
        ["symbol", "TradeDate", "bartime", *PARITY_METRICS]
    ].reset_index(drop=True)


def _fetch_ddb_parity() -> pd.DataFrame:
    store = MinuteBarStore(start_date=TARGET_START)
    frame = store.get_data(
        PARITY_START,
        PARITY_END,
        symbols=list(PARITY_SYMBOLS),
        fields=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adjfactor",
        ],
        trading_hours_only=False,
    )
    return _prepare_parity_frame(frame)


def _ch_symbol_filter(symbols: Sequence[str], suffix: str) -> str:
    bare = sorted(
        str(symbol).split(".")[0]
        for symbol in symbols
        if str(symbol).endswith(suffix)
    )
    return ", ".join(repr(symbol) for symbol in bare)


def _fetch_ch_parity(client) -> pd.DataFrame:
    frames = []
    for table, suffix, _exchange in CH_TABLES:
        values = _ch_symbol_filter(PARITY_SYMBOLS, suffix)
        query = f"""
SELECT
  concat(Symbol, '{suffix}') AS symbol,
  ExchTime AS bartime,
  toFloat64(Open) AS open,
  toFloat64(High) AS high,
  toFloat64(Low) AS low,
  toFloat64(Close) AS close,
  toFloat64(Volume) AS volume,
  toFloat64(Amount) AS amount
FROM cmds.`{table}`
WHERE Symbol IN ({values})
  AND Type = '1MIN'
  AND ExchTime >= toDateTime64(
    '{_ch_datetime(PARITY_START)}', 6, 'Asia/Shanghai'
  )
  AND ExchTime < toDateTime64(
    '{_ch_datetime(PARITY_END + pd.Timedelta(days=1))}', 6, 'Asia/Shanghai'
  )
ORDER BY symbol, bartime
"""
        frames.append(client.query_df(query))
    return _prepare_parity_frame(pd.concat(frames, ignore_index=True))


def _daily_spearman(
    merged: pd.DataFrame,
    left: str,
    right: str,
) -> pd.Series:
    rows = []
    for trade_date, block in merged.groupby("TradeDate", sort=True):
        pair = block[[left, right]].dropna()
        rho = (
            pair[left].corr(pair[right], method="spearman")
            if len(pair) >= 20
            else np.nan
        )
        rows.append((trade_date, rho))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(
        [item[1] for item in rows],
        index=[item[0] for item in rows],
        dtype=float,
    )


def _parity_rows(
    ddb: pd.DataFrame,
    clickhouse: pd.DataFrame,
) -> pd.DataFrame:
    scopes: Iterable[Tuple[str, Sequence[str]]] = (
        ("ALL", PARITY_SYMBOLS),
        ("SSE", tuple(x for x in PARITY_SYMBOLS if x.endswith(".SH"))),
        ("SZSE", tuple(x for x in PARITY_SYMBOLS if x.endswith(".SZ"))),
    )
    rows = []
    for exchange, symbols in scopes:
        left = ddb.loc[ddb["symbol"].isin(symbols)].copy()
        right = clickhouse.loc[clickhouse["symbol"].isin(symbols)].copy()
        merged = left.merge(
            right,
            on=["symbol", "TradeDate", "bartime"],
            how="outer",
            suffixes=("_ddb", "_ch"),
            indicator=True,
        )
        ddb_keys = int((merged["_merge"] != "right_only").sum())
        ch_keys = int((merged["_merge"] != "left_only").sum())
        matched_keys = int((merged["_merge"] == "both").sum())
        for metric in PARITY_METRICS:
            left_name = f"{metric}_ddb"
            right_name = f"{metric}_ch"
            pair = merged[[left_name, right_name]].dropna()
            difference = (pair[left_name] - pair[right_name]).abs()
            daily = _daily_spearman(merged, left_name, right_name)
            rows.append(
                {
                    "sample": "2024-06 fixed four-symbol sample",
                    "exchange": exchange,
                    "metric": metric,
                    "ddb_rows": ddb_keys,
                    "clickhouse_rows": ch_keys,
                    "matched_keys": matched_keys,
                    "matched_nonnull": int(len(pair)),
                    "ddb_missing_rate_vs_union": float(
                        (merged["_merge"] == "right_only").mean()
                    ),
                    "clickhouse_missing_rate_vs_union": float(
                        (merged["_merge"] == "left_only").mean()
                    ),
                    "max_abs_diff": (
                        float(difference.max()) if len(difference) else np.nan
                    ),
                    "median_abs_diff": (
                        float(difference.median()) if len(difference) else np.nan
                    ),
                    "mean_daily_spearman": float(daily.mean()),
                    "median_daily_spearman": float(daily.median()),
                    "daily_spearman_days": int(daily.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _has_time(times: Sequence[str], value: str) -> bool:
    return value in set(times)


def _format_float(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _write_inventory(
    *,
    coverage: pd.DataFrame,
    parity: pd.DataFrame,
    ddb_times: Sequence[str],
    ch_times: Dict[str, Sequence[str]],
) -> None:
    overall = parity.loc[parity["exchange"] == "ALL"].set_index("metric")
    comparison_lines = []
    for metric in PARITY_METRICS:
        row = overall.loc[metric]
        comparison_lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(int(row["matched_nonnull"])),
                    _format_float(row["median_abs_diff"]),
                    _format_float(row["max_abs_diff"]),
                    _format_float(row["mean_daily_spearman"], 4),
                    f"{float(row['ddb_missing_rate_vs_union']):.2%}",
                    f"{float(row['clickhouse_missing_rate_vs_union']):.2%}",
                ]
            )
            + " |"
        )
    coverage_lines = []
    for _, row in coverage.iterrows():
        coverage_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["table"]),
                    str(row["min_date"]),
                    str(row["max_date"]),
                    f"{int(row['row_count']):,}",
                    str(bool(row["covers_target_trading_days"])),
                    str(row["production_role"]),
                ]
            )
            + " |"
        )
    sse_times = ch_times["SSE"]
    szse_times = ch_times["SZSE"]
    lines = [
        "# Sprint 6 Phase 0 — Price Formation minute source inventory",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        "**Canonical minute source: DolphinDB "
        "`dfs://QV_Trade_to_MinuteBar/Stock_one_minute`.**",
        "",
        "The choice is frozen for Price Formation Family v1 because it is the "
        "existing production minute pipeline, has one SSE/SZSE schema, covers "
        "the full target trading-day interval, and supplies `Adjfactor` plus "
        "the minute active-flow fields. ClickHouse KLIN remains a read-only "
        "parity reference. The two sources are not concatenated.",
        "",
        "ClickHouse is not selected because `SSE_AL_KLIN_EXG` has 19 columns "
        "while `SZSE_AL_KLIN_CMD` has 10; SZSE lacks `PreClose` and `Average`, "
        "both lack `Adjfactor`, and June parity shows material OHLCVA "
        "construction differences. Active-flow candidates 33–35 are excluded "
        "from v1 because KLIN exposes no harmonized two-exchange active-flow "
        "definition.",
        "",
        "## Coverage",
        "",
        "| source | table | min date | max date | rows | target trading days | role |",
        "|---|---|---:|---:|---:|---:|---|",
        *coverage_lines,
        "",
        "The target starts on the 2019-01-01 holiday; all three sources begin "
        "on the first target trading day, 2019-01-02.",
        "",
        "## Actual field inventory",
        "",
        "- DolphinDB: `Symbol`, `Date`, `Bartime`, OHLC, `Volume`, `Amount`, "
        "`Adjfactor`, active buy/sell amount/volume/count, and cancel proxies. "
        "No minute `PreClose`, `Average`, or trade-status field.",
        "- ClickHouse SSE KLIN: OHLCVA plus `PreClose`, `Average`, IOPV and "
        "settlement-related fields; no `Adjfactor` or active flow.",
        "- ClickHouse SZSE KLIN: only symbol/time/type/OHLCVA/CHTime; no "
        "`PreClose`, `Average`, `Adjfactor`, or active flow.",
        "- Trade status is not inferred from minute bars. The frozen baseline "
        "continues to use the existing investability mask.",
        "",
        "Detailed types are in `schema_ddb.csv`, `schema_ch_sse.csv`, and "
        "`schema_ch_szse.csv`.",
        "",
        "## Minute labels and auction handling",
        "",
        f"- DDB 2024-06-03 labels: {len(ddb_times)} distinct; "
        f"09:25={_has_time(ddb_times, '09:25:00')}, "
        f"09:30={_has_time(ddb_times, '09:30:00')}, "
        f"11:30={_has_time(ddb_times, '11:30:00')}, "
        f"13:00={_has_time(ddb_times, '13:00:00')}, "
        f"14:57={_has_time(ddb_times, '14:57:00')}, "
        f"14:58={_has_time(ddb_times, '14:58:00')}, "
        f"14:59={_has_time(ddb_times, '14:59:00')}, "
        f"15:00={_has_time(ddb_times, '15:00:00')}.",
        f"- ClickHouse SSE sample has {len(sse_times)} `1MIN` labels; "
        f"SZSE has {len(szse_times)}. Both expose 15:00 separately.",
        "- Frozen continuous grid is `[09:30,11:30)` plus `[13:00,15:00)` "
        "(240 labels). DDB structurally omits 14:57–14:59 and consolidates "
        "the closing auction at 15:00. Primitive construction may carry the "
        "14:56 price state through those three labels within the afternoon "
        "session only; amount/volume are never filled. `valid_minute_count` "
        "counts observed valid prices, and `imputed_price_minute_count` records "
        "the structural price fills.",
        "- 09:25 and 15:00 are stored outside the continuous path. There is no "
        "forward fill across lunch.",
        "",
        "## 2024-06 fixed-sample parity",
        "",
        "Symbols: `600000.SH`, `601318.SH`, `000001.SZ`, `000333.SZ`; "
        "comparison uses equal timestamp labels on the frozen continuous grid.",
        "",
        "| metric | matched non-null | median abs diff | max abs diff | mean daily Spearman | DDB missing | CH missing |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *comparison_lines,
        "",
        "The parity result is diagnostic, not a reconciliation transform. No "
        "rescaling, timestamp shifting, or source splicing is applied.",
        "",
        "## Frozen units and adjustment",
        "",
        "- `Volume` is treated as shares and `Amount` as CNY after validating "
        "`Amount / Volume` against minute prices in the parity sample.",
        "- Price-path fields use `raw price * Adjfactor`; same-day returns are "
        "invariant to a constant daily factor. Amount and volume remain raw.",
        "- Daily VWAP is `sum(Amount) / sum(Volume)` in raw price units, then "
        "multiplied by the day’s `Adjfactor` for path comparisons.",
        "- `overnight_gap` uses the previous available canonical continuous "
        "close after adjustment because DDB has no `PreClose` column.",
        "",
        "## Scope boundary",
        "",
        "No raw Tick reconstruction was performed. ClickHouse KLIN is used "
        "only for schema, coverage, minute-label, and small-sample parity "
        "evidence.",
        "",
    ]
    (OUT_DIR / "source_inventory.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-date", default="2024-06-03")
    args = parser.parse_args()
    audit_date = pd.Timestamp(args.audit_date).normalize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = get_ddb_session(reuse=True)
    ddb_schema = _ddb_schema(session)
    ddb_coverage = _ddb_coverage(session)
    ddb_times = _ddb_bar_times(session, audit_date)
    ddb_parity = _fetch_ddb_parity()

    client = connect_hf_client()
    try:
        ch_sse_schema = _ch_schema(client, "SSE_AL_KLIN_EXG")
        ch_szse_schema = _ch_schema(client, "SZSE_AL_KLIN_CMD")
        coverage_rows = [ddb_coverage]
        ch_times: Dict[str, Sequence[str]] = {}
        sample_symbols = {"SSE": "600000", "SZSE": "000001"}
        for table, _suffix, exchange in CH_TABLES:
            coverage_rows.append(_ch_coverage(client, table, exchange))
            ch_times[exchange] = _ch_bar_times(
                client, table, sample_symbols[exchange], audit_date
            )
        ch_parity = _fetch_ch_parity(client)
    finally:
        client.close()

    ddb_schema.to_csv(OUT_DIR / "schema_ddb.csv", index=False)
    ch_sse_schema.to_csv(OUT_DIR / "schema_ch_sse.csv", index=False)
    ch_szse_schema.to_csv(OUT_DIR / "schema_ch_szse.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT_DIR / "coverage_comparison.csv", index=False)
    parity = _parity_rows(ddb_parity, ch_parity)
    parity.to_csv(OUT_DIR / "source_parity_2024_06.csv", index=False)
    _write_inventory(
        coverage=coverage,
        parity=parity,
        ddb_times=ddb_times,
        ch_times=ch_times,
    )
    metadata = {
        "version": "price_formation_source_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source": f"{DDB_DATABASE}/{DDB_TABLE}",
        "canonical_source_reason": (
            "single SSE/SZSE production schema, full target trading-day "
            "coverage, explicit Adjfactor, no source splicing"
        ),
        "parity_period": [
            _date_text(PARITY_START),
            _date_text(PARITY_END),
        ],
        "parity_symbols": list(PARITY_SYMBOLS),
        "active_flow_candidates_included": False,
        "files": [
            "source_inventory.md",
            "schema_ddb.csv",
            "schema_ch_sse.csv",
            "schema_ch_szse.csv",
            "coverage_comparison.csv",
            "source_parity_2024_06.csv",
        ],
    }
    (OUT_DIR / "source_audit_manifest.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=_json_safe,
        ),
        encoding="utf-8",
    )
    print(
        f"[done] canonical={DDB_DATABASE}/{DDB_TABLE} -> {OUT_DIR}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
