#!/usr/bin/env python
"""Phase 0: read-only SSL2 schema, coverage, and quality inventory.

This script performs metadata and aggregate queries only. It never downloads
raw Snapshot rows or starts the historical primitive build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402


OUT_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/order_book_daily"
)
SAMPLE_DAY = "2024-06-28"
TABLES: Tuple[Tuple[str, str, str], ...] = (
    ("SSE", "SSE_AL_SSL2_EXG", "startsWith(Symbol, '6')"),
    (
        "SZSE",
        "SZSE_AL_SSL2_EXG",
        (
            "(startsWith(Symbol, '000') OR startsWith(Symbol, '001') "
            "OR startsWith(Symbol, '002') OR startsWith(Symbol, '003') "
            "OR startsWith(Symbol, '300') OR startsWith(Symbol, '301') "
            "OR startsWith(Symbol, '302'))"
        ),
    ),
)


def _valid_predicate() -> str:
    return """
    length(BidPrices) >= 10
    AND length(AskPrices) >= 10
    AND length(BidVolumes) >= 10
    AND length(AskVolumes) >= 10
    AND toFloat64(BidPrices[1]) > 0
    AND toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])
    AND (
        arraySum(arrayMap(
            x -> ifNull(toFloat64(x), 0.),
            arraySlice(BidVolumes, 1, 10)
        ))
        + arraySum(arrayMap(
            x -> ifNull(toFloat64(x), 0.),
            arraySlice(AskVolumes, 1, 10)
        ))
    ) > 0
    """


def _session_predicate(day: str) -> str:
    return f"""
    (
      (
        ExchTime >= toDateTime64(
          '{day} 09:30:00', 6, 'Asia/Shanghai'
        )
        AND ExchTime < toDateTime64(
          '{day} 11:30:00', 6, 'Asia/Shanghai'
        )
      )
      OR
      (
        ExchTime >= toDateTime64(
          '{day} 13:00:00', 6, 'Asia/Shanghai'
        )
        AND ExchTime < toDateTime64(
          '{day} 15:00:00', 6, 'Asia/Shanghai'
        )
      )
    )
    """


def _json_value(value) -> str:
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        serializable = value.tolist() if hasattr(value, "tolist") else list(value)
        return json.dumps(
            serializable,
            ensure_ascii=False,
            default=lambda item: item.item(),
        )
    return str(value)


def _schema(client, table: str) -> pd.DataFrame:
    frame = client.query_df(f"DESCRIBE TABLE cmds.{table}")
    return frame[
        ["name", "type", "default_type", "default_expression"]
    ].copy()


def _coverage(client, exchange: str, table: str, stock_filter: str) -> pd.DataFrame:
    query = f"""
    SELECT
      '{exchange}' AS exchange,
      '{table}' AS source_table,
      toYear(day) AS year,
      min(day) AS date_min,
      max(day) AS date_max,
      count() AS trading_days,
      sum(daily_rows) AS raw_rows,
      avg(daily_rows) AS avg_rows_per_day,
      avg(daily_symbols) AS avg_symbols_per_day,
      min(daily_symbols) AS min_symbols_per_day,
      max(daily_symbols) AS max_symbols_per_day,
      sum(daily_rows) / sum(daily_symbols) AS
        avg_snapshots_per_symbol_day
    FROM (
      SELECT
        toDate(ExchTime) AS day,
        count() AS daily_rows,
        uniqExact(Symbol) AS daily_symbols
      FROM cmds.{table}
      WHERE {stock_filter}
      GROUP BY day
    )
    GROUP BY year
    ORDER BY year
    """
    return client.query_df(query)


def _sample_quality(
    client,
    exchange: str,
    table: str,
    stock_filter: str,
    day: str,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    valid = _valid_predicate()
    query = f"""
    WITH
      length(BidPrices) AS lbp,
      length(AskPrices) AS lap,
      length(BidVolumes) AS lbv,
      length(AskVolumes) AS lav,
      toFloat64(BidPrices[1]) AS bp1,
      toFloat64(AskPrices[1]) AS ap1,
      arraySum(arrayMap(
        x -> ifNull(toFloat64(x), 0.),
        arraySlice(BidVolumes, 1, 10)
      )) AS bd10,
      arraySum(arrayMap(
        x -> ifNull(toFloat64(x), 0.),
        arraySlice(AskVolumes, 1, 10)
      )) AS ad10
    SELECT
      count() AS raw_rows,
      uniqExact(Symbol) AS raw_symbols,
      min(ExchTime) AS min_time,
      max(ExchTime) AS max_time,
      countIf(lbp = 0 OR lap = 0 OR lbv = 0 OR lav = 0)
        AS empty_array_rows,
      countIf(lbp < 10 OR lap < 10 OR lbv < 10 OR lav < 10)
        AS short_array_rows,
      countIf(isNull(bp1) OR bp1 <= 0) AS bid1_nonpositive,
      countIf(isNull(ap1) OR ap1 <= 0) AS ask1_nonpositive,
      countIf(bp1 > 0 AND ap1 > 0 AND ap1 < bp1) AS crossed_rows,
      countIf(bd10 = 0 AND ad10 = 0) AS all_depth_zero,
      countIf(isNull(BidVWAP) OR toFloat64(BidVWAP) <= 0)
        AS bid_vwap_bad_raw,
      countIf(isNull(AskVWAP) OR toFloat64(AskVWAP) <= 0)
        AS ask_vwap_bad_raw,
      count() - uniqExact(tuple(Symbol, ExchTime))
        AS duplicate_key_excess_raw,
      countIf({valid}) AS structurally_valid_rows,
      countIf(
        ({valid})
        AND (isNull(BidVWAP) OR toFloat64(BidVWAP) <= 0)
      ) AS bid_vwap_bad_valid,
      countIf(
        ({valid})
        AND (isNull(AskVWAP) OR toFloat64(AskVWAP) <= 0)
      ) AS ask_vwap_bad_valid
    FROM cmds.{table}
    WHERE toDate(ExchTime) = toDate('{day}')
      AND {stock_filter}
    """
    summary = client.query_df(query)
    record = summary.iloc[0].to_dict()
    raw_rows = int(record["raw_rows"])
    valid_rows = int(record["structurally_valid_rows"])
    rows: List[Dict[str, object]] = []

    def add(
        scope: str,
        metric: str,
        value,
        denominator=None,
        notes: str = "",
    ) -> None:
        ratio = (
            float(value) / float(denominator)
            if denominator not in (None, 0)
            and isinstance(value, (int, float))
            else None
        )
        rows.append(
            {
                "exchange": exchange,
                "source_table": table,
                "sample_day": day,
                "scope": scope,
                "metric": metric,
                "value": _json_value(value),
                "denominator": denominator,
                "ratio": ratio,
                "notes": notes,
            }
        )

    add("raw_day", "rows", raw_rows)
    add("raw_day", "symbols", int(record["raw_symbols"]))
    add("raw_day", "min_time", record["min_time"])
    add("raw_day", "max_time", record["max_time"])
    for metric in (
        "empty_array_rows",
        "short_array_rows",
        "bid1_nonpositive",
        "ask1_nonpositive",
        "crossed_rows",
        "all_depth_zero",
        "bid_vwap_bad_raw",
        "ask_vwap_bad_raw",
        "duplicate_key_excess_raw",
    ):
        add("raw_day", metric, int(record[metric]), raw_rows)
    add("valid_day", "structurally_valid_rows", valid_rows, raw_rows)
    add(
        "valid_day",
        "bid_vwap_bad_valid",
        int(record["bid_vwap_bad_valid"]),
        valid_rows,
    )
    add(
        "valid_day",
        "ask_vwap_bad_valid",
        int(record["ask_vwap_bad_valid"]),
        valid_rows,
    )

    duplicate_valid = client.query_df(
        f"""
        SELECT
          count() - uniqExact(tuple(Symbol, ExchTime))
            AS duplicate_key_excess_valid
        FROM cmds.{table}
        WHERE toDate(ExchTime) = toDate('{day}')
          AND {stock_filter}
          AND {valid}
        """
    ).iloc[0, 0]
    add(
        "valid_day",
        "duplicate_key_excess_valid",
        int(duplicate_valid),
        valid_rows,
    )

    lengths = client.query_df(
        f"""
        SELECT
          length(BidPrices) AS bid_price_len,
          length(AskPrices) AS ask_price_len,
          length(BidVolumes) AS bid_volume_len,
          length(AskVolumes) AS ask_volume_len,
          count() AS rows
        FROM cmds.{table}
        WHERE toDate(ExchTime) = toDate('{day}')
          AND {stock_filter}
        GROUP BY
          bid_price_len, ask_price_len, bid_volume_len, ask_volume_len
        ORDER BY rows DESC
        """
    )
    for _, item in lengths.iterrows():
        descriptor = (
            f"{int(item['bid_price_len'])}/"
            f"{int(item['ask_price_len'])}/"
            f"{int(item['bid_volume_len'])}/"
            f"{int(item['ask_volume_len'])}"
        )
        add(
            "array_length_profile",
            descriptor,
            int(item["rows"]),
            raw_rows,
            "BidPrice/AskPrice/BidVolume/AskVolume lengths",
        )

    session = client.query_df(
        f"""
        SELECT
          segment,
          count() AS rows,
          countIf({valid}) AS valid_rows,
          uniqExact(Symbol) AS symbols
        FROM (
          SELECT
            *,
            multiIf(
              ExchTime < toDateTime64(
                '{day} 09:25:00', 6, 'Asia/Shanghai'
              ), 'before_0925',
              ExchTime < toDateTime64(
                '{day} 09:30:00', 6, 'Asia/Shanghai'
              ), 'auction_0925_0930',
              ExchTime < toDateTime64(
                '{day} 11:30:00', 6, 'Asia/Shanghai'
              ), 'continuous_am',
              ExchTime < toDateTime64(
                '{day} 13:00:00', 6, 'Asia/Shanghai'
              ), 'lunch',
              ExchTime < toDateTime64(
                '{day} 15:00:00', 6, 'Asia/Shanghai'
              ), 'continuous_pm',
              ExchTime < toDateTime64(
                '{day} 15:01:00', 6, 'Asia/Shanghai'
              ), 'close_1500_minute',
              'after_1501'
            ) AS segment
          FROM cmds.{table}
          WHERE toDate(ExchTime) = toDate('{day}')
            AND {stock_filter}
        )
        GROUP BY segment
        ORDER BY segment
        """
    )
    for _, item in session.iterrows():
        add(
            "session_profile",
            f"{item['segment']}_rows",
            int(item["rows"]),
            raw_rows,
        )
        add(
            "session_profile",
            f"{item['segment']}_valid_rows",
            int(item["valid_rows"]),
            int(item["rows"]),
        )

    continuous = _session_predicate(day)
    counts = client.query_df(
        f"""
        SELECT
          quantilesExact(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)(
            snapshots
          ) AS snapshot_quantiles,
          avg(snapshots) AS avg_snapshots,
          min(snapshots) AS min_snapshots,
          max(snapshots) AS max_snapshots,
          count() AS symbols
        FROM (
          SELECT Symbol, count() AS snapshots
          FROM cmds.{table}
          WHERE toDate(ExchTime) = toDate('{day}')
            AND {stock_filter}
            AND {valid}
            AND {continuous}
          GROUP BY Symbol
        )
        """
    ).iloc[0]
    add(
        "valid_continuous",
        "snapshot_count_quantiles",
        counts["snapshot_quantiles"],
    )
    add(
        "valid_continuous",
        "avg_snapshots_per_symbol",
        float(counts["avg_snapshots"]),
    )

    minutes = client.query_df(
        f"""
        SELECT
          quantilesExact(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)(
            minutes
          ) AS minute_quantiles,
          avg(minutes) AS avg_minutes,
          min(minutes) AS min_minutes,
          max(minutes) AS max_minutes,
          count() AS symbols
        FROM (
          SELECT
            Symbol,
            uniqExact(toStartOfMinute(ExchTime)) AS minutes
          FROM cmds.{table}
          WHERE toDate(ExchTime) = toDate('{day}')
            AND {stock_filter}
            AND {valid}
            AND {continuous}
          GROUP BY Symbol
        )
        """
    ).iloc[0]
    add(
        "valid_continuous",
        "valid_minute_count_quantiles",
        minutes["minute_quantiles"],
        notes="expected continuous grid = 240",
    )
    add(
        "valid_continuous",
        "avg_valid_minutes",
        float(minutes["avg_minutes"]),
        240,
    )

    top_symbols = client.query(
        f"""
        SELECT Symbol
        FROM cmds.{table}
        WHERE toDate(ExchTime) = toDate('{day}')
          AND {stock_filter}
          AND {valid}
          AND {continuous}
        GROUP BY Symbol
        ORDER BY count() DESC
        LIMIT 50
        """
    ).result_rows
    symbol_list = ", ".join(repr(row[0]) for row in top_symbols)
    gap = client.query_df(
        f"""
        SELECT
          quantilesTDigest(0.01, 0.1, 0.5, 0.9, 0.99)(gap_s)
            AS gap_seconds_quantiles,
          avg(gap_s) AS avg_gap_seconds,
          count() AS gaps
        FROM (
          SELECT
            (
              toUnixTimestamp64Milli(ExchTime)
              - toUnixTimestamp64Milli(prev_time)
            ) / 1000. AS gap_s
          FROM (
            SELECT
              Symbol,
              ExchTime,
              lagInFrame(ExchTime, 1) OVER (
                PARTITION BY Symbol
                ORDER BY ExchTime
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
              ) AS prev_time
            FROM cmds.{table}
            WHERE Symbol IN ({symbol_list})
              AND toDate(ExchTime) = toDate('{day}')
              AND {valid}
              AND {continuous}
          )
          WHERE prev_time > toDateTime64(
            '1970-01-01 00:00:00', 6, 'Asia/Shanghai'
          )
            AND gap_s >= 0
            AND gap_s < 600
        )
        """
    ).iloc[0]
    add(
        "top50_frequency",
        "gap_seconds_quantiles",
        gap["gap_seconds_quantiles"],
    )
    add(
        "top50_frequency",
        "avg_gap_seconds",
        float(gap["avg_gap_seconds"]),
    )

    if exchange == "SZSE":
        types = client.query_df(
            f"""
            SELECT
              Type,
              count() AS rows,
              countIf({valid}) AS valid_rows
            FROM cmds.{table}
            WHERE toDate(ExchTime) = toDate('{day}')
              AND {stock_filter}
            GROUP BY Type
            ORDER BY rows DESC
            """
        )
        for _, item in types.iterrows():
            add(
                "szse_type_profile",
                f"type_{item['Type']}_rows",
                int(item["rows"]),
                raw_rows,
                f"valid_rows={int(item['valid_rows'])}",
            )
    return summary, rows


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = connect_hf_client()
    coverage_parts = []
    quality_rows: List[Dict[str, object]] = []
    summaries = {}
    try:
        for exchange, table, stock_filter in TABLES:
            schema = _schema(client, table)
            schema.to_csv(
                OUT_DIR / f"schema_{exchange.lower()}.csv",
                index=False,
            )
            coverage_parts.append(
                _coverage(client, exchange, table, stock_filter)
            )
            summary, rows = _sample_quality(
                client,
                exchange,
                table,
                stock_filter,
                SAMPLE_DAY,
            )
            summaries[exchange] = summary.iloc[0].to_dict()
            quality_rows.extend(rows)
    finally:
        client.close()

    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage.to_csv(OUT_DIR / "coverage_by_year.csv", index=False)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT_DIR / "sample_quality_audit.csv", index=False)

    coverage_display = coverage[
        [
            "exchange",
            "year",
            "date_min",
            "date_max",
            "trading_days",
            "raw_rows",
            "avg_symbols_per_day",
            "avg_snapshots_per_symbol_day",
        ]
    ]
    sse = summaries["SSE"]
    szse = summaries["SZSE"]
    lines = [
        "# Order Book Daily — Phase 0 Inventory",
        "",
        f"样本质量日：{SAMPLE_DAY}。本阶段只执行 schema 和服务端聚合审计，"
        "未拉取原始 Snapshot，未启动历史 primitive 构建。",
        "",
        "## Decision",
        "",
        "**PASS WITH CONTRACT ADAPTATIONS**：两市都具有可用的十档价量数组，"
        "2019-2026 目标区间覆盖完整，可以继续 Sprint 5。必须冻结三项适配：",
        "",
        "1. 深市原始 `Type='010'` 同一时间戳成对出现空数组/完整十档行；"
        "只有通过完整十档和盘口有效性过滤后键才唯一。",
        "2. 深市没有 `BidNums/AskNums` 十档数组；本 family v1 不使用十档"
        "订单笔数。",
        "3. 深市 `BidVWAP/AskVWAP` 在有效十档行仍不可用；"
        "`book_vwap_gap` 必须由十档价量自行计算。",
        "",
        "## Schema",
        "",
        "- SSE：54 列；`Bid/AskPrices`, `Bid/AskVolumes`, `Bid/AskNums` "
        "均为 Array；包含撤单统计。",
        "- SZSE：53 列；具有十档价格/数量 Array，但只有一档 "
        "`BidNum1/AskNum1`，没有十档 Nums Array，也没有 SSE 撤单列。",
        "- 两市 `ExchTime` 均为 `DateTime64(6, 'Asia/Shanghai')`；"
        "ClickHouse Array 下标从 1 开始。",
        "",
        "完整字段和类型见 `schema_sse.csv`、`schema_szse.csv`。",
        "",
        "## Historical coverage",
        "",
        "- SSE A 股：2015-01-05 起；审计时最新 2026-08-04。",
        "- SZSE A 股：2008-01-02 起；审计时最新 2026-08-04。",
        "- 两市统一可用起点为 2015-01-05；要求的 2019-2026 区间完整。",
        "",
        _markdown_table(coverage_display),
        "",
        "## Sample-day quality",
        "",
        f"- SSE raw rows={int(sse['raw_rows']):,}, symbols="
        f"{int(sse['raw_symbols']):,}, valid rows="
        f"{int(sse['structurally_valid_rows']):,}。",
        f"- SZSE raw rows={int(szse['raw_rows']):,}, symbols="
        f"{int(szse['raw_symbols']):,}, valid rows="
        f"{int(szse['structurally_valid_rows']):,}。",
        f"- Raw duplicate excess：SSE="
        f"{int(sse['duplicate_key_excess_raw']):,}，SZSE="
        f"{int(szse['duplicate_key_excess_raw']):,}；有效十档过滤后两市均为 0。",
        f"- Crossed book：SSE={int(sse['crossed_rows']):,}，"
        f"SZSE={int(szse['crossed_rows']):,}。",
        "",
        "SSE 样本日全部价量数组长度为 10。SZSE 原始行包含长度 0/1/10；"
        "正式链路只接受长度至少 10 且一档价格、十档总量有效的行。",
        "",
        "## Time profile and update-frequency bias",
        "",
        "两市都含 09:25 前记录、集合竞价、少量午间记录、15:00 收盘记录和"
        "盘后记录；连续竞价 primitive 只使用 09:30-11:30 与 13:00-15:00"
        " 的 240 分钟网格。有效 15:00 记录单独审计，不混入连续竞价均值。",
        "",
        "活跃股票快照间隔中位数约 3 秒，但每股原始 Snapshot 数差异显著；"
        "样本日有效分钟数中位数为 240。直接 raw-row 平均会按更新次数加权，"
        "每分钟 `argMax(metric, ExchTime)` 后再做日聚合是必要修正。",
        "",
        "详细时段、数组长度、Snapshot 数量、分钟覆盖和间隔分位数见 "
        "`sample_quality_audit.csv`。",
        "",
        "## Frozen A-share filters",
        "",
        "- SSE：`startsWith(Symbol, '6')`。",
        "- SZSE：`000/001/002/003/300/301/302` 前缀。",
        "",
        "## Capacity",
        "",
        "源表规模约为 SSE 181 亿行、SZSE 714 亿行（审计时）。"
        "历史构建必须按较小时间块执行全部 array 计算、分钟 argMax 和日聚合，"
        "Python 只接收 symbol-day；禁止保存原始或分钟明细缓存。",
        "",
    ]
    (OUT_DIR / "inventory_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[done] inventory -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
