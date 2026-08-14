#!/usr/bin/env python3
"""Audit accessible DolphinDB L2/minute data and map alpha opportunities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from COMMON_CONST import DATA_DB_CONN  # noqa: E402

MINUTE_DB = "dfs://QV_Trade_to_MinuteBar"
CANDIDATE_DATABASES = [
    MINUTE_DB,
    "dfs://QV_SSL2",
    "dfs://QV_OrderBook",
    "dfs://QV_Tick",
    "dfs://SSL2",
    "dfs://OrderBook",
    "dfs://Level2",
    "dfs://QV_Transaction",
    "dfs://QV_Order",
    "dfs://QV_Snapshot",
]
TABLES = [
    "Stock_one_minute",
    "Fund_one_minute",
    "Cbond_one_minute",
    "Future_one_minute",
]
FIELDS = [
    ("Symbol", "identity", "instrument key", "grouping and universe"),
    ("Date", "time", "trading date", "session grouping"),
    ("Bartime", "time", "bar timestamp", "intraday state and seasonality"),
    ("Barstart", "time", "bar start", "bar-boundary validation"),
    ("Barend", "time", "bar end", "bar-boundary validation"),
    ("Open", "price", "minute open", "returns and path"),
    ("High", "price", "minute high", "range and jump"),
    ("Low", "price", "minute low", "range and downside risk"),
    ("Close", "price", "minute close", "returns, impact, volatility, skew"),
    ("Volume", "liquidity", "traded shares", "volume curve and volume impact"),
    ("Amount", "liquidity", "traded currency amount", "Amihud and auction share"),
    ("Active_buy_volume", "flow", "aggressive buy shares", "volume OFI"),
    ("Active_sell_volume", "flow", "aggressive sell shares", "volume OFI"),
    ("Active_buy_amount", "flow", "aggressive buy amount", "amount OFI and shocks"),
    ("Active_sell_amount", "flow", "aggressive sell amount", "amount OFI and shocks"),
    ("Active_buy_count", "behavior", "aggressive buy trade count", "buy ticket size"),
    ("Active_sell_count", "behavior", "aggressive sell trade count", "sell ticket size"),
    ("Bid_cancel_volume", "cancel intent", "bid-side canceled shares", "cancel imbalance"),
    ("Bid_cancel_count", "cancel intent", "bid-side cancel count", "cancel size/intensity"),
    ("Ask_cancel_volume", "cancel intent", "ask-side canceled shares", "cancel imbalance"),
    ("Ask_cancel_count", "cancel intent", "ask-side cancel count", "cancel size/intensity"),
    ("Adjfactor", "reference", "adjustment factor", "cross-date price normalization"),
]
OPPORTUNITIES = [
    ("flow", "amount OFI / persistence / shock", "available", "P0"),
    ("flow", "volume OFI / persistence / shock", "available", "P0"),
    ("trade behavior", "buy/sell ticket size and ATS imbalance", "available", "P0"),
    ("trade behavior", "large-active-buy bar proxy", "proxy_only", "P1"),
    ("liquidity", "Amihud / signed impact / volume impact", "available", "P0"),
    ("liquidity", "liquidity shock and impact decay", "available_proxy", "P1"),
    ("cancel intent", "cancel imbalance / persistence / shock", "available", "P0"),
    ("temporal", "volume curve deviation", "available", "P0"),
    ("auction", "opening cancel imbalance", "available_open_only", "P1"),
    ("auction", "closing auction amount share", "available", "P1"),
    ("distribution", "realized vol / skew / downside vol / jump ratio", "available", "P0"),
    ("order book", "Bid-Ask depth imbalance", "blocked", "BLOCKED"),
    ("order book", "depth-weighted imbalance", "blocked", "BLOCKED"),
    ("price formation", "microprice deviation / spread / queue", "blocked", "BLOCKED"),
    ("tick behavior", "true large-trade ratio / OrderID reconstruction", "blocked", "BLOCKED"),
]
ROADMAP = [
    (1, "close_vwap_deviation", "price formation", "production", "existing"),
    (2, "volume_front_loading", "temporal", "production", "existing"),
    (3, "volume_back_loading", "temporal", "production", "existing"),
    (4, "late_session_strength", "flow", "production", "existing"),
    (5, "active_buy_sell_imbalance", "flow", "production", "existing"),
    (6, "bartime_ofi", "flow", "candidate_ddb", "Batch 1"),
    (7, "ofi_persistence", "flow", "candidate_ddb", "Batch 1"),
    (8, "active_buy_shock", "flow", "candidate_ddb", "Batch 1"),
    (9, "average_active_trade_size", "trade behavior", "candidate_ddb", "Batch 1"),
    (10, "large_active_buy_ratio", "trade behavior", "proxy_candidate", "Batch 1"),
    (11, "intraday_amihud", "liquidity", "candidate_ddb", "Batch 1"),
    (12, "realized_volatility", "distribution", "candidate_ddb", "Batch 1"),
    (13, "minute_skew", "distribution", "candidate_ddb", "Batch 1"),
    (14, "active_sell_trade_size", "trade behavior", "next_P0", "new"),
    (15, "active_trade_size_imbalance", "trade behavior", "next_P0", "new"),
    (16, "signed_price_impact", "liquidity", "next_P0", "new"),
    (17, "cancel_imbalance", "cancel intent", "next_P0", "new"),
    (18, "cancel_persistence", "cancel intent", "next_P1", "new"),
    (19, "volume_curve_deviation", "temporal", "next_P0", "new"),
    (20, "closing_auction_amount_share", "auction", "next_P1", "new"),
]


def _connect():
    import dolphindb as ddb
    import dolphindb.settings as keys

    session = ddb.session(protocol=keys.PROTOCOL_DDB)
    session.connect(**DATA_DB_CONN)
    return session


def _table_coverage(session, table: str) -> dict:
    script = f"""
t=loadTable("{MINUTE_DB}","{table}")
d=exec distinct Date from t
sy=exec distinct Symbol from t
table(min(exec Date from t) as minDate,max(exec Date from t) as maxDate,
    long(exec count(*) from t) as rows,long(size(d)) as dates,
    long(size(sy)) as symbols)
"""
    return pd.DataFrame(session.run(script)).iloc[0].to_dict()


def _field_completeness(session, audit_date: str) -> pd.DataFrame:
    fields = [name for name, *_ in FIELDS if name not in {"Symbol", "Date", "Bartime", "Barstart", "Barend"}]
    expressions = []
    for field in fields:
        expressions.extend(
            [
                f"avg(iif(isValid({field}),1.0,0.0)) as {field}_valid",
                f"avg(iif(isValid({field}) and {field}>0,1.0,0.0)) as {field}_positive",
                f"avg(iif(isValid({field}) and {field}<0,1.0,0.0)) as {field}_negative",
            ]
        )
    query = f"""
select count(*) as rows,{",".join(expressions)}
from loadTable("{MINUTE_DB}","Stock_one_minute")
where Date={audit_date} and (Symbol like "0%" or Symbol like "3%" or Symbol like "6%")
"""
    row = pd.DataFrame(session.run(query)).iloc[0]
    return pd.DataFrame(
        [
            {
                "field": field,
                "valid_rate": float(row[f"{field}_valid"]),
                "positive_rate": float(row[f"{field}_positive"]),
                "negative_rate": float(row[f"{field}_negative"]),
                "sample_rows": int(row["rows"]),
            }
            for field in fields
        ]
    )


def _session_coverage(session, audit_date: str) -> pd.DataFrame:
    fields = [
        "Volume", "Amount", "Active_buy_amount", "Active_sell_amount",
        "Active_buy_count", "Active_sell_count", "Bid_cancel_volume",
        "Ask_cancel_volume", "Bid_cancel_count", "Ask_cancel_count",
    ]
    periods = [
        ("open_auction", "second(Bartime)=09:25:00"),
        (
            "continuous",
            "second(Bartime) between 09:30:00:14:56:00 "
            "and not(second(Bartime) between 11:30:00:12:59:59)",
        ),
        ("close_auction", "second(Bartime)=15:00:00"),
    ]
    rows = []
    for label, condition in periods:
        expressions = [
            f"avg(iif(isValid({field}) and {field}>0,1.0,0.0)) as {field}"
            for field in fields
        ]
        query = f"""
select count(*) as rows,{",".join(expressions)}
from loadTable("{MINUTE_DB}","Stock_one_minute")
where Date={audit_date} and (Symbol like "0%" or Symbol like "3%" or Symbol like "6%")
  and {condition}
"""
        result = pd.DataFrame(session.run(query)).iloc[0]
        for field in fields:
            rows.append(
                {
                    "period": label,
                    "field": field,
                    "positive_rate": float(result[field]),
                    "rows": int(result["rows"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-date", default="2024.06.03")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument(
        "--output", default="research/results/l2_feature_inventory"
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    session = _connect()

    try:
        visible_catalog = list(session.run("getClusterDFSDatabases()"))
    except Exception:
        visible_catalog = []
    database_rows = []
    for path in CANDIDATE_DATABASES:
        exists = bool(session.run(f'existsDatabase("{path}")'))
        tables = (
            list(session.run(f'getTables(database("{path}"))')) if exists else []
        )
        database_rows.append(
            {
                "database": path,
                "exists": exists,
                "tables": "|".join(map(str, tables)),
            }
        )

    table_rows = []
    schema_rows = []
    for table in TABLES:
        try:
            schema = pd.DataFrame(
                session.run(
                    f'select name,typeString from schema(loadTable("{MINUTE_DB}","{table}")).colDefs'
                )
            )
            readable = True
            for record in schema.to_dict(orient="records"):
                schema_rows.append({"table": table, **record})
        except Exception as exc:
            readable = False
            schema = pd.DataFrame()
            error = str(exc)
        row = {
            "database": MINUTE_DB,
            "table": table,
            "readable": readable,
            "columns": int(len(schema)),
            "error": "" if readable else error,
        }
        if readable and args.coverage:
            row.update(_table_coverage(session, table))
        table_rows.append(row)

    fields = pd.DataFrame(
        FIELDS, columns=["field", "dimension", "meaning", "candidate_use"]
    )
    schema_names = set(
        pd.DataFrame(schema_rows)
        .query("table == 'Stock_one_minute'")["name"]
        .astype(str)
    )
    fields["available"] = fields["field"].isin(schema_names)
    completeness = _field_completeness(session, args.audit_date)
    fields = fields.merge(completeness, on="field", how="left")
    sessions = _session_coverage(session, args.audit_date)
    opportunities = pd.DataFrame(
        OPPORTUNITIES,
        columns=["dimension", "candidate_factors", "status", "priority"],
    )
    roadmap = pd.DataFrame(
        ROADMAP,
        columns=["rank", "factor", "dimension", "status", "batch"],
    )

    pd.DataFrame(database_rows).to_csv(output / "database_inventory.csv", index=False)
    pd.DataFrame(table_rows).to_csv(output / "table_inventory.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(output / "table_schemas.csv", index=False)
    fields.to_csv(output / "field_inventory.csv", index=False)
    sessions.to_csv(output / "session_coverage.csv", index=False)
    opportunities.to_csv(output / "opportunity_map.csv", index=False)
    roadmap.to_csv(output / "factor_roadmap_v2.csv", index=False)
    summary = {
        "audit_date": args.audit_date,
        "catalog_listing_visible_count": len(visible_catalog),
        "catalog_listing_note": (
            "getClusterDFSDatabases returned no paths for this read-only account; "
            "known candidate paths were probed explicitly."
        ),
        "databases": database_rows,
        "tables": table_rows,
        "stock_fields": fields.where(pd.notna(fields), None).to_dict(orient="records"),
        "session_coverage": sessions.to_dict(orient="records"),
        "opportunities": opportunities.to_dict(orient="records"),
        "factor_roadmap_v2": roadmap.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"L2 feature inventory → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
