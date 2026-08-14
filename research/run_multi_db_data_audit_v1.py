#!/usr/bin/env python3
"""Audit every database endpoint declared in COMMON_CONST.py.

Focus: find which endpoint (especially ClickHouse HF) carries snapshot/tick/L2.
Uses the same client patterns as DB_Demo.py. Never writes passwords to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import COMMON_CONST as const  # noqa: E402

L2_NAME_HINTS = (
    "snapshot",
    "tick",
    "l2",
    "level2",
    "orderbook",
    "order_book",
    "depth",
    "quote",
    "bid",
    "ask",
    "offer",
    "entrust",
    "transaction",
    "trade",
    "queue",
    "lob",
    "md",
    "hf",
    "minute",
    "klin",
    "kline",
    "bar",
)
L2_COL_HINTS = (
    "bidprice",
    "offerprice",
    "askprice",
    "bidorderqty",
    "offerorderqty",
    "askorderqty",
    "bidvolume",
    "askvolume",
    "lastpx",
    "tradeprice",
    "orderid",
    "buyorderid",
    "sellorderid",
    "totalvolumetrade",
    "weightedavgbid",
    "applseq",
)


def _safe_conn_meta(name: str, cfg: dict) -> dict:
    out = {"name": name, "keys": sorted(cfg.keys())}
    for k in ("host", "port", "dsn", "database", "userid", "user", "username"):
        if k in cfg:
            out[k] = cfg[k]
    return out


def _looks_l2_name(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in L2_NAME_HINTS)


def _score_columns(columns: List[str]) -> Dict[str, Any]:
    lower = [c.lower() for c in columns]
    hits = [c for c in columns if any(h in c.lower() for h in L2_COL_HINTS)]
    return {
        "n_columns": len(columns),
        "l2_column_hits": hits,
        "has_bid_price": any("bidprice" in c or c.startswith("bid_p") for c in lower),
        "has_ask_or_offer": any(
            ("askprice" in c) or ("offerprice" in c) or c.startswith("ask_p")
            for c in lower
        ),
        "has_order_id": any("orderid" in c or "orderno" in c for c in lower),
        "has_trade_price": any("tradeprice" in c or c == "lastpx" for c in lower),
    }


def audit_dolphindb() -> dict:
    import dolphindb as ddb
    import dolphindb.settings as keys
    import pandas as pd

    cfg = const.DATA_DB_CONN
    row = {
        "endpoint": "DATA_DB_CONN",
        "engine": "DolphinDB",
        "role": "intraday minute bars (project default)",
        "connection": _safe_conn_meta("DATA_DB_CONN", cfg),
        "ok": False,
        "error": "",
        "l2_relevant": False,
        "details": {},
    }
    try:
        s = ddb.session(protocol=keys.PROTOCOL_DDB)
        s.connect(**cfg)
        sch = pd.DataFrame(
            s.run(
                'select name from schema(loadTable('
                '"dfs://QV_Trade_to_MinuteBar","Stock_one_minute")).colDefs'
            )
        )
        cols = sch["name"].astype(str).tolist()
        score = _score_columns(cols)
        row["ok"] = True
        row["details"] = {
            "readable_table": "dfs://QV_Trade_to_MinuteBar/Stock_one_minute",
            "columns": cols,
            **score,
            "snapshot_db_exists": {
                "dfs://QV_Snapshot": bool(s.run('existsDatabase("dfs://QV_Snapshot")')),
                "dfs://QV_Tick": bool(s.run('existsDatabase("dfs://QV_Tick")')),
            },
        }
        row["l2_relevant"] = False  # minute aggregated only
        s.close()
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:500]
    return row


def audit_oracle(name: str, cfg: dict, sample_sql: str, role: str) -> dict:
    import oracledb

    row = {
        "endpoint": name,
        "engine": "Oracle",
        "role": role,
        "connection": _safe_conn_meta(name, cfg),
        "ok": False,
        "error": "",
        "l2_relevant": False,
        "details": {},
    }
    try:
        with oracledb.connect(**cfg) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sample_sql)
                rows = cursor.fetchmany(3)
                # Search for L2-ish objects in accessible schemas (bounded).
                cursor.execute(
                    """
                    select owner, object_name, object_type
                    from all_objects
                    where object_type in ('TABLE','VIEW')
                      and (
                        upper(object_name) like '%SNAPSHOT%'
                        or upper(object_name) like '%TICK%'
                        or upper(object_name) like '%LEVEL2%'
                        or upper(object_name) like '%L2%'
                        or upper(object_name) like '%ORDERBOOK%'
                        or upper(object_name) like '%ENTRUST%'
                      )
                      and rownum <= 30
                    """
                )
                hits = cursor.fetchall()
        row["ok"] = True
        row["details"] = {
            "sample_sql": sample_sql,
            "sample_n": len(rows),
            "l2_object_hits": [
                {"owner": a, "object": b, "type": c} for a, b, c in hits
            ],
        }
        row["l2_relevant"] = bool(hits)
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:500]
    return row


def audit_mysql(name: str, cfg: dict, sample_sql: str, role: str) -> dict:
    import pymysql

    row = {
        "endpoint": name,
        "engine": "MySQL",
        "role": role,
        "connection": _safe_conn_meta(name, cfg),
        "ok": False,
        "error": "",
        "l2_relevant": False,
        "details": {},
    }
    try:
        with pymysql.connect(**cfg) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sample_sql)
                rows = cursor.fetchmany(3)
                cursor.execute("show tables")
                tables = [r[0] for r in cursor.fetchall()]
                interesting = [t for t in tables if _looks_l2_name(str(t))]
                # Cap deep describe
                table_scores = []
                for t in interesting[:40]:
                    try:
                        cursor.execute(f"describe `{t}`")
                        cols = [r[0] for r in cursor.fetchall()]
                        score = _score_columns(cols)
                        if score["l2_column_hits"] or _looks_l2_name(str(t)):
                            table_scores.append(
                                {"table": t, **score, "columns_head": cols[:25]}
                            )
                    except Exception as exc:  # noqa: BLE001
                        table_scores.append({"table": t, "error": str(exc)[:160]})
        row["ok"] = True
        row["details"] = {
            "sample_sql": sample_sql,
            "sample_n": len(rows),
            "n_tables": len(tables),
            "name_hint_tables": interesting[:80],
            "scored_tables": table_scores,
        }
        row["l2_relevant"] = any(
            t.get("has_bid_price") or t.get("has_trade_price") for t in table_scores
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:500]
    return row


def audit_clickhouse_hf() -> dict:
    import clickhouse_connect

    cfg = const.DATA_DB_HFDATA
    row = {
        "endpoint": "DATA_DB_HFDATA",
        "engine": "ClickHouse",
        "role": "高频行情 / L2 candidate (cmds)",
        "connection": _safe_conn_meta("DATA_DB_HFDATA", cfg),
        "ok": False,
        "error": "",
        "l2_relevant": False,
        "details": {},
    }
    try:
        client = clickhouse_connect.get_client(**cfg)
        # databases
        dbs = [
            r[0]
            for r in client.query(
                "select name from system.databases order by name"
            ).result_rows
        ]
        # tables in configured database + any db with l2-ish name
        target_dbs = [cfg.get("database", "cmds")]
        for d in dbs:
            if d not in target_dbs and _looks_l2_name(d):
                target_dbs.append(d)

        tables_by_db: Dict[str, List[str]] = {}
        for d in target_dbs:
            q = (
                "select name from system.tables "
                f"where database = '{d}' order by name"
            )
            tables_by_db[d] = [r[0] for r in client.query(q).result_rows]

        # Score interesting tables
        scored = []
        for d, tables in tables_by_db.items():
            candidates = [t for t in tables if _looks_l2_name(t)]
            # If few tables, inspect all; else name-filtered + keyword extras
            inspect = candidates if len(tables) > 80 else tables
            # Always force-include names containing stock/snapshot/tick/l2
            for t in tables:
                low = t.lower()
                if any(
                    k in low
                    for k in (
                        "snapshot",
                        "tick",
                        "l2",
                        "level2",
                        "order",
                        "entrust",
                        "trans",
                        "quote",
                        "depth",
                        "stock",
                        "ashare",
                        "szse",
                        "sse",
                    )
                ):
                    if t not in inspect:
                        inspect.append(t)
            for t in inspect[:200]:
                try:
                    cols = [
                        r[0]
                        for r in client.query(
                            "select name from system.columns "
                            f"where database='{d}' and table='{t}' "
                            "order by position"
                        ).result_rows
                    ]
                    score = _score_columns(cols)
                    interesting = (
                        score["l2_column_hits"]
                        or score["has_bid_price"]
                        or score["has_trade_price"]
                        or _looks_l2_name(t)
                    )
                    if not interesting:
                        continue
                    # tiny sample
                    sample_n = 0
                    sample_error = ""
                    try:
                        sample = client.query(
                            f"select * from `{d}`.`{t}` limit 2"
                        )
                        sample_n = len(sample.result_rows)
                    except Exception as exc:  # noqa: BLE001
                        sample_error = str(exc)[:200]
                    scored.append(
                        {
                            "database": d,
                            "table": t,
                            **score,
                            "columns": cols,
                            "sample_n": sample_n,
                            "sample_error": sample_error,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    scored.append(
                        {
                            "database": d,
                            "table": t,
                            "error": str(exc)[:200],
                        }
                    )

        # Demo table from DB_Demo.py
        demo_ok = False
        demo_err = ""
        try:
            demo = client.query("select * from CFFEX_AL_KLIN_RTH limit 3")
            demo_ok = len(demo.result_rows) > 0
            demo_cols = list(demo.column_names)
        except Exception as exc:  # noqa: BLE001
            demo_err = str(exc)[:200]
            demo_cols = []

        snapshot_like = [
            x
            for x in scored
            if x.get("has_bid_price") and x.get("has_ask_or_offer")
        ]
        tick_like = [
            x
            for x in scored
            if x.get("has_trade_price") or x.get("has_order_id")
        ]

        row["ok"] = True
        row["l2_relevant"] = bool(snapshot_like or tick_like)
        row["details"] = {
            "databases": dbs,
            "tables_by_db_counts": {d: len(v) for d, v in tables_by_db.items()},
            "tables_by_db_names_head": {
                d: v[:60] for d, v in tables_by_db.items()
            },
            "demo_table": {
                "name": "CFFEX_AL_KLIN_RTH",
                "ok": demo_ok,
                "error": demo_err,
                "columns": demo_cols,
            },
            "scored_tables": scored,
            "snapshot_like_tables": [
                f"{x['database']}.{x['table']}" for x in snapshot_like
            ],
            "tick_like_tables": [
                f"{x['database']}.{x['table']}" for x in tick_like
            ],
        }
        client.close()
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:500]
        row["details"]["traceback"] = traceback.format_exc()[-800:]
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/results/multi_db_data_audit.json",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=ROOT / "research/docs/multi_db_data_audit_v1.md",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip Oracle endpoints (thick client / network heavy)",
    )
    parser.add_argument(
        "--skip-mysql",
        action="store_true",
    )
    args = parser.parse_args(argv)

    # Oracle thick mode once, matching DB_Demo.py
    results: List[dict] = []
    print("[audit] DolphinDB DATA_DB_CONN ...", flush=True)
    results.append(audit_dolphindb())

    if not args.skip_oracle:
        try:
            import oracledb

            oracledb.init_oracle_client(lib_dir=None)
        except Exception as exc:  # noqa: BLE001
            print(f"[audit] oracle client init warning: {exc}", flush=True)

        oracle_jobs = [
            (
                "DATA_DB_WIND",
                const.DATA_DB_WIND,
                "select * from wind.windcustomcode where rownum < 3",
                "Wind EOD / reference",
            ),
            (
                "DATA_DB_JUYUAN",
                const.DATA_DB_JUYUAN,
                "select * from jydb.secumain where rownum < 3",
                "聚源 fundamentals",
            ),
            (
                "DATA_DB_JUYUAN_2",
                const.DATA_DB_JUYUAN_2,
                "select * from jydb.secumain where rownum < 3",
                "聚源 fundamentals (replica)",
            ),
            (
                "DATA_DB_ZYYX2",
                const.DATA_DB_ZYYX2,
                "select * from zyyq.con_forecast_stk where rownum < 3",
                "朝阳永续 consensus",
            ),
            (
                "DATA_DB_ORCL",
                const.DATA_DB_ORCL,
                "select * from dual where rownum < 3",
                "generic ORCL",
            ),
            (
                "DATA_DB_CAIHUI",
                const.DATA_DB_CAIHUI,
                "select * from finchina.TQ_OA_STCODE where rownum < 3",
                "财汇",
            ),
            (
                "DATA_DB_CAIHUI_2",
                const.DATA_DB_CAIHUI_2,
                "select * from finchina.TQ_OA_STCODE where rownum < 3",
                "财汇 replica",
            ),
            (
                "DATA_DB_PAIPAI",
                const.DATA_DB_PAIPAI,
                "select * from java.pvn_fund_info where rownum < 3",
                "排排网",
            ),
            (
                "DATA_DB_PAIPAI_2",
                const.DATA_DB_PAIPAI_2,
                "select * from java.pvn_fund_info where rownum < 3",
                "排排网 replica",
            ),
            (
                "DATA_DB_PUYI",
                const.DATA_DB_PUYI,
                "select * from pystandard.bank_base_info where rownum < 3",
                "普益",
            ),
            (
                "DATA_DB_PUYI_2",
                const.DATA_DB_PUYI_2,
                "select * from pystandard.bank_base_info where rownum < 3",
                "普益 replica",
            ),
        ]
        for name, cfg, sql, role in oracle_jobs:
            print(f"[audit] Oracle {name} ...", flush=True)
            results.append(audit_oracle(name, cfg, sql, role))

    if not args.skip_mysql:
        mysql_jobs = [
            (
                "DATA_DB_DATAYES",
                const.DATA_DB_DATAYES,
                "select * from datayes.con_index limit 3",
                "通联 datayes",
            ),
            (
                "DATA_DB_YECHEN",
                const.DATA_DB_YECHEN,
                "select * from yechen.industry_chain limit 3",
                "野尘",
            ),
            (
                "DATA_DB_YECHEN_2",
                const.DATA_DB_YECHEN_2,
                "select * from yechen.industry_chain limit 3",
                "野尘 replica",
            ),
            (
                "DATA_DB_EMDATA",
                const.DATA_DB_EMDATA,
                "select * from emdata.fund_bs_cfinfo limit 3",
                "东方财富",
            ),
            (
                "DATA_DB_EMDATA_2",
                const.DATA_DB_EMDATA_2,
                "select * from emdata.fund_bs_cfinfo limit 3",
                "东方财富 replica",
            ),
        ]
        for name, cfg, sql, role in mysql_jobs:
            print(f"[audit] MySQL {name} ...", flush=True)
            results.append(audit_mysql(name, cfg, sql, role))

    print("[audit] ClickHouse DATA_DB_HFDATA ...", flush=True)
    results.append(audit_clickhouse_hf())

    summary = {
        "audit_id": "multi_db_data_audit_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": ["COMMON_CONST.py", "DB_Demo.py"],
        "endpoints": results,
        "l2_candidates": [
            r["endpoint"]
            for r in results
            if r.get("l2_relevant") or r.get("endpoint") == "DATA_DB_HFDATA"
        ],
    }
    # Promote HF details into top-level convenience fields
    hf = next((r for r in results if r["endpoint"] == "DATA_DB_HFDATA"), None)
    if hf and hf.get("ok"):
        summary["clickhouse_snapshot_like"] = hf["details"].get(
            "snapshot_like_tables", []
        )
        summary["clickhouse_tick_like"] = hf["details"].get("tick_like_tables", [])
        summary["clickhouse_ok"] = True
    else:
        summary["clickhouse_ok"] = False
        summary["clickhouse_error"] = (hf or {}).get("error", "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    # Markdown report
    lines = [
        "# Multi-DB Data Audit v1",
        "",
        f"Audited at (UTC): `{summary['audited_at_utc']}`",
        "",
        "Scope: every connection constant in `COMMON_CONST.py`, using client",
        "patterns from `DB_Demo.py`. Passwords are never written to artifacts.",
        "",
        "## Endpoint status",
        "",
        "| Endpoint | Engine | OK | L2-relevant | Role |",
        "|----------|--------|----|-------------|------|",
    ]
    for r in results:
        lines.append(
            f"| `{r['endpoint']}` | {r['engine']} | "
            f"{'yes' if r['ok'] else 'no'} | "
            f"{'yes' if r.get('l2_relevant') else 'no'} | {r.get('role','')} |"
        )
        if r.get("error"):
            lines.append(f"| | | error | | `{r['error'][:160]}` |")

    lines.extend(["", "## ClickHouse HF findings", ""])
    if hf and hf.get("ok"):
        d = hf["details"]
        lines.append(f"- Databases visible: {', '.join(d.get('databases', []))}")
        lines.append(
            f"- Table counts: `{json.dumps(d.get('tables_by_db_counts', {}))}`"
        )
        lines.append(
            f"- Demo `CFFEX_AL_KLIN_RTH` ok={d.get('demo_table', {}).get('ok')}"
        )
        snap = d.get("snapshot_like_tables") or []
        tick = d.get("tick_like_tables") or []
        lines.append(f"- Snapshot-like tables ({len(snap)}):")
        for name in snap[:40]:
            lines.append(f"  - `{name}`")
        lines.append(f"- Tick-like tables ({len(tick)}):")
        for name in tick[:40]:
            lines.append(f"  - `{name}`")
        # Detail top scored
        scored = sorted(
            d.get("scored_tables") or [],
            key=lambda x: (
                int(bool(x.get("has_bid_price"))),
                int(bool(x.get("has_trade_price"))),
                len(x.get("l2_column_hits") or []),
            ),
            reverse=True,
        )
        lines.extend(["", "### Top scored HF tables", ""])
        for x in scored[:25]:
            if x.get("error"):
                lines.append(
                    f"- `{x.get('database')}.{x.get('table')}` ERROR: {x['error']}"
                )
                continue
            lines.append(
                f"- `{x['database']}.{x['table']}` cols={x.get('n_columns')} "
                f"bid={x.get('has_bid_price')} ask/offer={x.get('has_ask_or_offer')} "
                f"trade={x.get('has_trade_price')} orderid={x.get('has_order_id')} "
                f"sample_n={x.get('sample_n')} hits={x.get('l2_column_hits')}"
            )
    else:
        lines.append(f"- ClickHouse audit failed: `{summary.get('clickhouse_error')}`")

    lines.extend(
        [
            "",
            "## Interpretation for Sprint 4.4",
            "",
            "- DolphinDB `DATA_DB_CONN` remains the minute-bar research path.",
            "- Native snapshot/tick for equities, if present, is expected under",
            "  ClickHouse `DATA_DB_HFDATA` (`cmds`), not under DDB `QV_*` DFS paths.",
            "- Oracle/MySQL endpoints are primarily fundamental / alternative;",
            "  L2 object hits there would be unexpected and are listed if found.",
            "",
            f"JSON: `{args.output}`",
            "",
        ]
    )
    args.md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {args.output}", flush=True)
    print(f"[audit] wrote {args.md}", flush=True)

    # Exit code: 0 if HF ok, 2 if HF failed
    if hf and hf.get("ok"):
        print(
            "[audit] clickhouse_snapshot_like="
            f"{len(summary.get('clickhouse_snapshot_like') or [])} "
            "tick_like="
            f"{len(summary.get('clickhouse_tick_like') or [])}",
            flush=True,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
