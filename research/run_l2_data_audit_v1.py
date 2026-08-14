#!/usr/bin/env python3
"""Sprint 4.4 Phase 0 — DolphinDB L2 data availability audit.

Mandatory first step before any native snapshot factor implementation.
Probes candidate DFS databases/tables and marks which L2 families are
reachable under the current account. Does not compute factors.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from COMMON_CONST import DATA_DB_CONN  # noqa: E402

MINUTE_DB = "dfs://QV_Trade_to_MinuteBar"

# Explicit probes — getClusterDFSDatabases() is often empty for RO accounts.
CANDIDATE_DATABASES = [
    MINUTE_DB,
    "dfs://QV_SSL2",
    "dfs://QV_OrderBook",
    "dfs://QV_Tick",
    "dfs://QV_Transaction",
    "dfs://QV_Order",
    "dfs://QV_Snapshot",
    "dfs://QV_L2",
    "dfs://QV_Level2",
    "dfs://QV_Snapshot_L2",
    "dfs://SSL2",
    "dfs://OrderBook",
    "dfs://Level2",
    "dfs://L2",
    "dfs://Snapshot",
    "dfs://Tick",
    "dfs://Transaction",
    "dfs://Trade",
    "dfs://Order",
    "dfs://SH_L2",
    "dfs://SZ_L2",
    "dfs://L2Stock",
    "dfs://L2Snapshot",
    "dfs://L2Transaction",
    "dfs://WIND.ASHAREEODPRICES",
]

SNAPSHOT_MARKERS = [
    "BidPrice0",
    "OfferPrice0",
    "AskPrice0",
    "BidOrderQty0",
    "OfferOrderQty0",
    "AskOrderQty0",
    "BidVolume0",
    "AskVolume0",
    "TotalBidQty",
    "TotalOfferQty",
    "WeightedAvgBidPx",
    "WeightedAvgOfferPx",
]
TRANSACTION_MARKERS = [
    "TradePrice",
    "TradeVolume",
    "TradeAmount",
    "TradeBuyNo",
    "TradeSellNo",
    "OrderID",
    "BuyOrderID",
    "SellOrderID",
    "SeqNo",
    "ApplSeqNum",
    "ChannelNo",
]
ORDER_MARKERS = [
    "OrderPrice",
    "OrderQty",
    "OrderType",
    "Side",
    "OrderID",
    "OrderNO",
]
MINUTE_MARKERS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Amount",
    "Active_buy_amount",
    "Active_sell_amount",
    "Bid_cancel_volume",
    "Ask_cancel_volume",
]

CAPABILITY_MATRIX = [
    ("ten_level_book", "十档盘口 Bid/Ask Price/Qty 0-9", "snapshot"),
    ("trade_volume", "成交量 Volume / TradeVolume", "minute_or_tick"),
    ("trade_amount", "成交额 Amount / TradeAmount", "minute_or_tick"),
    ("vwap", "VWAP（可由 amount/volume 计算）", "derived"),
    ("wap_microprice", "WAP / microprice（需一档盘口）", "snapshot"),
    ("order_imbalance", "盘口 order imbalance（需 depth）", "snapshot"),
    ("active_buy_sell", "主动买卖 Active_buy/sell_*", "minute"),
    ("cancel_order", "撤单 Bid/Ask_cancel_*", "minute_proxy"),
    ("tick_trade_size", "逐笔成交量 / true trade size", "transaction"),
    ("order_id", "OrderID / BuyOrderID / SellOrderID", "transaction"),
]


def _connect():
    import dolphindb as ddb
    import dolphindb.settings as keys

    session = ddb.session(protocol=keys.PROTOCOL_DDB)
    session.connect(**DATA_DB_CONN)
    return session


def _safe_run(session, script: str) -> Any:
    return session.run(script)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    try:
        return [str(x) for x in list(value)]
    except TypeError:
        return [str(value)]


def _probe_catalog(session) -> Dict[str, Any]:
    note = ""
    try:
        visible = _as_str_list(_safe_run(session, "getClusterDFSDatabases()"))
    except Exception as exc:  # noqa: BLE001
        visible = []
        note = f"getClusterDFSDatabases failed: {exc}"
    if not visible and not note:
        note = (
            "getClusterDFSDatabases returned no paths for this account; "
            "candidate paths were probed with existsDatabase."
        )
    return {"visible_databases": visible, "catalog_note": note}


def _probe_database(session, path: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "database": path,
        "exists": False,
        "tables": [],
        "error": "",
    }
    try:
        exists = bool(_safe_run(session, f'existsDatabase("{path}")'))
        row["exists"] = exists
        if not exists:
            return row
        tables = _as_str_list(
            _safe_run(session, f'getTables(database("{path}"))')
        )
        row["tables"] = tables
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)
    return row


def _load_schema(session, database: str, table: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "database": database,
        "table": table,
        "readable": False,
        "columns": [],
        "column_types": {},
        "error": "",
        "row_count": None,
        "min_date": None,
        "max_date": None,
        "n_symbols": None,
    }
    try:
        schema = pd.DataFrame(
            _safe_run(
                session,
                "select name, typeString from "
                f'schema(loadTable("{database}", "{table}")).colDefs',
            )
        )
        cols = [str(x) for x in schema["name"].tolist()]
        types = {
            str(r["name"]): str(r["typeString"])
            for _, r in schema.iterrows()
        }
        out["readable"] = True
        out["columns"] = cols
        out["column_types"] = types
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out

    # Coverage stats only for the known minute bar (cheap enough).
    if database == MINUTE_DB and table.endswith("_one_minute"):
        try:
            cov = pd.DataFrame(
                _safe_run(
                    session,
                    f"""
t=loadTable("{database}","{table}")
d=exec distinct Date from t
sy=exec distinct Symbol from t
table(
    min(exec Date from t) as minDate,
    max(exec Date from t) as maxDate,
    long(exec count(*) from t) as rows,
    long(size(sy)) as symbols
)
""",
                )
            ).iloc[0]
            out["row_count"] = int(cov["rows"])
            out["min_date"] = str(cov["minDate"])
            out["max_date"] = str(cov["maxDate"])
            out["n_symbols"] = int(cov["symbols"])
        except Exception as exc:  # noqa: BLE001
            out["error"] = (out["error"] + "; " if out["error"] else "") + str(
                exc
            )
    return out


def _classify_columns(columns: List[str]) -> Dict[str, List[str]]:
    colset = set(columns)
    return {
        "snapshot_markers_present": [c for c in SNAPSHOT_MARKERS if c in colset],
        "transaction_markers_present": [
            c for c in TRANSACTION_MARKERS if c in colset
        ],
        "order_markers_present": [c for c in ORDER_MARKERS if c in colset],
        "minute_markers_present": [c for c in MINUTE_MARKERS if c in colset],
        "bid_price_levels": sorted(
            c for c in columns if c.startswith("BidPrice")
        ),
        "offer_price_levels": sorted(
            c
            for c in columns
            if c.startswith("OfferPrice") or c.startswith("AskPrice")
        ),
        "bid_qty_levels": sorted(
            c
            for c in columns
            if c.startswith("BidOrderQty") or c.startswith("BidVolume")
        ),
        "offer_qty_levels": sorted(
            c
            for c in columns
            if c.startswith("OfferOrderQty")
            or c.startswith("AskOrderQty")
            or c.startswith("AskVolume")
        ),
    }


def _capability_status(
    tables: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    all_cols: set[str] = set()
    for t in tables:
        if t.get("readable"):
            all_cols.update(t.get("columns") or [])

    has_snapshot = any(
        len(_classify_columns(t.get("columns") or [])["snapshot_markers_present"])
        > 0
        for t in tables
        if t.get("readable")
    )
    has_transaction = any(
        len(
            _classify_columns(t.get("columns") or [])[
                "transaction_markers_present"
            ]
        )
        >= 2
        for t in tables
        if t.get("readable")
    )
    has_minute = "Close" in all_cols and "Amount" in all_cols
    has_active = "Active_buy_amount" in all_cols
    has_cancel = "Bid_cancel_volume" in all_cols

    def status(flag: bool, note: str) -> Dict[str, Any]:
        return {"available": bool(flag), "note": note}

    return {
        "ten_level_book": status(
            has_snapshot,
            "requires BidPrice0-9 / OfferPrice0-9 style columns",
        ),
        "trade_volume": status(
            "Volume" in all_cols or "TradeVolume" in all_cols,
            "minute Volume or tick TradeVolume",
        ),
        "trade_amount": status(
            "Amount" in all_cols or "TradeAmount" in all_cols,
            "minute Amount or tick TradeAmount",
        ),
        "vwap": status(
            has_minute,
            "computable from minute Amount/Volume or Close path",
        ),
        "wap_microprice": status(
            has_snapshot,
            "blocked without top-of-book prices and sizes",
        ),
        "order_imbalance": status(
            has_snapshot,
            "blocked without displayed depth",
        ),
        "active_buy_sell": status(
            has_active,
            "minute Active_buy/sell amount/volume/count",
        ),
        "cancel_order": status(
            has_cancel,
            "minute cancel volume/count proxy; not raw cancel events",
        ),
        "tick_trade_size": status(
            has_transaction and "TradeVolume" in all_cols,
            "requires tick transaction feed",
        ),
        "order_id": status(
            any(c in all_cols for c in ("OrderID", "BuyOrderID", "SellOrderID")),
            "usually absent on aggregated minute bars",
        ),
    }


def _supported_families(capabilities: Dict[str, Dict[str, Any]]) -> List[str]:
    families = []
    if capabilities["active_buy_sell"]["available"]:
        families.append("trade_flow_minute")
    if capabilities["cancel_order"]["available"]:
        families.append("cancel_intent_minute")
    if capabilities["vwap"]["available"]:
        families.append("price_path_minute")
    if capabilities["trade_amount"]["available"]:
        families.append("liquidity_impact_minute")
    if capabilities["ten_level_book"]["available"]:
        families.extend(
            ["order_book", "microprice", "spread", "queue_pressure"]
        )
    if capabilities["tick_trade_size"]["available"]:
        families.append("tick_transaction")
    return families


def build_audit(session) -> Dict[str, Any]:
    catalog = _probe_catalog(session)
    databases = [_probe_database(session, path) for path in CANDIDATE_DATABASES]

    # Also probe any unexpected visible catalog paths not in the candidate list.
    for path in catalog["visible_databases"]:
        if path not in CANDIDATE_DATABASES:
            databases.append(_probe_database(session, path))

    tables: List[Dict[str, Any]] = []
    for db in databases:
        if not db["exists"]:
            continue
        for table in db["tables"]:
            schema = _load_schema(session, db["database"], table)
            schema["classification"] = _classify_columns(schema["columns"])
            tables.append(schema)

    capabilities = _capability_status(tables)
    snapshot_available = capabilities["ten_level_book"]["available"]
    transaction_available = capabilities["tick_trade_size"][
        "available"
    ] or capabilities["order_id"]["available"]
    # Broader transaction marker check
    transaction_available = transaction_available or any(
        len(t.get("classification", {}).get("transaction_markers_present", []))
        >= 2
        for t in tables
        if t.get("readable")
    )

    supported = _supported_families(capabilities)
    blocked = [
        name
        for name, meta in capabilities.items()
        if not meta["available"]
        and name
        in {
            "ten_level_book",
            "wap_microprice",
            "order_imbalance",
            "tick_trade_size",
            "order_id",
        }
    ]

    stop_reason = None
    if not snapshot_available:
        stop_reason = (
            "Native Level-2 snapshot columns are unavailable under the current "
            "DolphinDB account. Phase 1+ snapshot factor implementation is "
            "blocked. Continue only with minute-bar derived expansion or "
            "request snapshot entitlement."
        )

    return {
        "audit_id": "l2_data_audit_v1",
        "sprint": "4.4_phase0",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "account_scope": "current DolphinDB credentials in COMMON_CONST.DATA_DB_CONN",
        "catalog": catalog,
        "snapshot_available": snapshot_available,
        "transaction_available": bool(transaction_available),
        "minute_bar_available": any(
            t.get("readable") and t.get("database") == MINUTE_DB for t in tables
        ),
        "databases_probed": databases,
        "tables": [
            {
                "database": t["database"],
                "table": t["table"],
                "readable": t["readable"],
                "columns": t["columns"],
                "column_types": t["column_types"],
                "classification": t.get("classification", {}),
                "row_count": t.get("row_count"),
                "min_date": t.get("min_date"),
                "max_date": t.get("max_date"),
                "n_symbols": t.get("n_symbols"),
                "error": t.get("error") or "",
            }
            for t in tables
        ],
        "capabilities": capabilities,
        "capability_matrix_labels": [
            {"key": k, "label": lab, "layer": layer}
            for k, lab, layer in CAPABILITY_MATRIX
        ],
        "supported_factor_family": supported,
        "blocked_factor_family": [
            "order_book",
            "microprice",
            "spread",
            "queue_pressure",
            "tick_transaction",
        ]
        if not snapshot_available
        else [],
        "phase0_stop": not snapshot_available,
        "stop_reason": stop_reason,
        "recommended_next_step": (
            "Do not implement native snapshot factors. Either (a) request "
            "Level-2 snapshot / tick entitlement and re-audit, or (b) expand "
            "the minute-bar factory on available Active_* and cancel fields "
            "(cancel_imbalance, signed_price_impact, downside_RV) while keeping "
            "evaluation via intraday_evaluation_v2."
            if not snapshot_available
            else "Snapshot available — proceed to Phase 1 DDB-native feature engine."
        ),
        "prior_inventory": {
            "doc": "research/docs/l2_feature_inventory_20260730.md",
            "runner": "research/run_l2_feature_inventory.py",
            "note": "Sprint 4.0 inventory; this Phase 0 re-audit supersedes for Sprint 4.4 gating.",
        },
    }


def write_markdown(audit: Dict[str, Any], path: Path) -> None:
    caps = audit["capabilities"]
    lines = [
        "# Sprint 4.4 Phase 0 — DDB L2 Data Audit",
        "",
        f"Audited at (UTC): `{audit['audited_at_utc']}`",
        "",
        "## Gate decision",
        "",
        f"- `snapshot_available`: **{audit['snapshot_available']}**",
        f"- `transaction_available`: **{audit['transaction_available']}**",
        f"- `minute_bar_available`: **{audit['minute_bar_available']}**",
        f"- `phase0_stop`: **{audit['phase0_stop']}**",
        "",
    ]
    if audit.get("stop_reason"):
        lines.extend(["> " + audit["stop_reason"], ""])
    lines.extend(
        [
            "## Capability matrix",
            "",
            "| Feature | Available | Note |",
            "|---------|-----------|------|",
        ]
    )
    labels = {x["key"]: x["label"] for x in audit["capability_matrix_labels"]}
    for key, meta in caps.items():
        lines.append(
            f"| {labels.get(key, key)} | "
            f"{'yes' if meta['available'] else 'no'} | {meta['note']} |"
        )
    lines.extend(
        [
            "",
            "## Supported vs blocked factor families",
            "",
            "Supported under current grant:",
            "",
        ]
    )
    for fam in audit["supported_factor_family"]:
        lines.append(f"- `{fam}`")
    lines.extend(["", "Blocked (native L2 snapshot / tick):", ""])
    for fam in audit["blocked_factor_family"]:
        lines.append(f"- `{fam}`")
    lines.extend(
        [
            "",
            "## Readable tables",
            "",
        ]
    )
    readable = [t for t in audit["tables"] if t["readable"]]
    if not readable:
        lines.append("_No readable candidate tables._")
    for t in readable:
        lines.append(
            f"- `{t['database']}/{t['table']}` — {len(t['columns'])} columns"
            + (
                f", rows={t['row_count']}, {t['min_date']}→{t['max_date']}"
                if t.get("row_count") is not None
                else ""
            )
        )
        markers = t.get("classification", {})
        snap = markers.get("snapshot_markers_present") or []
        txn = markers.get("transaction_markers_present") or []
        if snap:
            lines.append(f"  - snapshot markers: {', '.join(snap)}")
        if txn:
            lines.append(f"  - transaction markers: {', '.join(txn)}")
    lines.extend(
        [
            "",
            "## Recommended next step",
            "",
            audit["recommended_next_step"],
            "",
            "## Artifacts",
            "",
            "- `research/results/l2_data_audit.json`",
            "- `research/docs/l2_data_audit_v1.md`",
            "- Runner: `research/run_l2_data_audit_v1.py`",
            "",
            "Phase 1+ snapshot factor code was **not** implemented.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "research/results/l2_data_audit.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "research/docs/l2_data_audit_v1.md",
    )
    args = parser.parse_args(argv)

    session = _connect()
    audit = build_audit(session)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(audit, args.output_md)

    print(
        f"[l2_data_audit] snapshot_available={audit['snapshot_available']} "
        f"transaction_available={audit['transaction_available']} "
        f"phase0_stop={audit['phase0_stop']}",
        flush=True,
    )
    print(f"[l2_data_audit] wrote {args.output_json}", flush=True)
    print(f"[l2_data_audit] wrote {args.output_md}", flush=True)
    if audit["phase0_stop"]:
        print(f"[l2_data_audit] STOP: {audit['stop_reason']}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
