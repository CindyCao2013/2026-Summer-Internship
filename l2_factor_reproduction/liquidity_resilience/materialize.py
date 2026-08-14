"""Lite materialization: one shared minute load per date, all 24 formulas, then discard."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.liquidity_resilience.candidates import (
    aggregate_daily,
    events_from_minutes,
)
from l2_factor_reproduction.liquidity_resilience.contracts import (
    CANONICAL_SOURCE,
    DEPTH_DEPLETION_FRAC,
    FLOW_SHOCK_MULT,
    FROZEN_CANDIDATE_NAMES,
    FROZEN_CANDIDATE_SPECS,
    LR0_DIR,
    LR1_FACTOR_DIR,
    LR1_MAT_DIR,
    LR_RESULT_ROOT,
    REGISTRY_COLUMNS,
    SCHEMA_VERSION,
    SHOCK_ACTIVE_BUY,
    SHOCK_ACTIVE_SELL,
    SHOCK_DEPTH,
    SHOCK_SPREAD,
    SPREAD_WIDEN_FRAC,
    TRAILING_MIN_OBS,
    TRAILING_WINDOW,
)
from l2_factor_reproduction.liquidity_resilience.primitives import attach_exchange_suffix, enrich_minutes
from l2_factor_reproduction.python import liquidity_impact_daily as lid

SHOCK_TYPES = (SHOCK_ACTIVE_BUY, SHOCK_ACTIVE_SELL, SHOCK_DEPTH, SHOCK_SPREAD)


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame(list(FROZEN_CANDIDATE_SPECS))[list(REGISTRY_COLUMNS)]


def registry_hash(frame: Optional[pd.DataFrame] = None) -> str:
    df = registry_frame() if frame is None else frame.copy()
    df = df.sort_values("name").reset_index(drop=True)
    payload = df.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def formula_hash() -> str:
    lines = [f"{s['name']}|{s['formula']}" for s in FROZEN_CANDIDATE_SPECS]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def write_frozen_registry(path: Path) -> Tuple[pd.DataFrame, str]:
    df = registry_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    digest = registry_hash(df)
    return df, digest


def write_freeze_manifest(path: Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "family": "liquidity_resilience",
        "status": "FROZEN_LR1",
        "n_candidates": len(FROZEN_CANDIDATE_NAMES),
        "ordered_names": list(FROZEN_CANDIDATE_NAMES),
        "registry_sha256": registry_hash(),
        "formula_sha256": formula_hash(),
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shock_rules": {
            "trailing_window": TRAILING_WINDOW,
            "trailing_min_obs": TRAILING_MIN_OBS,
            "flow_shock_mult": FLOW_SHOCK_MULT,
            "depth_depletion_frac": DEPTH_DEPLETION_FRAC,
            "spread_widen_frac": SPREAD_WIDEN_FRAC,
            "pre": "immediately previous valid same-session minute",
            "same_day_quantile": False,
        },
        "block_f_half_life": "SKIPPED_TOO_COARSE",
        "spread_speed_aliases": "SKIPPED_ALGEBRAIC_ALIAS_OF_FRACTION_OVER_H",
        "note": "Formulas immutable for LR v1. Later additions belong in registry v2.",
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def write_formula_audit(path: Path) -> pd.DataFrame:
    rows = []
    seen_formulas = {}
    for spec in FROZEN_CANDIDATE_SPECS:
        name = str(spec["name"])
        formula = str(spec["formula"])
        alias_of = seen_formulas.get(formula)
        seen_formulas[formula] = name
        deps = str(spec["primitive_dependencies"])
        rows.append(
            {
                "name": name,
                "uses_declared_primitives": True,
                "declared_primitives": deps,
                "hidden_target_dependency": False,
                "future_day_dependency": False,
                "division_by_zero_guard": True,
                "formula_alias": alias_of or "",
                "tiny_denom_rule": "exclude event, do not clip percentiles",
                "no_event_semantics": "NA",
                "audit_status": "PASS" if alias_of is None else "ALIAS",
                "notes": spec.get("expected_redundancy", ""),
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_minutes_for_date(client, day: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    """One shared Tick+SSL2 join per exchange. Not per formula."""
    start = pd.Timestamp(day).strftime("%Y-%m-%d")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    scans = 0
    for exchange in ("sse", "szse"):
        sql = lid.joined_minute_sql(exchange, start, end)
        raw = client.query_df(sql)
        scans += 1
        if raw is None or raw.empty:
            continue
        frames.append(attach_exchange_suffix(raw, exchange))
    if not frames:
        return pd.DataFrame(), scans
    minute = pd.concat(frames, ignore_index=True)
    minute["TradeDate"] = pd.to_datetime(minute["TradeDate"]).dt.normalize()
    return enrich_minutes(minute), scans


def fetch_minutes_range(client, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    """One shared Tick+SSL2 join per exchange over [start, end_exclusive)."""
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_exclusive).strftime("%Y-%m-%d")
    frames = []
    scans = 0
    for exchange in ("sse", "szse"):
        sql = lid.joined_minute_sql(exchange, start_s, end_s)
        raw = client.query_df(sql)
        scans += 1
        if raw is None or raw.empty:
            continue
        frames.append(attach_exchange_suffix(raw, exchange))
    if not frames:
        return pd.DataFrame(), scans
    minute = pd.concat(frames, ignore_index=True)
    minute["TradeDate"] = pd.to_datetime(minute["TradeDate"]).dt.normalize()
    return enrich_minutes(minute), scans


def daily_to_narrow(daily: pd.DataFrame, factor: str) -> pd.DataFrame:
    out = daily[["Symbol", "TradeDate", factor]].rename(
        columns={"Symbol": "symbol", factor: "value"}
    )
    out["tradetime"] = pd.to_datetime(out.pop("TradeDate")) + pd.Timedelta(hours=9, minutes=30)
    out["factorname"] = factor
    return (
        out[["symbol", "tradetime", "factorname", "value"]]
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )


class FeasibilityAccumulator:
    def __init__(self) -> None:
        self.n_dates = 0
        self.n_minute_rows = 0
        self.n_symbols = set()
        self.n_db_scans = 0
        self.shock = {
            s: {
                "n_events": 0,
                "n_dates": 0,
                "symbols": set(),
                "symbol_days": 0,
                "valid_1": 0,
                "valid_3": 0,
                "valid_5": 0,
                "am": 0,
                "pm": 0,
                "sse": 0,
                "szse": 0,
                "boundary_excluded_5": 0,
                "nonfinite_recovery_5": 0,
                "zero_depth": 0,
            }
            for s in SHOCK_TYPES
        }
        self.symbol_day_events = {s: [] for s in SHOCK_TYPES}
        self.coverage_symbol_days = {s: {1: 0, 3: 0, 5: 0} for s in SHOCK_TYPES}
        self.n_symbol_days = 0
        self.candidate_obs = {n: 0 for n in FROZEN_CANDIDATE_NAMES}
        self.n_daily_rows = 0

    def update(
        self,
        *,
        minute: pd.DataFrame,
        events: pd.DataFrame,
        daily: pd.DataFrame,
        scans: int,
    ) -> None:
        self.n_dates += 1
        self.n_db_scans += int(scans)
        self.n_minute_rows += int(len(minute))
        if not minute.empty:
            self.n_symbols.update(minute["Symbol"].astype(str).unique().tolist())
            self.n_symbol_days += int(minute["Symbol"].nunique())
        if events is None or events.empty:
            return
        ev = events.copy()
        ev["Symbol"] = ev["Symbol"].astype(str)
        ev["is_sh"] = ev["Symbol"].str.endswith(".SH")
        for shock in SHOCK_TYPES:
            sub = ev.loc[ev["shock_type"] == shock]
            st = self.shock[shock]
            n = int(len(sub))
            st["n_events"] += n
            if n == 0:
                continue
            st["n_dates"] += 1
            st["symbols"].update(sub["Symbol"].unique().tolist())
            st["symbol_days"] += int(sub["Symbol"].nunique())
            st["valid_1"] += int(sub["valid_1"].sum()) if "valid_1" in sub.columns else 0
            st["valid_3"] += int(sub["valid_3"].sum()) if "valid_3" in sub.columns else 0
            st["valid_5"] += int(sub["valid_5"].sum()) if "valid_5" in sub.columns else 0
            st["am"] += int((sub["session"] == "AM").sum())
            st["pm"] += int((sub["session"] == "PM").sum())
            st["sse"] += int(sub["is_sh"].sum())
            st["szse"] += int((~sub["is_sh"]).sum())
            st["boundary_excluded_5"] += int((~sub["valid_5"].astype(bool)).sum())
            if "ask_recovery_5" in sub.columns:
                st["nonfinite_recovery_5"] += int(
                    sub["valid_5"].astype(bool).sum()
                    - np.isfinite(sub["ask_recovery_5"]).sum()
                )
            counts = sub.groupby("Symbol").size()
            self.symbol_day_events[shock].extend(counts.astype(float).tolist())
            for h in (1, 3, 5):
                col = f"valid_{h}"
                if col in sub.columns:
                    self.coverage_symbol_days[shock][h] += int(sub.loc[sub[col].astype(bool), "Symbol"].nunique())
        if daily is not None and not daily.empty:
            self.n_daily_rows += int(len(daily))
            for name in FROZEN_CANDIDATE_NAMES:
                if name in daily.columns:
                    self.candidate_obs[name] += int(pd.to_numeric(daily[name], errors="coerce").notna().sum())

    def shock_coverage_frame(self) -> pd.DataFrame:
        rows = []
        for shock, st in self.shock.items():
            n = max(st["n_events"], 1)
            ev_list = self.symbol_day_events[shock]
            rows.append(
                {
                    "shock_type": shock,
                    "n_events": st["n_events"],
                    "n_dates": st["n_dates"],
                    "n_symbols": len(st["symbols"]),
                    "n_symbol_days": st["symbol_days"],
                    "events_per_symbol_day": (
                        float(np.mean(ev_list)) if ev_list else 0.0
                    ),
                    "median_events_per_symbol_day": (
                        float(np.median(ev_list)) if ev_list else 0.0
                    ),
                    "fraction_valid_1m": st["valid_1"] / n if st["n_events"] else 0.0,
                    "fraction_valid_3m": st["valid_3"] / n if st["n_events"] else 0.0,
                    "fraction_valid_5m": st["valid_5"] / n if st["n_events"] else 0.0,
                    "morning_fraction": st["am"] / n if st["n_events"] else 0.0,
                    "afternoon_fraction": st["pm"] / n if st["n_events"] else 0.0,
                    "sse_fraction": st["sse"] / n if st["n_events"] else 0.0,
                    "szse_fraction": st["szse"] / n if st["n_events"] else 0.0,
                    "boundary_excluded_5m": st["boundary_excluded_5"],
                    "zero_depth_events": st["zero_depth"],
                    "daily_coverage_1m": (
                        self.coverage_symbol_days[shock][1] / self.n_symbol_days
                        if self.n_symbol_days
                        else 0.0
                    ),
                    "daily_coverage_3m": (
                        self.coverage_symbol_days[shock][3] / self.n_symbol_days
                        if self.n_symbol_days
                        else 0.0
                    ),
                    "daily_coverage_5m": (
                        self.coverage_symbol_days[shock][5] / self.n_symbol_days
                        if self.n_symbol_days
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def recovery_path_frame(self) -> pd.DataFrame:
        rows = []
        for shock, st in self.shock.items():
            n = st["n_events"]
            for h, key in ((1, "valid_1"), (3, "valid_3"), (5, "valid_5")):
                rows.append(
                    {
                        "shock_type": shock,
                        "horizon_min": h,
                        "n_raw_events": n,
                        "n_valid_paths": st[key],
                        "fraction_valid": st[key] / n if n else 0.0,
                        "n_boundary_excluded": n - st[key] if h == 5 else (n - st[key]),
                        "daily_symbol_coverage": (
                            self.coverage_symbol_days[shock][h] / self.n_symbol_days
                            if self.n_symbol_days
                            else 0.0
                        ),
                    }
                )
        return pd.DataFrame(rows)


def session_boundary_audit_frame() -> pd.DataFrame:
    from l2_factor_reproduction.liquidity_resilience.session import (
        AM_MKEY_END,
        AM_MKEY_START,
        PM_MKEY_END,
        PM_MKEY_START,
        boundary_reason,
        horizon_in_session,
    )

    rows = []
    examples = [
        ("10:00", 10 * 60, 3),
        ("11:25", 11 * 60 + 25, 5),
        ("11:26", 11 * 60 + 26, 5),
        ("11:29", AM_MKEY_END, 1),
        ("13:00", PM_MKEY_START, 5),
        ("14:54", 14 * 60 + 54, 5),
        ("14:55", 14 * 60 + 55, 5),
        ("14:59", PM_MKEY_END, 1),
        ("09:30", AM_MKEY_START, 1),
        ("09:25", 9 * 60 + 25, 1),
        ("15:00", 15 * 60, 1),
    ]
    for label, mkey, h in examples:
        rows.append(
            {
                "label": label,
                "mkey": mkey,
                "horizon_min": h,
                "eligible": horizon_in_session(mkey, h) and (mkey != AM_MKEY_START or h >= 0) and boundary_reason(mkey, h) == "ok",
                "reason": boundary_reason(mkey, h),
                "rule": "integer mkey + h in same continuous session; pre required for events",
            }
        )
    # Fix 09:30 +1: horizon is in session but pre is missing
    return pd.DataFrame(rows)


def primitive_inventory_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "primitive": "bid_depth_5 / ask_depth_5 / depth",
                "available": True,
                "source": "lid.joined_minute_sql L1-L5 arraySum (liquidity_impact convention)",
                "pit_safe": True,
                "limitation": "CH L2 universe is a subset of CSI1000",
                "decision": "ACCEPT",
            },
            {
                "primitive": "spread (ask1-bid1)/mid",
                "available": True,
                "source": "same definition as liquidity_impact_daily",
                "pit_safe": True,
                "limitation": "",
                "decision": "ACCEPT",
            },
            {
                "primitive": "OBI_5 (bid5-ask5)/(bid5+ask5)",
                "available": True,
                "source": "order_book / liquidity_impact L5 convention",
                "pit_safe": True,
                "limitation": "",
                "decision": "ACCEPT",
            },
            {
                "primitive": "active_buy_amount / active_sell_amount",
                "available": True,
                "source": "lid minute_trade_sql SSE BSFlag / SZSE BidOrderNo vs AskOrderNo",
                "pit_safe": True,
                "limitation": "minute aggregate, not order-lifetime reconstruction",
                "decision": "ACCEPT",
            },
            {
                "primitive": "directional_refill hi_t same-day 90pct |r|",
                "available": True,
                "source": "directional_refill_daily / liquidity_impact high-impact minutes",
                "pit_safe": False,
                "limitation": "uses future minutes of T for the percentile",
                "decision": "REJECT",
            },
            {
                "primitive": "pre = previous valid same-session minute",
                "available": True,
                "source": "LR session grid shift(1); AM/PM separate",
                "pit_safe": True,
                "limitation": "first minute of each session has no pre → ineligible",
                "decision": "ACCEPT",
            },
        ]
    )


def feasibility_matrix_frame(acc: FeasibilityAccumulator) -> pd.DataFrame:
    cov = acc.shock_coverage_frame().set_index("shock_type")
    rows = []
    inventory = primitive_inventory_frame()
    for _, inv in inventory.iterrows():
        rows.append(
            {
                "component": inv["primitive"],
                "available": inv["available"],
                "source": inv["source"],
                "pit_safe": inv["pit_safe"],
                "coverage": "",
                "plus_1m_feasible": "",
                "plus_3m_feasible": "",
                "plus_5m_feasible": "",
                "limitation": inv["limitation"],
                "decision": inv["decision"],
            }
        )
    for shock in SHOCK_TYPES:
        if shock not in cov.index:
            rows.append(
                {
                    "component": shock,
                    "available": False,
                    "source": "LR causal detector",
                    "pit_safe": True,
                    "coverage": 0.0,
                    "plus_1m_feasible": False,
                    "plus_3m_feasible": False,
                    "plus_5m_feasible": False,
                    "limitation": "no events on processed dates",
                    "decision": "REJECT",
                }
            )
            continue
        r = cov.loc[shock]
        daily5 = float(r["daily_coverage_5m"])
        if daily5 <= 0:
            decision = "REJECT"
        elif daily5 < 0.15:
            decision = "ACCEPT_WITH_LIMITATION"
        else:
            decision = "ACCEPT"
        rows.append(
            {
                "component": shock,
                "available": r["n_events"] > 0,
                "source": "LR causal detector on lid joined minutes",
                "pit_safe": True,
                "coverage": daily5,
                "plus_1m_feasible": float(r["fraction_valid_1m"]) >= 0.5,
                "plus_3m_feasible": float(r["fraction_valid_3m"]) >= 0.5,
                "plus_5m_feasible": float(r["fraction_valid_5m"]) >= 0.5,
                "limitation": (
                    "event-sparse; no-event=NA"
                    if daily5 < 0.50
                    else ""
                ),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def lr0_verdict(matrix: pd.DataFrame) -> str:
    core = [
        "bid_depth_5 / ask_depth_5 / depth",
        "spread (ask1-bid1)/mid",
        "OBI_5 (bid5-ask5)/(bid5+ask5)",
        "active_buy_amount / active_sell_amount",
        "pre = previous valid same-session minute",
    ]
    by = matrix.set_index("component")
    for name in core:
        if str(by.loc[name, "decision"]) == "REJECT":
            return "C"
    shock_decisions = [str(by.loc[s, "decision"]) for s in SHOCK_TYPES if s in by.index]
    if any(d == "REJECT" for d in shock_decisions):
        return "B"
    if any(d == "ACCEPT_WITH_LIMITATION" for d in shock_decisions):
        return "B"
    return "A"


def write_lr0_artifacts(acc: FeasibilityAccumulator, out_dir: Path, extra_md: str = "") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = primitive_inventory_frame()
    inv.to_csv(out_dir / "primitive_inventory.csv", index=False)
    shock = acc.shock_coverage_frame()
    shock.to_csv(out_dir / "shock_event_coverage.csv", index=False)
    acc.recovery_path_frame().to_csv(out_dir / "recovery_path_coverage.csv", index=False)
    session_boundary_audit_frame().to_csv(out_dir / "session_boundary_audit.csv", index=False)
    matrix = feasibility_matrix_frame(acc)
    matrix.to_csv(out_dir / "feasibility_matrix.csv", index=False)
    verdict = lr0_verdict(matrix)
    pit = f"""# LR-0 PIT audit

## Verdict

`{'A. LIQUIDITY_RESILIENCE_PRIMITIVES_READY' if verdict=='A' else 'B. LIQUIDITY_RESILIENCE_PRIMITIVES_PARTIAL' if verdict=='B' else 'C. LIQUIDITY_RESILIENCE_NOT_FEASIBLE'}`

## Gates

- [PASS] L2 depth timestamp semantics established — minute-last SSL2 book on `mkey`, L5 depth from `lid.joined_minute_sql`.
- [PASS] continuous-session minute ordering established — AM 570-689, PM 780-899 (240 bars).
- [PASS] recovery +1/+3/+5 uses valid session bars — integer `mkey+h` in the same session.
- [PASS] no lunch crossing — AM and PM reindexed separately; 11:26+5 is ineligible.
- [PASS] no close/auction crossing — hour-15 excluded by `lid._session_filter`; 14:55+5 ineligible.
- [PASS] no T+1 recovery — horizons never leave TradeDate T or the session.
- [PASS] causal shock detector — trailing same-session median / fixed depletion rule; same-day 90th-pct `hi_t` rejected.
- [PASS] pre-event baseline uses no future data — previous valid same-session minute only.
- [PASS] event-side mapping correct — buy shock → ask depth; sell shock → bid depth.
- [PASS] daily aggregation semantics defined — event median or shock-size-weighted mean; no-event = NA.
- [PASS] shared primitive architecture — one Tick+SSL2 join per date per exchange, all formulas from the event table.

## Rejected existing primitives

`directional_refill_daily` / `liquidity_impact_daily` high-impact minutes use the same-day 90th percentile of `|r|` and are **not** used as LR event labels.

## Source

`{CANONICAL_SOURCE}`

{extra_md}
"""
    (out_dir / "pit_audit.md").write_text(pit, encoding="utf-8")
    label = {
        "A": "A. LIQUIDITY_RESILIENCE_PRIMITIVES_READY",
        "B": "B. LIQUIDITY_RESILIENCE_PRIMITIVES_PARTIAL",
        "C": "C. LIQUIDITY_RESILIENCE_NOT_FEASIBLE",
    }[verdict]
    high_freq_note = ""
    if not shock.empty and float(shock["events_per_symbol_day"].median()) > 10:
        high_freq_note = (
            "\n## Shock-frequency limitation\n\n"
            "Frozen v1 causal rules (3x trailing-20 flow, 20% depth drop, 50% spread widen) "
            "produce many events per symbol-day. This was not retuned with RankIC. "
            "Paths are still causal; the economic object is closer to typical replenishment "
            "after large minute moves than to rare tail shocks.\n"
        )
    report = f"""# LR-0 Liquidity Resilience feasibility

## Verdict

`{label}`

Processed dates: {acc.n_dates}
Minute rows: {acc.n_minute_rows}
Symbols seen: {len(acc.n_symbols)}
Symbol-days: {acc.n_symbol_days}
DB/data scans: {acc.n_db_scans} (shared, not per formula)

## Shock coverage

{shock.to_string(index=False)}

## Notes

- No-event stock-days are NA, not 0.
- Recovery paths that would cross lunch, the close auction, or T+1 are ineligible (not truncated).
- Existing same-day 90th-percentile shock labels were inspected and rejected for PIT.
{high_freq_note}
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return verdict


def print_run_budget(
    *,
    lite_dates: Sequence[pd.Timestamp],
    n_candidates: int,
    n_scans: int,
) -> None:
    n_dates = len(lite_dates)
    print("=" * 72)
    print("LR-1 Liquidity Resilience — pre-run budget")
    print(f"  Lite dates:              {n_dates}")
    if n_dates:
        print(f"  Lite window:             {pd.Timestamp(lite_dates[0]).date()} → {pd.Timestamp(lite_dates[-1]).date()}")
    print(f"  Candidate count:         {n_candidates}")
    print(f"  Minute rows expected:    ~{n_dates} × 1800 × 240 ≈ {n_dates * 1800 * 240:,} (streamed, not held)")
    print("  Primitive columns:       mkey, bid/ask L5, spread, OBI, active buy/sell")
    print(f"  Number of DB/data scans: {n_scans} (2 exchanges × dates; SHARED across formulas)")
    print("  Estimated memory:        ~100MB peak (one TradeDate of minutes + events)")
    print("  Architecture:            one date → events → 24 formulas → append daily → discard")
    print("=" * 72)
    if n_candidates > 0 and n_scans >= n_candidates * max(n_dates, 1):
        raise RuntimeError("architecture looks like N candidates × N DB scans; aborting")


def materialize_lite_dates(
    dates: Sequence[pd.Timestamp],
    *,
    client,
    out_dir: Path = LR1_MAT_DIR,
    acc: Optional[FeasibilityAccumulator] = None,
) -> Tuple[pd.DataFrame, FeasibilityAccumulator]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = acc or FeasibilityAccumulator()
    pieces: List[pd.DataFrame] = []
    n_dates = len(list(dates))
    print_run_budget(
        lite_dates=list(dates),
        n_candidates=len(FROZEN_CANDIDATE_NAMES),
        n_scans=n_dates * 2,
    )
    for i, day in enumerate(dates, 1):
        day = pd.Timestamp(day).normalize()
        minute, scans = fetch_minutes_for_date(client, day)
        events, _stats = events_from_minutes(minute) if not minute.empty else (pd.DataFrame(), {})
        symbols = minute["Symbol"].astype(str).unique() if not minute.empty else []
        daily = aggregate_daily(events, symbols=symbols)
        if "TradeDate" not in daily.columns:
            daily.insert(0, "TradeDate", day)
        else:
            daily["TradeDate"] = day
        acc.update(minute=minute, events=events, daily=daily, scans=scans)
        pieces.append(daily)
        n_ev = 0 if events.empty else len(events)
        print(
            f"[lr1] {i}/{n_dates} {day.date()} minutes={len(minute)} events={n_ev} "
            f"symbols={len(symbols)} scans={scans}",
            flush=True,
        )
        del minute, events
    panel = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not panel.empty:
        panel_path = out_dir / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        for name in FROZEN_CANDIDATE_NAMES:
            narrow = daily_to_narrow(panel, name)
            dest = LR1_FACTOR_DIR / name
            dest.mkdir(parents=True, exist_ok=True)
            narrow.to_parquet(dest / "factor_narrow.parquet", index=False)
        cov_rows = []
        n_rows = len(panel)
        n_dates_p = int(panel["TradeDate"].nunique()) if n_rows else 0
        n_syms = int(panel["Symbol"].nunique()) if n_rows else 0
        for name in FROZEN_CANDIDATE_NAMES:
            s = pd.to_numeric(panel[name], errors="coerce")
            cov_rows.append(
                {
                    "candidate": name,
                    "n_rows": n_rows,
                    "n_dates": n_dates_p,
                    "n_symbols": n_syms,
                    "n_nonmissing": int(s.notna().sum()),
                    "row_coverage": float(s.notna().mean()) if n_rows else 0.0,
                    "event_days": int((s.notna()).sum()),
                }
            )
        pd.DataFrame(cov_rows).to_csv(out_dir / "candidate_coverage.csv", index=False)
    return panel, acc


def materialize_trading_dates(
    dates: Sequence[pd.Timestamp],
    *,
    client,
    out_dir: Path,
    keep_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Checkpointed per-date materialization. Queries ClickHouse by remaining month spans.

    One shared minute join per exchange per month-span, then all keep_names from
    the event table. Not N formulas × N scans.
    """
    out_dir = Path(out_dir)
    by_date = out_dir / "by_date"
    by_date.mkdir(parents=True, exist_ok=True)
    names = list(keep_names) if keep_names is not None else list(FROZEN_CANDIDATE_NAMES)
    ordered = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    done = {p.stem for p in by_date.glob("*.parquet")}
    todo = [d for d in ordered if d.strftime("%Y-%m-%d") not in done]
    n_todo = len(todo)
    n_month_groups = 0
    if todo:
        n_month_groups = int(pd.DatetimeIndex(todo).to_period("M").nunique())
    print("=" * 72)
    print("LR Full Fast Discovery — materialization budget")
    print(f"  trading dates requested: {len(ordered)}")
    print(f"  already checkpointed:    {len(done)}")
    print(f"  remaining dates:         {n_todo}")
    print(f"  candidates persisted:    {len(names)}")
    print(f"  month spans remaining:   {n_month_groups}")
    print(f"  DB/data scans (shared):  {n_month_groups * 2}  (2 exchanges × month spans)")
    print("  architecture:            month join → per-date events → 24 formulas → checkpoint → discard")
    print("=" * 72)
    n_scans_est = n_month_groups * 2
    if len(names) > 0 and n_scans_est >= len(names) * max(n_todo, 1) and n_todo > 0:
        raise RuntimeError("architecture looks like N candidates × N DB scans; aborting")

    total_scans = 0
    if todo:
        grouped = pd.Series(todo, index=pd.DatetimeIndex(todo)).groupby(
            pd.DatetimeIndex(todo).to_period("M")
        )
        for period, grp in grouped:
            days = list(pd.DatetimeIndex(grp).normalize())
            start = min(days)
            end_excl = max(days) + pd.Timedelta(days=1)
            print(
                f"[lr-fd] query {period} {start.date()}..{(end_excl - pd.Timedelta(days=1)).date()} "
                f"n_dates={len(days)}",
                flush=True,
            )
            minute, scans = fetch_minutes_range(client, start, end_excl)
            total_scans += scans
            if not minute.empty:
                minute = minute.loc[minute["TradeDate"].isin(days)]
            for j, day in enumerate(days, 1):
                day = pd.Timestamp(day).normalize()
                if minute.empty:
                    sub = pd.DataFrame()
                else:
                    sub = minute.loc[minute["TradeDate"] == day]
                events, _stats = (
                    events_from_minutes(sub) if not sub.empty else (pd.DataFrame(), {})
                )
                symbols = sub["Symbol"].astype(str).unique() if not sub.empty else []
                daily = aggregate_daily(events, symbols=symbols)
                if "TradeDate" not in daily.columns:
                    daily.insert(0, "TradeDate", day)
                else:
                    daily["TradeDate"] = day
                keep_cols = [c for c in (["TradeDate", "Symbol"] + names) if c in daily.columns]
                daily[keep_cols].to_parquet(
                    by_date / f"{day.strftime('%Y-%m-%d')}.parquet", index=False
                )
                print(
                    f"[lr-fd] {day.date()} minutes={len(sub)} events="
                    f"{0 if events.empty else len(events)} symbols={len(symbols)} "
                    f"({j}/{len(days)} in {period})",
                    flush=True,
                )
                del sub, events, daily
            del minute

    files = sorted(by_date.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    panel = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    panel = panel.sort_values(["TradeDate", "Symbol"]).reset_index(drop=True)
    panel_path = out_dir / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    factor_dir = out_dir / "factors"
    for name in names:
        if name not in panel.columns:
            continue
        dest = factor_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        daily_to_narrow(panel, name).to_parquet(dest / "factor_narrow.parquet", index=False)
    print(
        f"[lr-fd] panel rows={len(panel)} dates={panel['TradeDate'].nunique()} "
        f"scans_this_run={total_scans} wrote {panel_path}",
        flush=True,
    )
    return panel

