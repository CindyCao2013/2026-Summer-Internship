"""TC-1 operator smoke orchestration. No RankIC / Sharpe / backtest."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.apply import (
    apply_tc1_recipes,
    availability_for_recipe,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    PRODUCTION_EXECUTION_CONTRACT,
    REGISTRY_COLUMNS,
    TC1_RECIPES,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.loaders import (
    build_tc1_panel,
    load_ch_ssl2_202406,
    load_ch_tick_large_order_202406,
    load_ddb_minutes_202406,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    assert_candidate_pool_unchanged,
    registry_row,
    snapshot_candidate_pool,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import TC1_OUTPUT

LOW_COVERAGE = 0.10
ZERO_VAR = 1e-8
DUP_CORR = 0.999
EXTREME_Z = 10.0
EXTREME_SHARE = 0.05


def diagnose_wide(wide: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["TradeDate", "symbol"]
    names = [c for c in wide.columns if c not in id_cols]
    n_dates = int(wide["TradeDate"].nunique()) if "TradeDate" in wide.columns else 0
    n_symbols = int(wide["symbol"].nunique()) if "symbol" in wide.columns else 0
    n = len(wide)
    rows = []
    finite_map = {}
    for name in names:
        arr = pd.to_numeric(wide[name], errors="coerce").to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        finite_map[name] = finite
        inf_n = int(np.isinf(arr).sum()) if arr.size else 0
        nan_rate = float(np.mean(~np.isfinite(arr))) if n else float("nan")
        coverage = float(np.mean(np.isfinite(arr))) if n else float("nan")
        if finite.size:
            mu = float(np.mean(finite))
            sd = float(np.std(finite, ddof=0))
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            p1, p99 = np.quantile(finite, [0.01, 0.99])
            z = (finite - mu) / sd if sd > 0 else np.zeros_like(finite)
            extreme_share = float(np.mean(np.abs(z) > EXTREME_Z))
        else:
            mu = sd = mn = mx = p1 = p99 = extreme_share = float("nan")
        status = "OK"
        if finite.size == 0:
            status = "SKIPPED_NO_DATA"
        elif np.isfinite(coverage) and coverage < LOW_COVERAGE:
            status = "LOW_COVERAGE"
        elif np.isfinite(sd) and sd < ZERO_VAR:
            status = "ZERO_VARIANCE"
        elif np.isfinite(extreme_share) and extreme_share > EXTREME_SHARE:
            status = "EXTREME_OUTLIERS"
        rows.append(
            {
                "candidate_name": name,
                "n": n,
                "n_finite": int(finite.size),
                "coverage": coverage,
                "nan_rate": nan_rate,
                "inf_count": inf_n,
                "mean": mu,
                "std": sd,
                "min": mn,
                "max": mx,
                "p1": float(p1) if np.isfinite(p1) else float("nan"),
                "p99": float(p99) if np.isfinite(p99) else float("nan"),
                "extreme_z10_share": extreme_share,
                "coverage_n_dates": n_dates,
                "coverage_n_symbols": n_symbols,
                "status": status,
                "duplicate_of": "",
            }
        )
    # pairwise stacked pearson on overlapping finite rows
    for i, a in enumerate(names):
        xa = pd.to_numeric(wide[a], errors="coerce").to_numpy(dtype=float)
        for b in names[i + 1 :]:
            xb = pd.to_numeric(wide[b], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(xa) & np.isfinite(xb)
            if int(ok.sum()) < 20:
                continue
            corr = float(np.corrcoef(xa[ok], xb[ok])[0, 1])
            if np.isfinite(corr) and abs(corr) > DUP_CORR:
                for rec in rows:
                    if rec["candidate_name"] == b and rec["status"] == "OK":
                        rec["status"] = "DUPLICATE"
                        rec["duplicate_of"] = a
    return pd.DataFrame(rows)


def build_registry(
    metas: Sequence[Mapping[str, object]],
    diagnostics: pd.DataFrame,
    tick_meta: Mapping[str, object],
) -> pd.DataFrame:
    diag = diagnostics.set_index("candidate_name")
    rows = []
    for meta in metas:
        spec = dict(meta)
        spec["base_family"] = {
            "net_active_flow": "trade_flow",
            "obi_5": "order_book",
            "large_order_amount": "order_size",
            "minute_return": "price_formation",
            "relative_spread": "liquidity",
            "cancel_imbalance": "cancel_lifecycle",
        }.get(str(spec.get("base_primitive")), "")
        spec["cut_definition"] = spec.get("cut_name")
        spec["economic_interpretation"] = spec.get("reason")
        spec["generation_reason"] = spec.get("reason")
        name = spec["candidate_name"]
        status = spec.get("status") or "OK"
        if name in diag.index:
            dstat = str(diag.loc[name, "status"])
            if status == "OK":
                status = dstat
        if str(spec.get("base_primitive")) == "large_order_amount":
            spec["generation_reason"] = (
                str(spec.get("generation_reason") or "")
                + " | large_order_definition: {}".format(
                    tick_meta.get("large_order_definition") or "within_stock_day_top20_notional_share"
                )
            )
            if tick_meta.get("requires_ch_tick"):
                spec["generation_reason"] += " | proxy_source: DDB_AvgTradeSize"
        spec["status"] = status
        row = registry_row(spec)
        # Schema boolean stays True; TC-1 also records the frozen contract name.
        row["execution_contract_compatible"] = True
        row["execution_contract"] = PRODUCTION_EXECUTION_CONTRACT
        for key in (
            "factor_available_after",
            "latest_source_timestamp",
            "contains_close_auction",
            "contains_1456_1500",
            "uses_last_5min",
            "uses_close_auction",
            "cut_start_time",
            "cut_end_time",
            "availability_timestamp",
        ):
            if key in spec and spec[key] != "":
                row[key] = spec[key]
        row["requires_ch_tick"] = bool(
            str(spec.get("base_primitive")) == "large_order_amount"
            and tick_meta.get("requires_ch_tick")
        )
        row["source_used"] = (
            tick_meta.get("source_used")
            if str(spec.get("base_primitive")) == "large_order_amount"
            else ""
        )
        row["proxy_source"] = (
            tick_meta.get("proxy_source") or ""
            if str(spec.get("base_primitive")) == "large_order_amount"
            else ""
        )
        extra_cols = [c for c in row.keys() if c not in REGISTRY_COLUMNS]
        ordered = {c: row.get(c, "") for c in list(REGISTRY_COLUMNS) + extra_cols}
        rows.append(ordered)
    return pd.DataFrame(rows)


def write_executive_summary(
    path: Path,
    *,
    n_candidates: int,
    diagnostics: pd.DataFrame,
    timings: Mapping[str, float],
    tick_meta: Mapping[str, object],
    issues: Sequence[str],
) -> None:
    n_ok = int((diagnostics["status"] == "OK").sum()) if len(diagnostics) else 0
    n_fail = int((diagnostics["status"] != "OK").sum()) if len(diagnostics) else 0
    lines = [
        "# TC-1 Operator Smoke — Executive Summary",
        "",
        "**Date:** 2026-08-14",
        "**Window:** 2024-06-01 to 2024-06-30",
        "**No RankIC / Sharpe / backtest.**",
        "",
        "## Counts",
        "",
        "- Recipes: {}".format(len(TC1_RECIPES)),
        "- Generated candidates: {}".format(n_candidates),
        "- Diagnostics OK: {}".format(n_ok),
        "- Diagnostics flagged: {}".format(n_fail),
        "",
        "## Timings (seconds)",
        "",
    ]
    for k, v in timings.items():
        lines.append("- {}: {:.1f}".format(k, v))
    lines.extend(
        [
            "",
            "## Large-order source",
            "",
            "- source_used: {}".format(tick_meta.get("source_used")),
            "- requires_ch_tick: {}".format(tick_meta.get("requires_ch_tick")),
            "- probe_sec: {}".format(tick_meta.get("probe_sec")),
            "- skipped_reason: {}".format(tick_meta.get("skipped_reason") or "none"),
            "",
            "## Flags",
            "",
        ]
    )
    if len(diagnostics):
        vc = diagnostics["status"].value_counts()
        for st, n in vc.items():
            lines.append("- {}: {}".format(st, int(n)))
    lines.extend(["", "## Issues", ""])
    if issues:
        for item in issues:
            lines.append("- {}".format(item))
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Next",
            "",
            "Phase TC-2 (nonlinear rescue feasibility) after executable V2V labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_tc1(
    *,
    out_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> Dict[str, object]:
    out_dir = Path(out_dir or TC1_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_before = snapshot_candidate_pool()
    log_lines: List[str] = []
    timings: Dict[str, float] = {}
    issues: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    t_all = time.time()
    log("TC-1 start window=2024-06")
    ddb, sec = load_ddb_minutes_202406(cache_dir=cache_dir, force=force_reload)
    timings["load_ddb_sec"] = sec
    log("DDB minutes: rows={} symbols={} sec={:.1f}".format(
        len(ddb), ddb["symbol"].nunique() if len(ddb) else 0, sec
    ))
    ssl2, sec = load_ch_ssl2_202406(cache_dir=cache_dir, force=force_reload)
    timings["load_ch_ssl2_sec"] = sec
    log("CH SSL2 minutes: rows={} symbols={} sec={:.1f}".format(
        len(ssl2), ssl2["symbol"].nunique() if len(ssl2) else 0, sec
    ))
    tick, tick_meta = load_ch_tick_large_order_202406(
        cache_dir=cache_dir, force=force_reload
    )
    timings["load_ch_tick_sec"] = float(tick_meta.get("load_sec") or 0.0)
    log("CH tick: source={} rows={} sec={:.1f} reason={}".format(
        tick_meta.get("source_used"),
        len(tick),
        timings["load_ch_tick_sec"],
        tick_meta.get("skipped_reason") or "ok",
    ))
    t1 = time.time()
    panel = build_tc1_panel(ddb, ssl2, tick, tick_meta)
    timings["join_sec"] = time.time() - t1
    log("joined panel rows={} cols={}".format(len(panel), list(panel.columns)))

    t1 = time.time()
    wide, metas = apply_tc1_recipes(panel, TC1_RECIPES)
    timings["generate_sec"] = time.time() - t1
    n_cand = len([c for c in wide.columns if c not in ("TradeDate", "symbol")])
    log("generated candidates={}".format(n_cand))
    if n_cand < 30:
        issues.append("generated {} < 30 candidates".format(n_cand))

    t1 = time.time()
    diagnostics = diagnose_wide(wide)
    zero_map = {m["candidate_name"]: m.get("zero_denominator_count", 0) for m in metas}
    diagnostics["zero_denominator_count"] = diagnostics["candidate_name"].map(zero_map)
    skip_map = {m["candidate_name"]: m.get("skip_reason", "") for m in metas}
    diagnostics["skip_reason"] = diagnostics["candidate_name"].map(skip_map)
    timings["diagnose_sec"] = time.time() - t1
    flagged = diagnostics.loc[diagnostics["status"] != "OK"]
    if len(flagged):
        for _, row in flagged.iterrows():
            msg = "WARN {} status={}".format(row["candidate_name"], row["status"])
            log(msg)
            issues.append(msg)
    log("DDB LATE_CLOSE note: Stock_one_minute omits observed 14:57-14:59; CLOSE/FULL exclude 15:00")

    registry = build_registry(metas, diagnostics, tick_meta)
    avail_rows = []
    for meta in metas:
        avail = availability_for_recipe(meta)
        avail["candidate_name"] = meta["candidate_name"]
        avail["cut_type"] = meta.get("cut_type")
        avail["cut_name"] = meta.get("cut_name")
        avail["execution_contract_compatible"] = PRODUCTION_EXECUTION_CONTRACT
        avail["close_auction_misuse_check"] = (
            "Fail" if avail["close_auction_misuse"] else "Pass"
        )
        avail_rows.append(avail)
    avail_df = pd.DataFrame(avail_rows)
    if (avail_df["close_auction_misuse_check"] == "Fail").any():
        raise RuntimeError("close_auction_misuse Fail")

    wide_path = out_dir / "tc1_candidates.parquet"
    wide.to_parquet(wide_path, index=False)
    registry.to_csv(out_dir / "tc1_registry.csv", index=False)
    diagnostics.to_csv(out_dir / "tc1_diagnostics.csv", index=False)
    avail_df.to_csv(out_dir / "tc1_availability_report.csv", index=False)
    timings["total_sec"] = time.time() - t_all
    (out_dir / "tc1_generation_log.txt").write_text(
        "\n".join(
            log_lines
            + ["", json.dumps(timings, indent=2), json.dumps(tick_meta, indent=2, default=str)]
        )
        + "\n"
    )
    write_executive_summary(
        out_dir / "tc1_executive_summary.md",
        n_candidates=n_cand,
        diagnostics=diagnostics,
        timings=timings,
        tick_meta=tick_meta,
        issues=issues,
    )
    assert_candidate_pool_unchanged(pool_before)
    if n_cand < 30:
        raise RuntimeError("TC-1 generated fewer than 30 candidates")
    return {
        "n_candidates": n_cand,
        "out_dir": str(out_dir),
        "timings": timings,
        "tick_meta": tick_meta,
        "diagnostics": diagnostics,
    }
