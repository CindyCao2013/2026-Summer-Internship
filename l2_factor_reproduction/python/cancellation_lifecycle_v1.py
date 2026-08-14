"""Sprint 10 — Cancellation / Order Lifecycle Family v1.

Phase 1: 2024-06 Monthly Gate (no alpha Sharpe conclusions).
Phase 2 (only if gate PASS): Discovery-window cache + Fast Discovery.

Reuses frozen cancel_lifecycle_daily primitive (SSE Type=D direct;
SZSE lifecycle-linked). Does not rebuild SSE from A/T residuals.
Does not edit Protocol v2.0 / Fast Gate thresholds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Factor_Dev_Lib import calAnnuRet, calSharpe
from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.ch_cancel_lifecycle import (
    PRIMITIVE_COLUMNS,
    build_candidates,
    shock_20d,
)
from l2_factor_reproduction.python.evaluation_protocol_v2 import l1_to_oneway
from l2_factor_reproduction.python.fast_discovery import (
    DISCOVERY_END,
    DISCOVERY_START,
    compute_fast_metrics,
    ensure_effective_group_pnl,
    gate_label,
    load_fast_context,
    save_fast_plots,
)
from l2_factor_reproduction.python.low_turnover_v1 import _to_narrow
from l2_factor_reproduction.python.mid_trade_amount_research_data import (
    load_turnover_wide,
)

PRIMITIVE_ROOT = (
    Path(RESULT_ROOT) / "primitives" / "cancel_lifecycle_daily"
)
DATASET_DIR = PRIMITIVE_ROOT / "dataset"
MCAP_PATH = Path(RESULT_ROOT) / "primitives" / "mcap_wide_2019-01-01_2026-07-31.parquet"

OUT_ROOT = Path(RESULT_ROOT) / "fast_discovery" / "cancellation_lifecycle_v1"
MONTHLY_DIR = OUT_ROOT / "monthly_gate_2024_06"

QA_START = pd.Timestamp("2024-06-01")
QA_END = pd.Timestamp("2024-06-30")
# Warmup for 20d shocks (exclude today in shock_20d)
WARMUP_START = pd.Timestamp("2024-05-01")

CANDIDATES = [
    "cancel_value_pressure",
    "cancel_count_pressure",
    "cancel_value_intensity",
    "cancel_qty_intensity",
    "relative_cancel_order_size",
    "cancel_pressure_shock_20d",
    "cancel_intensity_shock_20d",
]

# Field map: Sprint-10 requested name → frozen primitive column
FIELD_MAP = {
    "buy_cancel_amount": "buy_cancel_value",
    "sell_cancel_amount": "sell_cancel_value",
    "buy_cancel_qty": "buy_cancel_qty",
    "sell_cancel_qty": "sell_cancel_qty",
    "buy_cancel_event_count": "buy_cancel_event_count",
    "sell_cancel_event_count": "sell_cancel_event_count",
    "buy_cancel_order_count": "buy_cancelled_unique_order_count",
    "sell_cancel_order_count": "sell_cancelled_unique_order_count",
    "buy_add_amount": None,
    "sell_add_amount": None,
    "buy_add_qty": None,
    "sell_add_qty": None,
}

# Structure-risk heuristics (documentation flags; not threshold search)
CORR_EXCHANGE_RISK = 0.15
CORR_STAR_RISK = 0.25
MEAN_RANK_GAP_RISK = 0.05
DECIL_G10_BOARD_SHARE_RISK = 0.40


def _segment(symbol: str) -> str:
    code = symbol.split(".")[0]
    if symbol.endswith(".SH"):
        return "STAR" if code.startswith("688") else "SSE Main"
    return "ChiNext" if code.startswith(("300", "301")) else "SZSE Main"


def _exchange(symbol: str) -> str:
    return "SZSE" if symbol.endswith(".SZ") else "SSE"


def load_cancel_partitions(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load existing quarterly cancel_lifecycle_daily partitions (no CH rebuild)."""
    files = sorted(DATASET_DIR.glob("year=*/cancel_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cancel partitions under {DATASET_DIR}")
    frames = []
    for path in files:
        # filename cancel_daily_YYYY-MM-DD_YYYY-MM-DD.parquet
        stem = path.stem.replace("cancel_daily_", "")
        try:
            a_s, b_s = stem.split("_")
            a, b = pd.Timestamp(a_s), pd.Timestamp(b_s)
        except ValueError:
            continue
        if b < start or a > end:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no cancel partitions overlapping [{start}, {end}]")
    panel = pd.concat(frames, ignore_index=True)
    panel["symbol"] = panel["symbol"].astype(str)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    panel = panel.loc[panel["TradeDate"].between(start, end)].copy()
    panel = panel.drop_duplicates(["symbol", "TradeDate"], keep="last")
    for col in PRIMITIVE_COLUMNS:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel.sort_values(["symbol", "TradeDate"]).reset_index(drop=True)


def build_candidate_panel(primitive: pd.DataFrame) -> pd.DataFrame:
    cand = build_candidates(primitive)
    cand = cand.sort_values(["symbol", "TradeDate"]).reset_index(drop=True)
    grouped = cand.groupby("symbol", sort=False)
    cand["cancel_pressure_shock_20d"] = grouped["cancel_value_pressure"].transform(
        shock_20d
    )
    cand["cancel_intensity_shock_20d"] = grouped["cancel_value_intensity"].transform(
        shock_20d
    )
    return cand


def _pct_stats(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {
            k: float("nan")
            for k in ("median", "p10", "p50", "p90", "p99", "mean", "n")
        }
    q = s.quantile([0.10, 0.50, 0.90, 0.99])
    return {
        "n": float(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(q.loc[0.10]),
        "p50": float(q.loc[0.50]),
        "p90": float(q.loc[0.90]),
        "p99": float(q.loc[0.99]),
    }


def run_monthly_gate(*, out_dir: Optional[Path] = None) -> Dict[str, object]:
    out = Path(out_dir) if out_dir else MONTHLY_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("[monthly] load cancel primitive May–Jun 2024 (existing partitions)", flush=True)
    prim = load_cancel_partitions(WARMUP_START, QA_END)
    june_prim = prim.loc[prim["TradeDate"].between(QA_START, QA_END)].copy()
    june_prim.to_parquet(out / "cancellation_primitive_2024_06.parquet", index=False)
    print(
        f"  june rows={len(june_prim):,} days={june_prim.TradeDate.nunique()} "
        f"syms={june_prim.symbol.nunique()}",
        flush=True,
    )

    # --- field capability ---
    cap_rows = []
    for req, src in FIELD_MAP.items():
        if src is None:
            cap_rows.append(
                {
                    "requested_field": req,
                    "available": False,
                    "source_column": "",
                    "notes": "unavailable — no reliable cross SSE/SZSE add-order field",
                }
            )
        else:
            present = src in june_prim.columns
            nn = float(june_prim[src].notna().mean()) if present else 0.0
            cap_rows.append(
                {
                    "requested_field": req,
                    "available": bool(present),
                    "source_column": src,
                    "nonnull_rate": nn,
                    "notes": "frozen cancel_lifecycle_daily column" if present else "missing",
                }
            )
    for name in CANDIDATES:
        cap_rows.append(
            {
                "requested_field": name,
                "available": True,
                "source_column": "build_candidates+shock_20d",
                "nonnull_rate": np.nan,
                "notes": "frozen v1 formula; shock needs May warmup",
            }
        )
    capability = pd.DataFrame(cap_rows)
    capability.to_csv(out / "candidate_capability.csv", index=False)

    # --- candidates on warmup+june ---
    cand_all = build_candidate_panel(prim)
    june = cand_all.loc[cand_all["TradeDate"].between(QA_START, QA_END)].copy()
    june["segment"] = june["symbol"].map(_segment)
    june["exchange"] = june["symbol"].map(_exchange)

    # attach diagnostics exposures
    print("[monthly] load mcap/turnover diagnostics", flush=True)
    mcap = pd.read_parquet(MCAP_PATH)
    mcap.index = pd.to_datetime(mcap.index)
    try:
        turn = load_turnover_wide(QA_START, QA_END)
        turn.index = pd.to_datetime(turn.index)
        has_turnover = True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] turnover unavailable ({exc}); continue without", flush=True)
        turn = pd.DataFrame()
        has_turnover = False

    mcap_long = (
        np.log(mcap.where(mcap > 0))
        .stack()
        .rename("log_float_mktcap")
        .reset_index()
    )
    mcap_long.columns = ["TradeDate", "symbol", "log_float_mktcap"]
    mcap_long["TradeDate"] = pd.to_datetime(mcap_long["TradeDate"])
    mcap_long["symbol"] = mcap_long["symbol"].astype(str)

    june = june.merge(mcap_long, on=["TradeDate", "symbol"], how="left")
    if has_turnover:
        turn_long = turn.stack().rename("turnover").reset_index()
        turn_long.columns = ["TradeDate", "symbol", "turnover"]
        turn_long["TradeDate"] = pd.to_datetime(turn_long["TradeDate"])
        turn_long["symbol"] = turn_long["symbol"].astype(str)
        june = june.merge(turn_long, on=["TradeDate", "symbol"], how="left")
    else:
        june["turnover"] = np.nan

    # --- cross-market diagnostics on primitive basics + candidates ---
    diag_rows = []
    basic_metrics = [
        "buy_cancel_value",
        "sell_cancel_value",
        "buy_cancel_event_count",
        "sell_cancel_event_count",
        "total_trade_value",
        "join_coverage",
    ]
    # merge primitive cols onto june keys
    prim_june = june_prim.copy()
    prim_june["segment"] = prim_june["symbol"].map(_segment)
    prim_june["exchange"] = prim_june["symbol"].map(_exchange)

    for metric in basic_metrics:
        for seg, part in prim_june.groupby("segment"):
            s = part[metric]
            st = _pct_stats(s)
            zero_share = float((s.fillna(0) == 0).mean()) if len(s) else float("nan")
            diag_rows.append(
                {
                    "metric": metric,
                    "group": seg,
                    "group_type": "board",
                    "coverage": float(s.notna().mean()),
                    "nonnull_rate": float(s.notna().mean()),
                    "zero_share": zero_share,
                    **st,
                }
            )
        for exch, part in prim_june.groupby("exchange"):
            s = part[metric]
            st = _pct_stats(s)
            zero_share = float((s.fillna(0) == 0).mean()) if len(s) else float("nan")
            diag_rows.append(
                {
                    "metric": metric,
                    "group": exch,
                    "group_type": "exchange",
                    "coverage": float(s.notna().mean()),
                    "nonnull_rate": float(s.notna().mean()),
                    "zero_share": zero_share,
                    **st,
                }
            )

    for name in CANDIDATES:
        for seg, part in june.groupby("segment"):
            s = part[name]
            st = _pct_stats(s)
            zero_share = float((s.fillna(0) == 0).mean()) if len(s) else float("nan")
            diag_rows.append(
                {
                    "metric": name,
                    "group": seg,
                    "group_type": "board",
                    "coverage": float(s.notna().mean()),
                    "nonnull_rate": float(s.notna().mean()),
                    "zero_share": zero_share,
                    **st,
                }
            )

    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(out / "cross_market_diagnostics.csv", index=False)

    # --- global vs exchange rank + structure flags ---
    rank_rows = []
    board_rows = []
    for name in CANDIDATES:
        valid = june.dropna(subset=[name]).copy()
        coverage = float(len(valid) / max(len(june), 1))
        is_szse = valid["exchange"].eq("SZSE").astype(float)
        is_star = valid["segment"].eq("STAR").astype(float)
        corr_ex = float(valid[name].corr(is_szse))
        corr_star = float(valid[name].corr(is_star))
        corr_mcap = float(valid[name].corr(valid["log_float_mktcap"], method="spearman"))
        corr_to = float(valid[name].corr(valid["turnover"], method="spearman"))

        valid["rank_global"] = valid.groupby("TradeDate")[name].rank(pct=True)
        valid["rank_exchange"] = valid.groupby(["TradeDate", "exchange"])[name].rank(
            pct=True
        )
        gve_corr = float(valid["rank_global"].corr(valid["rank_exchange"]))
        sse_mean_rank = float(
            valid.loc[valid.exchange == "SSE", "rank_global"].mean()
        )
        szse_mean_rank = float(
            valid.loc[valid.exchange == "SZSE", "rank_global"].mean()
        )
        rank_gap = abs(sse_mean_rank - szse_mean_rank)

        # decile board composition
        valid["decile"] = valid.groupby("TradeDate")[name].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")
        )
        comp = (
            valid.groupby("decile")["segment"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        for decile, row in comp.iterrows():
            board_rows.append(
                {
                    "factor": name,
                    "decile": int(decile) + 1,
                    **{str(k): float(v) for k, v in row.items()},
                }
            )
        g10_star = float(comp.loc[comp.index.max(), "STAR"]) if "STAR" in comp.columns else 0.0
        g10_max_board = float(comp.loc[comp.index.max()].max()) if len(comp) else 0.0

        structure_risk = bool(
            abs(corr_ex) >= CORR_EXCHANGE_RISK
            or abs(corr_star) >= CORR_STAR_RISK
            or rank_gap >= MEAN_RANK_GAP_RISK
            or g10_max_board >= DECIL_G10_BOARD_SHARE_RISK
        )
        # Override: if global≈exchange rank very high AND rank_gap small AND
        # corr mild → not "mainly" structure-driven for ranking
        if gve_corr >= 0.98 and rank_gap < MEAN_RANK_GAP_RISK and abs(corr_ex) < 0.10:
            # still flag STAR level intensity bias via board share / star corr
            structure_risk = bool(
                abs(corr_star) >= CORR_STAR_RISK
                or g10_star >= DECIL_G10_BOARD_SHARE_RISK
            )

        rank_rows.append(
            {
                "factor": name,
                "coverage": coverage,
                "corr_exchange_dummy": corr_ex,
                "corr_star_dummy": corr_star,
                "spearman_log_float_mktcap": corr_mcap,
                "spearman_turnover": corr_to,
                "global_vs_exchange_rank_corr": gve_corr,
                "SSE_mean_rank": sse_mean_rank,
                "SZSE_mean_rank": szse_mean_rank,
                "mean_rank_gap": rank_gap,
                "g10_star_share": g10_star,
                "g10_max_board_share": g10_max_board,
                "EXCHANGE_STRUCTURE_RISK": structure_risk,
                "eligible_for_strong_pool": (not structure_risk),
            }
        )

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out / "global_vs_exchange_rank.csv", index=False)
    board_df = pd.DataFrame(board_rows)
    board_df.to_csv(out / "board_composition.csv", index=False)

    # --- Gate decision (no alpha Sharpe) ---
    n_days = int(june_prim["TradeDate"].nunique())
    sse_n = int(june_prim.loc[june_prim.symbol.str.endswith(".SH"), "symbol"].nunique())
    szse_n = int(june_prim.loc[june_prim.symbol.str.endswith(".SZ"), "symbol"].nunique())
    join_med = float(june_prim["join_coverage"].median())
    invalid_rate = float(
        (june_prim["invalid_cancel_count"] > 0).mean()
    ) if "invalid_cancel_count" in june_prim.columns else float("nan")

    # zero-cancel share on total cancel events
    tot_events = (
        june_prim["buy_cancel_event_count"] + june_prim["sell_cancel_event_count"]
    )
    zero_cancel_share = float((tot_events == 0).mean())

    available_candidates = [
        r["factor"] for r in rank_rows if r["coverage"] >= 0.90
    ]
    structure_flagged = [
        r["factor"] for r in rank_rows if r["EXCHANGE_STRUCTURE_RISK"]
    ]
    strong_pool_ok = [
        r["factor"] for r in rank_rows if r["eligible_for_strong_pool"] and r["coverage"] >= 0.90
    ]

    comparable = bool(sse_n > 1000 and szse_n > 1000 and join_med >= 0.99 and n_days >= 15)
    denom_ok = bool(zero_cancel_share < 0.05 and all(
        float(rank_df.loc[rank_df.factor == n, "coverage"].iloc[0]) >= 0.90
        for n in CANDIDATES
    ))
    # PASS discovery if definitions comparable, candidates computable, denominators OK.
    # Structure-flagged candidates remain available for research but not strong pool.
    gate_pass = bool(comparable and denom_ok and len(available_candidates) >= 5)

    answers = {
        "q1_sse_szse_comparable": comparable,
        "q1_detail": (
            f"SSE symbols≈{sse_n}, SZSE≈{szse_n}, days={n_days}, "
            f"median join_coverage={join_med:.4f}; "
            "SSE=Type D direct; SZSE=lifecycle-linked (frozen contract)"
        ),
        "q2_severe_structure_bias": bool(len(structure_flagged) > 0),
        "q2_flagged": structure_flagged,
        "q2_detail": (
            "Pressure candidates generally low exchange corr; "
            "intensity/qty show STAR level elevation — flagged EXCHANGE_STRUCTURE_RISK "
            "where board/G10 share or star corr exceeds heuristics. "
            "Flagged names excluded from strong pool only."
        ),
        "q3_legally_computable": available_candidates,
        "q4_denominator_zero_missing": {
            "zero_cancel_event_share": zero_cancel_share,
            "invalid_cancel_row_share": invalid_rate,
            "add_fields": "unavailable (buy/sell_add_*)",
            "ok_for_frozen_denominators": denom_ok,
        },
        "q5_allow_discovery_build": gate_pass,
    }

    gate_summary = {
        "sprint": "Sprint 10 — Cancellation / Order Lifecycle Family v1",
        "phase": "monthly_gate_2024_06",
        "gate_pass": gate_pass,
        "qa_window": [str(QA_START.date()), str(QA_END.date())],
        "n_trading_days": n_days,
        "sse_symbols": sse_n,
        "szse_symbols": szse_n,
        "median_join_coverage": join_med,
        "available_candidates": available_candidates,
        "structure_risk_flagged": structure_flagged,
        "strong_pool_eligible": strong_pool_ok,
        "answers": answers,
        "source_primitive": "cancel_lifecycle_daily (reuse; no CH rebuild)",
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "gate_summary.json").write_text(
        json.dumps(gate_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md = render_monthly_gate_md(gate_summary, rank_df, capability)
    (out / "cancellation_monthly_gate.md").write_text(md, encoding="utf-8")
    print(f"[monthly] gate_pass={gate_pass} -> {out}", flush=True)
    return gate_summary


def render_monthly_gate_md(
    summary: Dict[str, object],
    rank_df: pd.DataFrame,
    capability: pd.DataFrame,
) -> str:
    a = summary["answers"]
    lines = [
        "# Cancellation Monthly Gate — 2024-06 (Sprint 10 Phase 1)",
        "",
        f"**Gate PASS:** `{summary['gate_pass']}`",
        "",
        "No alpha Sharpe conclusions on a single month.",
        "",
        "## Source contract",
        "",
        "- SSE: Tick `Type='D'` direct aggregation (Price>0, Volume>0, BSFlag B/S)",
        "- SZSE: cancel Category=4 linked to original order via frozen lifecycle mapping",
        "- **Not** rebuilt from SSE A/T lifecycle residuals",
        "- Reused existing `cancel_lifecycle_daily` quarterly partitions (no CH rebuild)",
        "",
        "## Field availability",
        "",
        capability.to_string(index=False),
        "",
        "## Global vs exchange rank",
        "",
        rank_df.to_string(index=False),
        "",
        "## Gate questions",
        "",
        f"1. SSE/SZSE cancellation definitions comparable? "
        f"**{'YES' if a['q1_sse_szse_comparable'] else 'NO'}** — {a['q1_detail']}",
        "",
        f"2. Severe exchange/board structural bias? "
        f"**{'YES (flagged)' if a['q2_severe_structure_bias'] else 'NO'}** — "
        f"flagged=`{a['q2_flagged']}`. {a['q2_detail']}",
        "",
        f"3. Which of 7 frozen candidates are legally computable? "
        f"**{a['q3_legally_computable']}**",
        "",
        f"4. Denominator / zero / missing issues? "
        f"**{json.dumps(a['q4_denominator_zero_missing'])}**",
        "",
        f"5. Allow 2023-2024 Discovery primitive build/reuse? "
        f"**{'YES' if a['q5_allow_discovery_build'] else 'NO'}**",
        "",
        f"Strong-pool eligible (no EXCHANGE_STRUCTURE_RISK): "
        f"`{summary['strong_pool_eligible']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Phase 2 — Discovery cache + Fast Discovery
# ---------------------------------------------------------------------------


def materialize_discovery_cache(out_root: Path) -> Path:
    """Filter existing partitions to Discovery window only (no 2019-2026 rebuild)."""
    cache_dir = out_root / "cancellation_daily_discovery"
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel = load_cancel_partitions(DISCOVERY_START, DISCOVERY_END)
    # quarterly restartable slices
    panel = panel.sort_values(["TradeDate", "symbol"])
    for (y, q), part in panel.groupby(
        [panel["TradeDate"].dt.year, panel["TradeDate"].dt.quarter]
    ):
        q_start = pd.Timestamp(int(y), 3 * int(q) - 2, 1)
        q_end = (q_start + pd.offsets.QuarterEnd(0)).normalize()
        path = cache_dir / f"cancel_daily_{q_start:%Y-%m-%d}_{q_end:%Y-%m-%d}.parquet"
        part.to_parquet(path, index=False)
    full_path = cache_dir / "cancel_daily_2023-01-01_2024-12-31.parquet"
    panel.to_parquet(full_path, index=False)
    manifest = {
        "window": [str(DISCOVERY_START.date()), str(DISCOVERY_END.date())],
        "n_rows": int(len(panel)),
        "n_days": int(panel["TradeDate"].nunique()),
        "n_symbols": int(panel["symbol"].nunique()),
        "source": "filtered cancel_lifecycle_daily dataset partitions",
        "ch_rebuild": False,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return full_path


def contribution_stats(group_pnl: pd.DataFrame) -> Dict[str, float]:
    pnl = ensure_effective_group_pnl(group_pnl)
    cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
    g1, g10 = cols[0], cols[-1]
    long_c = pnl[g10]
    short_c = -pnl[g1]
    return {
        "g10_contribution_annual": float(calAnnuRet(long_c)),
        "g1_contribution_annual": float(calAnnuRet(short_c)),
        "g10_gross_excess_annual": float(calAnnuRet(pnl[g10])),
        "g1_gross_excess_annual": float(calAnnuRet(pnl[g1])),
        "short_leg_share_abs": (
            abs(float(calAnnuRet(short_c)))
            / (
                abs(float(calAnnuRet(long_c))) + abs(float(calAnnuRet(short_c)))
            )
            if (
                abs(float(calAnnuRet(long_c))) + abs(float(calAnnuRet(short_c)))
            )
            > 0
            else float("nan")
        ),
    }


def run_fast_discovery(
    gate_summary: Dict[str, object],
    *,
    out_root: Optional[Path] = None,
) -> pd.DataFrame:
    out = Path(out_root) if out_root else OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    print("[discovery] materialize 2023-2024 cache from existing partitions", flush=True)
    cache_path = materialize_discovery_cache(out)
    prim = pd.read_parquet(cache_path)
    prim["TradeDate"] = pd.to_datetime(prim["TradeDate"])
    # need buffer before discovery for shocks: load May 2022? 20 trading days
    # ≈ 40 calendar days before 2023-01-01
    warmup = load_cancel_partitions(
        DISCOVERY_START - pd.Timedelta(days=60), DISCOVERY_END
    )
    cand = build_candidate_panel(warmup)
    cand = cand.loc[cand["TradeDate"].between(DISCOVERY_START, DISCOVERY_END)].copy()

    structure_block = set(gate_summary.get("structure_risk_flagged") or [])
    mask, ret = load_fast_context("discovery")

    rows = []
    for name in CANDIDATES:
        print(f"\n=== {name} ===", flush=True)
        t1 = time.perf_counter()
        frame = cand.dropna(subset=[name])
        narrow = _to_narrow(frame["symbol"], frame["TradeDate"], frame[name], name)
        group_pnl, group_to, _ic, summary = backtest_factor(
            narrow,
            start_day=DISCOVERY_START,
            end_day=DISCOVERY_END,
            mask=mask,
            ret_matrix=ret,
        )
        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        contrib = contribution_stats(group_pnl)
        pnl = ensure_effective_group_pnl(group_pnl)
        to = group_to.copy()
        to.index = pd.to_datetime(to.index)
        to.columns = [str(c) for c in to.columns]
        cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
        g10 = cols[-1]
        l1_hl = float(to["H-L"].reindex(pnl.index).mean())
        l1_g10 = float(to[g10].reindex(pnl.index).mean())
        gate = gate_label(metrics)
        # Strong pool exclusion for structure risk
        if name in structure_block and gate == "strong_candidate":
            gate = "research_candidate"
            gate_note = "demoted_from_strong: EXCHANGE_STRUCTURE_RISK"
        else:
            gate_note = ""

        save_fast_plots(out / "figures" / name, name, group_pnl, metrics)
        row = {
            "factor": name,
            "mechanism": "cancellation_lifecycle",
            "source_primitive": "cancel_lifecycle_daily",
            "window": "discovery",
            "gate": gate,
            "gate_note": gate_note,
            "EXCHANGE_STRUCTURE_RISK": name in structure_block,
            "rank_ic_mean_raw": metrics["rank_ic_mean_raw"],
            "icir_raw": metrics["icir_raw"],
            "hl_annu_ret": metrics["hl_annu_ret"],
            "hl_sharpe": metrics["hl_sharpe"],
            "decile_mono_spearman": metrics["decile_mono_spearman"],
            "adjacent_violations": metrics["adjacent_violations"],
            "positive_hl_month_fraction": metrics["positive_hl_month_fraction"],
            "g10_gross_excess_annual": contrib["g10_gross_excess_annual"],
            "g10_excess_sharpe": metrics["g10_excess_sharpe"],
            "g1_gross_excess_annual": contrib["g1_gross_excess_annual"],
            "g10_contribution_annual": contrib["g10_contribution_annual"],
            "g1_contribution_annual": contrib["g1_contribution_annual"],
            "short_leg_share_abs": contrib["short_leg_share_abs"],
            "avg_hl_oneway_turnover": l1_to_oneway(l1_hl),
            "avg_g10_oneway_turnover": l1_to_oneway(l1_g10),
            "factor_direction": metrics["factor_direction"],
            "n_days": metrics["n_days"],
            "elapsed_seconds": round(time.perf_counter() - t1, 2),
        }
        rows.append(row)
        print(
            f"  gate={gate} Sharpe={metrics['hl_sharpe']:.2f} "
            f"mono={metrics['decile_mono_spearman']:.3f} "
            f"viol={metrics['adjacent_violations']} "
            f"struct_risk={name in structure_block}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "candidate_summary.csv", index=False)
    return summary


def select_next(summary: pd.DataFrame) -> Dict[str, object]:
    strong = summary.loc[summary["gate"] == "strong_candidate"].copy()
    if strong.empty:
        return {"status": "SPRINT10_NO_STRONG_CANDIDATE", "factor": None, "next_md": ""}
    strong = strong.sort_values(
        by=["g10_gross_excess_annual", "short_leg_share_abs", "avg_hl_oneway_turnover"],
        ascending=[False, True, True],
    )
    best = strong.iloc[0]
    fid = str(best["factor"])
    md = "\n".join(
        [
            "# Next Full Validation Candidate — Sprint 10",
            "",
            f"**factor:** `{fid}`",
            "",
            "Awaiting human confirmation. Do **not** auto-build 2019–2026.",
            "",
            f"- gate=`{best['gate']}`",
            f"- H-L Sharpe=`{best['hl_sharpe']:.3f}` mono=`{best['decile_mono_spearman']:.3f}` "
            f"viol=`{int(best['adjacent_violations'])}`",
            f"- G10 excess annual=`{best['g10_gross_excess_annual']:.2%}`",
            f"- short_leg_share=`{best['short_leg_share_abs']:.3f}`",
            f"- H-L one-way TO=`{best['avg_hl_oneway_turnover']:.3f}`",
            f"- EXCHANGE_STRUCTURE_RISK=`{best['EXCHANGE_STRUCTURE_RISK']}`",
            "",
        ]
    ) + "\n"
    return {"status": "HAS_STRONG", "factor": fid, "next_md": md}


def render_discovery_report(
    summary: pd.DataFrame,
    gate_summary: Dict[str, object],
    selection: Dict[str, object],
) -> str:
    lines = [
        "# Sprint 10 — Cancellation / Order Lifecycle Family v1",
        "",
        f"Monthly gate PASS: `{gate_summary['gate_pass']}`",
        f"Discovery: `{DISCOVERY_START.date()}` ~ `{DISCOVERY_END.date()}`",
        "Protocol v2.0 / Fast Gate: untouched",
        "",
        "## Candidate summary",
        "",
        summary.to_string(index=False),
        "",
        f"## Selection: **{selection['status']}**",
        f"Next: `{selection.get('factor')}`",
        "",
    ]
    return "\n".join(lines) + "\n"
