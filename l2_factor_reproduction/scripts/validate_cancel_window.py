#!/usr/bin/env python
"""Sprint 6B Phase 1 validation chain — cancellation / order lifecycle.

Modes (execution boundary from the frozen contract; the 2019-2026
full-history scan is forbidden before the monthly gate passes):

  smoke    2024-06-28 full-market day: fetch both exchanges, sanity-band
           event counts vs the Phase 0 audit (full-day numbers: SSE 34.48M
           events / SZSE 26.59M; this module counts the continuous-auction
           window only, so a tolerance band is used), join_coverage,
           candidate distribution stats. -> smoke_2024-06-28.md/csv

  monthly  2024-06 monthly gate: build symbol-day primitives for the warm-up
           window (2024-05-01..2024-06-30, needed for the 20d shocks) day by
           day, persist to primitives/cancel_lifecycle_daily/daily/, then
           evaluate the 7 frozen candidates on June with the contract's
           cross-market exposure audit. -> monthly_gate_2024-06.md +
           monthly primitives parquet.

Usage:
    python validate_cancel_window.py --mode smoke
    python validate_cancel_window.py --mode monthly
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_cancel_lifecycle import (  # noqa: E402
    PRIMITIVE_COLUMNS,
    build_candidates,
    fetch_cancel_daily,
    shock_20d,
)

OUT_DIR = (
    PROJ_ROOT / "research" / "results" / "l2_reproduction" / "primitives"
    / "cancel_lifecycle_daily"
)
DAILY_DIR = OUT_DIR / "daily"

CANDIDATES = [
    "cancel_value_pressure", "cancel_count_pressure",
    "cancel_value_intensity", "cancel_qty_intensity",
    "relative_cancel_order_size",
    "cancel_pressure_shock_20d", "cancel_intensity_shock_20d",
]

# Phase 0 audit (2024-06-28, full calendar day, ALL symbols): SSE 34.48M /
# SZSE 26.59M cancel events. Stock-universe full-day references (verified
# 2026-08-07): SSE 14,846,572 / SZSE 16,475,184. The continuous-auction
# window keeps ~98-100% of stock cancel events, hence the band below.
EXPECTED_EVENTS = {"SSE": 14.847e6, "SZSE": 16.475e6}


def _segment(symbol: str) -> str:
    code = symbol.split(".")[0]
    if symbol.endswith(".SH"):
        return "STAR" if code.startswith("688") else "SSE main"
    return "ChiNext" if code.startswith(("300", "301")) else "SZSE main"


def run_smoke(client, day: str) -> bool:
    print(f"[smoke] full-market fetch {day}", flush=True)
    t0 = time.time()
    prim = fetch_cancel_daily(client, day)
    elapsed = time.time() - t0
    print(f"[smoke] rows={len(prim):,} elapsed={elapsed:.0f}s", flush=True)
    prim.to_csv(OUT_DIR / f"smoke_{day}.csv", index=False)

    ok = True
    lines = [f"# Cancellation smoke {day}", ""]
    for exch, suffix in [("SSE", ".SH"), ("SZSE", ".SZ")]:
        part = prim[prim["symbol"].str.endswith(suffix)]
        events = (
            part["buy_cancel_event_count"] + part["sell_cancel_event_count"]
        ).sum()
        band = EXPECTED_EVENTS[exch]
        ratio = events / band
        status = "OK" if 0.85 <= ratio <= 1.0 else "FAIL"
        if status == "FAIL":
            ok = False
        cov = part["join_coverage"].median()
        invalid = part["invalid_cancel_count"].sum()
        lines.append(
            f"- {exch}: symbols={len(part):,} cancel_events={events:,.0f} "
            f"(phase0 full-day ref {band:,.0f}, ratio {ratio:.2f} "
            f"[{status}]) median_join_coverage={cov:.4f} "
            f"invalid_cancel_total={invalid:,.0f}"
        )
        if cov < 0.99:
            ok = False

    cand = build_candidates(prim)
    lines += ["", "## candidate distribution (smoke day)", "", "```"]
    stats = cand[CANDIDATES[:5]].describe(percentiles=[0.01, 0.5, 0.99]).T
    lines.append(stats[["count", "mean", "std", "1%", "50%", "99%"]]
                 .to_string(float_format=lambda x: f"{x:.4g}"))
    lines.append("```")
    (OUT_DIR / f"smoke_{day}.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[smoke {'PASS' if ok else 'FAIL'}]", flush=True)
    return ok


def run_monthly(client, start: str, end: str) -> bool:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start, end)  # upper bound; non-trading days are empty
    for day in days:
        out_path = DAILY_DIR / f"cancel_daily_{day:%Y-%m-%d}.parquet"
        if out_path.exists():
            continue
        t0 = time.time()
        prim = fetch_cancel_daily(client, day)
        if len(prim) == 0:
            print(f"[{day:%Y-%m-%d}] non-trading or empty", flush=True)
            continue
        prim.to_parquet(out_path, index=False)
        print(f"[{day:%Y-%m-%d}] rows={len(prim):,} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    frames = [pd.read_parquet(p) for p in sorted(DAILY_DIR.glob("*.parquet"))]
    if not frames:
        print("[monthly FAIL] no daily primitives built")
        return False
    panel = pd.concat(frames, ignore_index=True)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"])

    cand = build_candidates(panel)
    cand = cand.sort_values(["symbol", "TradeDate"]).reset_index(drop=True)
    grouped = cand.groupby("symbol")
    cand["cancel_pressure_shock_20d"] = grouped[
        "cancel_value_pressure"].transform(shock_20d)
    cand["cancel_intensity_shock_20d"] = grouped[
        "cancel_value_intensity"].transform(shock_20d)

    june = cand[cand["TradeDate"] >= "2024-06-01"].copy()
    june["segment"] = june["symbol"].map(_segment)

    ok = True
    lines = ["# Cancellation 2024-06 monthly gate", ""]
    june_days = june["TradeDate"].nunique()
    lines.append(f"- trading days in scope: {june_days}")
    if june_days < 15:
        lines.append("- **FAIL**: fewer than 15 trading days")
        ok = False

    # contract: distribution / coverage / decile board composition /
    # exchange & board dummy corr / full-market vs intra-exchange rank corr
    lines += ["", "## per-candidate cross-market exposure", ""]
    exposure_rows = []
    for name in CANDIDATES:
        valid = june.dropna(subset=[name])
        share = len(valid) / len(june)
        seg_share = valid.groupby("segment")[name].mean()
        is_szse = valid["symbol"].str.endswith(".SZ").astype(float)
        is_star = valid["symbol"].str.startswith("688").astype(float)
        dummy_corr = valid[name].corr(is_szse)
        star_corr = valid[name].corr(is_star)
        # full-market rank vs intra-exchange rank
        valid = valid.copy()
        valid["rank_full"] = valid.groupby("TradeDate")[name].rank(pct=True)
        valid["rank_intra"] = (
            valid.groupby(["TradeDate",
                           valid["symbol"].str.endswith(".SZ")])[name]
            .rank(pct=True)
        )
        rank_corr = valid["rank_full"].corr(valid["rank_intra"])
        exposure_rows.append({
            "factor": name, "coverage": share,
            "sse_main_mean": seg_share.get("SSE main", np.nan),
            "star_mean": seg_share.get("STAR", np.nan),
            "szse_main_mean": seg_share.get("SZSE main", np.nan),
            "chinext_mean": seg_share.get("ChiNext", np.nan),
            "corr_exchange_dummy": dummy_corr,
            "corr_star_dummy": star_corr,
            "rank_full_vs_intra_exchange": rank_corr,
        })
    exposure = pd.DataFrame(exposure_rows)
    exposure.to_csv(OUT_DIR / "monthly_gate_2024-06_exposure.csv",
                    index=False)
    lines.append("```")
    lines.append(exposure.to_string(index=False,
                                    float_format=lambda x: f"{x:.4f}"))
    lines.append("```")

    # decile board composition for the two pressure candidates
    lines += ["", "## decile board composition (share of segment per decile)",
              ""]
    comp_rows = []
    for name in ["cancel_value_pressure", "cancel_value_intensity"]:
        valid = june.dropna(subset=[name]).copy()
        valid["decile"] = valid.groupby("TradeDate")[name].transform(
            lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
        comp = (valid.groupby("decile")["segment"]
                .value_counts(normalize=True).unstack().fillna(0))
        comp.index = [f"{name}#G{i + 1}" for i in comp.index]
        comp_rows.append(comp)
    comp_frame = pd.concat(comp_rows)
    comp_frame.to_csv(OUT_DIR / "monthly_gate_2024-06_decile_board.csv")
    lines.append("```")
    lines.append(comp_frame.to_string(float_format=lambda x: f"{x:.3f}"))
    lines.append("```")

    # ---- June T+1 RankIC via the shared backtest context
    lines += ["", "## June 2024 T+1 baseline (signal_shift=1)", ""]
    from l2_factor_reproduction.python.backtest import (
        backtest_factor,
        load_backtest_context,
    )
    mask, ret_matrix = load_backtest_context(
        pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"))
    ic_rows = []
    for name in CANDIDATES:
        frame = june.dropna(subset=[name])[
            ["symbol", "TradeDate", name]].rename(
            columns={"TradeDate": "tradetime", name: "value"})
        _, _, ic, summary = backtest_factor(
            frame, start_day=pd.Timestamp("2024-06-01"),
            end_day=pd.Timestamp("2024-06-30"),
            signal_shift=1, mask=mask, ret_matrix=ret_matrix,
        )
        ic_rows.append({
            "factor": name,
            "rank_ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hl_sharpe": float(summary.get("hl_sharpe_flipped", np.nan)),
            "g10_excess_sharpe": float(
                summary.get("g10_excess_sharpe", np.nan)),
        })
        print(f"[ic] {name}: IC={ic_rows[-1]['rank_ic']:+.4f}", flush=True)
    ic_frame = pd.DataFrame(ic_rows)
    ic_frame.to_csv(OUT_DIR / "monthly_gate_2024-06_ic.csv", index=False)
    lines.append("```")
    lines.append(ic_frame.to_string(index=False,
                                    float_format=lambda x: f"{x:.4f}"))
    lines.append("```")

    # ---- peer comparison (contract Part B.6 focus list)
    lines += ["", "## peer correlation (mean daily cross-sectional Spearman,"
                  " June dates)", ""]
    peers = _peer_narrows()
    peer_rows = []
    wide = june.pivot_table(index=["symbol", "TradeDate"],
                            values=CANDIDATES, aggfunc="first")
    for label, peer, path in peers:
        if not path.exists():
            continue
        narrow = pd.read_parquet(path, columns=["symbol", "tradetime",
                                                "value"])
        narrow["TradeDate"] = pd.to_datetime(narrow["tradetime"]).dt.normalize()
        block = narrow.loc[
            (narrow["TradeDate"] >= "2024-06-01")
            & (narrow["TradeDate"] <= "2024-06-30")
        ][["symbol", "TradeDate", "value"]].rename(
            columns={"value": peer})
        merged = wide.reset_index().merge(
            block, on=["symbol", "TradeDate"], how="inner")
        for name in CANDIDATES:
            rho = merged.groupby("TradeDate").apply(
                lambda g: g[name].corr(g[peer], method="spearman")
                if len(g) >= 100 else np.nan
            ).mean()
            peer_rows.append({"family": label, "peer": peer,
                              "factor": name, "rho": rho})
        print(f"[peer] {label}/{peer} done", flush=True)
    peer_frame = pd.DataFrame(peer_rows)
    peer_frame.to_csv(OUT_DIR / "monthly_gate_2024-06_peer_corr.csv",
                      index=False)
    pivot = peer_frame.pivot_table(index=["family", "peer"],
                                   columns="factor", values="rho")
    lines.append("```")
    lines.append(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))
    lines.append("```")

    (OUT_DIR / "monthly_gate_2024-06.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"[monthly gate written] {OUT_DIR / 'monthly_gate_2024-06.md'}",
          flush=True)
    return ok


def _peer_narrows():
    """Representative peers for the Part B.6 comparison: Order Book shock /
    OBI, Trade Flow direction, Order Size, Liquidity."""
    from l2_factor_reproduction.config.settings import RESULT_ROOT
    pool = Path(RESULT_ROOT) / "candidate_pool_v1"
    root = Path(RESULT_ROOT)
    ob = pool / "order_book_family" / "factors"
    li = pool / "liquidity_impact_family" / "factors"
    peers = [
        ("order_book", "obi_shock_20d", ob / "obi_shock_20d" / "factor_narrow.parquet"),
        ("order_book", "spread_shock_20d", ob / "spread_shock_20d" / "factor_narrow.parquet"),
        ("order_book", "depth_shock_20d", ob / "depth_shock_20d" / "factor_narrow.parquet"),
        ("order_book", "weighted_obi_mean", ob / "weighted_obi_mean" / "factor_narrow.parquet"),
        ("trade_flow", "net_buy_ratio", root / "net_buy_ratio" / "factor_narrow.parquet"),
        ("trade_flow", "buy_dominance", root / "buy_dominance" / "factor_narrow.parquet"),
        ("order_size", "small_order_ratio", root / "small_order_ratio" / "factor_narrow.parquet"),
        ("order_size", "mid_order_ratio", root / "mid_order_ratio" / "factor_narrow.parquet"),
        ("liquidity_impact", "depth_per_amount", li / "depth_per_amount" / "factor_narrow.parquet"),
        ("liquidity_impact", "effective_spread_proxy", li / "effective_spread_proxy" / "factor_narrow.parquet"),
    ]
    return [(label, peer, path) for label, peer, path in peers
            if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "monthly"],
                        required=True)
    parser.add_argument("--day", default="2024-06-28")
    parser.add_argument("--start", default="2024-04-15",
                        help="monthly: warm-up start (>=20 trading days of "
                             "shock history before 2024-06-01)")
    parser.add_argument("--end", default="2024-06-30")
    args = parser.parse_args()

    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client
    client = connect_hf_client()
    if args.mode == "smoke":
        return 0 if run_smoke(client, args.day) else 1
    return 0 if run_monthly(client, args.start, args.end) else 1


if __name__ == "__main__":
    raise SystemExit(main())
