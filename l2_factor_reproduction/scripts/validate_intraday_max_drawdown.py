#!/usr/bin/env python
"""Single-Factor Full Validation v2.0 — intraday_max_drawdown.

ORDINARY_OHLCV_CONTROL = True (Price Formation strong baseline/control).
Reuse existing price_formation cache. No formula edits.
Does not validate close_auction_return. Does not modify Sprint 11 / tail FV.

Usage:
    python -m l2_factor_reproduction.scripts.validate_intraday_max_drawdown
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import calAnnuRet, calMDD, calSharpe  # noqa: E402
from Factor_Dev_Lib import groupTest  # noqa: E402
from l2_factor_reproduction.config.settings import BACKTEST_SILENT, RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    backtest_factor,
    compute_rank_ic,
    narrow_to_wide,
    prepare_factor_signal,
)
from l2_factor_reproduction.python.cache_coverage import (  # noqa: E402
    coverage_bounds_ok,
)
from l2_factor_reproduction.python.evaluation_protocol_v2 import (  # noqa: E402
    BENCHMARK,
    FEE_RATE_L1,
    FREEZE_DATE,
    FULL_END,
    FULL_START,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    SAMPLES,
    SIGNAL_SHIFT,
    assign_factor_grade,
    assign_long_only_status,
    check_effective_turnover_parity,
    ensure_effective_group_to,
    factor_layer_metrics,
    label_block,
    load_benchmark_return,
    long_only_metrics,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    PLOT_DPI,
    _configure_plot_fonts,
    _decile_cmap_colors,
    _group_columns,
    ensure_effective_group_pnl,
    load_fast_context,
)

FACTOR = "intraday_max_drawdown"
ORDINARY_OHLCV_CONTROL = True

EXACT_FORMULA = (
    "intraday_max_drawdown = max(1 - price_t / running_max_price_t) over "
    "continuous-auction session minutes from price_formation_daily; "
    "running_max_price_t = cummax(closePx); "
    "primitive column = max_drawdown_intraday; signal_shift = T+1; "
    "no rolling transform; no parameter grid; "
    "formula_version = price_formation_level_formulas_v1; "
    "ORDINARY_OHLCV_CONTROL = True"
)
FACTOR_HASH = hashlib.sha256(EXACT_FORMULA.encode("utf-8")).hexdigest()
SOURCE_PRIMITIVE = "price_formation_daily.max_drawdown_intraday"
FORMULA_VERSION = "price_formation_level_formulas_v1"
SCHEMA_VERSION = "l2_primitive_price_formation_daily_v1"
EXPECTED_RAW_DIRECTION = -1

DISCOVERY_METRICS = {
    "hl_sharpe": 3.11,
    "mono": 1.00,
    "violations": 0,
    "g10_gross_excess_annual": 0.230,
}

OUT_DIR = Path(RESULT_ROOT) / "full_validation" / FACTOR
FAMILY_ROOT = Path(RESULT_ROOT) / "candidate_pool_v1" / "price_formation_family"
FACTOR_NARROW_PATH = FAMILY_ROOT / "factors" / FACTOR / "factor_narrow.parquet"
FAMILY_MANIFEST = FAMILY_ROOT / "manifest.json"
PRIM_MANIFEST = (
    Path(RESULT_ROOT) / "primitives" / "price_formation_daily" / "manifest.json"
)
FORMULA_MODULE = (
    PROJ_ROOT / "l2_factor_reproduction" / "python" / "price_formation_factors.py"
)
PRIMITIVE_MODULE = (
    PROJ_ROOT / "l2_factor_reproduction" / "python" / "price_formation_daily.py"
)

MARKERS = (
    (pd.Timestamp("2022-12-31"), "end-2022 / PRE|DISCOVERY"),
    (pd.Timestamp("2024-12-31"), "end-2024 / DISCOVERY|POST"),
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2%}"


def _fmt_num(x: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# PART A/B — freeze + cache reuse gate
# ---------------------------------------------------------------------------


def build_factor_freeze() -> Dict[str, Any]:
    return {
        "factor_id": FACTOR,
        "exact_formula": EXACT_FORMULA,
        "factor_hash_sha256": FACTOR_HASH,
        "formula_hash": FACTOR_HASH,
        "source_primitive": SOURCE_PRIMITIVE,
        "formula_version": FORMULA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "raw_direction": EXPECTED_RAW_DIRECTION,
        "signal_shift": SIGNAL_SHIFT,
        "ORDINARY_OHLCV_CONTROL": ORDINARY_OHLCV_CONTROL,
        "ordinary_ohlcv_control_reason": (
            "Price Formation strong baseline/control; not automatically counted "
            "as a distinct L2-specific microstructure alpha."
        ),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_status": PROTOCOL_STATUS,
        "freeze_date": FREEZE_DATE,
        "sprint11_discovery_metrics": DISCOVERY_METRICS,
        "evidence_labels": {
            "DISCOVERY": {
                "window": ["2023-01-01", "2024-12-31"],
                "role": "frozen Fast Discovery selection sample",
            },
            "PRE": {
                "window": ["2019-01-01", "2022-12-31"],
                "role": "pre-discovery retrospective segment",
            },
            "POST": {
                "window": ["2025-01-01", "2026-07-31"],
                "role": "recent decay diagnostic (NOT pristine unseen)",
            },
            "FULL": {
                "window": ["2019-01-01", "2026-07-31"],
                "role": "retrospective robustness validation (NOT pristine unseen)",
            },
        },
        "optimization_policy": "NO_PARAMETER_OPTIMIZATION",
        "exclusions": [
            "do_not_validate_close_auction_return",
            "do_not_rerun_tail_return_share",
            "do_not_modify_sprint11",
            "do_not_start_sprint12_automatically",
        ],
    }


def validate_existing_cache() -> Dict[str, Any]:
    """Hard-fail if frozen cache / lineage inconsistent. Never rebuild."""
    errors: List[str] = []
    if not FACTOR_NARROW_PATH.exists():
        errors.append(f"missing factor_narrow: {FACTOR_NARROW_PATH}")
    if not FAMILY_MANIFEST.exists():
        errors.append(f"missing family manifest: {FAMILY_MANIFEST}")
    if not PRIM_MANIFEST.exists():
        errors.append(f"missing primitive manifest: {PRIM_MANIFEST}")
    if errors:
        raise RuntimeError("CACHE_ILLEGAL:\n" + "\n".join(errors))

    prim = json.loads(PRIM_MANIFEST.read_text(encoding="utf-8"))
    fam = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))

    expected_formula_sha = prim["lineage"]["factor_formula_module"]["sha256"]
    expected_prim_mod_sha = prim["lineage"]["primitive_module"]["sha256"]
    got_formula_sha = _sha256_file(FORMULA_MODULE)
    got_prim_mod_sha = _sha256_file(PRIMITIVE_MODULE)
    if got_formula_sha != expected_formula_sha:
        errors.append(
            f"formula_module sha mismatch: got={got_formula_sha} "
            f"expected={expected_formula_sha}"
        )
    if got_prim_mod_sha != expected_prim_mod_sha:
        errors.append(
            f"primitive_module sha mismatch: got={got_prim_mod_sha} "
            f"expected={expected_prim_mod_sha}"
        )

    if prim.get("formula_version") != FORMULA_VERSION:
        errors.append(
            f"formula_version mismatch: {prim.get('formula_version')} "
            f"!= {FORMULA_VERSION}"
        )
    if prim.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version mismatch: {prim.get('schema_version')} "
            f"!= {SCHEMA_VERSION}"
        )

    cov = prim.get("date_coverage", {})
    if cov.get("requested_start") != "2019-01-01" or cov.get("requested_end") != "2026-07-31":
        errors.append(f"sample coverage request mismatch: {cov}")
    if str(cov.get("actual_max")) < "2026-07-31" or str(cov.get("actual_min")) > "2019-01-02":
        errors.append(f"sample coverage actual incomplete: {cov}")

    prim_manifest_sha = _sha256_file(PRIM_MANIFEST)
    if fam.get("primitive_manifest_sha256") != prim_manifest_sha:
        errors.append(
            "family.primitive_manifest_sha256 mismatch vs current primitive manifest"
        )
    if fam.get("formula_module_sha256") != expected_formula_sha:
        errors.append("family.formula_module_sha256 mismatch")
    if FACTOR not in set(fam.get("factors", [])):
        errors.append(f"{FACTOR} missing from family manifest factors list")

    narrow = pd.read_parquet(FACTOR_NARROW_PATH)
    if "factorname" in narrow.columns and set(narrow["factorname"].unique()) != {FACTOR}:
        errors.append(f"factorname column unexpected: {narrow['factorname'].unique()}")
    tt = pd.to_datetime(narrow["tradetime"])
    ok_cov, cov_err = coverage_bounds_ok(
        tt, expected_min="2019-01-02", expected_max="2026-07-31"
    )
    if not ok_cov:
        errors.append(f"narrow coverage incomplete: {cov_err}; raw={tt.min()}..{tt.max()}")
    if narrow["value"].isna().all():
        errors.append("narrow values all NaN")

    if errors:
        raise RuntimeError(
            "CACHE_ILLEGAL — STOP without rebuild:\n" + "\n".join(errors)
        )

    return {
        "status": "CACHE_OK_REUSE",
        "factor_narrow_path": str(FACTOR_NARROW_PATH),
        "n_rows": int(len(narrow)),
        "tradetime_min": str(tt.min()),
        "tradetime_max": str(tt.max()),
        "primitive_manifest_sha256": prim_manifest_sha,
        "formula_module_sha256": got_formula_sha,
        "primitive_module_sha256": got_prim_mod_sha,
        "formula_version": FORMULA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "date_coverage": cov,
        "rebuild": False,
        "raw_minute_scan": False,
        "clickhouse_scan": False,
    }


def load_factor_narrow_full() -> pd.DataFrame:
    df = pd.read_parquet(FACTOR_NARROW_PATH)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    out = df.loc[:, ["symbol", "tradetime", "factorname", "value"]].copy()
    out["factorname"] = FACTOR
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def contribution_for_sample(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    *,
    sample: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Any]:
    pnl = ensure_effective_group_pnl(group_pnl).loc[start:end]
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl.index)
    cols = _group_columns(pnl)
    g1, g10 = cols[0], cols[-1]
    long_c = pnl[g10]
    short_c = -pnl[g1]
    long_l1 = to[g10].astype(float)
    # long net excess ≈ G10 gross excess - long L1 * 7.5bps (vs excess baseline)
    # Report G10 gross excess and G10 net excess after long-leg cost on excess series:
    # Protocol long-only nets absolute returns; for leg table we also report
    # g10_net_excess = g10_gross_excess - long_l1*fee (fee on long leg only).
    g10_gross = pnl[g10].astype(float)
    g10_net = g10_gross - long_l1.reindex(g10_gross.index).fillna(0.0) * FEE_RATE_L1
    long_annu = float(calAnnuRet(long_c))
    short_annu = float(calAnnuRet(short_c))
    hl_annu = float(calAnnuRet(pnl["H-L"]))
    return {
        "sample": sample,
        "g10_gross_excess_annual": float(calAnnuRet(g10_gross)),
        "g10_net_excess_annual": float(calAnnuRet(g10_net)),
        "g1_gross_excess_annual": float(calAnnuRet(pnl[g1])),
        "long_contribution": long_annu,
        "short_contribution": short_annu,
        "gross_hl_annual": hl_annu,
        "hl_minus_legs": hl_annu - (long_annu + short_annu),
        "short_leg_share_abs": (
            abs(short_annu) / (abs(long_annu) + abs(short_annu))
            if (abs(long_annu) + abs(short_annu)) > 0
            else float("nan")
        ),
        "dominant_leg": (
            "SHORT"
            if abs(short_annu) > abs(long_annu) * 1.25
            else ("LONG" if abs(long_annu) > abs(short_annu) * 1.25 else "BALANCED")
        ),
    }


def raw_and_effective(
    narrow: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.Series]:
    signal_raw, ret_aln = prepare_factor_signal(
        narrow_to_wide(narrow),
        start=FULL_START,
        end=FULL_END,
        mask=mask,
        ret_matrix=ret,
    )
    rank_ic_raw = compute_rank_ic(signal_raw, ret_aln)
    info = "silent" if BACKTEST_SILENT else "silent"
    _r, pnl_raw, to_raw = groupTest(signal_raw, ret_aln, n=10, info=info)
    direction = 1 if float(pnl_raw["H-L"].mean()) > 0 else -1

    group_pnl, group_to, _rank_ic_eff, summary = backtest_factor(
        narrow,
        start_day=FULL_START,
        end_day=FULL_END,
        mask=mask,
        ret_matrix=ret,
    )
    assert int(summary["factor_direction"]) == direction
    pnl_eff = ensure_effective_group_pnl(group_pnl)
    to_eff = ensure_effective_group_to(group_to, group_pnl)
    return pnl_raw, to_raw, pnl_eff, to_eff, direction, rank_ic_raw


# ---------------------------------------------------------------------------
# Plots (3 standard)
# ---------------------------------------------------------------------------


def _add_segment_markers(ax: plt.Axes, y_ref: float) -> None:
    for mark, label in MARKERS:
        ax.axvline(mark, color="crimson", linewidth=1.2, linestyle="--", alpha=0.85)
        ax.text(
            mark,
            y_ref,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="crimson",
        )


def _stats_box(ax: plt.Axes, text: str, cum_hl_end: float) -> None:
    box_loc = "lower right" if cum_hl_end >= 0 else "upper right"
    ax.text(
        0.98 if "right" in box_loc else 0.02,
        0.02 if "lower" in box_loc else 0.98,
        text,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom" if "lower" in box_loc else "top",
        ha="right" if "right" in box_loc else "left",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.4",
            "alpha": 0.92,
        },
        zorder=6,
        family="monospace",
    )


def save_validation_plots_v2(
    out_dir: Path,
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    benchmark: pd.Series,
    full_metrics: Dict[str, Any],
) -> Dict[str, str]:
    """Three standard plots + compatibility cumulative_hl.png."""
    _configure_plot_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)
    pnl = ensure_effective_group_pnl(group_pnl)
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl.index)
    group_cols = _group_columns(pnl)
    labels = [f"G{c}" for c in group_cols]
    hl = pnl["H-L"]
    cum = pnl.cumsum()

    annu = float(full_metrics["gross_hl_annual"])
    sharpe = float(full_metrics["gross_hl_sharpe"])
    mdd = float(full_metrics["gross_hl_mdd"])
    ow = float(full_metrics["avg_daily_hl_oneway_turnover"])
    l1 = float(full_metrics["avg_daily_hl_l1_traded_notional"])
    mono = float(full_metrics["decile_mono_gross"])
    viol = int(full_metrics["adjacent_violations_gross"])

    stats_text = (
        f"H-L Ann. Ret: {annu:.2%}\n"
        f"H-L Sharpe:   {sharpe:.2f}\n"
        f"H-L Max DD:   {mdd:.2%}\n"
        f"L1 daily:     {l1:.2f}\n"
        f"one-way daily:{ow:.2f}\n"
        f"mono/viol:    {mono:.3f}/{viol}"
    )

    # --- 1. cumulative_deciles_hl.png ---
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    colors = _decile_cmap_colors(len(group_cols), kind="warm")
    for col, color, label in zip(group_cols, colors, labels):
        ax1.plot(
            cum.index, cum[col].to_numpy(), color=color, linewidth=1.15, alpha=0.85, label=label
        )
    ax1.plot(
        cum.index,
        cum["H-L"].to_numpy(),
        color="black",
        linewidth=2.6,
        alpha=0.95,
        label="H-L",
        zorder=5,
    )
    ax1.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    _add_segment_markers(ax1, float(cum.max().max()) * 0.98)
    ax1.set_title(
        f"{FACTOR} — Decile + H-L Cumulative (effective; FULL 2019–2026)\n"
        "G1=low effective factor … G10=high effective factor",
        fontsize=12,
    )
    ax1.set_xlabel("TradeDate")
    ax1.set_ylabel("Cumulative return (cumsum)")
    ax1.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)
    _stats_box(ax1, stats_text, float(cum["H-L"].iloc[-1]))
    fig1.tight_layout()
    path_dec = out_dir / "cumulative_deciles_hl.png"
    fig1.savefig(path_dec, dpi=PLOT_DPI)
    # compatibility copy
    fig1.savefig(out_dir / "cumulative_hl.png", dpi=PLOT_DPI)
    plt.close(fig1)

    # --- 2. decile_bar.png ---
    means = pnl[group_cols + ["H-L"]].mean()
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    bar_colors = _decile_cmap_colors(len(group_cols), kind="cool") + ["#C44E52"]
    x_labels = labels + ["H-L"]
    values = means.to_numpy(dtype=float)
    bars = ax2.bar(x_labels, values, color=bar_colors, width=0.75)
    ax2.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
    ax2.set_title(
        f"{FACTOR} — Decile Mean Daily Return (monotonicity; effective)",
        fontsize=12,
    )
    ax2.set_xlabel("Decile (G1=low effective factor … G10=high effective factor)")
    ax2.set_ylabel("Mean daily return")
    ax2.grid(True, axis="y", alpha=0.3)
    y_span = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    offset = 0.02 * y_span if y_span > 0 else 1e-5
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            continue
        label = f"{val * 1e4:.1f}bp" if abs(val) < 0.01 else f"{val:.4f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (offset if val >= 0 else -offset),
            label,
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    fig2.tight_layout()
    path_bar = out_dir / "decile_bar.png"
    fig2.savefig(path_bar, dpi=PLOT_DPI)
    plt.close(fig2)

    # --- 3. cumulative_deciles_hl_excess.png ---
    # G1..G10 are already excess vs equal-weight universe in groupTest;
    # still plot vs frozen benchmark active for G10 narrative, and keep H-L.
    # Per requirement: G1~G10 relative to benchmark cumulative excess.
    # group_pnl columns are already excess returns vs EW cross-section;
    # for benchmark-relative: g_abs = g_excess + bench? No —
    # In this engine, group returns are excess vs cap/EW universe, NOT vs 000852.
    # Protocol long-only reconstructs abs = g10_excess + bench.
    # For plot3: show (G_k absolute - benchmark) = G_k excess + (EW_universe - benchmark)?
    # User asked: "G1~G10 相对 benchmark 的累计超额收益".
    # Closest protocol-consistent construction matching long_only_metrics:
    #   g_abs = g_excess + bench  is WRONG because g_excess is vs EW not zero.
    # Looking at groupTest: group returns are typically absolute or excess?
    # In Factor_Dev_Lib groupTest, returns are usually stock excess or raw.
    # From long_only_metrics: g10_abs = g10_excess + bench — so they treat
    # group_pnl as excess vs something that +bench recovers absolute.
    # That implies group_pnl is excess vs the same benchmark series used.
    # So plot3 can use group_pnl directly as "excess", with H-L retained.
    # We'll title it clearly as engine excess (+ H-L LS), benchmark=000852.SH.
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    bench = benchmark.reindex(pnl.index).astype(float)
    # Reconstruct absolute then subtract frozen benchmark (same as long-only layer).
    excess_vs_bench = pd.DataFrame(index=pnl.index)
    for col in group_cols:
        g_abs = pnl[col].astype(float) + bench
        excess_vs_bench[col] = g_abs - bench  # == pnl[col] if aligned
    # Numerically identical to pnl[col] when bench aligns; keep explicit for audit.
    for col, color, label in zip(group_cols, colors, labels):
        ax3.plot(
            excess_vs_bench.index,
            excess_vs_bench[col].cumsum().to_numpy(),
            color=color,
            linewidth=1.15,
            alpha=0.85,
            label=label,
        )
    ax3.plot(
        cum.index,
        cum["H-L"].to_numpy(),
        color="black",
        linewidth=2.6,
        alpha=0.95,
        label="H-L (long-short)",
        zorder=5,
    )
    ax3.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    _add_segment_markers(ax3, float(cum.max().max()) * 0.98)
    ax3.set_title(
        f"{FACTOR} — Decile Cumulative Excess vs {BENCHMARK} + H-L "
        f"(effective; FULL)\n"
        "G1=low effective factor … G10=high effective factor",
        fontsize=12,
    )
    ax3.set_xlabel("TradeDate")
    ax3.set_ylabel("Cumulative excess / H-L (cumsum)")
    ax3.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
    ax3.grid(True, axis="y", alpha=0.3)
    g10_ex = float(calAnnuRet(pnl[group_cols[-1]]))
    stats3 = (
        f"bench: {BENCHMARK}\n"
        f"G10 excess ann: {g10_ex:.2%}\n"
        f"H-L Ann. Ret: {annu:.2%}\n"
        f"H-L Sharpe:   {sharpe:.2f}\n"
        f"H-L Max DD:   {mdd:.2%}\n"
        f"mono/viol:    {mono:.3f}/{viol}"
    )
    _stats_box(ax3, stats3, float(cum["H-L"].iloc[-1]))
    fig3.tight_layout()
    path_ex = out_dir / "cumulative_deciles_hl_excess.png"
    fig3.savefig(path_ex, dpi=PLOT_DPI)
    plt.close(fig3)

    return {
        "cumulative_deciles_hl.png": str(path_dec),
        "decile_bar.png": str(path_bar),
        "cumulative_deciles_hl_excess.png": str(path_ex),
        "cumulative_hl.png": str(out_dir / "cumulative_hl.png"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    freeze: Dict[str, Any],
    cache: Dict[str, Any],
    factor_df: pd.DataFrame,
    long_df: pd.DataFrame,
    leg_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    factor_grade: str,
    long_only_status: str,
    flags: List[str],
    strat_warns: List[str],
    retention: Dict[str, float],
    direction: int,
) -> str:
    by_f = {r["sample"]: r for _, r in factor_df.iterrows()}
    by_l = {r["sample"]: r for _, r in long_df.iterrows()}
    by_leg = {r["sample"]: r for _, r in leg_df.iterrows()}
    full = by_f["FULL"]
    post = by_f["POST"]
    disc = by_f["DISCOVERY"]
    long_full = by_l["FULL"]
    leg_full = by_leg["FULL"]
    leg_post = by_leg["POST"]
    leg_disc = by_leg["DISCOVERY"]

    if factor_grade.startswith("A"):
        freeze_status = "FROZEN_A_STRONG"
    elif factor_grade.startswith("B"):
        freeze_status = "FROZEN_B_RESEARCH"
    else:
        freeze_status = "FROZEN_C_NOT_CONFIRMED"

    cost_s_ret = float(full["cost_sharpe_retention"])
    cost_r_ret = float(full["cost_return_retention"])

    # Q10: control vs standalone research factor
    if factor_grade.startswith("A"):
        q10 = (
            "**Preserve as strong standalone research factor** while retaining "
            "`ORDINARY_OHLCV_CONTROL=True` taxonomy tag (not L2 microstructure-specific)."
        )
    elif factor_grade.startswith("B"):
        q10 = (
            "**Preserve as research factor with ORDINARY_OHLCV_CONTROL tag** — "
            "useful Price Formation baseline; do not count as distinct L2 microstructure alpha."
        )
    else:
        q10 = (
            "**Remain ORDINARY_OHLCV_CONTROL only** — gross may look strong, but "
            f"grade `{factor_grade}` does not justify treating it as a confirmed "
            "standalone research production candidate."
        )

    lines = [
        "# Single-Factor Full Validation v2.0 — `intraday_max_drawdown`",
        "",
        "Protocol v2.0 FINAL FROZEN. No formula edits. "
        "`ORDINARY_OHLCV_CONTROL = True`.",
        "",
        "## Evidence label roles (NOT pristine unseen for FULL/POST)",
        "",
        "- **DISCOVERY** (2023-01-01~2024-12-31): frozen Fast Discovery selection sample",
        "- **PRE** (2019-01-01~2022-12-31): pre-discovery retrospective segment",
        "- **POST** (2025-01-01~2026-07-31): recent decay diagnostic",
        "- **FULL** (2019-01-01~2026-07-31): retrospective robustness validation",
        "",
        "This family already had FULL history computed; FULL/POST are **not** pristine unseen.",
        "",
        "## Factor freeze",
        "",
        f"- factor_id: `{FACTOR}`",
        f"- exact_formula: `{EXACT_FORMULA}`",
        f"- formula_hash: `{FACTOR_HASH}`",
        f"- source_primitive: `{SOURCE_PRIMITIVE}`",
        f"- formula_version: `{FORMULA_VERSION}`",
        f"- raw_direction: `{direction}`",
        f"- signal_shift: `{SIGNAL_SHIFT}`",
        f"- ORDINARY_OHLCV_CONTROL: `{ORDINARY_OHLCV_CONTROL}`",
        f"- cache reuse: `{cache['status']}` (rebuild={cache['rebuild']})",
        "",
        "## Sprint 11 discovery snapshot (must approximately match)",
        "",
        f"- H-L Sharpe ≈ {DISCOVERY_METRICS['hl_sharpe']}",
        f"- mono = {DISCOVERY_METRICS['mono']}",
        f"- violations = {DISCOVERY_METRICS['violations']}",
        f"- G10 gross excess annual ≈ {DISCOVERY_METRICS['g10_gross_excess_annual']:.1%}",
        f"- FV DISCOVERY recompute: Sharpe={_fmt_num(disc['gross_hl_sharpe'], 2)}, "
        f"mono={_fmt_num(disc['decile_mono_gross'], 3)}, "
        f"viol={int(disc['adjacent_violations_gross'])}",
        "",
        "## Factor Layer (Protocol v2.0)",
        "",
        "| sample | RankIC | ICIR | gross H-L ann | gross Sharpe | net H-L ann | net Sharpe | mono | viol | pos-month | cost_S_ret | cost_R_ret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in ("DISCOVERY", "PRE", "POST", "FULL"):
        r = by_f[sample]
        lines.append(
            f"| {sample} | {_fmt_num(r['rank_ic'], 4)} | {_fmt_num(r['icir'], 2)} | "
            f"{_fmt_pct(r['gross_hl_annual'])} | {_fmt_num(r['gross_hl_sharpe'], 2)} | "
            f"{_fmt_pct(r['net_hl_annual'])} | {_fmt_num(r['net_hl_sharpe'], 2)} | "
            f"{_fmt_num(r['decile_mono_gross'], 3)} | {int(r['adjacent_violations_gross'])} | "
            f"{_fmt_pct(r['positive_hl_month_fraction'])} | "
            f"{_fmt_num(r['cost_sharpe_retention'], 3)} | "
            f"{_fmt_num(r['cost_return_retention'], 3)} |"
        )

    lines += [
        "",
        f"- FULL daily H-L L1 / one-way = `{full['avg_daily_hl_l1_traded_notional']:.4f}` / "
        f"`{full['avg_daily_hl_oneway_turnover']:.4f}`",
        f"- FULL annualized one-way = `{full['annualized_hl_oneway_turnover']:.1f}`",
        f"- FULL fee annualized (L1×7.5bps) = `{full['fee_annualized']:.2%}`",
        f"- FULL cost diagnostics (no thresholds): "
        f"cost_sharpe_retention=`{cost_s_ret:.3f}`, "
        f"cost_return_retention=`{cost_r_ret:.3f}`",
        "",
        "## Decay diagnostics",
        "",
        f"- IC_retention = `{retention['IC_retention']:.3f}`",
        f"- Sharpe_retention = `{retention['Sharpe_retention']:.3f}`",
        f"- flags: `{', '.join(flags) if flags else '(none)'}`",
        "",
        "## Long-only Strategy Layer (G10 vs 000852.SH)",
        "",
        "| sample | long_net_ann | net_excess_ann | IR | TE | excess_MDD | pos-excess-month | long one-way daily |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in ("DISCOVERY", "PRE", "POST", "FULL"):
        r = by_l[sample]
        lines.append(
            f"| {sample} | {_fmt_pct(r['long_net_annual_return'])} | "
            f"{_fmt_pct(r['excess_annual_return'])} | {_fmt_num(r['IR'], 2)} | "
            f"{_fmt_num(r['tracking_error'], 3)} | {_fmt_pct(r['excess_mdd'])} | "
            f"{_fmt_pct(r['positive_excess_month_fraction'])} | "
            f"{_fmt_num(r['avg_daily_long_oneway_turnover'], 3)} |"
        )

    lines += [
        "",
        f"- strategy warnings: `{', '.join(strat_warns) if strat_warns else '(none)'}`",
        "",
        "## Leg contribution",
        "",
        "| sample | G10 gross ex | G10 net ex | G1 gross | long contrib | short contrib | gross H-L | dominant |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sample in ("DISCOVERY", "PRE", "POST", "FULL"):
        r = by_leg[sample]
        lines.append(
            f"| {sample} | {_fmt_pct(r['g10_gross_excess_annual'])} | "
            f"{_fmt_pct(r['g10_net_excess_annual'])} | {_fmt_pct(r['g1_gross_excess_annual'])} | "
            f"{_fmt_pct(r['long_contribution'])} | {_fmt_pct(r['short_contribution'])} | "
            f"{_fmt_pct(r['gross_hl_annual'])} | {r['dominant_leg']} |"
        )

    lines += [
        "",
        f"- Discovery legs: long={_fmt_pct(leg_disc['long_contribution'])}, "
        f"short={_fmt_pct(leg_disc['short_contribution'])}",
        f"- FULL dominant: **{leg_full['dominant_leg']}**; "
        f"POST dominant: **{leg_post['dominant_leg']}**",
        "",
        "## Grades",
        "",
        f"- **factor_grade:** `{factor_grade}`",
        f"- **long_only_status:** `{long_only_status}`",
        f"- **freeze_status:** `{freeze_status}`",
        f"- **ORDINARY_OHLCV_CONTROL:** `{ORDINARY_OHLCV_CONTROL}`",
        "",
        "## FINAL QUESTIONS",
        "",
        f"1. Discovery gross Sharpe 3.11 → FULL = **{_fmt_num(full['gross_hl_sharpe'], 2)}** "
        f"(ann={_fmt_pct(full['gross_hl_annual'])}).",
        f"2. FULL mono / violations = **{_fmt_num(full['decile_mono_gross'], 3)} / "
        f"{int(full['adjacent_violations_gross'])}**.",
        f"3. Gross vs net survival: cost_sharpe_retention=**{_fmt_num(cost_s_ret, 3)}**, "
        f"cost_return_retention=**{_fmt_num(cost_r_ret, 3)}** "
        f"(net Sharpe={_fmt_num(full['net_hl_sharpe'], 2)}, net ann={_fmt_pct(full['net_hl_annual'])}, "
        f"fee={_fmt_pct(full['fee_annualized'])}).",
        f"4. Conventional one-way turnover (FULL daily) = "
        f"**{_fmt_num(full['avg_daily_hl_oneway_turnover'], 4)}** "
        f"(L1={_fmt_num(full['avg_daily_hl_l1_traded_notional'], 4)}).",
        f"5. POST stability: RankIC={_fmt_num(post['rank_ic'], 4)}, "
        f"Sharpe={_fmt_num(post['gross_hl_sharpe'], 2)}, "
        f"mono={_fmt_num(post['decile_mono_gross'], 3)}, "
        f"viol={int(post['adjacent_violations_gross'])}; "
        f"IC_ret={retention['IC_retention']:.3f}, Sharpe_ret={retention['Sharpe_retention']:.3f}; "
        f"flags=`{', '.join(flags) if flags else 'none'}`.",
        f"6. FULL G10 net excess / IR = **{_fmt_pct(long_full['excess_annual_return'])} / "
        f"{_fmt_num(long_full['IR'], 2)}**.",
        f"7. Long vs short — FULL **{leg_full['dominant_leg']}** "
        f"(long={_fmt_pct(leg_full['long_contribution'])}, "
        f"short={_fmt_pct(leg_full['short_contribution'])}); "
        f"POST **{leg_post['dominant_leg']}**.",
        f"8. factor_grade = **{factor_grade}**.",
        f"9. long_only_status = **{long_only_status}**.",
        f"10. {q10}",
        "",
        "## STOP",
        "",
        "Do not auto-start Sprint 12. Do not validate `close_auction_return`. "
        "Do not re-optimize `tail_return_share`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[A] factor freeze", flush=True)
    freeze = build_factor_freeze()
    (OUT_DIR / "factor_freeze.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("[B] existing full-history reuse gate", flush=True)
    cache = validate_existing_cache()
    print(f"  {cache['status']} rows={cache['n_rows']}", flush=True)

    print("[D] load FULL context + reused narrow", flush=True)
    narrow = load_factor_narrow_full()
    mask, ret = load_fast_context("full")
    benchmark = load_benchmark_return("full")

    print("[E] backtest FULL (reuse values; no CH rebuild)", flush=True)
    pnl_raw, to_raw, group_pnl, group_to, direction, rank_ic_raw = raw_and_effective(
        narrow, mask, ret
    )
    if direction != EXPECTED_RAW_DIRECTION:
        raise RuntimeError(
            f"raw_direction={direction} != expected {EXPECTED_RAW_DIRECTION}"
        )
    parity = check_effective_turnover_parity(
        factor_id=FACTOR,
        raw_direction=direction,
        to_raw=to_raw,
        to_eff=group_to,
    )
    print(f"  direction={direction} turnover_parity={parity['pass']}", flush=True)
    if not parity["pass"]:
        raise RuntimeError(f"turnover parity fail: {parity}")

    # Persist engine outputs for audit
    group_pnl.to_csv(OUT_DIR / "group_pnl.csv")
    group_to.to_csv(OUT_DIR / "group_to.csv")
    rank_ic_raw.to_frame("rank_ic_raw").to_csv(OUT_DIR / "rank_ic_raw.csv")

    factor_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    leg_rows: List[Dict[str, Any]] = []
    by_sample: Dict[str, Dict[str, Any]] = {}

    for sample_name, (s0, s1) in SAMPLES.items():
        fmet = factor_layer_metrics(
            group_pnl, group_to, rank_ic_raw, sample=sample_name, start=s0, end=s1
        )
        # diagnostic only — no thresholds
        g_s = float(fmet["gross_hl_sharpe"])
        n_s = float(fmet["net_hl_sharpe"])
        g_a = float(fmet["gross_hl_annual"])
        n_a = float(fmet["net_hl_annual"])
        fmet["cost_sharpe_retention"] = (
            n_s / g_s if abs(g_s) > 1e-12 else float("nan")
        )
        fmet["cost_return_retention"] = (
            n_a / g_a if abs(g_a) > 1e-12 else float("nan")
        )
        lmet = long_only_metrics(
            group_pnl, group_to, benchmark, sample=sample_name, start=s0, end=s1
        )
        cmet = contribution_for_sample(
            group_pnl, group_to, sample=sample_name, start=s0, end=s1
        )
        # PART H naming aliases
        cmet["g10_gross_excess"] = cmet["g10_gross_excess_annual"]
        cmet["g10_net_excess"] = cmet["g10_net_excess_annual"]
        cmet["g1_gross_excess"] = cmet["g1_gross_excess_annual"]
        labels = label_block(
            factor_id=FACTOR,
            factor_hash=FACTOR_HASH,
            sample=sample_name,
            return_kind="GROSS+NET_7P5BPS_L1",
        )
        fmet = {**labels, **fmet, "factor_direction": direction}
        lmet = {**labels, **lmet, "factor_direction": direction}
        factor_rows.append(fmet)
        long_rows.append(lmet)
        leg_rows.append(cmet)
        by_sample[sample_name] = {"factor": fmet, "long": lmet, "leg": cmet}
        print(
            f"  [{sample_name}] grossS={fmet['gross_hl_sharpe']:.2f} "
            f"netS={fmet['net_hl_sharpe']:.2f} mono={fmet['decile_mono_gross']:.3f} "
            f"viol={fmet['adjacent_violations_gross']} "
            f"excess={lmet['excess_annual_return']:.2%} IR={lmet['IR']:.2f}",
            flush=True,
        )

    # Discovery metric gate vs Sprint 11 record
    disc_m = by_sample["DISCOVERY"]["factor"]
    disc_leg = by_sample["DISCOVERY"]["leg"]
    sharpe_ok = abs(float(disc_m["gross_hl_sharpe"]) - DISCOVERY_METRICS["hl_sharpe"]) <= 0.25
    mono_ok = abs(float(disc_m["decile_mono_gross"]) - DISCOVERY_METRICS["mono"]) <= 0.05
    viol_ok = int(disc_m["adjacent_violations_gross"]) <= DISCOVERY_METRICS["violations"] + 1
    g10_ok = abs(
        float(disc_leg["g10_gross_excess_annual"]) - DISCOVERY_METRICS["g10_gross_excess_annual"]
    ) <= 0.05
    if not (sharpe_ok and mono_ok and viol_ok and g10_ok):
        raise RuntimeError(
            "DISCOVERY METRIC MISMATCH vs Sprint 11 — STOP AND INVESTIGATE: "
            f"sharpe={disc_m['gross_hl_sharpe']:.3f} (expect~{DISCOVERY_METRICS['hl_sharpe']}), "
            f"mono={disc_m['decile_mono_gross']:.3f}, "
            f"viol={disc_m['adjacent_violations_gross']}, "
            f"g10={disc_leg['g10_gross_excess_annual']:.3%}"
        )
    print("  discovery metrics match Sprint 11 record (within tolerance)", flush=True)

    full = by_sample["FULL"]["factor"]
    post = by_sample["POST"]["factor"]
    long_full = by_sample["FULL"]["long"]
    long_post = by_sample["POST"]["long"]
    factor_grade, flags, retention = assign_factor_grade(full, post)
    long_only_status, strat_warns = assign_long_only_status(long_full, long_post)

    factor_df = pd.DataFrame(factor_rows)
    long_df = pd.DataFrame(long_rows)
    leg_df = pd.DataFrame(leg_rows)

    decay_rows = [
        {
            "factor": FACTOR,
            "IC_retention": retention["IC_retention"],
            "Sharpe_retention": retention["Sharpe_retention"],
            "POST_rank_ic": post["rank_ic"],
            "FULL_rank_ic": full["rank_ic"],
            "POST_gross_hl_sharpe": post["gross_hl_sharpe"],
            "FULL_gross_hl_sharpe": full["gross_hl_sharpe"],
            "POST_mono": post["decile_mono_gross"],
            "POST_violations": post["adjacent_violations_gross"],
            "flags": "|".join(flags),
        }
    ]
    decay_df = pd.DataFrame(decay_rows)

    # segment_validation / full_summary convenience views
    segment_validation = factor_df.copy()
    full_summary = pd.DataFrame(
        [
            {
                "factor": FACTOR,
                "factor_hash": FACTOR_HASH,
                "ORDINARY_OHLCV_CONTROL": ORDINARY_OHLCV_CONTROL,
                "factor_grade": factor_grade,
                "long_only_status": long_only_status,
                "raw_direction": direction,
                "discovery_hl_sharpe_record": DISCOVERY_METRICS["hl_sharpe"],
                "full_gross_hl_sharpe": full["gross_hl_sharpe"],
                "full_gross_hl_annual": full["gross_hl_annual"],
                "full_net_hl_sharpe": full["net_hl_sharpe"],
                "full_net_hl_annual": full["net_hl_annual"],
                "cost_sharpe_retention": full["cost_sharpe_retention"],
                "cost_return_retention": full["cost_return_retention"],
                "full_mono": full["decile_mono_gross"],
                "full_violations": full["adjacent_violations_gross"],
                "full_rank_ic": full["rank_ic"],
                "full_icir": full["icir"],
                "full_daily_hl_oneway_turnover": full["avg_daily_hl_oneway_turnover"],
                "full_daily_hl_l1": full["avg_daily_hl_l1_traded_notional"],
                "post_gross_hl_sharpe": post["gross_hl_sharpe"],
                "post_mono": post["decile_mono_gross"],
                "IC_retention": retention["IC_retention"],
                "Sharpe_retention": retention["Sharpe_retention"],
                "long_net_excess_annual": long_full["excess_annual_return"],
                "long_IR": long_full["IR"],
                "long_pos_excess_month": long_full["positive_excess_month_fraction"],
                "flags": "|".join(flags),
                "strategy_warnings": "|".join(strat_warns),
                "protocol_version": PROTOCOL_VERSION,
            }
        ]
    )

    factor_df.to_csv(OUT_DIR / "factor_layer.csv", index=False)
    long_df.to_csv(OUT_DIR / "long_only_summary.csv", index=False)
    leg_df.to_csv(OUT_DIR / "leg_contribution.csv", index=False)
    decay_df.to_csv(OUT_DIR / "decay_flags.csv", index=False)
    segment_validation.to_csv(OUT_DIR / "segment_validation.csv", index=False)
    full_summary.to_csv(OUT_DIR / "full_summary.csv", index=False)

    print("[I] plots", flush=True)
    plot_paths = save_validation_plots_v2(
        OUT_DIR, group_pnl, group_to, benchmark, full
    )

    if factor_grade.startswith("A"):
        freeze_status = "FROZEN_A_STRONG"
    elif factor_grade.startswith("B"):
        freeze_status = "FROZEN_B_RESEARCH"
    else:
        freeze_status = "FROZEN_C_NOT_CONFIRMED"

    report = render_report(
        freeze,
        cache,
        factor_df,
        long_df,
        leg_df,
        decay_df,
        factor_grade,
        long_only_status,
        flags,
        strat_warns,
        retention,
        direction,
    )
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "task": "Single-Factor Full Validation v2.0",
        "factor": FACTOR,
        "formula_hash": FACTOR_HASH,
        "factor_hash_sha256": FACTOR_HASH,
        "ORDINARY_OHLCV_CONTROL": ORDINARY_OHLCV_CONTROL,
        "status": freeze_status,
        "factor_grade": factor_grade,
        "long_only_status": long_only_status,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_status": PROTOCOL_STATUS,
        "signal_shift": SIGNAL_SHIFT,
        "raw_direction": direction,
        "cache_reuse": cache,
        "evidence_roles": freeze["evidence_labels"],
        "flags": flags,
        "strategy_warnings": strat_warns,
        "retention": retention,
        "cost_diagnostics_full": {
            "cost_sharpe_retention": full["cost_sharpe_retention"],
            "cost_return_retention": full["cost_return_retention"],
            "note": "diagnostic only; no thresholds",
        },
        "turnover_parity": parity,
        "plots": plot_paths,
        "close_auction_return": "ANOMALOUS_ARTIFACT_WATCHLIST / NO_FULL_VALIDATION",
        "tail_return_share": "FROZEN_C_NOT_CONFIRMED / NO_PARAMETER_OPTIMIZATION",
        "sprint12_auto_start": False,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n[done] grade={factor_grade} long_only={long_only_status} "
        f"freeze={freeze_status} -> {OUT_DIR} ({manifest['elapsed_seconds']}s)",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    main()
