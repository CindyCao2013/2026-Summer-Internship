#!/usr/bin/env python
"""Sprint 14B — QA + overlap for sweep_penetration_daily (no alpha)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python import sweep_penetration_daily as spd  # noqa: E402

OUT = Path(RESULT_ROOT) / "sprint14_sweep_penetration" / "full_history_build"
PRIM = Path(RESULT_ROOT) / "primitives" / "sweep_penetration_daily"
DS = PRIM / "dataset"
LIQ = Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily" / "dataset"
NEAR_ALIAS = 0.90


def load_all() -> pd.DataFrame:
    files = sorted(DS.glob("quarter=*/sweep_penetration_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no partitions under {DS}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    return df.sort_values(["TradeDate", "symbol"]).reset_index(drop=True)


def load_liq_overlap_cols() -> pd.DataFrame:
    cols = [
        "symbol",
        "TradeDate",
        "amount_to_depth",
        "large_trade_impact",
        "daily_amount",
    ]
    files = sorted(LIQ.glob("quarter=*/liquidity_impact_daily_*.parquet"))
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    return df


def xs_spearman(df: pd.DataFrame, a: str, b: str) -> float:
    daily = []
    for _, g in df.groupby("TradeDate"):
        x = g[a].astype(float)
        y = g[b].astype(float)
        m = x.notna() & y.notna()
        if m.sum() < 30:
            continue
        daily.append(x[m].corr(y[m], method="spearman"))
    return float(np.nanmean(daily)) if daily else float("nan")


def run_qa(panel: pd.DataFrame) -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": spd.SCHEMA_VERSION,
        "formula_version": spd.FORMULA_VERSION,
        "columns": list(spd.DAILY_COLUMNS),
        "formulas": spd.PRIMITIVE_FORMULAS,
        "side_direction_contract": {
            "buy": "ASK",
            "sell": "BID",
            "inversion_forbidden": True,
        },
        "analysis_unit": "trade_print",
        "event_level_persisted": False,
        "note": "estimated vs strictly-before ~3s-stale SSL2 book",
    }
    (OUT / "primitive_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    key_dup = int(panel.duplicated(["symbol", "TradeDate"]).sum())
    panel = panel.copy()
    panel["year"] = panel["TradeDate"].dt.year

    cov_rows = []
    for year, g in panel.groupby("year"):
        cov_rows.append(
            {
                "year": int(year),
                "rows": int(len(g)),
                "symbols": int(g["symbol"].nunique()),
                "n_dates": int(g["TradeDate"].nunique()),
                "mean_usable_event_count": float(g["usable_event_count"].mean()),
                "mean_ambiguous_event_share": float(g["ambiguous_event_share"].mean()),
                "p50_ambiguous_event_share": float(g["ambiguous_event_share"].median()),
                "mean_sweep_2plus_share": float(g["sweep_2plus_share"].dropna().mean()),
                "mean_median_alignment_lag_ms": float(
                    g["median_alignment_lag_ms"].dropna().mean()
                ),
                "sse_rows": int((g["exchange"] == ".SH").sum()),
                "szse_rows": int((g["exchange"] == ".SZ").sum()),
            }
        )
    # field-level NA
    for col in [
        "sweep_2plus_share",
        "sweep_notional_share",
        "mean_estimated_levels_penetrated",
        "mean_depth_consumed_ratio",
        "sweep_directional_asymmetry",
        "usable_event_count",
        "ambiguous_event_share",
        "median_alignment_lag_ms",
    ]:
        s = panel[col]
        cov_rows.append(
            {
                "year": "ALL",
                "field": col,
                "rows": int(len(s)),
                "na_rate": float(s.isna().mean()),
                "inf_rate": float(
                    np.isinf(pd.to_numeric(s, errors="coerce")).mean()
                ),
            }
        )
    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(OUT / "primitive_coverage.csv", index=False)

    dist_rows = []
    for col in [
        "sweep_2plus_share",
        "sweep_notional_share",
        "mean_estimated_levels_penetrated",
        "mean_depth_consumed_ratio",
        "sweep_directional_asymmetry",
        "ambiguous_event_share",
        "median_alignment_lag_ms",
        "usable_event_count",
    ]:
        s = pd.to_numeric(panel[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        dist_rows.append(
            {
                "field": col,
                "count": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p1": float(s.quantile(0.01)),
                "p5": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }
        )
    distribution = pd.DataFrame(dist_rows)
    distribution.to_csv(OUT / "primitive_distribution.csv", index=False)

    exch = (
        panel.groupby("exchange")
        .agg(
            rows=("symbol", "size"),
            symbols=("symbol", "nunique"),
            mean_ambiguous=("ambiguous_event_share", "mean"),
            mean_sweep2=("sweep_2plus_share", "mean"),
            mean_levels=("mean_estimated_levels_penetrated", "mean"),
            mean_asym=("sweep_directional_asymmetry", "mean"),
            mean_lag=("median_alignment_lag_ms", "mean"),
        )
        .reset_index()
    )

    # Ambiguity drift: year-to-year change
    year_amb = (
        panel.groupby("year")["ambiguous_event_share"].mean().sort_index()
    )
    amb_drift = float(year_amb.max() - year_amb.min()) if len(year_amb) else np.nan

    qa = {
        "passed": bool(
            key_dup == 0
            and panel["TradeDate"].min() <= pd.Timestamp("2019-01-05")
            and panel["TradeDate"].max() >= pd.Timestamp("2026-07-01")
            and float(panel["ambiguous_event_share"].mean()) < 0.45
            and amb_drift < 0.25
            and float(panel["sweep_2plus_share"].dropna().mean()) > 0.01
            and float(panel["sweep_2plus_share"].dropna().mean()) < 0.50
        ),
        "symbol_date_duplicates": key_dup,
        "date_min": str(panel["TradeDate"].min().date()),
        "date_max": str(panel["TradeDate"].max().date()),
        "n_rows": int(len(panel)),
        "n_symbols": int(panel["symbol"].nunique()),
        "mean_ambiguous_event_share": float(panel["ambiguous_event_share"].mean()),
        "ambiguous_share_year_range": float(amb_drift),
        "mean_sweep_2plus_share": float(panel["sweep_2plus_share"].dropna().mean()),
        "mean_median_alignment_lag_ms": float(
            panel["median_alignment_lag_ms"].dropna().mean()
        ),
        "side_contract": "BUY=ASK / SELL=BID enforced in SQL; no counter-check failures possible at daily layer",
        "exchange_consistency": exch.to_dict(orient="records"),
        "year_ambiguous_means": {
            str(int(k)): float(v) for k, v in year_amb.items()
        },
    }

    (OUT / "primitive_QA.md").write_text(
        "\n".join(
            [
                "# Sprint 14B — Sweep Penetration Daily Primitive QA",
                "",
                f"**PASS = {qa['passed']}**",
                "",
                f"- rows: `{qa['n_rows']}` symbols: `{qa['n_symbols']}`",
                f"- dates: `{qa['date_min']}` → `{qa['date_max']}`",
                f"- duplicates: `{qa['symbol_date_duplicates']}`",
                f"- mean ambiguous_event_share: `{qa['mean_ambiguous_event_share']:.4f}`",
                f"- year ambiguous range (max-min): `{qa['ambiguous_share_year_range']:.4f}`",
                f"- mean sweep_2plus_share: `{qa['mean_sweep_2plus_share']:.4f}`",
                f"- mean median_alignment_lag_ms: `{qa['mean_median_alignment_lag_ms']:.1f}`",
                "",
                "## Coverage by year",
                "",
                coverage.loc[coverage["year"] != "ALL"].to_string(index=False)
                if "year" in coverage.columns
                else "",
                "",
                "## Distribution",
                "",
                distribution.to_string(index=False),
                "",
                "## Exchange consistency",
                "",
                exch.to_string(index=False),
                "",
                "## Notes",
                "",
                "- Side contract enforced in build SQL (BUY→ASK, SELL→BID).",
                "- Metrics are estimated vs strictly-before ~3s-stale book.",
                "- Event-level full history not persisted; daily aggregates only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "primitive_QA.json").write_text(
        json.dumps(qa, indent=2, default=str), encoding="utf-8"
    )
    return qa


def run_overlap(panel: pd.DataFrame) -> pd.DataFrame:
    liq = load_liq_overlap_cols()
    # discovery-ish window for speed/stability
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
    a = panel.loc[panel["TradeDate"].between(start, end)].copy()
    b = liq.loc[liq["TradeDate"].between(start, end)].copy()
    m = a.merge(b, on=["symbol", "TradeDate"], how="inner")
    m["log_daily_amount"] = np.log1p(m["daily_amount"].astype(float))

    lefts = [
        "sweep_2plus_share",
        "mean_estimated_levels_penetrated",
        "sweep_directional_asymmetry",
        "mean_depth_consumed_ratio",
        "sweep_notional_share",
    ]
    rights = [
        "log_daily_amount",
        "amount_to_depth",
        "large_trade_impact",
        "mean_trade_amount_usable",
    ]
    rows = []
    for L in lefts:
        for R in rights:
            if R not in m.columns:
                continue
            rho = xs_spearman(m, L, R)
            rows.append(
                {
                    "left": L,
                    "right": R,
                    "mean_daily_xs_spearman": rho,
                    "near_alias_risk": bool(
                        np.isfinite(rho) and abs(rho) >= NEAR_ALIAS
                    ),
                    "window": "2023-01-01..2024-12-31",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "overlap_diagnostics.csv", index=False)
    return out


def decide(qa: Dict[str, Any], overlap: pd.DataFrame) -> str:
    core = overlap.loc[
        overlap["left"].isin(
            [
                "sweep_2plus_share",
                "mean_estimated_levels_penetrated",
                "sweep_directional_asymmetry",
            ]
        )
        & overlap["right"].isin(
            ["log_daily_amount", "amount_to_depth", "large_trade_impact"]
        )
    ]
    size_alias = bool(core["near_alias_risk"].any()) if len(core) else True
    # at least one core sweep field not near-alias to size proxies
    non_alias_ok = (
        not core.loc[
            core["left"].isin(
                ["sweep_2plus_share", "mean_estimated_levels_penetrated"]
            ),
            "near_alias_risk",
        ].all()
        if len(core)
        else False
    )

    ready = (
        qa.get("passed", False)
        and non_alias_ok
        and not (
            # fail if BOTH key sweep shares are size aliases
            size_alias
            and core.loc[
                core["left"] == "sweep_2plus_share", "near_alias_risk"
            ].all()
            if len(core)
            else True
        )
    )
    # clearer rule:
    # READY if QA pass AND (sweep_2plus_share OR mean_levels) not near-alias to amount/amount_to_depth/large_trade
    key = core.loc[
        core["left"].isin(
            ["sweep_2plus_share", "mean_estimated_levels_penetrated"]
        )
    ]
    has_non_alias = (
        (~key["near_alias_risk"]).any() if len(key) else False
    )
    verdict = (
        "A. SWEEP_DAILY_PRIMITIVE_READY"
        if qa.get("passed") and has_non_alias
        else "B. SWEEP_DAILY_PRIMITIVE_NOT_READY"
    )

    report = [
        "# Sprint 14B — Full-History Sweep Daily Primitive",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Final questions",
        "",
        f"1. Full-history build successful? `{qa.get('passed') and qa.get('n_rows', 0) > 0}`",
        f"2. Coverage through? `{qa.get('date_min')} → {qa.get('date_max')}`",
        f"3. Mean ambiguous_event_share? `{qa.get('mean_ambiguous_event_share'):.4f}`",
        f"4. SSE/SZSE consistent? see exchange table in primitive_QA.md (definition shared; levels differ by market microstructure)",
        f"5. sweep_2plus_share distribution reasonable? mean=`{qa.get('mean_sweep_2plus_share'):.4f}` (expect mid-single-digit to teens %)",
        f"6. estimated_levels still distinct from trade size? `{has_non_alias}` (see overlap_diagnostics.csv)",
        f"7. sweep_directional_asymmetry independent? check overlap vs amount proxies",
        f"8. Worth 14C Fast Discovery? `{'YES' if verdict.startswith('A') else 'NO'}` (human-gated; not auto-started)",
        f"9. Final verdict: **{verdict.split('. ', 1)[-1] if '. ' in verdict else verdict}**",
        "",
        "## Overlap (2023–2024 mean daily XS Spearman)",
        "",
        overlap.to_string(index=False),
        "",
        "## Rules",
        "",
        "- No alpha backtest / discovery / parameter optimization / FV in 14B.",
        "- STOP after this decision.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")

    manifest = {
        "sprint": "14B",
        "verdict": verdict,
        "qa": qa,
        "has_non_alias_sweep_signal": has_non_alias,
        "primitive_dataset": str(DS.relative_to(PROJ_ROOT))
        if DS.is_relative_to(PROJ_ROOT)
        else str(DS),
        "artifacts": [
            "primitive_schema.json",
            "primitive_coverage.csv",
            "primitive_distribution.csv",
            "primitive_QA.md",
            "overlap_diagnostics.csv",
            "report.md",
            "manifest.json",
            "verdict.txt",
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return verdict


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] sweep_penetration_daily", flush=True)
    panel = load_all()
    print(
        f"[load] rows={len(panel)} {panel.TradeDate.min().date()}..{panel.TradeDate.max().date()}",
        flush=True,
    )
    qa = run_qa(panel)
    print(f"[QA] passed={qa['passed']}", flush=True)
    overlap = run_overlap(panel)
    print("[overlap] done", flush=True)
    verdict = decide(qa, overlap)
    print(f"[DONE] {verdict}", flush=True)
    return 0 if verdict.startswith("A") else 1


if __name__ == "__main__":
    raise SystemExit(main())
