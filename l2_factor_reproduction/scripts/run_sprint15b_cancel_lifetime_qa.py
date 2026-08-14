#!/usr/bin/env python
"""Sprint 15B QA + overlap + decision for cancel_lifetime_daily.

NO alpha / discovery / FV.
"""

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
from l2_factor_reproduction.python import cancel_lifetime_daily as cld  # noqa: E402

PRIM = Path(RESULT_ROOT) / "primitives" / "cancel_lifetime_daily"
DS = PRIM / "dataset"
OUT = Path(RESULT_ROOT) / "sprint15_order_lifetime" / "full_history_build"
CANCEL_DS = Path(RESULT_ROOT) / "primitives" / "cancel_lifecycle_daily" / "dataset"
NEAR_ALIAS = 0.90


def load_daily() -> pd.DataFrame:
    files = sorted(DS.glob("quarter=*/cancel_lifetime_daily_*.parquet"))
    files = [p for p in files if "smoke" not in p.parent.name]
    if not files:
        raise FileNotFoundError(f"no partitions under {DS}")
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    return df.sort_values(
        ["TradeDate", "source_exchange", "symbol"], kind="stable"
    ).reset_index(drop=True)


def write_schema() -> None:
    schema = {
        "schema_version": cld.SCHEMA_VERSION,
        "formula_version": cld.FORMULA_VERSION,
        "architecture_class": "EVENT_DRIVEN_L2",
        "primary_object": "cancel_age / order commitment (NOT full fill lifetime)",
        "columns": list(cld.DAILY_COLUMNS),
        "formulas": {
            "cancel_age_ms": "cancel_time - order_add_time for cancel-terminated orders",
            "partial_fill_then_cancel": "n_cancel>0 AND cancel_qty < order_size",
            "censored_order_share_v1": (
                "non_cancel / eligible (= FULL_FILL ∪ SESSION_END_CENSORED; "
                "not separated without fill join)"
            ),
            "universe": "continuous auction posted orders; SSE Type=A; SZSE Cat1/2",
            "exchange_separation": "source_exchange retained; no pooled cross-section norm",
        },
        "exclusions": [
            "auction boundary adds",
            "orders without Type=A / Cat1-2 add",
            "session-end censored from cancel_age sample",
            "no short_lived threshold in 15B",
        ],
    }
    (OUT / "primitive_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["year"] = df["TradeDate"].dt.year
    df["quarter"] = df["TradeDate"].dt.to_period("Q").astype(str)
    for (ex, y), g in df.groupby(["source_exchange", "year"]):
        rows.append(
            {
                "source_exchange": ex,
                "year": int(y),
                "rows": int(len(g)),
                "symbols": int(g["symbol"].nunique()),
                "n_dates": int(g["TradeDate"].nunique()),
                "mean_eligible": float(g["eligible_order_count"].mean()),
                "mean_cancel_age_median_ms": float(
                    g["cancel_age_median_ms"].dropna().mean()
                ),
                "p50_cancel_age_median_ms": float(
                    g["cancel_age_median_ms"].median()
                ),
                "mean_censored_share": float(g["censored_order_share"].mean()),
                "mean_partial_share": float(
                    g["partial_fill_then_cancel_share"].mean()
                ),
                "sum_negative_lifetime": int(
                    g["negative_lifetime_count"].fillna(0).sum()
                ),
                "dup_keys": int(
                    g.duplicated(["symbol", "TradeDate", "source_exchange"]).sum()
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(["source_exchange", "year"])
    out.to_csv(OUT / "primitive_coverage.csv", index=False)
    return out


def exchange_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ex, g in df.groupby("source_exchange"):
        age = g["cancel_age_median_ms"].dropna()
        rows.append(
            {
                "source_exchange": ex,
                "rows": int(len(g)),
                "symbols": int(g["symbol"].nunique()),
                "date_min": str(g["TradeDate"].min().date()),
                "date_max": str(g["TradeDate"].max().date()),
                "mean_eligible": float(g["eligible_order_count"].mean()),
                "cancel_age_median_p10": float(age.quantile(0.10)),
                "cancel_age_median_p25": float(age.quantile(0.25)),
                "cancel_age_median_p50": float(age.quantile(0.50)),
                "cancel_age_median_p75": float(age.quantile(0.75)),
                "cancel_age_median_p90": float(age.quantile(0.90)),
                "mean_cancel_age_median_ms": float(age.mean()),
                "mean_censored_share": float(g["censored_order_share"].mean()),
                "mean_partial_fill_then_cancel_share": float(
                    g["partial_fill_then_cancel_share"].mean()
                ),
                "mean_asymmetry_ms": float(
                    g["cancel_age_asymmetry_ms"].dropna().mean()
                ),
                "sum_negative_lifetime": int(
                    g["negative_lifetime_count"].fillna(0).sum()
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "exchange_diagnostics.csv", index=False)
    return out


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ex, g in df.groupby("source_exchange"):
        for col in [
            "cancel_age_median_ms",
            "cancel_age_p25_ms",
            "cancel_age_p75_ms",
            "cancel_age_asymmetry_ms",
            "censored_order_share",
            "partial_fill_then_cancel_share",
            "eligible_order_count",
        ]:
            s = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "source_exchange": ex,
                    "field": col,
                    "count": int(len(s)),
                    "mean": float(s.mean()) if len(s) else np.nan,
                    "std": float(s.std()) if len(s) else np.nan,
                    "min": float(s.min()) if len(s) else np.nan,
                    "p1": float(s.quantile(0.01)) if len(s) else np.nan,
                    "p25": float(s.quantile(0.25)) if len(s) else np.nan,
                    "p50": float(s.quantile(0.50)) if len(s) else np.nan,
                    "p75": float(s.quantile(0.75)) if len(s) else np.nan,
                    "p99": float(s.quantile(0.99)) if len(s) else np.nan,
                    "max": float(s.max()) if len(s) else np.nan,
                    "na_rate": float(g[col].isna().mean()),
                    "inf_rate": float(
                        np.isinf(pd.to_numeric(g[col], errors="coerce")).mean()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "primitive_distribution.csv", index=False)
    return out


def load_cancel_agg(start: str, end: str) -> pd.DataFrame:
    files = sorted(CANCEL_DS.glob("year=*/**/*.parquet"))
    if not files:
        files = sorted(CANCEL_DS.glob("**/*.parquet"))
    cols = [
        "symbol",
        "TradeDate",
        "buy_cancel_event_count",
        "sell_cancel_event_count",
        "buy_cancel_value",
        "sell_cancel_value",
        "total_trade_value",
        "total_trade_count",
    ]
    frames = []
    for path in files:
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "symbol"})
        keep = [c for c in cols if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"])
    panel = panel.loc[
        panel["TradeDate"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    # build representatives
    b = panel["buy_cancel_event_count"].astype(float)
    s = panel["sell_cancel_event_count"].astype(float)
    panel["cancel_count_pressure"] = (b - s) / (b + s).replace(0, np.nan)
    bv = panel["buy_cancel_value"].astype(float)
    sv = panel["sell_cancel_value"].astype(float)
    tv = panel["total_trade_value"].astype(float).replace(0, np.nan)
    panel["cancel_value_intensity"] = (bv + sv) / tv
    n_ev = b + s
    avg_c = (bv + sv) / n_ev.replace(0, np.nan)
    avg_t = tv / panel["total_trade_count"].astype(float).replace(0, np.nan)
    panel["relative_cancel_order_size"] = avg_c / avg_t
    return panel


def mean_daily_xs_spearman(
    left: pd.DataFrame, right: pd.DataFrame, col_l: str, col_r: str
) -> float:
    m = left[["symbol", "TradeDate", col_l]].merge(
        right[["symbol", "TradeDate", col_r]],
        on=["symbol", "TradeDate"],
        how="inner",
    )
    ics = []
    for _, g in m.groupby("TradeDate"):
        a = g[col_l].astype(float)
        b = g[col_r].astype(float)
        mask = a.notna() & b.notna()
        if int(mask.sum()) < 30:
            continue
        rho = a[mask].corr(b[mask], method="spearman")
        if np.isfinite(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


def overlap(df: pd.DataFrame) -> pd.DataFrame:
    # Discovery window only for overlap (consistent with prior sprints)
    start, end = "2023-01-01", "2024-12-31"
    sub = df.loc[df["TradeDate"].between(start, end)].copy()
    cancel = load_cancel_agg(start, end)
    rows = []
    if cancel.empty:
        pd.DataFrame(rows).to_csv(OUT / "overlap_diagnostics.csv", index=False)
        return pd.DataFrame(rows)
    reps = [
        "cancel_count_pressure",
        "cancel_value_intensity",
        "relative_cancel_order_size",
    ]
    for left in ("cancel_age_median_ms", "cancel_age_asymmetry_ms"):
        for right in reps:
            # per exchange separately (mandatory)
            for ex in ("SSE", "SZSE"):
                l = sub.loc[sub["source_exchange"] == ex]
                # cancel lifecycle symbols already have .SH/.SZ
                rho = mean_daily_xs_spearman(l, cancel, left, right)
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "source_exchange": ex,
                        "mean_daily_xs_spearman": rho,
                        "near_alias_risk": bool(
                            np.isfinite(rho) and abs(rho) >= NEAR_ALIAS
                        ),
                        "window": f"{start}..{end}",
                    }
                )
                print(
                    f"  {ex} {left} vs {right}: rho={rho:.4f}",
                    flush=True,
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "overlap_diagnostics.csv", index=False)
    return out


def decide(
    df: pd.DataFrame, cov: pd.DataFrame, exd: pd.DataFrame, ov: pd.DataFrame
) -> str:
    neg = int(df["negative_lifetime_count"].fillna(0).sum())
    dups = int(df.duplicated(["symbol", "TradeDate", "source_exchange"]).sum())
    na_age = float(df["cancel_age_median_ms"].isna().mean())
    # structural break: year-to-year median age ratio jump > 5x within exchange
    break_flag = False
    for ex in ("SSE", "SZSE"):
        sub = cov.loc[cov["source_exchange"] == ex].sort_values("year")
        vals = sub["p50_cancel_age_median_ms"].astype(float).to_numpy()
        for i in range(1, len(vals)):
            if vals[i - 1] > 0 and vals[i] > 0:
                ratio = max(vals[i], vals[i - 1]) / min(vals[i], vals[i - 1])
                if ratio > 5:
                    break_flag = True
    alias = bool(ov["near_alias_risk"].any()) if len(ov) else False
    sse_med = float(
        exd.loc[exd["source_exchange"] == "SSE", "cancel_age_median_p50"].iloc[0]
    )
    szse_med = float(
        exd.loc[exd["source_exchange"] == "SZSE", "cancel_age_median_p50"].iloc[0]
    )
    exchange_ok = np.isfinite(sse_med) and np.isfinite(szse_med) and sse_med > 0 and szse_med > 0

    ready = (
        len(df) > 0
        and neg == 0
        and dups == 0
        and na_age < 0.5
        and exchange_ok
        and not break_flag
        and not alias
    )
    verdict = (
        "A. CANCEL_LIFETIME_DAILY_PRIMITIVE_READY"
        if ready
        else "B. CANCEL_LIFETIME_DAILY_PRIMITIVE_NOT_READY"
    )
    (OUT / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    (OUT / "decision_aux.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "neg": neg,
                "dups": dups,
                "na_age": na_age,
                "break_flag": break_flag,
                "alias": alias,
                "sse_med_p50": sse_med,
                "szse_med_p50": szse_med,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return verdict


def write_report(
    df: pd.DataFrame,
    cov: pd.DataFrame,
    exd: pd.DataFrame,
    dist: pd.DataFrame,
    ov: pd.DataFrame,
    verdict: str,
) -> None:
    aux = json.loads((OUT / "decision_aux.json").read_text())
    sse = exd.loc[exd["source_exchange"] == "SSE"].iloc[0]
    szse = exd.loc[exd["source_exchange"] == "SZSE"].iloc[0]
    lines = [
        "# Sprint 15B — Conservative Cancel-Lifetime Full-History Build",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "## Build",
        "",
        f"- rows: `{len(df):,}`",
        f"- coverage: `{df['TradeDate'].min().date()}` → `{df['TradeDate'].max().date()}`",
        f"- symbols: `{df['symbol'].nunique()}`",
        f"- negative lifetime sum: `{aux['neg']}`",
        f"- duplicate keys: `{aux['dups']}`",
        "",
        "## Exchange separation (mandatory)",
        "",
        "```",
        exd.to_string(index=False),
        "```",
        "",
        f"- SSE median-of-daily-median cancel age: `{sse['cancel_age_median_p50']:.0f} ms`",
        f"- SZSE median-of-daily-median cancel age: `{szse['cancel_age_median_p50']:.0f} ms`",
        f"- ratio SSE/SZSE: `{sse['cancel_age_median_p50']/szse['cancel_age_median_p50']:.1f}x`",
        "",
        "## Coverage by year × exchange",
        "",
        "```",
        cov.to_string(index=False),
        "```",
        "",
        "## Overlap vs aggregate cancellation (2023–2024, per exchange)",
        "",
        "```",
        ov.to_string(index=False) if len(ov) else "unavailable",
        "```",
        "",
        "## Final Questions",
        "",
        f"1. Full-history build success? `{len(df)>0}`",
        f"2. Coverage: `{df['TradeDate'].min().date()}` → `{df['TradeDate'].max().date()}`",
        f"3. SSE vs SZSE cancel-age: SSE p50≈`{sse['cancel_age_median_p50']:.0f}ms`, "
        f"SZSE p50≈`{szse['cancel_age_median_p50']:.0f}ms` "
        f"(~{sse['cancel_age_median_p50']/max(szse['cancel_age_median_p50'],1):.1f}×)",
        "4. Difference stable across years? see coverage table (not a one-day fluke if persistent)",
        f"5. Censored/non-cancel share: SSE `{sse['mean_censored_share']:.3f}`, "
        f"SZSE `{szse['mean_censored_share']:.3f}` "
        "(v1 = non-cancel pool; fill vs censor not split)",
        f"6. Partial-fill-then-cancel share: SSE `{sse['mean_partial_fill_then_cancel_share']:.4f}`, "
        f"SZSE `{szse['mean_partial_fill_then_cancel_share']:.4f}`",
        "7. cancel_age_median vs cancel count/value: see overlap_diagnostics.csv",
        "8. cancel_age_asymmetry independence: see overlap vs pressure/intensity",
        f"9. Structural break flag: `{aux['break_flag']}`",
        "10. Worth Sprint 15C Fast Discovery? only if READY",
        f"11. Verdict: `{verdict}`",
        "",
        "## Hard stops",
        "",
        "- NO alpha backtest / discovery / FV / threshold / exchange-normalization",
        "- STOP after this Sprint",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "primitive_QA.md").write_text(
        "\n".join(
            [
                "# Sprint 15B — Primitive QA",
                "",
                f"**Verdict: `{verdict}`**",
                "",
                f"- negative lifetime = `{aux['neg']}` (require 0)",
                f"- duplicate lifecycle keys = `{aux['dups']}` (require 0)",
                "- cancel-age sample excludes session-end / non-cancel rows",
                "- QA reported separately for SSE and SZSE",
                "",
                "## Exchange diagnostics",
                "",
                "```",
                exd.to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(df: pd.DataFrame, verdict: str) -> None:
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    manifest = {
        "sprint": "15B",
        "title": "Conservative Cancel-Lifetime Full-History Build",
        "verdict": verdict,
        "primitive_path": str(PRIM),
        "row_count": int(len(df)),
        "date_min": str(df["TradeDate"].min().date()) if len(df) else None,
        "date_max": str(df["TradeDate"].max().date()) if len(df) else None,
        "outputs": files,
        "no_backtest": True,
        "no_discovery": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_schema()
    print("[load] cancel_lifetime_daily", flush=True)
    df = load_daily()
    print(
        f"  rows={len(df)} {df['TradeDate'].min().date()}→{df['TradeDate'].max().date()}",
        flush=True,
    )
    cov = coverage(df)
    exd = exchange_diagnostics(df)
    dist = distribution(df)
    print("[overlap]", flush=True)
    ov = overlap(df)
    verdict = decide(df, cov, exd, ov)
    write_report(df, cov, exd, dist, ov, verdict)
    write_manifest(df, verdict)
    print(f"[DONE] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
