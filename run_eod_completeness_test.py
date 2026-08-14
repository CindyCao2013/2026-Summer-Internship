#!/usr/bin/env python
"""EOD information completeness test: is the OHLCV manifold exhausted for alpha?"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

from eod_data_foundation import (
    EOD_DATA_STACK,
    PRIMITIVE_FEATURE_KEYS,
    build_primitive_features,
    foundation_summary_text,
)
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import EOD_ENGINE_REGISTRY, build_eod_engine_factor
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_REGISTRY,
    build_liquidity_norm_cache,
    build_liquidity_norm_factor,
)
from factor_taxonomy import (
    ALPHA_BUNDLE_V1_LIST,
    EOD_ENGINE_HF_V2_LIST,
    EOD_ENGINE_HF_V3_LIST,
    EOD_ENGINE_PRIORITY_A_LIST,
)
from liquidity_normalization import panel_cross_sectional_residual

EOD_ENGINE_FACTORS = set(EOD_ENGINE_REGISTRY.keys())
LIQUIDITY_NORM_FACTORS = set(LIQUIDITY_NORM_REGISTRY.keys())

# Representative alpha universe for subspace coverage (non-redundant intent)
ALPHA_UNIVERSE = sorted(
    set(ALPHA_BUNDLE_V1_LIST)
    | set(EOD_ENGINE_PRIORITY_A_LIST)
    | set(EOD_ENGINE_HF_V2_LIST)
    | set(EOD_ENGINE_HF_V3_LIST)
    | {"amount_stability_20d", "trend_consistency_20d", "volatility_level_20d"}
)


def _stack_sample(wide: pd.DataFrame, max_rows: int = 400_000) -> pd.Series:
    s = wide.stack(dropna=True)
    if len(s) > max_rows:
        s = s.sample(max_rows, random_state=42)
    return s


def _cs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1).replace(0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0)


def pca_effective_rank(stacked: pd.DataFrame, thresholds: Tuple[float, ...] = (0.8, 0.9, 0.95)) -> Dict:
    """PCA on standardized columns; return components needed for variance thresholds."""
    x = stacked.dropna()
    if len(x) < 100:
        return {"n_obs": len(x), "ranks": {}}
    x = (x - x.mean()) / x.std().replace(0, np.nan)
    x = x.dropna(axis=1, how="any")
    if x.shape[1] < 2:
        return {"n_obs": len(x), "ranks": {}}

    cov = np.cov(x.values, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = eigvals.sum()
    if total <= 0:
        return {"n_obs": len(x), "ranks": {}}

    cum = np.cumsum(eigvals) / total
    ranks = {}
    for thr in thresholds:
        ranks[f"pct_{int(thr * 100)}"] = int(np.searchsorted(cum, thr) + 1)

    return {
        "n_obs": len(x),
        "n_features": x.shape[1],
        "eigenvalues_top5": eigvals[:5].tolist(),
        "ranks": ranks,
    }


def daily_cs_r2(y: pd.DataFrame, xs: List[pd.DataFrame], max_dates: int = 250) -> float:
    """Mean daily cross-sectional R² of y on xs (date subsample for speed)."""
    dates = y.index
    if len(dates) > max_dates:
        step = max(1, len(dates) // max_dates)
        dates = dates[::step]
    y_sub = y.loc[dates]
    xs_sub = [x.loc[dates] for x in xs]
    resid = panel_cross_sectional_residual(y_sub, xs_sub)
    r2_list = []
    for dt_idx in dates:
        obs_y = y_sub.loc[dt_idx]
        obs_r = resid.loc[dt_idx]
        mask = obs_y.notna() & obs_r.notna()
        if mask.sum() < 50:
            continue
        yv = obs_y[mask]
        rv = obs_r[mask]
        ss_tot = ((yv - yv.mean()) ** 2).sum()
        ss_res = (rv ** 2).sum()
        if ss_tot > 0:
            r2_list.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2_list)) if r2_list else np.nan


def rank_ic_series(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    aligned = signal.loc[fwd_ret.index]
    return aligned.corrwith(fwd_ret, axis=1, method="spearman")


def build_alpha_panel(
    name: str,
    pv_cache,
    norm_cache,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    if name in LIQUIDITY_NORM_FACTORS:
        wide = build_liquidity_norm_factor(name, norm_cache)
    elif name in EOD_ENGINE_FACTORS:
        wide = build_eod_engine_factor(name, pv_cache)
    else:
        wide = build_factor(name, pv_cache)
    return _cs_zscore(wide.loc[start:end])


def build_all_panels(
    start: dt.datetime,
    end: dt.datetime,
    preheat: dt.datetime,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )

    primitives_raw = build_primitive_features(pv_cache)
    primitives = {
        k: _cs_zscore(v.loc[start:end]) for k, v in primitives_raw.items()
    }

    alphas = {}
    for name in ALPHA_UNIVERSE:
        try:
            alphas[name] = build_alpha_panel(name, pv_cache, norm_cache, start, end)
        except Exception as exc:
            print(f"[SKIP alpha] {name}: {exc}")

    close = enriched.close.loc[start:end]
    fwd_ret = close.shift(-1) / close - 1

    return primitives, alphas, fwd_ret


def primitive_coverage_by_alphas(
    primitives: Dict[str, pd.DataFrame],
    alphas: Dict[str, pd.DataFrame],
    top_k: int = 15,
) -> pd.DataFrame:
    alpha_names = sorted(alphas.keys())
    xs_top = [alphas[n] for n in alpha_names[:top_k]]
    xs_all = [alphas[n] for n in alpha_names]

    rows = []
    for pname, pwide in primitives.items():
        r2_top = daily_cs_r2(pwide, xs_top)
        r2_all = daily_cs_r2(pwide, xs_all)
        resid = panel_cross_sectional_residual(pwide, xs_all)
        rows.append(
            {
                "primitive": pname,
                "r2_top_alphas": r2_top,
                "r2_all_alphas": r2_all,
                "unexplained_pct": 1.0 - r2_all if pd.notna(r2_all) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def residual_ic_table(
    primitives: Dict[str, pd.DataFrame],
    alphas: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    max_dates: int = 250,
) -> pd.DataFrame:
    xs_all = [alphas[n] for n in sorted(alphas.keys())]
    dates = fwd_ret.index
    if len(dates) > max_dates:
        step = max(1, len(dates) // max_dates)
        dates = dates[::step]
    fwd_sub = fwd_ret.loc[dates]
    rows = []
    for pname, pwide in primitives.items():
        psub = pwide.loc[dates]
        resid = panel_cross_sectional_residual(psub, [x.loc[dates] for x in xs_all])
        ic = rank_ic_series(resid, fwd_sub).mean()
        raw_ic = rank_ic_series(psub, fwd_sub).mean()
        rows.append(
            {
                "primitive": pname,
                "raw_rank_ic": raw_ic,
                "residual_rank_ic": ic,
                "abs_residual_ic": abs(ic) if pd.notna(ic) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def max_primitive_alpha_corr(
    primitives: Dict[str, pd.DataFrame],
    alphas: Dict[str, pd.DataFrame],
    sample_rows: int = 200_000,
) -> pd.DataFrame:
    rows = []
    for pname, pwide in primitives.items():
        ps = _stack_sample(pwide, sample_rows)
        best = ("", 0.0)
        for aname, awide in alphas.items():
            as_ = _stack_sample(awide, sample_rows)
            joined = pd.concat([ps.rename("p"), as_.rename("a")], axis=1).dropna()
            if len(joined) < 1000:
                continue
            c = joined["p"].corr(joined["a"])
            if pd.notna(c) and abs(c) > abs(best[1]):
                best = (aname, c)
        rows.append(
            {
                "primitive": pname,
                "best_alpha_match": best[0],
                "max_corr": best[1],
            }
        )
    return pd.DataFrame(rows)


def completeness_verdict(
    prim_pca: Dict,
    alpha_pca: Dict,
    coverage: pd.DataFrame,
    residual_ic: pd.DataFrame,
) -> Dict[str, str]:
    prim_rank90 = prim_pca.get("ranks", {}).get("pct_90", 99)
    mean_r2 = coverage["r2_all_alphas"].mean()
    max_res_ic = residual_ic["abs_residual_ic"].max()

    if prim_rank90 <= 8:
        manifold = "LOW-DIM (~≤8 PCs explain 90% of primitive variance)"
    else:
        manifold = f"MEDIUM-DIM ({prim_rank90} PCs for 90% primitive variance)"

    if mean_r2 >= 0.75:
        transform = "HIGH — alpha universe spans most primitive subspace"
    elif mean_r2 >= 0.55:
        transform = "PARTIAL — room for residualization / interaction"
    else:
        transform = "LOW — many primitive directions not yet captured"

    if pd.notna(max_res_ic) and max_res_ic >= 0.02:
        ceiling = "NOT REACHED — primitive residuals still carry |IC| ≥ 2%"
        next_step = "Continue EOD: orthogonal residuals + regime conditioning"
    elif pd.notna(max_res_ic) and max_res_ic >= 0.01:
        ceiling = "APPROACHING — residual |IC| in 1–2% band"
        next_step = "Marginal EOD gains; prepare L2 / external data pipeline"
    else:
        ceiling = "NEAR CEILING for linear OHLCV projections"
        next_step = "Shift to L2 / fund flow / sentiment / options IV for new manifold"

    return {
        "primitive_manifold": manifold,
        "alpha_subspace_coverage": transform,
        "mean_primitive_r2_by_alphas": f"{mean_r2:.3f}" if pd.notna(mean_r2) else "n/a",
        "max_residual_primitive_ic": f"{max_res_ic:.4f}" if pd.notna(max_res_ic) else "n/a",
        "eod_ceiling_status": ceiling,
        "recommended_next": next_step,
    }


def write_report(
    path: Path,
    prim_pca: Dict,
    alpha_pca: Dict,
    coverage: pd.DataFrame,
    residual_ic: pd.DataFrame,
    corr_match: pd.DataFrame,
    verdict: Dict[str, str],
) -> None:
    lines = [
        "# EOD Information Completeness Test",
        "",
        foundation_summary_text(),
        "",
        "## Data stack",
        "",
    ]
    for layer in EOD_DATA_STACK:
        lines.append(f"- **{layer['layer']}**: {', '.join(layer['fields'][:6])}{'...' if len(layer['fields']) > 6 else ''}")
        lines.append(f"  - {layer['notes']}")

    lines.extend(["", "## PCA effective rank (stacked cross-sectional obs)", ""])
    lines.append(f"Primitives: {prim_pca}")
    lines.append(f"Alphas ({len(ALPHA_UNIVERSE)} candidates): {alpha_pca}")

    lines.extend(["", "## Primitive coverage by alpha subspace", ""])
    lines.append(_df_to_md(coverage.sort_values("r2_all_alphas", ascending=False)))

    lines.extend(["", "## Residual primitive IC (after projecting out all alphas)", ""])
    lines.append(_df_to_md(residual_ic.sort_values("abs_residual_ic", ascending=False)))

    lines.extend(["", "## Best alpha match per primitive (max |corr|)", ""])
    lines.append(_df_to_md(corr_match))

    lines.extend(["", "## Verdict", ""])
    for k, v in verdict.items():
        lines.append(f"- **{k}**: {v}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4g}" if pd.notna(val) else "")
            else:
                cells.append("" if pd.isna(val) else str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def main():
    parser = argparse.ArgumentParser(description="EOD OHLCV manifold completeness test")
    parser.add_argument("--sample-rows", type=int, default=400_000)
    args = parser.parse_args()

    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    out_dir = cfg.RESEARCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EOD panel + building primitive / alpha features...")
    primitives, alphas, fwd_ret = build_all_panels(start, end, preheat)
    print(f"  primitives: {len(primitives)}, alphas built: {len(alphas)}")

    prim_stack = pd.DataFrame(
        {k: _stack_sample(v, args.sample_rows) for k, v in primitives.items()}
    )
    alpha_stack = pd.DataFrame(
        {k: _stack_sample(v, args.sample_rows) for k, v in alphas.items()}
    )

    prim_pca = pca_effective_rank(prim_stack)
    alpha_pca = pca_effective_rank(alpha_stack)

    coverage = primitive_coverage_by_alphas(primitives, alphas)
    residual_ic = residual_ic_table(primitives, alphas, fwd_ret)
    corr_match = max_primitive_alpha_corr(primitives, alphas, args.sample_rows // 2)
    verdict = completeness_verdict(prim_pca, alpha_pca, coverage, residual_ic)

    coverage.to_csv(out_dir / "eod_primitive_coverage.csv", index=False)
    residual_ic.to_csv(out_dir / "eod_residual_primitive_ic.csv", index=False)
    corr_match.to_csv(out_dir / "eod_primitive_alpha_match.csv", index=False)
    pd.DataFrame([verdict]).to_csv(out_dir / "eod_completeness_verdict.csv", index=False)

    report_path = out_dir / "eod_completeness_report.md"
    write_report(
        report_path, prim_pca, alpha_pca, coverage, residual_ic, corr_match, verdict
    )

    print("\n=== EOD Completeness Verdict ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()
