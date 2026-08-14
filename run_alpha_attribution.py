#!/usr/bin/env python
"""Alpha Attribution Engine v1 — exposure chain + enhancer × dimension grid.

Answers:
  - Is D1 mostly size/vol/liquidity proxy, or independent return driver?
  - Which frozen dimension does each enhancer improve (cancel_shock → D4)?

Usage:
  OMP_NUM_THREADS=1 python run_alpha_attribution.py
  OMP_NUM_THREADS=1 python run_alpha_attribution.py --sample-days 504
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_attribution_engine import (
    DEFAULT_ENHANCER_LAMBDA_REPORT,
    ENGINE_VERSION,
    FROZEN_DIM_MAP,
    interpret_dimension_independence,
    run_enhancer_dimension_grid,
    run_exposure_attribution_batch,
    summarize_enhancer_targets,
)
from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from factor_data_loaders import (
    load_derivative_wide_tables,
    load_eod_enriched_tables,
    load_financial_ttmhis_long,
)
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_fundamental import build_fundamental_cache, build_fundamental_factor
from factor_formulas_value import build_value_factor
from factor_formulas_liquidity_norm import build_liquidity_norm_cache, build_liquidity_norm_factor
from l2_conditioning_layer import L2_STATE_LAYER
from run_l2_validation import build_any_eod, load_context as load_l2_context

OUT = cfg.RESEARCH_DIR
LOCK_PATH = OUT / ".alpha_attribution.lock"

ENHANCER_FACTORS = [x["factor"] for x in L2_STATE_LAYER] + [
    "quality_composite",
    "value_composite",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(f"Lock active: {LOCK_PATH}")
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def build_market_structure_exposures(enriched, pv_cache, norm_cache, start, end, sample_start) -> dict:
    """Four frozen proxies — separate from D1–D5 production reps."""
    size = np.log(enriched.float_mktcap.replace(0, np.nan))
    vol = build_any_eod("volatility_60d", pv_cache, norm_cache)
    liq = build_factor("amount_20d_mean", pv_cache)
    mom = build_factor("momentum_20d", pv_cache)
    return {
        "size": size.loc[start:end].iloc[sample_start:],
        "volatility": vol.loc[start:end].iloc[sample_start:],
        "liquidity": liq.loc[start:end].iloc[sample_start:],
        "momentum_reversal": mom.loc[start:end].iloc[sample_start:],
    }


def load_value_enhancer(enriched, session, start, end, sample_start) -> pd.DataFrame | None:
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    finance_long, _ = load_financial_ttmhis_long(preheat, end, session=session)
    der_tables, _ = load_derivative_wide_tables(preheat, end, session=session)
    cache = build_fundamental_cache(
        der_tables,
        close=enriched.close,
        finance_long=finance_long if len(finance_long) else None,
    )
    wide = build_value_factor("value_composite", cache)
    return wide.loc[start:end].iloc[sample_start:]


def load_quality_enhancer(enriched, session, start, end, sample_start) -> pd.DataFrame | None:
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    finance_long, _ = load_financial_ttmhis_long(preheat, end, session=session)
    if finance_long is None or len(finance_long) == 0:
        return None
    der_tables, _ = load_derivative_wide_tables(preheat, end, session=session)
    cache = build_fundamental_cache(
        der_tables,
        close=enriched.close,
        finance_long=finance_long,
    )
    wide = build_fundamental_factor("quality_composite", cache)
    return wide.loc[start:end].iloc[sample_start:]


def publish_information_space_v1(
    out_dir: Path,
    enhancer_summary: pd.DataFrame,
    *,
    stack_metrics_path: Optional[Path] = None,
) -> Path:
    """Freeze architecture spec from attribution + production stack v2."""
    base_dims = [
        {
            "dimension_id": spec.dimension_id,
            "name": spec.name,
            "representative": spec.representative,
            "layer": "base_alpha",
            "status": "frozen",
        }
        for spec in OHLCV_PRODUCTION_DIMENSIONS
    ]
    base_dims.append(
        {
            "dimension_id": "D6",
            "name": "Fundamental Value",
            "representative": "value_composite",
            "layer": "enhancer_only",
            "status": "frozen",
            "bricks": ["value_ep", "value_bp", "value_cfp"],
            "harness_verdict": "enhancer_not_standalone_dim",
        }
    )
    base_dims.append(
        {
            "dimension_id": "D7",
            "name": "Fundamental Quality",
            "representative": "quality_composite",
            "layer": "enhancer_only",
            "status": "frozen",
            "bricks": ["roe_stability", "gross_profitability", "cfo_quality"],
        }
    )
    enhancers = []
    for spec in L2_STATE_LAYER:
        enhancers.append(
            {
                "factor": spec["factor"],
                "layer": "conditional_enhancer",
                "source": "l2",
                "role": spec["role"],
                "tier": spec["tier"],
            }
        )
    enhancers.append(
        {
            "factor": "quality_composite",
            "layer": "conditional_enhancer",
            "source": "fundamental_quality",
            "role": "quality_state",
            "tier": "d7_representative",
        }
    )
    enhancers.append(
        {
            "factor": "value_composite",
            "layer": "conditional_enhancer",
            "source": "fundamental_value",
            "role": "value_state",
            "tier": "d6_representative",
        }
    )
    if len(enhancer_summary):
        for _, row in enhancer_summary.iterrows():
            for e in enhancers:
                if e["factor"] == row["enhancer_factor"]:
                    e["primary_target_dimension"] = row["best_dimension_id"]
                    e["sharpe_delta_at_lambda_0.2"] = float(row["sharpe_delta"])

    production_stack_v2 = {
        "version": "v2",
        "status": "production",
        "base": "equal_weight_cs_z(D1..D5)",
        "enhancers": [
            {"factor": "cn_cancel_shock", "lambda": 0.2, "source": "l2"},
            {"factor": "quality_composite", "lambda": 0.2, "source": "fundamental_quality"},
        ],
        "blend": "cs_z( eq_wt(D1..D5) + λ_cancel·z(cancel_shock) + λ_quality·z(quality_composite) )",
    }
    if stack_metrics_path and stack_metrics_path.exists():
        import json as _json

        production_stack_v2["artifact"] = str(stack_metrics_path.name)
        production_stack_v2["metrics"] = _json.loads(stack_metrics_path.read_text()).get(
            "metrics", {}
        )

    doc = {
        "version": "information_space_v1",
        "status": "frozen",
        "base_dimensions": base_dims,
        "enhancers": enhancers,
        "production_stack_v2": production_stack_v2,
        "deferred": [
            {"dimension_id": "D8", "pillar": "growth", "status": "not_started"},
        ],
        "closed_tracks": ["l2_level_v1", "l2.5_ssl2", "ml_score", "factor_zoo_expansion"],
    }
    path = out_dir / "alpha_information_space_v1.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Attribution Engine v1")
    parser.add_argument("--sample-days", type=int, default=504)
    args = parser.parse_args()

    acquire_lock()
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        log("=== Alpha Attribution Engine v1 ===")
        log("Loading L2 + frozen stack context...")
        ctx = load_l2_context(sample_days=args.sample_days, engine="v2", build_cluster_reps=False)

        start = cfg.START_DAY
        end = cfg.END_DAY
        preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
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
        sample_start = max(0, len(ctx["ret"]) - args.sample_days - 25)
        market_exp = build_market_structure_exposures(
            enriched, pv_cache, norm_cache, start, end, sample_start
        )

        quality_panel = load_quality_enhancer(enriched, session, start, end, sample_start)
        value_panel = load_value_enhancer(enriched, session, start, end, sample_start)
        gc.collect()

        frozen = ctx["frozen_panels"]
        candidates = dict(frozen)
        roles = {rep: "base_dimension" for rep in frozen}

        enhancer_panels = {
            k: v for k, v in ctx["l2_panels"].items() if k in ENHANCER_FACTORS
        }
        if quality_panel is not None:
            enhancer_panels["quality_composite"] = quality_panel
            candidates["quality_composite"] = quality_panel
            roles["quality_composite"] = "enhancer"
        if value_panel is not None:
            enhancer_panels["value_composite"] = value_panel
            candidates["value_composite"] = value_panel
            roles["value_composite"] = "enhancer"

        for k in enhancer_panels:
            if k not in candidates:
                candidates[k] = enhancer_panels[k]
                roles[k] = "enhancer"

        log("=== Step 1: Market-structure exposure chain ===")
        attr_df = run_exposure_attribution_batch(
            candidates,
            ctx["ret"],
            market_exp,
            frozen,
            roles=roles,
            sample_dates=args.sample_days,
        )
        attr_df["dimension_interpretation"] = attr_df.apply(interpret_dimension_independence, axis=1)

        attr_path = OUT / "alpha_attribution_v1.csv"
        attr_df.to_csv(attr_path, index=False)
        log(f"Saved -> {attr_path}")

        log("\n=== Frozen dimensions (exposure chain) ===")
        base_cols = [
            "dimension_id",
            "factor_name",
            "ic_raw",
            "ic_after_size",
            "ic_after_volatility",
            "ic_after_liquidity",
            "ic_after_momentum_reversal",
            "ic_true_residual",
            "ic_after_other_frozen_dims",
            "dominant_exposure_drop",
            "dimension_interpretation",
        ]
        show = attr_df[attr_df["role"] == "base_dimension"]
        log(show[[c for c in base_cols if c in show.columns]].to_string(index=False))

        log("\n=== Step 2: Enhancer × dimension grid (additive) ===")
        grid_df = run_enhancer_dimension_grid(frozen, enhancer_panels, ctx["ret"])
        grid_path = OUT / "alpha_enhancer_dimension_grid_v1.csv"
        grid_df.to_csv(grid_path, index=False)
        log(f"Saved -> {grid_path}")

        summary = summarize_enhancer_targets(grid_df, report_lambda=DEFAULT_ENHANCER_LAMBDA_REPORT)
        summary_path = OUT / "alpha_enhancer_targets_v1.csv"
        summary.to_csv(summary_path, index=False)
        log(f"Saved -> {summary_path}")

        if len(summary):
            log("\n=== Enhancer primary targets (λ=0.2) ===")
            log(summary.to_string(index=False))

        lam = DEFAULT_ENHANCER_LAMBDA_REPORT
        pivot = grid_df[np.isclose(grid_df["lambda"], lam)].pivot_table(
            index="enhancer_factor",
            columns="dimension_id",
            values="sharpe_delta",
            aggfunc="first",
        )
        pivot_path = OUT / "alpha_enhancer_sharpe_delta_matrix_v1.csv"
        pivot.to_csv(pivot_path)
        log(f"\nSharpe Δ matrix (λ={lam}) -> {pivot_path}")
        log(pivot.round(3).to_string())

        # Production Stack v2 evaluation (uses same sample window)
        from alpha_production_stack_v2 import (
            evaluate_stack_v2,
            load_enhancer_lambdas_from_attribution,
            publish_production_stack_v2,
        )

        stack_lambdas = load_enhancer_lambdas_from_attribution(OUT / "alpha_enhancer_targets_v1.csv")
        stack_enhancers = {
            "cn_cancel_shock": enhancer_panels.get("cn_cancel_shock"),
            "quality_composite": enhancer_panels.get("quality_composite"),
        }
        stack_enhancers = {k: v for k, v in stack_enhancers.items() if v is not None}
        stack_metrics = evaluate_stack_v2(frozen, stack_enhancers, ctx["ret"], lambdas=stack_lambdas)
        stack_path = publish_production_stack_v2(
            stack_metrics,
            stack_lambdas,
            out_dir=OUT,
            sample_days=args.sample_days,
            n_days=len(ctx["ret"]),
        )
        log(f"\n=== Production Stack v2 ===")
        log(f"  IC Δ {stack_metrics.ic_delta:+.4f}  |  H-L Sharpe Δ {stack_metrics.sharpe_delta:+.3f}")
        log(f"Published -> {stack_path}")

        info_path = publish_information_space_v1(
            OUT, summary, stack_metrics_path=stack_path
        )
        log(f"\nPublished -> {info_path}")

        meta = {
            "engine_version": ENGINE_VERSION,
            "sample_days": args.sample_days,
            "n_days": len(ctx["ret"]),
            "market_structure_exposures": list(market_exp.keys()),
            "enhancers_tested": list(enhancer_panels.keys()),
        }
        (OUT / "alpha_attribution_v1_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    finally:
        release_lock()


if __name__ == "__main__":
    main()
