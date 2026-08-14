#!/usr/bin/env python
"""Build the machine-readable L2 Candidate Pool v1 family index.

This index concatenates frozen family registries and baseline summaries. It
does not calculate cross-family correlations, select candidates, or combine
signals.

Every family summary is mapped onto ``CANDIDATE_SUMMARY_SCHEMA_V1``. Fields a
family cannot supply are filled from the factor's frozen per-factor artifacts
(summary.json / rank_ic.csv / group_pnl.csv / factor_narrow.parquet); any
remaining gap is recorded in the ``missing_reason`` column — nothing is
silently missing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import calAnnuRet, calSharpe  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.candidate_pool_registry import (  # noqa: E402
    BASELINE_POLICY,
    BRIDGE_CONFIG,
    BRIDGE_FACTOR,
    CANDIDATE_SUMMARY_SCHEMA_V1,
    MISSING_REASON_NO_CATEGORY,
    POOL_ROOT,
    FamilyConfig,
    active_families,
)


def _decile_monotonicity(summary: Dict[str, object]) -> float:
    group = summary.get("group_mean_annu", {})
    if not isinstance(group, dict):
        return float("nan")
    pairs = sorted(
        ((int(key), float(value)) for key, value in group.items()),
        key=lambda item: item[0],
    )
    return float(
        pd.Series([item[0] for item in pairs]).corr(
            pd.Series([item[1] for item in pairs]),
            method="spearman",
        )
    )


@lru_cache(maxsize=None)
def _factor_summary_json(directory: str) -> Dict[str, object]:
    path = Path(directory) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _narrow_coverage(path: str) -> Tuple[Optional[str], Optional[str], int]:
    frame = pd.read_parquet(path, columns=["symbol", "tradetime"])
    tradetime = pd.to_datetime(frame["tradetime"])
    return (
        str(tradetime.min().date()),
        str(tradetime.max().date()),
        int(frame["symbol"].nunique()),
    )


def _g10_excess_metrics(
    directory: Path,
) -> Tuple[float, float]:
    path = directory / "group_pnl.csv"
    if not path.exists():
        return float("nan"), float("nan")
    pnl = pd.read_csv(path, index_col=0)
    top = "10" if "10" in pnl.columns else pnl.columns[-2]
    series = pd.to_numeric(pnl[top], errors="coerce").dropna()
    return float(calAnnuRet(series)), float(calSharpe(series))


def _positive_ic_fraction(directory: Path, direction: int) -> float:
    path = directory / "rank_ic.csv"
    if not path.exists():
        return float("nan")
    ic = pd.read_csv(path, index_col=0).iloc[:, 0]
    raw = pd.to_numeric(ic, errors="coerce").dropna() * int(direction)
    if raw.empty:
        return float("nan")
    return float((raw > 0).mean())


def _bridge_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": BRIDGE_FACTOR,
                "formula": (
                    "(active_buy_amt-active_sell_amt)/float_market_cap"
                ),
                "mechanism": "trade_direction_amount_scaled_by_mcap",
                "lookback_days": 1,
                "signed": True,
                "expected_redundancy": (
                    "same flow numerator as net_buy_ratio; "
                    "Sprint-2 bridge candidate"
                ),
                "family": BRIDGE_CONFIG.name,
                "registry_status": "frozen_baseline",
            }
        ]
    )


def _bridge_summary() -> pd.DataFrame:
    factor_dir = Path(RESULT_ROOT) / BRIDGE_FACTOR
    summary = json.loads(
        (factor_dir / "summary.json").read_text(encoding="utf-8")
    )
    yearly = pd.read_csv(factor_dir / "yearly_ic.csv", index_col=0)
    yearly_mean = pd.to_numeric(yearly["mean"], errors="coerce").dropna()
    raw_mean = float(
        summary.get("rank_ic_mean_raw", summary["rank_ic_mean"])
    )
    same_sign = int((np.sign(yearly_mean) == np.sign(raw_mean)).sum())
    g10_ret, g10_sharpe = _g10_excess_metrics(factor_dir)
    direction = int(summary.get("factor_direction", 1))
    date_min, date_max, n_symbols = _narrow_coverage(
        str(factor_dir / "factor_narrow.parquet")
    )
    return pd.DataFrame(
        [
            {
                "factor": BRIDGE_FACTOR,
                "family": BRIDGE_CONFIG.name,
                "category": None,
                "mechanism": "trade_direction_amount_scaled_by_mcap",
                "lookback_days": 1,
                "factor_direction": direction,
                "rank_ic_raw": raw_mean,
                "icir_raw": float(summary["rank_icir"]),
                "rank_ic_std": float(summary["rank_ic_std"]),
                "positive_ic_fraction": _positive_ic_fraction(
                    factor_dir, direction
                ),
                "rank_ic_effective": float(summary["rank_ic_mean"]),
                "icir_effective": float(summary["rank_icir"]),
                "hl_annu_ret": float(summary["hl_annu_ret_flipped"]),
                "hl_sharpe": float(summary["hl_sharpe_flipped"]),
                "g10_excess_annu_ret": g10_ret,
                "g10_excess_sharpe": g10_sharpe,
                "hl_mdd": float(summary["hl_mdd_flipped"]),
                "avg_hl_turnover": float(summary["avg_hl_turnover"]),
                "implied_annu_fee": float(summary["implied_annu_fee"]),
                "net_annu_after_fee": (
                    float(summary["hl_annu_ret_flipped"])
                    - float(summary["implied_annu_fee"])
                ),
                "decile_mono_spearman": _decile_monotonicity(summary),
                "n_factor_rows": int(
                    pd.read_parquet(
                        factor_dir / "factor_narrow.parquet", columns=["symbol"]
                    ).shape[0]
                ),
                "date_min": date_min,
                "date_max": date_max,
                "n_symbols": n_symbols,
                "n_days": int(summary["n_days"]),
                "n_names_avg": float(summary["n_names_avg"]),
                "n_years": int(len(yearly_mean)),
                "same_sign_years": same_sign,
                "sign_consistency": float(same_sign / len(yearly_mean)),
                "positive_ic_years": int((yearly_mean > 0).sum()),
                "negative_ic_years": int((yearly_mean < 0).sum()),
                "yearly_ic_min": float(yearly_mean.min()),
                "yearly_ic_max": float(yearly_mean.max()),
                "redundancy_cluster_080": None,
                "max_corr_peer": None,
                "max_abs_corr": np.nan,
                "near_alias_observed": False,
                "group_pnl_saved_direction": summary.get(
                    "group_pnl_saved_direction"
                ),
            }
        ]
    )


def _legacy_schema_patch(
    config: FamilyConfig, summary: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Fill schema fields missing from legacy family summaries.

    Uses frozen per-factor artifacts only; no backtest is re-run. Returns the
    patched summary plus the list of schema fields that remain unavailable.
    """
    summary = summary.copy()
    permanently_missing: List[str] = []
    if "category" not in summary.columns:
        summary["category"] = None
        permanently_missing.append("category")

    patchable = {
        "rank_ic_std",
        "positive_ic_fraction",
        "g10_excess_annu_ret",
        "g10_excess_sharpe",
        "date_min",
        "date_max",
        "n_symbols",
    }
    needed = [
        column
        for column in patchable
        if column not in summary.columns
        or summary[column].isna().any()
    ]
    for column in needed:
        summary[column] = pd.Series(
            [np.nan] * len(summary), index=summary.index, dtype=object
        )

    for index, row in summary.iterrows():
        factor_dir = config.factor_result_dir(row["factor"])
        direction = int(row["factor_direction"])
        if "rank_ic_std" in needed:
            summary.at[index, "rank_ic_std"] = _factor_summary_json(
                str(factor_dir)
            ).get("rank_ic_std", np.nan)
        if "positive_ic_fraction" in needed:
            summary.at[index, "positive_ic_fraction"] = (
                _positive_ic_fraction(factor_dir, direction)
            )
        if "g10_excess_annu_ret" in needed or "g10_excess_sharpe" in needed:
            g10_ret, g10_sharpe = _g10_excess_metrics(factor_dir)
            if "g10_excess_annu_ret" in needed:
                summary.at[index, "g10_excess_annu_ret"] = g10_ret
            if "g10_excess_sharpe" in needed:
                summary.at[index, "g10_excess_sharpe"] = g10_sharpe
        narrow = factor_dir / "factor_narrow.parquet"
        if narrow.exists() and (
            "date_min" in needed
            or "date_max" in needed
            or "n_symbols" in needed
        ):
            date_min, date_max, n_symbols = _narrow_coverage(str(narrow))
            if "date_min" in needed:
                summary.at[index, "date_min"] = date_min
            if "date_max" in needed:
                summary.at[index, "date_max"] = date_max
            if "n_symbols" in needed:
                summary.at[index, "n_symbols"] = n_symbols
    return summary, permanently_missing


def _map_family_summary(
    config: FamilyConfig,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    summary = pd.read_csv(config.directory / config.summary_csv)
    summary["family"] = config.name
    missing_fields: List[str] = []
    if not config.has_categories or any(
        column not in summary.columns
        for column in CANDIDATE_SUMMARY_SCHEMA_V1
        if column not in ("factor", "family")
    ):
        summary, missing_fields = _legacy_schema_patch(config, summary)
    summary["family_redundancy_cluster_080"] = summary[
        "redundancy_cluster_080"
    ].map(
        lambda cluster: (
            f"{config.name}:{cluster}" if pd.notna(cluster) else None
        )
    )
    reasons: Dict[str, str] = {}
    for _, row in summary.iterrows():
        parts = []
        if "category" in missing_fields:
            parts.append(f"category: {MISSING_REASON_NO_CATEGORY}")
        reasons[row["factor"]] = "; ".join(parts)
    return summary, reasons


def main() -> int:
    registry_parts = []
    summary_parts = []
    family_counts: Dict[str, int] = {}
    family_cluster_counts: Dict[str, int] = {}
    missing_reasons: Dict[str, str] = {}

    families = active_families()
    for config in families:
        registry = pd.read_csv(config.directory / config.registry_csv)
        registry["family"] = config.name
        registry["registry_status"] = "frozen_baseline"
        registry_parts.append(registry)
        family_counts[config.name] = int(len(registry))

        summary, reasons = _map_family_summary(config)
        summary_parts.append(summary)
        missing_reasons.update(reasons)
        family_cluster_counts[config.name] = int(
            summary["redundancy_cluster_080"].nunique()
        )

    registry_parts.append(_bridge_registry())
    family_counts[BRIDGE_CONFIG.name] = 1
    registry = pd.concat(registry_parts, ignore_index=True, sort=False)
    bridge = _bridge_summary()
    missing_reasons[BRIDGE_FACTOR] = (
        f"category: {MISSING_REASON_NO_CATEGORY}; "
        "redundancy_cluster_080: bridge factor outside any family "
        "correlation cluster analysis"
    )
    summary = pd.concat(
        [*summary_parts, bridge],
        ignore_index=True,
        sort=False,
    )
    summary["missing_reason"] = summary["factor"].map(
        lambda name: missing_reasons.get(name, "")
    )
    if registry["name"].duplicated().any():
        duplicated = registry.loc[
            registry["name"].duplicated(keep=False), "name"
        ].tolist()
        raise ValueError(f"duplicate candidate names: {duplicated}")
    if set(registry["name"]) != set(summary["factor"]):
        raise ValueError(
            "registry and candidate summary factor sets do not match"
        )

    ordered = [
        column
        for column in CANDIDATE_SUMMARY_SCHEMA_V1
        if column in summary.columns
    ]
    remaining = [
        column
        for column in summary.columns
        if column not in CANDIDATE_SUMMARY_SCHEMA_V1
        and column != "missing_reason"
    ]
    summary = summary[[*ordered, "missing_reason", *remaining]]

    registry.to_csv(POOL_ROOT / "candidate_registry.csv", index=False)
    (POOL_ROOT / "candidate_registry.json").write_text(
        json.dumps(
            registry.to_dict("records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary.to_csv(POOL_ROOT / "candidate_summary.csv", index=False)

    cross_status = {
        config.name: (
            "complete"
            if all(
                (config.directory / name).exists()
                for name in config.cross_reference_files
            )
            else ("none_required" if not config.cross_reference_files else "pending")
        )
        for config in families
    }
    manifest = {
        "version": "l2_candidate_pool_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_summary_schema": "candidate_summary_schema_v1",
        "baseline_policy": BASELINE_POLICY,
        "n_formula_candidates": int(len(registry)),
        "family_counts": family_counts,
        "family_local_cluster_counts_080": family_cluster_counts,
        "sample": "2019-01-01 through 2026-07-31 baseline",
        "cross_family_correlation": cross_status,
        "exposure_audit": "pending",
        "selection": "not_started",
        "combination": "out_of_scope",
    }
    (POOL_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    family_dirs = "、".join(
        f"`{config.directory.name}/`" for config in families
    )
    readme = [
        "# L2 Candidate Pool v1",
        "",
        f"- 冻结候选口径：{len(registry)}",
        *[
            f"- {config.title}：{family_counts[config.name]}"
            for config in families
        ],
        f"- {BRIDGE_CONFIG.title}：1",
        "",
        "当前数量是公式数量，不是独立 alpha 数量。各 family 的"
        " |日截面 Spearman| ≥ 0.80 经验簇单独记录；跨家族仅提供只读"
        "相关参考，不作筛选或组合。",
        "",
        "## Machine-readable outputs",
        "",
        "- `candidate_registry.csv/json`：公式、机制与 family",
        "- `candidate_summary.csv`：统一 baseline 指标"
        "（candidate_summary_schema_v1，缺失字段带 missing_reason）",
        "- `manifest.json`：边界与阶段状态",
        f"- {family_dirs}：族内详细产物",
        "",
        "下一步：Liquidity / Price Impact 完成后进入统一 taxonomy / "
        "exposure audit；此处不作 KEEP/DROP 或组合结论。",
        "",
    ]
    (POOL_ROOT / "README.md").write_text(
        "\n".join(readme), encoding="utf-8"
    )
    print(
        f"[done] L2 Candidate Pool v1: {len(registry)} formulas -> "
        f"{POOL_ROOT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
