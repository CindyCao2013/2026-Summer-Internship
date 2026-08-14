"""FS-1 ML preprocessing adapter: HUATAI_STYLE_IND_CAP_Z_V1.

Reuses Factor_Dev_Lib.mad / zsc / cs_neutral_size_ind and the frozen
mcap_wide parquet. Industry from get_preheat_ind_data_citics (cached to disk).

Does NOT modify Fast Discovery RAW evaluation path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Factor_Dev_Lib import (
    cs_neutral_size_ind,
    get_preheat_ind_data_citics,
    mad,
    zsc,
)
from l2_factor_reproduction.feature_selection.contracts import (
    MAD_TANH,
    MAD_THRESHOLD,
    MCAP_PARQUET,
    PREPROCESS_CONTRACT_ID,
    PREPROCESS_STEPS,
)

logger = logging.getLogger(__name__)


def load_or_cache_industry(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path,
) -> pd.DataFrame:
    """Industry wide panel (dates × symbols), Citics L1 codes.

    Caches to ``cache_path`` after first DDB fetch so reruns are local.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if cache_path.exists():
        ind = pd.read_parquet(cache_path)
        ind.index = pd.to_datetime(ind.index).normalize()
        # ensure coverage
        if ind.index.min() <= start and ind.index.max() >= end:
            return ind.loc[start:end]
        logger.info("Industry cache incomplete; refreshing from DDB")

    raw = get_preheat_ind_data_citics(start, end)
    if "TradingDay" in raw.columns:
        raw = raw.set_index("TradingDay")
    raw.index = pd.to_datetime(raw.index).normalize()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_path)
    return raw.loc[start:end]


def load_log_mcap(
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """log(FloatMktCap) from frozen primitive; MAD+zsc on cap as in FV scripts."""
    mcap = pd.read_parquet(MCAP_PARQUET)
    mcap.index = pd.to_datetime(mcap.index).normalize()
    mcap = mcap.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if columns is not None:
        cols = [c for c in columns if c in mcap.columns]
        mcap = mcap.reindex(columns=cols)
    log_cap = np.log(mcap.where(mcap > 0))
    log_cap = mad(log_cap, threshold=MAD_THRESHOLD, tanh=MAD_TANH)
    log_cap = zsc(log_cap)
    return log_cap


def _industry_mean_impute_day(
    values: pd.Series,
    industry: pd.Series,
    impute_mask: pd.Series,
) -> pd.Series:
    """Fill ORDINARY missings with same-day industry mean; leave others NaN."""
    out = values.copy()
    need = impute_mask.fillna(False) & out.isna()
    if not need.any():
        return out
    # industry means from observed
    observed = out.notna() & industry.notna()
    if not observed.any():
        return out
    means = out[observed].groupby(industry[observed]).mean()
    fill_idx = need[need].index
    ind_codes = industry.reindex(fill_idx)
    mapped = ind_codes.map(means)
    out.loc[fill_idx] = mapped.values
    return out


def classify_impute_mask(
    values: pd.Series,
    trade_date: pd.Timestamp,
    factor_date_min: Optional[pd.Timestamp],
    lookback_days: int,
    day_coverage: float,
    *,
    structural_coverage_floor: float = 0.05,
) -> pd.Series:
    """Boolean mask: True where ORDINARY missing may be imputed.

    Non-missing → False. STRUCTURAL / WARMUP → False.
    """
    na = values.isna()
    out = pd.Series(False, index=values.index)
    if not na.any():
        return out

    # WARMUP
    if factor_date_min is not None and pd.notna(factor_date_min):
        warm_cut = pd.Timestamp(factor_date_min) + pd.Timedelta(
            days=max(int(lookback_days) - 1, 0)
        )
        if trade_date < warm_cut:
            return out  # all False — do not impute warmup

    # STRUCTURAL: near-empty day
    if day_coverage < structural_coverage_floor:
        return out

    out.loc[na] = True
    return out


def apply_huatai_style_ind_cap_z_v1(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    industry: pd.DataFrame,
    log_cap: pd.DataFrame,
    inventory: Optional[pd.DataFrame] = None,
    *,
    return_stages: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
    """Apply MAD → ordinary industry impute → IND_CAP residual → CS z-score.

    ``panel`` must contain TradeDate, Symbol + factor columns (aligned_raw).
    Returns processed panel with same key columns + factor columns.
    """
    if panel.empty:
        return panel.copy(), None

    inv = None
    if inventory is not None:
        inv = inventory.set_index("factor")

    # work in wide-per-date fashion but keep long output
    keys = panel[["TradeDate", "Symbol"]].copy()
    stages: Dict[str, pd.DataFrame] = {}

    # pivot each factor is heavy; process date-by-date on a wide matrix of factors
    dates = sorted(panel["TradeDate"].unique())
    out_blocks: List[pd.DataFrame] = []

    # optional stage sample storage
    stage_raw_sample: List[pd.DataFrame] = []
    stage_win_sample: List[pd.DataFrame] = []
    stage_imp_sample: List[pd.DataFrame] = []
    stage_nt_sample: List[pd.DataFrame] = []
    stage_z_sample: List[pd.DataFrame] = []

    for dt in dates:
        day = panel.loc[panel["TradeDate"] == dt].set_index("Symbol")
        feats = day.reindex(columns=list(factor_cols))
        # coverage per factor this day
        n_univ = len(feats)
        cov = feats.notna().sum(axis=0) / max(n_univ, 1)

        # 1) MAD winsor across symbols (row vector ops via DataFrame apply)
        # Factor_Dev_Lib.mad expects dates×symbols wide; we have symbols×factors.
        # Apply MAD on the transpose (1 "date" × symbols) per factor.
        winsor = feats.copy()
        for col in factor_cols:
            s = feats[col]
            if s.notna().sum() < 30:
                continue
            wide = s.to_frame().T
            wide.index = [dt]
            w = mad(wide, threshold=MAD_THRESHOLD, tanh=MAD_TANH)
            winsor[col] = w.iloc[0].reindex(winsor.index)

        # 2) ordinary industry-mean impute
        ind_row = industry.reindex(index=[dt]).iloc[0] if dt in industry.index else None
        imputed = winsor.copy()
        if ind_row is not None:
            ind_s = ind_row.reindex(imputed.index)
            for col in factor_cols:
                lookback = 1
                fmin = None
                if inv is not None and col in inv.index:
                    lb = inv.loc[col, "lookback_days"]
                    try:
                        lookback = int(float(lb)) if pd.notna(lb) and lb != "" else 1
                    except (TypeError, ValueError):
                        lookback = 1
                    fmin = pd.to_datetime(inv.loc[col, "date_min"], errors="coerce")
                mask = classify_impute_mask(
                    imputed[col],
                    pd.Timestamp(dt),
                    fmin,
                    lookback,
                    float(cov.get(col, 0.0)),
                )
                imputed[col] = _industry_mean_impute_day(
                    imputed[col], ind_s, mask
                )

        # 3) industry + ln(mktcap) neutralization
        cap_row = log_cap.reindex(index=[dt]).iloc[0] if dt in log_cap.index else None
        neut = imputed.copy() * np.nan
        if ind_row is not None and cap_row is not None:
            ind_s = ind_row.reindex(imputed.index)
            cap_s = cap_row.reindex(imputed.index)
            for col in factor_cols:
                resid = cs_neutral_size_ind(
                    imputed[col], ind_s, cap_s, nt_type="ind_cap"
                )
                neut.loc[resid.index, col] = resid.values

        # 4) CS z-score
        zwide = neut.copy()
        # zsc expects dates×symbols; per factor:
        for col in factor_cols:
            s = neut[col]
            if s.notna().sum() < 30:
                zwide[col] = np.nan
                continue
            w = s.to_frame().T
            w.index = [dt]
            zwide[col] = zsc(w).iloc[0].reindex(zwide.index)

        block = zwide.copy()
        block.insert(0, "Symbol", block.index)
        block = block.reset_index(drop=True)
        block.insert(0, "TradeDate", pd.Timestamp(dt))
        out_blocks.append(block[["TradeDate", "Symbol"] + list(factor_cols)])

        if return_stages:

            def _pack(df_stage: pd.DataFrame) -> pd.DataFrame:
                t = df_stage.copy()
                t.insert(0, "Symbol", t.index)
                t = t.reset_index(drop=True)
                t.insert(0, "TradeDate", pd.Timestamp(dt))
                return t

            stage_raw_sample.append(_pack(feats))
            stage_win_sample.append(_pack(winsor))
            stage_imp_sample.append(_pack(imputed))
            stage_nt_sample.append(_pack(neut))
            stage_z_sample.append(_pack(zwide))

    out = pd.concat(out_blocks, ignore_index=True) if out_blocks else keys.copy()
    # align column order
    out = keys.merge(out, on=["TradeDate", "Symbol"], how="left")

    stage_dict = None
    if return_stages:
        stage_dict = {
            "raw": pd.concat(stage_raw_sample, ignore_index=True) if stage_raw_sample else pd.DataFrame(),
            "winsorized": pd.concat(stage_win_sample, ignore_index=True) if stage_win_sample else pd.DataFrame(),
            "imputed": pd.concat(stage_imp_sample, ignore_index=True) if stage_imp_sample else pd.DataFrame(),
            "neutralized": pd.concat(stage_nt_sample, ignore_index=True) if stage_nt_sample else pd.DataFrame(),
            "zscore": pd.concat(stage_z_sample, ignore_index=True) if stage_z_sample else pd.DataFrame(),
        }
    return out, stage_dict


def audit_preprocessing_stages(
    stages: Dict[str, pd.DataFrame],
    factor_cols: Sequence[str],
    industry: pd.DataFrame,
    log_cap: pd.DataFrame,
    sample_dates: Sequence[pd.Timestamp],
    sample_factors: Sequence[str],
) -> pd.DataFrame:
    """Transform audit: zscore mean≈0/std≈1; neut vs ln_cap ≈0; industry means≈0."""
    rows = []
    z = stages.get("zscore", pd.DataFrame())
    neut = stages.get("neutralized", pd.DataFrame())
    for dt in sample_dates:
        dt = pd.Timestamp(dt).normalize()
        for fac in sample_factors:
            if fac not in factor_cols:
                continue
            rec: Dict[str, object] = {
                "TradeDate": str(dt.date()),
                "factor": fac,
            }
            if not z.empty and fac in z.columns:
                day = z.loc[z["TradeDate"] == dt, fac]
                rec["z_mean"] = float(day.mean()) if day.notna().any() else np.nan
                rec["z_std"] = float(day.std(ddof=0)) if day.notna().sum() > 1 else np.nan
                rec["z_n"] = int(day.notna().sum())
            else:
                rec["z_mean"] = rec["z_std"] = np.nan
                rec["z_n"] = 0

            if not neut.empty and fac in neut.columns and dt in industry.index and dt in log_cap.index:
                nseries = neut.loc[neut["TradeDate"] == dt].set_index("Symbol")[fac]
                cap_s = log_cap.loc[dt].reindex(nseries.index)
                ind_s = industry.loc[dt].reindex(nseries.index)
                valid = nseries.notna() & cap_s.notna()
                if valid.sum() >= 30:
                    rec["corr_vs_ln_cap"] = float(
                        nseries[valid].corr(cap_s[valid], method="spearman")
                    )
                else:
                    rec["corr_vs_ln_cap"] = np.nan
                # industry means abs mean
                valid_i = nseries.notna() & ind_s.notna()
                if valid_i.any():
                    means = nseries[valid_i].groupby(ind_s[valid_i]).mean().abs()
                    rec["mean_abs_industry_mean"] = float(means.mean())
                else:
                    rec["mean_abs_industry_mean"] = np.nan
            else:
                rec["corr_vs_ln_cap"] = np.nan
                rec["mean_abs_industry_mean"] = np.nan

            # pass heuristics
            z_ok = (
                pd.notna(rec["z_mean"])
                and abs(float(rec["z_mean"])) < 0.05
                and pd.notna(rec["z_std"])
                and abs(float(rec["z_std"]) - 1.0) < 0.15
            )
            cap_ok = pd.isna(rec["corr_vs_ln_cap"]) or abs(float(rec["corr_vs_ln_cap"])) < 0.15
            ind_ok = (
                pd.isna(rec["mean_abs_industry_mean"])
                or float(rec["mean_abs_industry_mean"]) < 0.15
            )
            rec["pass"] = bool(z_ok and cap_ok and ind_ok)
            rows.append(rec)
    return pd.DataFrame(rows)


def preprocess_contract_manifest() -> Dict[str, object]:
    return {
        "contract_id": PREPROCESS_CONTRACT_ID,
        "steps": list(PREPROCESS_STEPS),
        "mad_threshold": MAD_THRESHOLD,
        "mad_tanh": MAD_TANH,
        "mcap_source": str(MCAP_PARQUET),
        "neutralization": "cs_neutral_size_ind(nt_type='ind_cap')",
        "imputation": "ORDINARY only → same-day Citics industry mean",
        "fast_discovery_impact": "none",
    }
