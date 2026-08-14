"""Neutralization ladder for cutting factors — style vs pure alpha.

Modes:
  raw
  size          residual vs ln(float_mktcap)
  industry      within-industry demean
  size_industry industry demean then residual vs ln(size)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from factor_attribution import cs_zscore
from factor_cutting.cutting_analysis.knife_ic import ic_stats, monthly_ic_stats
from industry_neutral import panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual

NEUT_MODES = ("raw", "size", "industry", "size_industry")


def neutralize_panel(
    panel: pd.DataFrame,
    mode: str,
    *,
    industry: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply one neutralization mode. Missing exposures → return raw (with warning via NaN attrs)."""
    if mode == "raw":
        return panel

    if mode == "size":
        if float_mktcap is None:
            raise ValueError("size neutralization needs float_mktcap")
        log_size = np.log(float_mktcap.replace(0, np.nan)).reindex_like(panel)
        return panel_cross_sectional_residual(panel, [log_size])

    if mode == "industry":
        if industry is None:
            raise ValueError("industry neutralization needs industry panel")
        return panel_industry_demean(panel, industry.reindex_like(panel))

    if mode == "size_industry":
        if industry is None or float_mktcap is None:
            raise ValueError("size_industry needs industry + float_mktcap")
        ind = panel_industry_demean(panel, industry.reindex_like(panel))
        log_size = np.log(float_mktcap.replace(0, np.nan)).reindex_like(ind)
        return panel_cross_sectional_residual(ind, [log_size])

    raise KeyError(f"Unknown neut mode: {mode}")


def available_neut_modes(
    *,
    industry: Optional[pd.DataFrame],
    float_mktcap: Optional[pd.DataFrame],
) -> List[str]:
    modes = ["raw"]
    if float_mktcap is not None:
        modes.append("size")
    if industry is not None:
        modes.append("industry")
    if industry is not None and float_mktcap is not None:
        modes.append("size_industry")
    return modes


def neutralization_ladder(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    modes: Optional[List[str]] = None,
    zscore: bool = True,
) -> pd.DataFrame:
    """
    Compare RankIC / ICIR / monthly IC across neutralization modes.

    Returns one row per mode (factor_decay_report core table).
    """
    modes = modes or available_neut_modes(industry=industry, float_mktcap=float_mktcap)
    rows = []
    raw_ic = None
    for mode in modes:
        try:
            neut = neutralize_panel(
                panel, mode, industry=industry, float_mktcap=float_mktcap
            )
        except ValueError:
            continue
        if zscore:
            neut = cs_zscore(neut)
        st = ic_stats(neut, ret)
        mon = monthly_ic_stats(neut, ret)
        if mode == "raw":
            raw_ic = st["rank_ic"]
        retention = (
            float(st["rank_ic"] / raw_ic)
            if raw_ic is not None and pd.notna(raw_ic) and abs(raw_ic) > 1e-12 and pd.notna(st["rank_ic"])
            else np.nan
        )
        rows.append(
            {
                "mode": mode,
                "rank_ic": st["rank_ic"],
                "icir": st["icir"],
                "ic_pos_ratio": st["ic_pos_ratio"],
                "n_days": st["n_days"],
                "monthly_rank_ic": mon["monthly_rank_ic"],
                "monthly_icir": mon["monthly_icir"],
                "ic_retention_vs_raw": retention,
            }
        )
    return pd.DataFrame(rows)


def write_factor_decay_report(path, factor_name: str, ladder: pd.DataFrame) -> None:
    lines = [
        f"# Factor Decay Report — {factor_name}",
        "",
        "Neutralization ladder: does RankIC survive size / industry controls?",
        "",
        "| Mode | RankIC | ICIR | Monthly IC | Retention vs raw |",
        "|------|--------|------|------------|------------------|",
    ]
    for _, r in ladder.iterrows():
        ret_s = f"{r['ic_retention_vs_raw']:.2f}" if pd.notna(r["ic_retention_vs_raw"]) else "—"
        lines.append(
            f"| `{r['mode']}` | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{r['monthly_rank_ic']:.4f} | {ret_s} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- Large drop after **size** → small-cap contamination.",
        "- Drop after **industry** → sector tilt.",
        "- Stable after **size_industry** → cleaner microstructure / behavior alpha.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_neut_callable(
    mode: str,
    industry: Optional[pd.DataFrame],
    float_mktcap: Optional[pd.DataFrame],
) -> Optional[Callable[[pd.DataFrame], pd.DataFrame]]:
    if mode == "raw":
        return None

    def _fn(panel: pd.DataFrame) -> pd.DataFrame:
        return neutralize_panel(
            panel, mode, industry=industry, float_mktcap=float_mktcap
        )

    return _fn
