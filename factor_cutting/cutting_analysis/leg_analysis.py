"""Leg analysis: M_high / M_low / spread IC decomposition."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from factor_cutting.cutting_analysis.knife_ic import ic_stats
from factor_cutting.w_cut import w_cut


def decompose_legs(
    object_panel: pd.DataFrame,
    knife_panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    window: int = 20,
    signal_shift: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Compute W-cut legs and IC stats.

    Returns:
      summary_df with rows high / low / spread
      detail dict with ic_daily series per leg
    """
    spread, high, low = w_cut(object_panel, knife_panel, window=window, return_legs=True)
    detail = {}
    rows = []
    for name, panel in (("high", high), ("low", low), ("spread", spread)):
        st = ic_stats(panel, ret, signal_shift=signal_shift)
        detail[name] = st
        rows.append(
            {
                "leg": name,
                "rank_ic": st["rank_ic"],
                "icir": st["icir"],
                "ic_pos_ratio": st["ic_pos_ratio"],
                "n_days": st["n_days"],
            }
        )
    summary = pd.DataFrame(rows)

    # Separation score: how much the knife splits predictive power
    # K = IC(high) - IC(low)  (signed; for reversal both may be negative)
    ic_h = summary.loc[summary["leg"] == "high", "rank_ic"].iloc[0]
    ic_l = summary.loc[summary["leg"] == "low", "rank_ic"].iloc[0]
    sep = float(ic_h - ic_l) if pd.notna(ic_h) and pd.notna(ic_l) else np.nan
    # Effectiveness: |IC(spread)| relative to max(|IC high|, |IC low|)
    ic_s = summary.loc[summary["leg"] == "spread", "rank_ic"].iloc[0]
    denom = max(abs(ic_h) if pd.notna(ic_h) else 0.0, abs(ic_l) if pd.notna(ic_l) else 0.0, 1e-12)
    purity = float(abs(ic_s) / denom) if pd.notna(ic_s) else np.nan

    summary.attrs["knife_separation"] = sep
    summary.attrs["knife_purity"] = purity
    return summary, detail


def legs_ic_timeseries(
    detail: Dict[str, dict],
) -> pd.DataFrame:
    """Wide daily IC: columns high / low / spread."""
    frames = {}
    for name, st in detail.items():
        frames[name] = st["ic_daily"]
    return pd.DataFrame(frames)


def write_leg_mechanism_md(
    path,
    *,
    factor_name: str,
    knife_name: str,
    summary: pd.DataFrame,
    paper_note: str = "",
) -> None:
    sep = summary.attrs.get("knife_separation", np.nan)
    purity = summary.attrs.get("knife_purity", np.nan)
    lines = [
        f"# Mechanism — {factor_name}",
        "",
        f"Knife: `{knife_name}`",
        "",
        "| Leg | RankIC | ICIR | IC+ |",
        "|-----|--------|------|-----|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['leg']} | {r['rank_ic']:.4f} | {r['icir']:.2f} | {r['ic_pos_ratio']:.2f} |"
        )
    lines += [
        "",
        f"**Knife separation** (IC_high - IC_low): `{sep:.4f}`",
        f"**Knife purity** (|IC_spread| / max(|IC_h|,|IC_l|)): `{purity:.3f}`",
        "",
        "Interpretation:",
        "- High leg should carry the alpha; low leg should be near-noise.",
        "- Spread = high − low is information purification, not a second independent factor.",
        "",
    ]
    if paper_note:
        lines.append(paper_note)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
