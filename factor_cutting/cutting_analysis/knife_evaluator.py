"""Knife evaluator — rank candidate knives by separation / spread IC."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_cutting.cutting_analysis.leg_analysis import decompose_legs
from factor_cutting.knives import KNIFE_CANDIDATES, available_knives


def evaluate_knives(
    object_panel: pd.DataFrame,
    knives: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """
    For each knife, W-cut object and score:

      separation = IC(high) - IC(low)
      |spread IC|, ICIR, purity

    Higher |separation| + strong |spread IC| ⇒ better knife for this object.
    """
    rows = []
    for name, knife in knives.items():
        summary, _ = decompose_legs(object_panel, knife, ret, window=window)
        ic_h = float(summary.loc[summary["leg"] == "high", "rank_ic"].iloc[0])
        ic_l = float(summary.loc[summary["leg"] == "low", "rank_ic"].iloc[0])
        ic_s = float(summary.loc[summary["leg"] == "spread", "rank_ic"].iloc[0])
        icir_s = float(summary.loc[summary["leg"] == "spread", "icir"].iloc[0])
        sep = summary.attrs.get("knife_separation", ic_h - ic_l)
        purity = summary.attrs.get("knife_purity", np.nan)
        # effectiveness score: prefer large |spread IC| and large |separation|
        score = abs(ic_s) * 0.6 + abs(sep) * 0.4 if pd.notna(ic_s) and pd.notna(sep) else np.nan
        rows.append(
            {
                "knife": name,
                "ic_high": ic_h,
                "ic_low": ic_l,
                "ic_spread": ic_s,
                "icir_spread": icir_s,
                "separation": sep,
                "purity": purity,
                "effectiveness": score,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("effectiveness", ascending=False).reset_index(drop=True)
    return df


def evaluate_default_knives(
    object_panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    amount: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    trade_count: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    ret_1d: Optional[pd.DataFrame] = None,
    window: int = 20,
    knife_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    knives = available_knives(
        amount=amount,
        volume=volume,
        trade_count=trade_count,
        turnover=turnover,
        ret_1d=ret_1d if ret_1d is not None else object_panel,
    )
    if knife_names is not None:
        knives = {k: v for k, v in knives.items() if k in knife_names}
    cleaned = {}
    for k, panel in knives.items():
        aligned = panel.reindex_like(object_panel)
        if aligned.notna().sum().sum() < 1000:
            continue
        cleaned[k] = aligned
    return evaluate_knives(object_panel, cleaned, ret, window=window)


def knife_ranking_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Knife Evaluator Ranking",
        "",
        "Score = 0.6·|IC_spread| + 0.4·|IC_high − IC_low|",
        "",
        "| Rank | Knife | IC_high | IC_low | IC_spread | Sep | Effectiveness |",
        "|------|-------|---------|--------|-----------|-----|---------------|",
    ]
    for i, r in df.iterrows():
        lines.append(
            f"| {i+1} | `{r['knife']}` | {r['ic_high']:.4f} | {r['ic_low']:.4f} | "
            f"{r['ic_spread']:.4f} | {r['separation']:.4f} | {r['effectiveness']:.4f} |"
        )
    lines.append("")
    lines.append(f"Candidates considered: {', '.join(KNIFE_CANDIDATES)}")
    lines.append("")
    return "\n".join(lines)
