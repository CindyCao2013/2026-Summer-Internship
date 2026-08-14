"""Event knife — limit-up / limit-down filtering for Factor Cutting.

Uses Factor_Dev_Lib.get_EOD_Not_Limit (S_DQ_CLOSE within S_DQ_LIMIT / S_DQ_STOPPING).
1 = tradable (not limit), NaN = limit or missing.

Honesty: this is an *event filter* on the cross-section, not a replacement knife.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from factor_attribution import cs_zscore
from factor_cutting.cutting_analysis.knife_ic import apply_universe_mask, ic_stats, monthly_ic_stats
from factor_cutting.w_cut import w_cut


def load_not_limit_mask(
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    """Wide Date×stock: 1 if not limit at close, else NaN."""
    from Factor_Dev_Lib import get_EOD_Not_Limit

    mask = get_EOD_Not_Limit(start, end)
    if not isinstance(mask.index, pd.DatetimeIndex):
        mask.index = pd.to_datetime(mask.index)
    return mask.sort_index()


def apply_not_limit(
    panel: pd.DataFrame,
    not_limit: Optional[pd.DataFrame],
) -> pd.DataFrame:
    return apply_universe_mask(panel, not_limit)


def limit_coverage_stats(
    not_limit: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict:
    """Fraction of non-null reference cells that survive the not-limit mask."""
    ref = reference.notna()
    nl = not_limit.reindex(index=reference.index, columns=reference.columns)
    keep = ref & nl.notna() & (nl > 0)
    n_ref = int(ref.sum().sum())
    n_keep = int(keep.sum().sum())
    n_drop = n_ref - n_keep
    return {
        "n_ref_cells": n_ref,
        "n_keep_cells": n_keep,
        "n_drop_cells": n_drop,
        "drop_frac": float(n_drop / n_ref) if n_ref else np.nan,
        "mean_names_kept": float(keep.sum(axis=1).mean()) if len(keep) else np.nan,
        "mean_names_ref": float(ref.sum(axis=1).mean()) if len(ref) else np.nan,
    }


def eval_factor(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    label: str,
    mode: str,
) -> dict:
    z = cs_zscore(panel)
    st = ic_stats(z, ret)
    mst = monthly_ic_stats(z, ret)
    return {
        "label": label,
        "mode": mode,
        "rank_ic": st["rank_ic"],
        "icir": st["icir"],
        "ic_pos_ratio": st["ic_pos_ratio"],
        "n_days": st["n_days"],
        "monthly_rank_ic": mst["monthly_rank_ic"],
        "monthly_icir": mst["monthly_icir"],
        "n_months": mst["n_months"],
    }


def compare_limit_filter(
    object_panel: pd.DataFrame,
    knife_panel: pd.DataFrame,
    ret: pd.DataFrame,
    not_limit: pd.DataFrame,
    *,
    window: int = 20,
    label: str = "ideal_reversal",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Modes:
      raw              — no limit filter
      filter_signal    — W-cut on full data; mask factor on limit days for IC
      filter_cut       — mask object+knife on limit days before W-cut
      filter_cut_signal — filter_cut + also mask residual limit cells on factor
    """
    cov = limit_coverage_stats(not_limit, object_panel)

    raw_factor = w_cut(object_panel, knife_panel, window=window)
    obj_f = apply_not_limit(object_panel, not_limit)
    knife_f = apply_not_limit(knife_panel, not_limit)
    cut_factor = w_cut(obj_f, knife_f, window=window)

    panels = {
        "raw": raw_factor,
        "filter_signal": apply_not_limit(raw_factor, not_limit),
        "filter_cut": cut_factor,
        "filter_cut_signal": apply_not_limit(cut_factor, not_limit),
    }

    rows = []
    for mode, panel in panels.items():
        row = eval_factor(panel, ret, label=label, mode=mode)
        row.update({f"cov_{k}": v for k, v in cov.items()})
        if mode != "raw":
            base = rows[0]
            row["rank_ic_delta"] = row["rank_ic"] - base["rank_ic"]
            row["icir_delta"] = row["icir"] - base["icir"]
            row["monthly_ic_delta"] = row["monthly_rank_ic"] - base["monthly_rank_ic"]
            if pd.notna(base["rank_ic"]) and abs(base["rank_ic"]) > 1e-12:
                row["ic_retention_vs_raw"] = row["rank_ic"] / base["rank_ic"]
            else:
                row["ic_retention_vs_raw"] = np.nan
        else:
            row["rank_ic_delta"] = 0.0
            row["icir_delta"] = 0.0
            row["monthly_ic_delta"] = 0.0
            row["ic_retention_vs_raw"] = 1.0
        rows.append(row)

    return pd.DataFrame(rows), panels


def write_event_knife_report(
    path: Path,
    compare_df: pd.DataFrame,
    *,
    period: str,
    coverage: Optional[dict] = None,
    extra_notes: Optional[list] = None,
) -> None:
    lines = [
        "# Event Knife — Limit-Up / Limit-Down Filter",
        "",
        f"**Period:** `{period}`",
        "",
        "Source: `Factor_Dev_Lib.get_EOD_Not_Limit` "
        "(close strictly inside `S_DQ_LIMIT` / `S_DQ_STOPPING`).",
        "",
    ]
    if coverage:
        lines += [
            "## Coverage",
            "",
            f"- Drop fraction of object cells: **{coverage.get('drop_frac', float('nan')):.2%}**",
            f"- Mean names/day ref → kept: "
            f"{coverage.get('mean_names_ref', float('nan')):.0f} → "
            f"{coverage.get('mean_names_kept', float('nan')):.0f}",
            "",
        ]
    lines += [
        "## RankIC comparison",
        "",
        "| Mode | RankIC | ICIR | Monthly IC | ΔRankIC | Retention | n_days |",
        "|------|--------|------|------------|---------|-----------|--------|",
    ]
    for _, r in compare_df.iterrows():
        lines.append(
            f"| `{r['mode']}` | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{r['monthly_rank_ic']:.4f} | {r['rank_ic_delta']:+.4f} | "
            f"{r['ic_retention_vs_raw']:.3f} | {int(r['n_days'])} |"
        )
    lines += [
        "",
        "## Modes",
        "",
        "- `raw` — no filter",
        "- `filter_signal` — mask limit names on the finished factor before IC",
        "- `filter_cut` — exclude limit days from object/knife inside W-cut",
        "- `filter_cut_signal` — both",
        "",
        "## Interpretation",
        "",
        "- Retention > 1 and |IC| larger → limit days diluted alpha (paper-table-2 style).",
        "- Retention ≈ 1 → limit filter mostly cosmetic.",
        "- Retention << 1 → alpha concentrated on limit events (fragile / hard to trade).",
        "",
    ]
    if extra_notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in extra_notes] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
