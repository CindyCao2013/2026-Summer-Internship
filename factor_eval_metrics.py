"""Factor evaluation metrics schema — machine-readable research + production scores.

Research presentation looks at alpha strength (IC / ICIR / Gross Sharpe / MDD).
Production admission looks at investability (turnover / cost / Net Sharpe).
Both live in one schema; neither replaces the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Canonical metric definitions (boss / research dashboard)
FACTOR_METRICS_SCHEMA: Dict[str, Dict[str, str]] = {
    "daily_ic": {
        "definition": "mean Spearman corr(factor_t, r_{t+1}) across days (RankIC)",
        "purpose": "Predictive power (primary IC in this codebase)",
    },
    "rank_ic": {
        "definition": "Alias of daily_ic (Spearman cross-sectional)",
        "purpose": "Robust nonlinear prediction",
    },
    "icir": {
        "definition": "mean(IC) / std(IC) * sqrt(250)",
        "purpose": "IC stability / information ratio of IC series",
    },
    "annu_ic": {
        "definition": "mean(IC) * sqrt(250)",
        "purpose": "Annualized IC magnitude (signal strength)",
    },
    "hl_annu_ret": {
        "definition": "Annualized H-L (top−bottom decile) mean return",
        "purpose": "Portfolio separation / economic magnitude",
    },
    "hl_sharpe": {
        "definition": "Sharpe of direction-adjusted H-L daily PnL",
        "purpose": "Risk-adjusted tradable performance (gross)",
    },
    "hl_mdd": {
        "definition": "Max drawdown of direction-adjusted H-L",
        "purpose": "Path risk",
    },
    "daily_turnover": {
        "definition": "Mean daily |Δw| of H-L book (long+short)",
        "purpose": "Capacity / cost driver",
    },
    "implied_annu_fee": {
        "definition": "daily_TO × 7.5bps × 250",
        "purpose": "Fee drag at 7.5bp assumption",
    },
    "net_sharpe": {
        "definition": "Sharpe after round-trip cost on turnover",
        "purpose": "Production admission (investability)",
    },
    "monotonicity": {
        "definition": "Spearman(decile_id, mean_decile_return)",
        "purpose": "Factor quality / ordering",
    },
    "direction": {
        "definition": "+1 if mean H-L > 0 else −1",
        "purpose": "Sign of predicted relationship",
    },
}


def schema_table() -> pd.DataFrame:
    rows = [{"metric": k, **v} for k, v in FACTOR_METRICS_SCHEMA.items()]
    return pd.DataFrame(rows)


def pack_factor_metrics(
    *,
    factor: str,
    period: str,
    universe: str,
    mode: str,
    rank_ic: float,
    icir: float,
    hl_annu_ret: float,
    hl_sharpe: float,
    hl_mdd: float,
    daily_turnover: float,
    implied_annu_fee: Optional[float] = None,
    net_sharpe: Optional[float] = None,
    monotonicity: Optional[float] = None,
    direction: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build machine-readable factor metrics payload."""
    annu_ic = float(rank_ic) * np.sqrt(250.0) if pd.notna(rank_ic) else np.nan
    if implied_annu_fee is None and pd.notna(daily_turnover):
        implied_annu_fee = float(daily_turnover) * 7.5 / 1e4 * 250.0
    payload = {
        "factor": factor,
        "period": period,
        "universe": universe,
        "mode": mode,
        "ic": float(rank_ic) if pd.notna(rank_ic) else None,
        "rank_ic": float(rank_ic) if pd.notna(rank_ic) else None,
        "annu_ic": float(annu_ic) if pd.notna(annu_ic) else None,
        "icir": float(icir) if pd.notna(icir) else None,
        "portfolio": {
            "annual_return": float(hl_annu_ret) if pd.notna(hl_annu_ret) else None,
            "sharpe": float(hl_sharpe) if pd.notna(hl_sharpe) else None,
            "max_drawdown": float(hl_mdd) if pd.notna(hl_mdd) else None,
            "turnover": float(daily_turnover) if pd.notna(daily_turnover) else None,
            "implied_annu_fee": float(implied_annu_fee) if pd.notna(implied_annu_fee) else None,
            "net_sharpe": float(net_sharpe) if net_sharpe is not None and pd.notna(net_sharpe) else None,
        },
        "monotonicity": float(monotonicity) if monotonicity is not None and pd.notna(monotonicity) else None,
        "direction": int(direction),
        "research_score": {
            "rank_ic": float(rank_ic) if pd.notna(rank_ic) else None,
            "icir": float(icir) if pd.notna(icir) else None,
            "gross_sharpe": float(hl_sharpe) if pd.notna(hl_sharpe) else None,
            "mdd": float(hl_mdd) if pd.notna(hl_mdd) else None,
        },
        "production_score": {
            "net_sharpe": float(net_sharpe) if net_sharpe is not None and pd.notna(net_sharpe) else None,
            "turnover": float(daily_turnover) if pd.notna(daily_turnover) else None,
            "implied_annu_fee": float(implied_annu_fee) if pd.notna(implied_annu_fee) else None,
        },
    }
    if extra:
        payload["extra"] = extra
    return payload


def save_metrics(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def metrics_to_summary_row(payload: dict) -> dict:
    port = payload.get("portfolio", {})
    return {
        "factor": payload.get("factor"),
        "period": payload.get("period"),
        "universe": payload.get("universe"),
        "mode": payload.get("mode"),
        "rank_ic": payload.get("rank_ic"),
        "annu_ic": payload.get("annu_ic"),
        "icir": payload.get("icir"),
        "hl_annu_ret": port.get("annual_return"),
        "hl_sharpe": port.get("sharpe"),
        "hl_mdd": port.get("max_drawdown"),
        "daily_turnover": port.get("turnover"),
        "implied_annu_fee": port.get("implied_annu_fee"),
        "net_sharpe": port.get("net_sharpe"),
        "monotonicity": payload.get("monotonicity"),
        "direction": payload.get("direction"),
    }
