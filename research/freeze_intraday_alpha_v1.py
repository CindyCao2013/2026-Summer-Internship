#!/usr/bin/env python3
"""Create or verify the immutable Intraday Alpha Library v1 freeze spec."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "research/results/intraday_alpha_discovery_v1"
    / "intraday_alpha_library_v2_candidates.csv"
)
DEFAULT_OUTPUT = ROOT / "research/config/intraday_alpha_freeze_v1.json"
EXPECTED_FACTORS = [
    "close_vwap_deviation",
    "volume_front_loading",
    "volume_back_loading",
    "late_session_strength",
    "active_buy_sell_imbalance",
    "bartime_ofi",
    "ofi_persistence",
    "active_buy_shock",
    "average_active_trade_size",
    "intraday_amihud",
    "realized_volatility",
    "minute_skew",
]
RESIDUAL_CONTROLS_BY_SLOT = {
    "09:59": [
        "close_vwap_deviation",
        "volume_back_loading",
        "late_session_strength",
        "active_buy_sell_imbalance",
    ],
    "10:29": [
        "close_vwap_deviation",
        "volume_front_loading",
        "active_buy_sell_imbalance",
    ],
    "11:29": [
        "close_vwap_deviation",
        "active_buy_sell_imbalance",
    ],
    "13:29": [
        "close_vwap_deviation",
        "active_buy_sell_imbalance",
    ],
    "14:29": [
        "close_vwap_deviation",
        "active_buy_sell_imbalance",
    ],
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(document: dict) -> str:
    payload = dict(document)
    payload.pop("spec_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _direction(value: float, factor_name: str) -> int:
    if value == 0:
        raise ValueError(f"{factor_name}: zero train IC cannot define direction")
    return 1 if value > 0 else -1


def build_spec(source: Path) -> dict:
    candidates = pd.read_csv(source)
    actual = candidates["factor"].tolist()
    if actual != EXPECTED_FACTORS:
        raise ValueError(
            "Candidate set/order changed; freeze aborted. "
            f"expected={EXPECTED_FACTORS}, actual={actual}"
        )
    factors = {}
    for row in candidates.to_dict(orient="records"):
        factor_name = str(row["factor"])
        raw_ic = float(row["IC"])
        direction = _direction(raw_ic, factor_name)
        backend = str(row["backend"])
        bartime = str(row["best_bartime"])
        residual_controls = (
            RESIDUAL_CONTROLS_BY_SLOT[bartime]
            if backend == "candidate_ddb"
            else []
        )
        residual_ic = row.get("residual_IC")
        residual_icir = row.get("residual_ICIR")
        factors[factor_name] = {
            "family": str(row["family"]),
            "backend": backend,
            "research_role": str(row["role"]),
            "bartime": bartime,
            "horizon": str(row["best_horizon"]),
            "direction": direction,
            "portfolio_rule": (
                "long_high_short_low"
                if direction == 1
                else "long_low_short_high"
            ),
            "train_raw_ic": raw_ic,
            "train_raw_icir": float(row["ICIR"]),
            "train_fixed_direction_sharpe": float(row["Sharpe"]),
            "train_cost_sharpe_7p5bps": float(row["cost_sharpe"]),
            "residual_controls": residual_controls,
            "train_residual_raw_ic": (
                None if pd.isna(residual_ic) else float(residual_ic)
            ),
            "train_residual_raw_icir": (
                None if pd.isna(residual_icir) else float(residual_icir)
            ),
            "residual_direction": (
                None
                if pd.isna(residual_ic)
                else _direction(float(residual_ic), factor_name + "_residual")
            ),
        }
    document = {
        "freeze_id": "intraday_alpha_freeze_v1",
        "schema_version": 1,
        "locked": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_period": {
            "start": "2024-01-01",
            "end": "2024-06-30",
        },
        "universe": "000852.SH",
        "selection_policy": (
            "Maximum absolute annualized ICIR across eligible 2024H1 "
            "execution bartimes and six prespecified horizons"
        ),
        "locked_dimensions": [
            "factor",
            "bartime",
            "horizon",
            "direction",
            "residual_controls",
            "residual_direction",
        ],
        "cost_assumption_bps_per_one_way_turnover": 7.5,
        "source_candidate_path": str(source.relative_to(ROOT)),
        "source_candidate_sha256": _file_sha256(source),
        "excluded_factors": {
            "large_active_buy_ratio": {
                "status": "proxy_only",
                "reason": (
                    "No tick trade-size or OrderID source; bar-level ticket-size "
                    "proxy is not eligible for OOS promotion."
                ),
            }
        },
        "oos_periods": {
            "validation_2024H2": {
                "start": "2024-07-01",
                "end": "2024-12-31",
            },
            "test_2025_available": {
                "start": "2025-01-01",
                "end": "2025-08-18",
                "note": (
                    "Current Stock_one_minute inventory ends 2025-08-18; "
                    "this is not a complete calendar-year test."
                ),
            },
        },
        "residual_method": (
            "Daily cross-sectional rank-z OLS against the explicitly frozen "
            "same-bartime production controls"
        ),
        "factors": factors,
    }
    document["spec_sha256"] = _canonical_sha256(document)
    return document


def verify_spec(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = document.get("spec_sha256")
    actual = _canonical_sha256(document)
    if expected != actual:
        raise ValueError(
            f"Freeze spec hash mismatch: expected={expected}, actual={actual}"
        )
    if not document.get("locked"):
        raise ValueError("Freeze spec is not locked")
    factors = list(document.get("factors", {}))
    if factors != EXPECTED_FACTORS:
        raise ValueError(f"Frozen factor set changed: {factors}")
    if "large_active_buy_ratio" in factors:
        raise ValueError("Excluded proxy entered frozen factor set")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        document = verify_spec(args.output)
        print(
            f"Verified {document['freeze_id']} "
            f"sha256={document['spec_sha256']}",
            flush=True,
        )
        return 0
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists; freeze specs are immutable. "
            "Use --verify, or create a new version."
        )
    document = build_spec(args.source)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Created {args.output} sha256={document['spec_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
