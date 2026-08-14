#!/usr/bin/env python
"""Phase 0: freeze current research baseline (no more drift on architecture artifacts)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import factor_config as cfg

FROZEN_DIR = Path("research/frozen_state_v1")
SOURCE = cfg.RESEARCH_DIR

ARTIFACTS = [
    "alpha_information_space_v1.json",
    "alpha_production_stack_v2.json",
    "l2_trade_flow_state_verdict.json",
    "fundamental_quality_d7_verdict.json",
    "fundamental_value_d6_verdict.json",
    "alpha_enhancer_targets_v1.csv",
]

BASELINE = {
    "base_dimensions": {
        "D1": "low_vol_liquidity_quality_60d",
        "D2": "volatility_60d",
        "D3": "lower_shadow_support_20d",
        "D4": "winner_sentiment_reversal_5d",
        "D5": "upside_fragility_20d",
    },
    "enhancer_layers": {
        "L2_primary": "cn_cancel_shock",
        "D7_quality": "quality_composite",
        "D6_value": "value_composite",
    },
    "production_stack": "v2 (D1-D5 base + cancel + quality; NOT deployed)",
}

ROADMAP = {
    "paused": [
        "production_stack_v3_deploy",
        "D8_growth_block",
        "ML_MCTS",
        "L2_v3_expansion",
        "architecture_expansion",
    ],
    "active_phase": "guided_alpha_mining + alpha_research_report_v1",
    "next_steps": [
        "Tier-A factor reports (IC, decile, L/S curve)",
        "Alpha density expansion within D1-D5 families",
        "Harness filter on 30-50 new candidates",
        "Stack v3 enhancer calibration (after alpha library denser)",
    ],
    "target": "15-30 robust signals across 5-8 economic dimensions",
}


def main() -> None:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for name in ARTIFACTS:
        src = SOURCE / name
        dst = FROZEN_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(name)
        else:
            missing.append(name)

    manifest = {
        "frozen_at": datetime.now().isoformat(),
        "version": "frozen_state_v1",
        "baseline": BASELINE,
        "roadmap": ROADMAP,
        "artifacts_copied": copied,
        "artifacts_missing": missing,
        "note": "Architecture research frozen. Resume alpha discovery via guided mining + reports.",
    }
    (FROZEN_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Frozen state -> {FROZEN_DIR}/manifest.json")
    print(f"Copied {len(copied)} artifacts; missing: {missing}")


if __name__ == "__main__":
    main()
