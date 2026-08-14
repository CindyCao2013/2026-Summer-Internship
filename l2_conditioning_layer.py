"""L2 Trade Flow State Engine — frozen conditioning layer (NOT alpha mining).

Research objective:
  Extract market *state* from executed flow + cancel withdrawal; enhance D1–D5,
  not create standalone information dimensions.

  L2 Microstructure Alpha Mining → CLOSED
  L2 Trade Flow State Engine    → FROZEN (3 signals)

Status: CLOSED for library expansion. See research/results/l2_v2_closed.md,
        L2_DATA_LINEAGE.md, l2_trade_flow_state_verdict.json
"""

from __future__ import annotations

from typing import Dict, List

L2_LIBRARY_STATUS = "CLOSED_CONDITIONING_LAYER"

# Retained after L2 v2 validation — enhancer / strict-residual roles only
L2_STATE_LAYER: List[Dict[str, str]] = [
    {
        "factor": "cn_voi_shock",
        "role": "flow_shock",
        "tier": "strict_residual",
        "use": "confidence_overlay",
    },
    {
        "factor": "cn_mpb_shock",
        "role": "trade_shock",
        "tier": "strict_residual",
        "use": "confidence_overlay",
    },
    {
        "factor": "cn_cancel_shock",
        "role": "cancel_state",
        "tier": "enhancer_only",
        "use": "confidence_overlay_primary",
    },
]

L2_DROPPED_V2 = [
    "cn_flow_persistence",
    "cn_imbalance_duration",
    "cn_liquidity_consumption",  # research_only — flow intensity, not depth
]

DEFAULT_OVERLAY_LAMBDA = 0.25
