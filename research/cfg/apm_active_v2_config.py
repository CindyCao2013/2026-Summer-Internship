"""APM ActiveV2 Smart profile hyperparameters (Factor Factory).

Profiles are consumed by ``build_smartv2_panel(profile=...)``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LOCKED V2.1 delivery (2026-07-24)
# factor_id: APM_ActiveV2_SmartV2_1F
# q80 size filter + buy/sell intensity + ASC hard-gate OFF + daily EWM span=5
# ---------------------------------------------------------------------------
SMART_V2_1F_CONFIG = {
    "profile": "v2_1f",
    "status": "LOCKED_DELIVERY",
    "factor_id": "APM_ActiveV2_SmartV2_1F",
    "quantile": 0.80,
    "ewm_span": 5,
    "asc_min_rank": 0.0,  # ASC hard gate disabled
    "brick": "active_pressure_smartv2",
}

# Research / ablated (not delivery)
SMART_V2_CONFIG = {
    "profile": "v2",
    "status": "research",
    "quantile": 0.80,
    "ewm_span": 2,
    "asc_min_rank": 0.50,
    "brick": "active_pressure_smartv2",
}

# Abandoned: q90 was not run to completion; expected to over-shrink CSI1000 coverage
SMART_V2_1_CONFIG = {
    "profile": "v2_1",
    "status": "ABANDONED_DO_NOT_RUN",
    "quantile": 0.90,
    "ewm_span": 5,
    "asc_min_rank": 0.0,
    "brick": "active_pressure_smartv2_1",
}

CANONICAL_SMART_APM = SMART_V2_1F_CONFIG
