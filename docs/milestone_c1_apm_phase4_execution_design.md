# C1.4 — APM_SessionResidual Execution / Horizon Diagnosis

**Date:** 2026-07-21  
**Status:** COMPLETE — **Case A** · `testing_candidate` (execution recipe found)  
**Parent:** C1.3 Scout = `PASS_research_FAIL_invest`  
**Identity:** `APM_SessionResidual` · `adapted_replication`  
**Artifacts:** `research/reports/apm_session_v1/execution/`  
**Script:** `run_milestone_c1_apm_session_execution.py`

---

## Question

> Is APM unprofitable, or is the **rebalance horizon** wrong?

Scout already showed: alpha exists (ICIR≈4–6.5), daily TO≈0.75 → Net@15bp≈0.92.

---

## Setup

| Item | Value |
|------|-------|
| Panel | frozen CSI1000 `apm_cs` (2021–2025 scout cache) |
| Book | **long high / short low** (positive IC; no flip) |
| Cost | 15bp RT |
| Horizons | H ∈ {1, 3, 5, 10, 20} |
| Rebalance | daily, every_3d, every_5d, every_10d |
| Buffer | none, 5–15, 10–30 |

---

## Outputs

```text
research/reports/apm_session_v1/execution/
├── horizon_ic.csv
├── holding_horizon.csv
├── execution_grid.csv
├── turnover_curve.png
└── execution_verdict.json
```

---

## Verdict cases

| Case | Condition | Action |
|------|-----------|--------|
| A | Best Net Sharpe > 1 (or soft >0.5 + clear recipe) | optional Pack v1 `testing` with recipe |
| B | IC persistent; slow book still weak | `research_candidate` park (like SmartMoney) |
| C | IC unstable / dies | close paper track |

**Default status until Case A clear:** `research_candidate` — **no** `factor_library` auto-admit · **no** Registry.

---

## Result (2026-07-21)

| Gate | Outcome |
|------|---------|
| Horizon IC | Strengthens H1→H20 (2.4% → 4.0%); class `medium_or_longer_persistent` |
| Best execution | **`daily \| buffer_10_30`** Net Sharpe **1.50** · TO **0.28** |
| Case | **A** (Net > 1) |
| Recommended status | `testing_candidate` |
| Pack | Eligible for **C1 Pack v1** with documented recipe — not auto-written |

Also strong: `every_10d|buffer_5_15` Net≈1.37; plain `every_10d` Net≈1.09.

---

## Forbidden

- Formula change / sign flip  
- Pack / library / Registry without Case A  
- Composite  
- Touch ActiveTradeProxy  
