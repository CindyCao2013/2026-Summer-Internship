# FlowDensity20 — Data Source

## Input

| Item | Spec |
|------|------|
| Fields | L2 active buy/sell amounts, floating mktcap |
| Derived | Net active flow, amount/volume for orthogonality tests |
| Universe (research harvest) | ALL A-shares (0/3/6) |
| Target calendar | 2018–2025 |

## Actual coverage (this pack)

| Field | Value |
|-------|-------|
| Confirmation sample | **2022-01-28 → 2025-12-31** (951 trading days) |
| Yearly stability | 2022–2025 (4 years) |
| Exception | L2/active-flow confirmation window |

Requested vs actual is recorded in `summary.yaml` (`coverage_exception: true`).  
Do not claim 2018 start for headline IC metrics unless re-run.

## Pipeline layers

```text
L2 active buy/sell + mktcap
    → net_active_flow / mktcap
    → 20d rolling sum + cs_zscore
    → FlowDensity20 panel (date × symbol)
```

Cache / builders: `l2_data_loaders.L2DailyWideCache` + `factor_formulas_l2_flow_p2.py`
