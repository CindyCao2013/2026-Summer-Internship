# Selected Factor Research

This is the clean entry layer for factors judged strongest enough for focused
delivery. It does **not** duplicate implementation code or large datasets.
Each factor bundle contains manifests that point to the canonical source,
runners, data lineage, raw artifacts, plots, and report.

## Current selected set

| Factor | Engine identity | Selection reason |
|---|---|---|
| TGD20 | `TGD20` | validated; exact SI Group10 excess Sharpe **2.16** |
| FlowDensity20 | `net_active_flow_mktcap_20d` | candidate; exact SI Group10 excess Sharpe **0.50** |

The exact market-relative rerun confirms TGD20 as the clear strongest selected
factor. FlowDensity20 remains the leading flow candidate, but it is not equally
strong as a standalone long-book alpha: raw excess Sharpe is approximately
−0.05 and the positive 0.50 result depends on neutralization.

## Bundle contract

```text
selected_factors/<factor_id>/
├── factor.yaml       # identity, formula, status, direction
├── workflow.yaml     # source → runner → artifacts → report
├── metrics.csv       # headline metrics, including exact universe-excess Sharpe
└── README.md         # human entry point
```

Canonical implementation remains in its original module. Canonical experiment
outputs remain under `research/reports/`. This directory is the stable index,
so the repository can be reorganized later without copying or silently
diverging factor logic.

## Headline portfolio metric

For selected factors, the primary portfolio metric is:

```text
daily selected-book return
− daily equal-weight return of every valid stock in the test universe
→ annualized excess Sharpe
```

The old H–L Net Sharpe remains an execution/cost diagnostic, not the headline
measure of market-relative alpha.
