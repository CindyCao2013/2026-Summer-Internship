# Research Delivery Layer

> Showable **research-grade** factor cards. Not textbook checklist. Not Registry/OS.

Governance: [`GOVERNANCE.md`](GOVERNANCE.md) · Roadmap: [`ROADMAP.md`](ROADMAP.md) · Board: [`factor_delivery_plan.csv`](factor_delivery_plan.csv)

Focused selected-factor bundles: [`selected_factors/`](selected_factors/)

```text
Existing research asset → identity + formula → link artifacts → report.md → index
```

## Tier quick reference

| Tier | Role | Counts? |
|------|------|---------|
| **A** | Delivery candidate / delivered | yes |
| **B** | Research candidate (missing a link) | no until promote |
| **C** | Archive / baseline | never |

## Layout

```text
research_delivery/
├── GOVERNANCE.md
├── factor_delivery_plan.csv
├── factor_index.csv
├── selected_factor_metrics.csv
├── selected_factors/          # clean workflow/data/plot/report manifests
├── templates/
├── scripts/generate_factor_card.py
└── factors/<ID>/{report.md,formula.md,metrics.csv,plots/,artifacts/}
```

## Current

Batch 1 delivered · **AmihudShockReversal5d** promoted as Batch 2 #1 (`testing_candidate`).  
Next promote: **LiquidityResidual20d**.

## Headline portfolio metric

**Long-book / Group10 excess Sharpe** (not H–L Net Sharpe).  
See [`METRICS_G10_EXCESS.md`](METRICS_G10_EXCESS.md) ·
[`selected_factor_metrics.csv`](selected_factor_metrics.csv).
