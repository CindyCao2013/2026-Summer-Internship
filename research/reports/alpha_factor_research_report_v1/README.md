# Alpha Factor Research Report v1

**Audience:** Quant researcher / PM / internal reviewer  
**Style:** Institutional alpha research memo  
**Rule:** Metrics are harvested from existing Pack / CSV / YAML artifacts only — **no invented numbers**.

> **Note (2026-07-21):** This folder is the **Library Overview** memo.  
> **Individual Factor Research Reports** (delivery sprint) live in [`research_delivery/factors/*/report.md`](../../research_delivery/).  
> Active roadmap: [`research_delivery/ROADMAP.md`](../../research_delivery/ROADMAP.md).

## What this is

A deliverable packaging of completed single-factor research into a showable Factor Library memo:

```text
Idea → Identity → Formula → Implementation → IC / Quantile / Stability → Execution → Pack → Library
```

This is **Stage 1 (Factor Discovery)** documentation. It is **not** a production portfolio OS, and it does **not** promote testing factors.

## Target factors in this report

| Factor | Status | Pack |
|--------|--------|------|
| TGD20 | validated | `research/reports/factors/TGD20` |
| D1_LiquidityQuality60d | candidate | `research/reports/factors/D1_LiquidityQuality60d` |
| FlowDensity20 | candidate | `research/reports/factors/FlowDensity20` |
| APM_SessionResidual | testing_candidate | `research/reports/factors/APM_SessionResidual` |
| IdealReversal | testing | `research/reports/factors/IdealReversal` |
| IdealAmplitude | testing | `research/reports/factors/IdealAmplitude` |
| SmartMoney10d | research (parked) | `research/reports/smart_money_v1` |

## Document map

| File | Content |
|------|---------|
| [Executive_Summary.md](Executive_Summary.md) | 5-minute briefing |
| [sections/1_Research_Framework/](sections/1_Research_Framework/) | Universe, data, architecture |
| [sections/2_Factor_Universe/](sections/2_Factor_Universe/) | Factor inventory |
| [sections/3_Factor_Methodology/](sections/3_Factor_Methodology/) | IC / RankIC / ICIR / decile / execution |
| [sections/4_Factor_Research/](sections/4_Factor_Research/) | Per-factor memos |
| [sections/5_Cross_Factor_Analysis/](sections/5_Cross_Factor_Analysis/) | Signal / IC corr, info map |
| [sections/6_Portfolio_Construction/](sections/6_Portfolio_Construction/) | Lightweight combination only |
| [sections/7_Risk_and_Execution/](sections/7_Risk_and_Execution/) | Turnover / cost |
| [sections/8_Final_Factor_Library/](sections/8_Final_Factor_Library/) | Status classification |
| [Appendix/](Appendix/) | Formulas, schema, code map |
| [TODO.md](TODO.md) | Missing figures / joint backtests |
| figures/ | Optional schematics only — **factor backtest PNGs are not stored here** |

## Provenance / figure rule

- Library table: `research/reports/factors/factor_library.csv`
- Pack summaries: `research/reports/factors/*/summary.yaml`
- **All factor validation images in Section 4 link directly to Pack / scout experiment paths**, e.g.  
  `../factors/TGD20/ic_analysis/ic_curve.png`  
  not to duplicated copies under this report folder.
- APM peer heatmaps: `factors/APM_SessionResidual/ic_analysis/peer_*_heatmap.png` (from scout CSVs).
- Gaps: see [`TODO.md`](TODO.md).

## Status discipline

- Do **not** change factor status in this report.
- Do **not** promote `testing` / `research` to validated.
- Distinguish **paper replication** vs **paper adapted** vs **internal**.
- ActiveTradeProxy ≠ APM_SessionResidual (proxy excluded from headline library).
