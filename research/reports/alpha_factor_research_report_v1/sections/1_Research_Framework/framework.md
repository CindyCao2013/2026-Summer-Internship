# 1. Research Framework

## 1.1 Market universe

| Item | Value |
|------|-------|
| Market | China A-share |
| Primary research universes | ALL (broad), CSI1000 (session / APM), CSI300/500 as secondary where dual-benchmark exists |
| Frequency | Daily EOD + minute L2 (where required) |
| Typical research windows | 2018–2025 requested; actual coverage factor-specific (see Pack `data` block) |
| Benchmarks | CSI300 / CSI500 / CSI1000 (index residual uses 000852.SH for APM) |

**Coverage caveat:** Several L2 lineages (TGD Gu/Gd, Flow) harvest from ~2022; Pack summaries document `coverage_exception` explicitly. Do not treat all factors as same-panel.

## 1.2 Data architecture

```text
DolphinDB
  ├── Daily OHLCV / mktcap / industry
  ├── Minute L2 / Trade→MinuteBar
  └── Fundamental (SUE path; not in Report v1 body)
        ↓
Feature / Panel builders (core/l2_features, factor_cutting, …)
        ↓
Signal + shift(1) evaluation
        ↓
IC / Quantile / Stability / Execution grids
        ↓
Factor Pack v1 → factor_library.csv
```

## 1.3 Evaluation stages (shared)

| Stage | Gate question | Typical artifacts |
|-------|---------------|-------------------|
| Identity | What economic object is this? | `factor_definition.md`, identity docs |
| Formula freeze | Exact math + not-equal-to list | `formula.md`, `summary.yaml` |
| Sanity | Coverage, PIT, direction, no look-ahead | sanity runners / notes |
| Predictive | RankIC, ICIR, mono | `ic_analysis/`, `quantile_analysis/` |
| Stability | Yearly / block IC | `stability/` |
| Execution | Turnover, Net Sharpe @ cost, buffer | `execution/` |
| Pack | Library-ready asset | Pack folder + CSV row |

## 1.4 What Report v1 does **not** claim

- Production portfolio optimization / risk model OS
- Full TGD paper multi-factor additive reproduction (deferred; see TODO)
- Registry promotion of testing factors
- Invented metrics for missing Net Sharpe / SI cells

See also: [3_Factor_Methodology](../3_Factor_Methodology/evaluation.md).
