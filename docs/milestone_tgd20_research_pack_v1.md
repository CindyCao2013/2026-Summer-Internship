# Milestone — TGD20 Research Pack v1 Finalization

**Date:** 2026-07-21  
**Status:** DONE  
**Goal reset:** Alpha Factor Library (not Portfolio/Composite OS)

---

## Why this milestone

Project had drifted toward:

```text
Alpha Research OS → Portfolio → Composite → Production
```

Reset to original intent:

```text
Reproduce paper factor → validate → pack → library → next alpha
```

TGD20 is the first **complete** asset. Pack it as the template; do not block on Registry/Similarity.

---

## Delivered

### Canonical pack

```text
research/reports/factors/TGD20/
├── factor_definition.md
├── formula.md
├── data_source.md
├── implementation.md
├── validation.md
├── summary.yaml
├── ic_analysis/
├── quantile_analysis/
├── stability/
├── execution/
└── README.md
```

### Lightweight library

```text
research/reports/factors/factor_library.csv
```

No Registry write. No Composite. No Portfolio.

---

## TGD20 headline (frozen)

| Metric | Value |
|--------|------:|
| ICIR (size+industry) | **11.29** |
| RankIC (SI) | 0.0415 |
| Monotonicity | 0.988 |
| Yearly IC>0 | 6/6 |
| Best Net Sharpe (15bp) | 2.32 |
| Status | **validated** |

Formula identity:

```text
TGD20 = MA20( CS residual of εd on εu )
≠ Gd − Gu
```

---

## Strategic locks

| Item | Decision |
|------|----------|
| SmartMoney10d | stay `research` — do not optimize |
| Ideal* / ActiveTradeProxy | `testing` — pack later |
| SUE / Fundamental | **deferred** (PIT cost high; not next) |
| Portfolio / Composite / Registry / Topology | **paused** |

---

## Next

**Milestone C** — pick next paper factor for fast coverage:

1. ActiveTrade true replication (preferred if data ready)  
2. Or next cutting / microstructure paper with existing minute chain  

Target trajectory: grow library toward ~20 factors, then revisit Similarity/Composite.
