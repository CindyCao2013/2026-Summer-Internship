# Milestone C0 — Factor Pack Normalization

**Date:** 2026-07-21  
**Status:** DONE  
**Parent:** Alpha Factor Library reset (post-TGD Pack v1)

---

## Goal

Convert existing harvest packs to the **same Research Pack v1** format as TGD20, so the library does not grow into 10 report formats.

```text
TGD20 template
  → D1_LiquidityQuality60d
  → FlowDensity20
  → IdealReversal
  → IdealAmplitude
```

No new factor research. No ActiveTrade. No SUE. No Registry.

---

## Canonical layout (all five)

```text
research/reports/factors/<FACTOR>/
├── factor_definition.md
├── formula.md
├── data_source.md
├── implementation.md
├── validation.md
├── summary.yaml
├── README.md
├── ic_analysis/
├── quantile_analysis/
├── stability/
└── execution/
```

---

## Results

| Factor | Status | Headline ICIR | Pack v1 |
|--------|--------|--------------:|---------|
| TGD20 | validated | 11.29 (SI) | ✅ (prior) |
| D1_LiquidityQuality60d | candidate | 6.01 (raw) | ✅ |
| FlowDensity20 | candidate | 4.85 (SI) | ✅ |
| IdealReversal | testing | 9.46 (SI, signed) | ✅ |
| IdealAmplitude | testing | 9.97 (SI, signed) | ✅ |

Notes:

- IdealAmplitude **monotonicity ≈ 0.11** — strong ICIR, weak decile shape; kept as `testing`.  
- Legacy Template-v2 files left in place; canonical docs are the Pack v1 root files.  
- `factor_library.csv` updated.

---

## Library count

```text
Before C0:  1 standardized asset (TGD20) + scattered packs
After C0:   5 standardized Pack v1 assets
```

---

## Next — Milestone C1

**ActiveTrade / APM** (not ActiveTradeProxy):

```text
Phase0 ✅ ADAPTED GO     → docs/milestone_c1_activetrade_phase0_identity.md
Phase1 panel design NOW  → docs/milestone_c1_activetrade_phase1_panel_design.md
Phase1 code (after accept) → research/cache/apm_session/  (no IC yet)
```

Proposed id: `APM_SessionResidual` (`adapted_replication`).  
SmartMoney stays parked. SUE stays deferred.
