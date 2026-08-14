# Milestone 1D — Representative Research Pack Validation

**Date:** 2026-07-20  
**Status:** Complete (validation pass with one documented chart gap)  
**Scope:** Prove Report Generator v2 works across factor families — **not** full batch migration, **not** Registry.

---

## Goal

Validate:

```
factor_spec.yaml + *_report_content.yaml
        ↓
schema-driven renderer (factor_report_generator_v2.py)
        ↓
research/reports/factors/{factor_id}/
```

across four representative families before any Registry work.

---

## Constraints (honored)

| Do not | Result |
|--------|--------|
| Modify Protocol v1 | untouched |
| Retune / recompute TGD20 | harvest + narrative polish only |
| Create Registry | deferred |
| Change factor formulas | harvest-only |

---

## Pack set

| Factor | Family role | Pack path | Generator validation |
|--------|-------------|-----------|----------------------|
| **TGD20** | Temporal (Golden, 1C) | `research/reports/factors/TGD20/` | ✅ `ok=true`, 7/7 charts, 32 metrics |
| **FlowDensity20** | Microstructure × liquidity interaction | `research/reports/factors/FlowDensity20/` | ⚠ structure OK; **3 charts missing** (see below) |
| **D1_LiquidityQuality60d** | Classical EOD liquidity | `research/reports/factors/D1_LiquidityQuality60d/` | ✅ `ok=true`, 7/7 charts, 32 metrics |
| **IdealReversal** | Paper / cutting replication | `research/reports/factors/IdealReversal/` | ✅ `ok=true`, 7/7 charts, 32 metrics |

**Run:**

```bash
/opt/conda/anaconda3/envs/base_93/bin/python assemble_representative_packs_1d.py
/opt/conda/anaconda3/envs/base_93/bin/python factor_report_generator_v2.py --factor FlowDensity20
/opt/conda/anaconda3/envs/base_93/bin/python factor_report_generator_v2.py --factor D1_LiquidityQuality60d
/opt/conda/anaconda3/envs/base_93/bin/python factor_report_generator_v2.py --factor IdealReversal
```

Specs / narrative (no Python factor branches):

- `factor_specs/{FlowDensity20,D1_LiquidityQuality60d,IdealReversal}.yaml`
- `factor_specs/{id}_report_content.yaml` (+ construction_steps for diagram)

---

## Validation matrix

### 1. Metric Union completeness

All four packs write `metrics.json` with `schema_version: factor_report_v2` and **32** Metric Union entries. Missing values are explicit `N/A` / `missing_reason` (never silently dropped).

| Factor | Headline harvest note |
|--------|------------------------|
| TGD20 | Full neut ladder + execution; Pearson IC etc. N/A |
| FlowDensity20 | Full neut ladder + execution; monotonicity N/A in summary row |
| D1 | Universe ladder as diagnostic modes; size/industry neut N/A |
| IdealReversal | Cutting legs in mechanism; Sharpe≈1.70 / mono≈0.44 (below soft bar — intentional stress pack) |

### 2. Chart registry compatibility

| Chart | TGD | Flow | D1 | Ideal |
|-------|-----|------|----|-------|
| construction_diagram | ✅ content-driven | ✅ | ✅ | ✅ |
| ic_curve | ✅ | ❌ no legacy PNG | ✅ | ✅ |
| decile_return | ✅ | ❌ | ✅ | ✅ |
| cumulative_long_short | ✅ | ❌ | ✅ | ✅ |
| neutralization_compare | ✅ | ✅ from summary | ✅ universe modes | ✅ |
| stability_yearly | ✅ | ✅ (`year`→`period` normalize) | ✅ block | ✅ block |
| turnover | ✅ | ✅ execution grid | ✅ summary fallback | ✅ summary fallback |

**Flow chart gap (expected, not invented):**  
Confirmation research under `l2_flow_density_v1/` never exported daily IC / decile / H–L PNGs. Chart registry rule: *never invent a chart that does not exist*. Gaps are listed under **Missing Artifacts** in `factor_report.md`. Fill only via a future Protocol re-run that writes those figures — not by fabricating curves.

### 3. Formula section availability

All four reports include Template v2 §4 (raw → intermediate → transform → final) from `*_report_content.yaml`. Construction diagrams use per-factor `construction_steps` (no hardcoded TGD boxes in the generator).

### 4. Mechanism section availability

| Factor | Mechanism content |
|--------|-------------------|
| TGD20 | Residual timing ladder (τ/υ rejected; εd / TGD accepted) |
| FlowDensity20 | Amount / GrossActive / Flow⊥Amount + `amount_orth_*` under `mechanism/` & `diagnostics/` |
| D1 | Canonical signal row (no L2 residual ladder — documents that absence) |
| IdealReversal | Object / Knife / Output legs (M_high, M_low, spread, Ret20 baseline) |

### 5. Diagnostics separation

| Dir | Role |
|-----|------|
| `mechanism/` | Hypothesis tests / component tables |
| `execution/` | Trade-efficiency grid (when present) |
| `diagnostics/` | Stability, universe/IC-decay ladders, orthogonality (TGD/Flow only) |
| `artifacts/` | Pack validation JSON + copied summaries |

Stale TGD×Flow orthogonality copies were removed from D1 / Ideal packs (diagnostics must not cross-contaminate families).

---

## What 1D proved

1. **Interaction factor (Flow)** — Template v2 can host amount-orth mechanism + execution without treating Flow as pure flow.  
2. **Simple EOD (D1)** — Works with thin mechanism and universe-ladder diagnostics instead of a full neut ladder.  
3. **Paper / cutting (Ideal Reversal)** — Works for Object–Knife–Output docs even when soft-bar fails (testing pack).  
4. **Schema-driven path** — Same generator for all four; narrative lives in YAML, not `if factor_id == ...`.

---

## Human review checklist (before Batch / Registry)

1. **Executive Summary** — research language first; numbers as support (not “ICIR = X” as the whole story).  
2. **Risk Adjustment** — full ladder / universe modes; never best-cell-only.  
3. **Execution** — framed as portfolio construction / trade efficiency, not new alpha.  
4. **Flow Missing Artifacts** — accept gap or schedule Production Track figure export.  
5. **IdealReversal** — keep status `testing`; do not auto-admit on Sharpe&lt;2 / weak mono.

---

## Explicit non-goals (still deferred)

- Registry CSV / admission gate  
- Full batch of 8–15 packs  
- Dual Benchmark Production re-run for all factors  
- Composite / Factor Matrix as main line  

---

## Verdict

**Milestone 1D: pass for representative-family validation.**  
Generator v2 is not TGD-only. Proceed to human review of the three new packs; then a controlled Batch (1E) before Registry.

**Next (suggested name):** Milestone 1E — Controlled Batch Packs → then Registry.
