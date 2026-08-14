# Factor Research Report Template v2

**Status:** schema freeze (Milestone 1B-Report Schema)  
**Date:** 2026-07-20  
**Contract detail:** [`factor_report_template_v2_requirements.md`](factor_report_template_v2_requirements.md)  
**Machine schemas:**

| File | Role |
|------|------|
| [`schemas/metric_registry.yaml`](schemas/metric_registry.yaml) | Metric Union catalog |
| [`schemas/chart_registry.yaml`](schemas/chart_registry.yaml) | Standard chart slots |
| [`schemas/factor_report.schema.yaml`](schemas/factor_report.schema.yaml) | Pack + report document shape |
| [`schemas/factor_card.example.yaml`](schemas/factor_card.example.yaml) | Card v2 fields |

**Do not (this milestone):** migrate factors · change Protocol/Harness/formulas · write Registry.

**Milestone 1C:** ✅ TGD20 Golden Pack (`factor_report_generator_v2.py`). Next: human review → 1D batch.

---

## 1. Role in Alpha Research OS

```
Validation / experiments
        ↓
Research Asset Layer   ← Template v2 (this document)
        ↓
Registry
        ↓
Orthogonality / Composite
```

A Research Pack is the **assetization interface**: evidence a PM and a researcher can both read.

---

## 2. Metric Union (non-negotiable)

```
ReportMetrics = UNION(
  template PDF fields,
  factor_summary.csv,
  metrics.json,
  portfolio / group-test outputs,
  execution_summary.csv,
  mechanism*.csv,
  stability*.csv,
  diagnostics,
  chart-implied metrics,
  Protocol v1 required metrics
)
```

- Present every known metric (headline / ladder / appendix).  
- Missing → `N/A` + `missing_reason`.  
- **No silent drop.**  
- Canonical IDs live in `metric_registry.yaml`.

### Required groups (summary)

| Group | Examples |
|-------|----------|
| Signal | IC, RankIC, ICIR, RankICIR, IC t-stat, IC mean/std, positive IC ratio, annu IC |
| Portfolio | annual/cum return, Sharpe/Sortino/Calmar, MDD, vol, H–L, turnover, gross/net Sharpe, implied fee |
| Structure | monotonicity, decile spread, direction, stability |
| Risk ladder | raw · size · industry · size+industry (show all modes, not best-only) |

---

## 3. Layer discipline (anti-confusion rule)

A Research Pack documents **one** investable factor identity. Mechanism rows and
execution rows are **not** additional factors.

```
Factor Identity          →  exactly one factor_id (Registry row)
Signal Construction      →  Gu / Gd / ε / … explain the formula
Mechanism Validation     →  diagnostic variants that accept/reject hypotheses
Portfolio Implementation →  how to trade the same signal (buffer / frequency)
Production Evaluation    →  Metric Union + Dual Benchmark on that factor_id
```

| Layer | Examples (TGD) | Registry? |
|-------|----------------|-----------|
| Factor Identity | `TGD20` | **yes — only this** |
| Signal / diagnostic | `Gu_MA20`, `τ`, `εd`, `tgd_eps` | no |
| Implementation | `daily`, `buffer_5_15` | no (pack `execution/`) |

Never treat higher ICIR on `tgd_eps` as “another factor that beats TGD20.”
Unsmoothed residuals often win ICIR and lose on turnover / net Sharpe.

---

## 4. Frozen chapter architecture

`factor_report.md` **must** use this order:

1. Executive Summary  
2. Factor Thesis  
3. Economic Intuition  
4. Formula Construction  
5. Signal Pipeline  
6. Mechanism Validation  
7. IC Analysis  
8. Portfolio Analysis  
9. Risk Adjustment  
10. Stability  
11. Execution *(Portfolio Implementation)*  
12. Limitations  
13. Final Verdict  

Appendices:

- A. Complete Metric Dump (union)  
- B. Data Dictionary & Code Map  

Also emit: `summary.md` (1-pager for PM).

### Reading guides

| Reader | Must answer |
|--------|-------------|
| Researcher | Why exists? Why not noise? Why not size/liquidity? Cost tradable? |
| PM | Sharpe? MDD? Turnover? Enter composite? |

### Prose rule

Numbers support research language. Ban empty lines like “Factor has high ICIR.”

---

## 5. Formula requirements

Every report supports four layers (LaTeX in §4):

```
raw variables
      ↓
intermediate variables
      ↓
transformation / residual / neutralization
      ↓
final investable signal (+ signal_shift)
```

Cutting factors: Object / Knife / Output explicit.

---

## 6. Mechanism Validation

Required verdict table: Hypothesis | Test | Result | Conclusion.

Full mechanism CSV (if present) must be introduced as **diagnostic variants /
signal representations**, not as a list of competing factor_ids.

TGD-shaped chain example:

```
tau → failed → upsilon → failed → epsilon_d → passed → TGD20 → accepted
```

Empty mechanism ⇒ status ≤ `candidate`.

---

## 7. Chart registry

Canonical slots in `chart_registry.yaml`:

1. factor construction diagram  
2. IC curve  
3. decile return  
4. cumulative long–short  
5. neutralization comparison  
6. yearly stability  
7. turnover / cost  

Missing slot → list under Missing Artifacts (never fake).

---

## 8. Factor Card v2

Required extra fields (see example YAML):

- `hypothesis`  
- `mechanism`  
- `formula`  
- `data_requirement`  
- `known_failure_modes`  
- `correlation_cluster` (nullable until Matrix)  

Plus Protocol fields: family, status, data_coverage, admission, benchmarks, frozen_formula.

**Card / Registry rule:** one `factor_id` only. Mechanism diagnostics and execution
labels never become Registry rows.

---

## 9. Pack directory

```
research/reports/factors/{factor_id}/
  factor_card.yaml
  metrics.json              # schema_version: factor_report_v2
  summary.md
  factor_report.md
  charts/
  mechanism/                # diagnostics — not sibling factors
  execution/                # portfolio implementation variants
  diagnostics/
  artifacts/
```

---

## 10. Implementation milestones (locked)

| ID | Name | Scope |
|----|------|--------|
| **1B** | Report Schema | ✅ this doc + YAML registries |
| **1C** | TGD20 Golden Pack | renderer + TGD pack only; no formula change |
| **1D** | Representative packs | ✅ Flow / D1 / IdealReversal |
| 1E | Controlled Batch → Registry | only after human review of 1D |

**Anti-pattern:** special-case report logic per factor (`if TGD… elif Flow…`).  
Generator must be **schema-driven** + factor_spec / artifact ingest.

**1D.6 lock:** artifact copies live in `factor_specs/{id}.yaml → artifacts:`;  
Appendix B paths live in `*_report_content.yaml → code_map:`. No `factor_id` branches in Python.

---

## 10. Acceptance (1B Schema only)

- [x] `docs/factor_report_template_v2.md`  
- [x] `docs/schemas/metric_registry.yaml`  
- [x] `docs/schemas/chart_registry.yaml`  
- [x] `docs/schemas/factor_report.schema.yaml`  
- [x] Factor Card v2 fields in example YAML  
- [ ] No factor calculation changes  
- [ ] No evaluation logic changes  
- [ ] No registry changes  
- [ ] No pack migration  

---

## 11. One-liner

> Template v2 freezes the **research asset interface** (metrics + chapters + charts + card) before any batch migration — so Alpha Factory does not grow a new report-island per factor.
