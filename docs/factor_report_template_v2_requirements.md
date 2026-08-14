# Factor Research Report Contract v2

**Document type:** Research Report Contract (detail behind Template v2)  
**Status:** absorbed into schema freeze — see human entrypoint below  
**Date:** 2026-07-20  

**Human entrypoint (1B done):** [`factor_report_template_v2.md`](factor_report_template_v2.md)  
**Schemas:** `docs/schemas/{metric_registry,chart_registry,factor_report.schema}.yaml`

**Depends on:** Protocol v1, Harness 1A  
**Must not modify:** Protocol v1 · TGD20 formula · FlowDensity formula · Harness compute/eval logic  

**Gold-standard narrative:**  
`research/reports/tgd_v1/日内分钟收益率时序特征_TGD20因子研究报告.md`

**Phase II order (locked):**

```
Protocol v1 → Harness 1A
    → 1B Report Schema          ✅ NOW complete
    → 1C TGD20 Golden Pack      ← next (renderer only)
    → 1D Batch Migration (8–15 packs)
    → Registry → Orthogonality → Composite
```

---

## 0. Product intent

Report Generator v2 does **not** produce a metrics dump or a thin markdown summary.

It produces an **institutional factor research report** that answers:

| Question | Section |
|----------|---------|
| What? | Executive Summary + Thesis |
| Why economically? | Economic Intuition |
| How built? | Formula + Signal Pipeline |
| Why does it work? | Mechanism Analysis |
| Does it predict? | IC Analysis |
| Can we trade the spread? | Portfolio Analysis |
| Is it just risk? | Risk Adjustment |
| Is it stable? | Stability |
| Is it investable? | Execution |
| What can go wrong? | Limitations |
| Ship / hold / kill? | Final Verdict |

Registry registers **evidence packs**, not Sharpe numbers.

---

## 1. Core principle — Metric Union Schema

### 1.1 Definition

```
ReportMetrics = UNION(
  PDF template fields (floor),
  factor_summary.csv columns,
  metrics.json keys (recursive),
  portfolio / group-test simulator outputs,
  execution_layer / execution_summary.csv columns,
  mechanism*.csv columns,
  stability / yearly_*.csv columns,
  diagnostic artifact metrics,
  metrics implied by chart titles/axes,
  Protocol v1 required metrics
)
```

### 1.2 Hard rules

1. **No silent drop.** Any metric that appears in any source must appear in the report (headline table, ladder, or appendix dump).  
2. **Missing → `N/A`** with `missing_reason` (`not_computed` | `not_in_artifacts` | `series_unavailable`).  
3. Template PDF fields are a **floor**, not a ceiling.  
4. Prefer **explicit labels**: if today’s `icir` is computed on RankIC, print it as **RankICIR**, and show Pearson **ICIR** separately (or N/A).  
5. Implementation harvests existing artifacts; it does **not** invent a second backtester in the report layer. New metrics (Sortino/Calmar/signal decay) are N/A until the eval stack produces them.

### 1.3 Metric catalog (must support)

#### A. Signal Quality

| Metric | Notes |
|--------|--------|
| IC (Pearson) | required by Protocol |
| RankIC | required |
| ICIR | on Pearson series |
| RankICIR | on RankIC series — label clearly |
| IC mean / IC std | from daily series |
| IC t-stat | from daily series |
| IC positive ratio | win rate |
| Annualized IC / Annualized RankIC | |
| Signal decay | if horizon ladder exists; else N/A |

Headline table example:

| Metric | Value |
|--------|-------|
| RankIC | |
| IC (Pearson) | |
| RankICIR | |
| ICIR | |
| IC t-stat | |
| Positive IC ratio | |
| Annualized RankIC | |

#### B. Portfolio Performance

**Return:** annual return · cumulative return · excess return (if available)  
**Risk:** Sharpe · Sortino · Calmar · MDD · volatility (N/A if not computed)  
**Long–short:** H–L return · H–L Sharpe · long leg · short leg · decile spread  
**Trading:** daily turnover · annual turnover · implied fee · gross Sharpe · net Sharpe  

#### C. Factor Structure

monotonicity · decile spread · quantile returns · direction · stability score (if defined)

#### D. Risk Adjustment (mandatory ladder, not best-only)

| Mode | RankIC | RankICIR | H–L Sharpe | MDD | Net Sharpe |
|------|--------|----------|------------|-----|------------|
| Raw | | | | | |
| Size | | | | | |
| Industry | | | | | |
| Size+Industry | | | | | |

Narrative must state whether alpha **survives or disappears** after removing exposures — never show only the best cell.

#### E. Execution / Mechanism / Diagnostics

Preserve **all columns** from `execution_summary.csv`, `mechanism*.csv`, `stability*.csv` in pack dirs; report body summarizes, appendix dumps full union.

---

## 2. Report structure contract (`factor_report.md`)

Fixed section order (English titles; Chinese subtitle optional):

```
1.  Executive Summary
2.  Factor Thesis
3.  Economic Intuition
4.  Formula Construction
5.  Signal Pipeline
6.  Mechanism Analysis
7.  IC Analysis
8.  Portfolio Analysis
9.  Risk Adjustment
10. Stability Analysis
11. Execution Analysis
12. Limitations
13. Final Verdict
Appendix A. Complete Metric Dump (union)
Appendix B. Data Dictionary & Code Map
```

Also emit `summary.md` (1-page) and keep Protocol dirs: `charts/` `mechanism/` `execution/` `diagnostics/` `artifacts/`.

### 2.1 Research language (mandatory)

**Wrong:** “Factor has high ICIR.”  

**Right:** explain persistence, monotonicity, and survival after size/industry controls in research prose (Alpha Mining style). Numbers support the prose; prose is not optional.

Boss reading guide at top of every report:

- **Research ranking:** RankIC, RankICIR, Gross Sharpe, MDD, Monotonicity  
- **Admission ranking:** Net Sharpe, Turnover, Implied Fee, Execution  

---

## 3. Formula Construction contract

Section 4 must include **all four layers** (LaTeX):

### 3.1 Raw variables

e.g. \(Amount_t\), \(TradeCount_t\), minute \(r_t\), …

### 3.2 Intermediate variables

e.g. \(\mathrm{ATS}_t = Amount_t / TradeCount_t\), \(G_u\), \(G_d\), …

### 3.3 Transformation

ranking / z-score / cross-sectional residualization:

\[
F_t = X_t\beta + \varepsilon_t
\]

Cutting: Object / Knife / Output explicitly.

### 3.4 Final investable signal

\[
\mathrm{Signal}_t = \varepsilon_t \quad\text{or}\quad \mathrm{Signal}_t = \mathrm{MA}_{20}(F_t)
\]

plus **no-lookahead** policy (`signal_shift`).

Section 5 (Signal Pipeline) must show the diagram:

```
raw data → feature → residual / transform → signal
```

---

## 4. Mechanism Analysis contract

Every factor must answer **why it works**, not only Sharpe.

### 4.1 Verdict table (required)

| Hypothesis | Test | Result | Conclusion |
|------------|------|--------|------------|
| … | neutralize amount / size / … | pass/fail + number | … |

### 4.2 Chain display (TGD-shaped)

```
tau        → failed
upsilon    → failed
epsilon_d  → passed
TGD20      → accepted
```

Flow-shaped: Flow raw / Amount / Flow⊥Amount → interaction conclusion.

**Empty mechanism ⇒ status cannot exceed `candidate`.**

---

## 5. Charts contract (Alpha Mining style)

Auto-generate or attach; if missing, list under Missing Artifacts (do not pretend present).

| # | Chart | Path |
|---|--------|------|
| 1 | Factor construction diagram | `charts/construction_diagram.png` |
| 2 | IC curve | `charts/ic_curve.png` |
| 3 | Decile return | `charts/decile_return.png` |
| 4 | Cumulative long–short | `charts/cumulative_long_short.png` |
| 5 | Neutralization comparison | `charts/neutralization_compare.png` |
| 6 | Yearly stability | `charts/stability_yearly.png` |
| 7 | Turnover / cost | `charts/turnover.png` |

Protocol’s four required charts ⊆ this set.

---

## 6. Factor Card contract (v2 fields)

Extend `factor_card.yaml` (see `docs/schemas/factor_card.example.yaml`) with:

```yaml
hypothesis: |
  ...
mechanism:
  - temporal_information
  - downside_timing
formula: |
  short human-readable formula pointer / LaTeX one-liner
data_requirement:
  - close
  - ...
known_failure_modes:
  - high_turnover
  - short_sample
correlation_cluster: null   # filled after Matrix; optional in v2 migration
```

Keep Protocol fields: `family`, `status`, `data_coverage`, `admission`, `frozen_formula`, benchmarks.

---

## 7. Pack layout

```
research/reports/factors/{factor_id}/
  factor_card.yaml
  metrics.json                 # schema_version: factor_report_v2
  summary.md
  factor_report.md             # full contract narrative
  charts/
  mechanism/
  execution/
  diagnostics/
  artifacts/                   # full CSV/JSON = metric union source of truth
```

During migration, rename Template v1 checklist report to `factor_report_v1.md` if needed.

---

## 8. Generator scope (when implementing)

New module preferred: `factor_report_generator_v2.py` (wrap/reuse v1 ingest; **do not** change formulas).

Pipeline:

1. Collect artifacts → build Metric Union  
2. Render 13 sections + appendices  
3. Ensure chart slots / Missing Artifacts  
4. Write card + metrics.json (`factor_report_v2`)  
5. Validate checklist  

Do **not** write Registry. Do **not** change Harness compute path in this milestone.

---

## 9. Migration priority (after generator works)

### Priority A

1. TGD20  
2. FlowDensity20  
3. D1 `low_vol_liquidity_quality_60d`  
4. D4 `winner_sentiment_reversal_5d`  
5. D5 `upside_fragility_20d`  

### Priority B

6. `amount_stability_20d` (tag likely_redundant_to_D1)  
7. `reversal_20d`  
8. `price_volume_divergence_20d`  
9. `low_attention_reversal_20d`  
10. Ideal Reversal  

**Then** Milestone 1B Registry.

---

## 10. Acceptance (TGD20 golden)

For `research/reports/factors/TGD20/`:

- [ ] `factor_report.md` follows §2 section order  
- [ ] Metric count ≥ Template v1 (union ⊇ `factor_summary` columns)  
- [ ] Charts ≥ 7 slots (present or Missing Artifacts listed)  
- [ ] Formula layers 3.1–3.4 complete  
- [ ] Mechanism chain tau/upsilon/εd/TGD present  
- [ ] Neutralization ladder shown (not best-only)  
- [ ] Research prose present (not number-only)  
- [ ] **Factor values / formulas unchanged**  
- [ ] **No Registry writes**

---

## 11. One-line contract

> **Report Generator v2 implements a Research Report Contract: Metric Union + institutional narrative + complete formulas + mechanism verdicts + Alpha Mining charts — so quality-bar factors become Registry-ready assets without silent metric loss.**
