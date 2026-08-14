# Research Pack Audit v1 — Milestone 1D.5

**Date:** 2026-07-20  
**Scope:** Representative packs only (not full inventory, **not Registry**)  
**Audited against:**

| Contract | Path |
|----------|------|
| Protocol v1 | `docs/factor_research_protocol_v1.md` |
| Report Template v2 | `docs/factor_report_template_v2.md` |
| Metric Registry | `docs/schemas/metric_registry.yaml` |
| Chart Registry | `docs/schemas/chart_registry.yaml` |
| Pack Schema | `docs/schemas/factor_report.schema.yaml` |

**Constraints honored:** no Registry write · no formula changes · no factor recompute.

---

## 0. One-line verdict

| Question | Answer |
|----------|--------|
| Does Report OS + Schema + Generator work across factor families? | **Yes** |
| Are all four packs Registry-ready? | **No** — see blockers / gaps |
| Ready for Controlled Batch → Registry? | **After** closing P0/P1 items below + human status review |

```
1D proved generalization
        ↓
1D.5 audits asset quality   ← you are here
        ↓
Controlled Batch (optional fill gaps)
        ↓
Registry (admission by status)
```

---

## 1. Pack inventory under audit

| factor_id | Family role | Card status | Frozen formula | Generator `validation.ok` |
|-----------|-------------|-------------|----------------|---------------------------|
| TGD20 | Temporal / Golden | `validated` | true | ✅ true |
| FlowDensity20 | Microstructure × liquidity interaction | `candidate` | false | ❌ false (3 charts) |
| D1_LiquidityQuality60d | Classical EOD liquidity | `testing` | true | ✅ true |
| IdealReversal | Paper / cutting replication | `testing` | false | ✅ true |

All four have `metrics.json` → `schema_version: factor_report_v2` and **complete Metric Union key set** (every registry headline/portfolio/cost id present; missing values are explicit N/A).

---

## 2. Cross-pack scorecard

Legend: ✅ pass · ⚠ partial / documented gap · ❌ fail

| Check | TGD20 | Flow | D1 | Ideal |
|-------|-------|------|----|-------|
| Pack required files | ✅ | ✅ | ✅ | ✅ |
| Pack required dirs | ✅ | ✅ | ✅ | ✅ |
| 13 chapters | ✅ | ✅ | ✅ | ✅ |
| Factor card required fields | ✅ | ✅ | ✅ | ✅ |
| Metric Union keys complete | ✅ | ✅ | ✅ | ✅ |
| Metric Union values filled* | ⚠ 18/32 | ⚠ 16/32 | ⚠ 15/32 | ⚠ 12/32 |
| Chart registry (7 slots) | ✅ 7/7 | ❌ 4/7 | ✅ 7/7 | ✅ 7/7 |
| Formula §4.1–4.4 + LaTeX | ✅ | ✅ | ✅ | ✅ |
| Layer discipline (Identity ≠ mech ≠ exec) | ✅ | ✅ | ✅ | ✅ |
| Mechanism section + CSV | ✅ rich | ✅ rich + amount-orth | ⚠ thin canonical | ✅ cutting legs |
| Risk adjustment ladder | ✅ full neut | ✅ full neut | ⚠ universe modes | ✅ neut modes |
| Execution completeness | ✅ grid | ✅ grid | ❌ empty | ❌ empty |
| Missing Artifacts honesty | ✅ none | ✅ listed | ✅ none** | ✅ none** |
| Dual Benchmark Production Track | ❌ harvest legacy | ❌ | ❌ | ❌ |
| Appendix B code map (factor-specific) | ✅ correct for TGD | ❌ **TGD-hardcoded** | ❌ **TGD-hardcoded** | ❌ **TGD-hardcoded** |

\* Filled = non-null `metric_union` values. N/A for Pearson IC / Sortino / Calmar etc. is **compliant** (never silent drop).  
\*\* D1/Ideal chart slots exist; execution CSVs do not — empty `execution/` dir still satisfies schema dir requirement but **execution completeness fails**.

---

## 3. Per-factor audit

### 3.1 TGD20 — Golden reference

| Dimension | Result | Notes |
|-----------|--------|-------|
| Schema compliance | ✅ | Full pack + 13 chapters + layer banners |
| Metric completeness | ✅ structure / ⚠ values | RankIC family filled; Pearson IC, Sortino, Calmar, leg returns N/A |
| Missing artifacts | ✅ | None |
| Formula | ✅ | Full residual / orthogonalization / MA20 |
| Mechanism | ✅ | Verdict table + diagnostics (τ/υ rejected; εd key; tgd_eps ≠ second factor) |
| Risk adjustment | ✅ | raw → size → industry → size_industry (not best-only) |
| Execution | ✅ | Implementation grid; Net Sharpe best ≈ 2.32 framed as trade efficiency |
| Registry readiness | **Conditional** | Status `validated` OK for Registry **row** after Production Track note; still harvest ALL/2022–2025 |

**Strength:** Complete evidence chain; layer discipline locked.

**Gap:** Not Protocol CSI1000 Production Track; Pearson IC family absent.

---

### 3.2 FlowDensity20 — Interaction / attribution

| Dimension | Result | Notes |
|-----------|--------|-------|
| Schema compliance | ⚠ | Structure OK; chart validation fails |
| Metric completeness | ✅ / ⚠ | Neut + execution metrics strong; mono N/A in summary |
| Missing artifacts | ❌→documented | `ic_curve`, `decile_return`, `cumulative_long_short` — **no fake charts** |
| Formula | ✅ | MA20(net active / mktcap) |
| Mechanism | ✅ | Amount / GrossActive / Flow⊥Amount; amount-orth under `mechanism/` |
| Risk adjustment | ✅ | Full neut ladder |
| Execution | ✅ | Buffer implementations; Net Sharpe ≈ 2.88 |
| Registry readiness | **Not yet** | Status `candidate`; chart P0; formula not frozen; interaction classification required |

**Strength:** Best demonstration that the OS can record *mechanism failure / attribution* (not only “pass then ship”).

**Gap (P0):** Three protocol charts absent from legacy research export — fill via future portfolio-simulator Production Track, not invented PNGs.

---

### 3.3 D1_LiquidityQuality60d — Classical EOD

| Dimension | Result | Notes |
|-----------|--------|-------|
| Schema compliance | ✅ structure | Generator ok=true |
| Metric completeness | ⚠ | Soft-bar metrics present; net Sharpe / execution N/A |
| Missing artifacts | ⚠ | Charts present; **no execution_summary** |
| Formula | ✅ | 4-layer present (library pointer; not L2 residual ladder) |
| Mechanism | ⚠ | Canonical row only — acceptable for simple EOD if narrative says so |
| Risk adjustment | ⚠ | No size/industry modes; **universe ladder** (ALL/CSI1000/500/300) used as diagnostic substitute |
| Execution | ❌ | Empty `execution/` |
| Registry readiness | **Not yet** | Card status `testing` (soft-bar numbers look candidate-grade — **status needs human reconcile**) |

**Strength:** Proves Template v2 does not require L2/minute/residual.

**Gaps:** Status mismatch vs research narrative; no execution layer; no classical neut ladder; Appendix B wrong (see §5).

---

### 3.4 IdealReversal — Paper / cutting track

| Dimension | Result | Notes |
|-----------|--------|-------|
| Schema compliance | ✅ structure | Generator ok=true |
| Metric completeness | ⚠ thinnest fill | MDD / turnover / net Sharpe largely N/A |
| Missing artifacts | ⚠ | Charts present; no execution grid |
| Formula | ✅ | Object / Knife / Output explicit |
| Mechanism | ✅ | M_high / M_low / spread / Ret20 baseline |
| Risk adjustment | ✅ | raw/size/industry/size_industry present |
| Execution | ❌ | Empty |
| Registry readiness | **testing only** | Sharpe≈1.70, mono≈0.44 — below soft bar; correct as replication stress pack |

**Strength:** Paper → spec → cutting → pack path works.

**Gaps:** Execution absent; many Metric Union N/As; must not auto-admit.

---

## 4. Dimension roll-up (audit checklist)

### 4.1 Schema compliance

- Required files/dirs/chapters: **4/4 packs pass**.
- Factor cards: all required fields present; statuses in Protocol enum.
- Layer discipline present on all four reports.
- **Defect:** Appendix B in generator hardcodes TGD implementation / essay paths for every factor (§5 P1).

### 4.2 Metric completeness

- Union **keys**: complete for all packs (Metric Registry contract satisfied).
- Union **values**: harvest-limited; common N/A cluster across packs:
  - Pearson `IC`, `IC_tstat`, `IC_mean`, `IC_std`
  - `Sortino`, `Calmar`, `volatility`, leg returns, `cumulative_return`, `excess_return`, `stability_score`
- This is **acceptable for harvest packs** under Protocol “N/A never silent”; Production Track should fill headline cost/risk fields for Registry comparison.

### 4.3 Missing artifacts

| Pack | Listed / actual gaps |
|------|----------------------|
| TGD20 | None |
| FlowDensity20 | 3 protocol charts (honest Missing Artifacts) |
| D1 | Execution CSV absent (dir empty; not always listed as Missing Artifacts — **should be**) |
| IdealReversal | Execution CSV absent (same) |

### 4.4 Formula completeness

All four: §4.1–4.4 + pipeline + construction diagram steps from content YAML. **Pass.**

### 4.5 Mechanism completeness

| Pack | Grade | Comment |
|------|-------|---------|
| TGD20 | A | Full reject/accept ladder |
| Flow | A | Interaction attribution |
| Ideal | B+ | Cutting legs |
| D1 | C+ | Thin but family-appropriate if kept explicit |

Empty mechanism would force status ≤ `candidate` (Protocol). D1 is borderline thin — keep narrative that EOD library factors may have shallow mechanism until dedicated tests exist.

### 4.6 Risk adjustment completeness

| Pack | Grade | Comment |
|------|-------|---------|
| TGD / Flow / Ideal | Pass | Mode ladder shown |
| D1 | Partial | Universe ladder ≠ size/industry neut; OK as diagnostic analogue if labeled (currently labeled in report) |

### 4.7 Execution completeness

| Pack | Grade |
|------|-------|
| TGD / Flow | Pass (implementation variants, not new factors) |
| D1 / Ideal | Fail — no `execution_summary.csv` |

---

## 5. Findings (priority)

### P0 — Block Registry admission for affected packs

1. **FlowDensity20:** missing `ic_curve` / `decile_return` / `cumulative_long_short`. Do not invent; schedule Production Track chart export.
2. **Status governance:** do not dump all four into Registry as `validated`. Proposed admission intent:

| factor_id | Suggested Registry status (after human confirm) | Admit now? |
|-----------|--------------------------------------------------|------------|
| TGD20 | `validated` | Yes (with coverage exception note) |
| FlowDensity20 | `candidate` | Only after chart P0 or explicit exception flag |
| D1_LiquidityQuality60d | `candidate` (reconcile from `testing`) | After status + execution policy decision |
| IdealReversal | `testing` | No production admission |

### P1 — Fix before / with Controlled Batch

1. **Generator Appendix B** hardcodes TGD paths (`core/l2_features/...tgd`, `tgd_v1` essay) for every pack — violates schema-driven rule. Move to `*_report_content.yaml` (`code_map` / `appendix_b`).
2. **D1 / Ideal:** either run minimal execution harvest or list `execution/execution_summary.csv` under Missing Artifacts explicitly.
3. **D1 card status** (`testing`) vs soft-bar narrative — human reconcile to `candidate` or keep `testing` with reason.
4. **Metric Union N/A cluster** — document as “Production Track backlog,” not silent improvement by faking.

### P2 — Nice-to-have

1. Pearson IC family when simulator supports it.
2. Flow monotonicity into `factor_summary.csv` (exists on mechanism row).
3. Correlation cluster still null until Factor Matrix.

---

## 6. What this audit does *not* claim

- Full factor inventory assetized  
- Dual Benchmark Production Track complete  
- Orthogonality Matrix / Composite ready  
- Numeric auto-admission (Protocol: `requires_manual_review`)

---

## 7. Recommended next steps

```
DONE: 1D Representative packs (generalization proof)
DONE: 1D.5 Research Pack Audit (this document)
        |
        v
NOW:   Human review of P0/P1 + status table
        |
        v
NEXT:  Generator fix (Appendix B) + optional Flow chart Production fill
        |
        v
THEN:  Milestone 1E — Registry (admit by status, not by “has a folder”)
```

**Do not** start Registry until:

1. Flow chart gap accepted-with-exception **or** filled  
2. Appendix B generator defect fixed  
3. Human signs status column for all four  

---

## 8. Closing

1D proved the **factory can manufacture packs across families**.  
1D.5 shows which packs are **assets vs drafts**.

TGD20 remains the only pack that is structurally and evidentially near Registry-grade.  
Flow proves mechanism honesty (including failure modes) but fails chart completeness.  
D1/Ideal prove family coverage but lack execution depth and (for Ideal) soft-bar strength.

That is the correct posture for an Alpha Factory: **generalized tooling first, selective admission second.**
