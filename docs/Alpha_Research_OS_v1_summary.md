# Alpha Research OS v1 — Phase I Completion Report

**Checkpoint date:** 2026-07-20  
**Status:** **FROZEN** — do not start Milestone 1B / Registry / Ideal Reversal / Composite until Phase II kickoff.  
**Codename:** Alpha Research OS v1 (first stage wrap-up)

---

## 0. One-line verdict

Phase I delivered a **research contract + harness skeleton + two deep factor case studies + a cutting discovery framework**.  
The project moved from *factor experiments* to *factor assets under a single protocol*.  
**Pause coding here.** Resume at Phase II from Registry, not from new random factors.

---

## 1. Objective (what Phase I set out to do)

Build an **institutional-style factor research framework** for FACTOR_DEV:

```
Paper / Search / LLM
        ↓
Factor Specification
        ↓
Standard Evaluation (Dual Benchmark)
        ↓
Report Pack (charts + mechanism + execution)
        ↓
Registry  →  Orthogonality  →  Composite  →  Production
```

Phase I **froze the left half** (contract + skeleton + proven research chains).  
Phase II will extend the right half (registry → paper track → matrix → composite).

---

## 2. Architecture completion map

```
                 Factor Research OS v1 (Phase I)

                          Protocol v1
                              |
                     Research Contract
                              |
        ---------------------------------------------
        |                    |                      |
   Factor Spec         Evaluation Contract     Report Pack Schema
   (factor_specs/)     (Dual Benchmark)        (metrics / charts /
                              |                 mechanism / execution)
                              |
                    Unified Harness skeleton
                    (run_factor_research.py)
                              |
                    ---------------------------------
                    |              |                |
                 TGD20        FlowDensity      Cutting / others
               (validated*)    (candidate)      (pre-pack / testing)
```

| Module | Status | Path / artifact |
|--------|--------|-----------------|
| Research Protocol v1 | ✅ **Frozen** | `docs/factor_research_protocol_v1.md` |
| Factor lifecycle | ✅ | discovery → … → retired |
| Dual Benchmark | ✅ | Research ALL / Production CSI1000 @ 20D |
| Metrics schema | ✅ | IC, RankIC, ICIR, RankICIR, Sharpe, … |
| Factor pack schema | ✅ | card / metrics / charts / mechanism / diagnostics / execution |
| Mechanism layer | ✅ (concept + TGD/Flow practice) | Protocol §E2 |
| Execution layer concept | ✅ | `execution_layer.py` + pack `execution/` |
| Unified Harness skeleton | ✅ Milestone 1A | `run_factor_research.py`, `factor_research_harness.py` |
| Factor Spec examples | ✅ | `factor_specs/TGD20.yaml`, `docs/schemas/` |
| Registry | ✅ **1E** | `research/registry/` |
| Composite engine | ✅ **2.0** | `research/reports/composite_alpha_v1/` — incremental A/B/C |
| Production portfolio pipeline | ⏸ | deferred |

\* Legacy card status `validated_single_factor` maps to Protocol lifecycle **`validated`** (manual review; formula frozen). Full Dual-Benchmark re-run under Protocol is **not** required for this checkpoint.

---

## 3. Why Protocol + Harness matter (not “just YAML”)

### Before

```
Factor A → run_A.py → result_A.csv → plot_A.png
Factor B → run_B.py → result_B.csv
Factor C → notebook
```

Different universes, horizons, costs, and metrics → **no fair answer to “which factor is better?”**

### After (contract)

```
factor_spec.yaml
      ↓
run_factor_research.py  (--mode research|production)
      ↓
metrics.json  (Production = comparable; Diagnostic = research only)
      ↓
factor pack
      ↓
registry  (Phase II)
```

Experiments become **assets**. Cherry-picked diagnostic ICIR is explicitly non-comparable.

---

## 4. Completed research assets

### 4.1 TGD20 — highest completeness

| Field | Value |
|-------|--------|
| Protocol family | `temporal_information` |
| Legacy status | `validated_single_factor` |
| Protocol status (mapped) | **`validated`** (formula frozen; Production admission still manual / Dual Benchmark migration later) |
| Spec | `factor_specs/TGD20.yaml` |
| Pack | `research/reports/factors/TGD20/` |
| Long-form | `research/reports/tgd_v1/` |

**Economic claim:** not “up vs down magnitude”, but **when** intraday return pressure forms (residualized downside timing + MA20).

| Track / view | Key numbers |
|--------------|-------------|
| Research-style (legacy ALL, ~2022–2025) | RankIC ≈ 4.3%; ICIR raw 6.98; H–L Sharpe 2.77; Monotonicity 0.988 |
| size+industry | ICIR **11.29**; H–L Sharpe **4.06** |
| Execution | TO 0.65 → **0.297** (buffer 5/15); Net Sharpe ~1.0 → **2.32** |

**Mechanism chain (complete):** Gu / Gd → τ / υ → εu / εd → TGD20.  
Simple \(G_d - G_u\) is **not** the alpha; residualized downside timing is.

**Do not:** retune MA / residual controls / invent `TGD20_v2` under the same id.

---

### 4.2 FlowDensity20 — research closed, formula not frozen

| Field | Value |
|-------|--------|
| Protocol family | `[microstructure, liquidity]` |
| Protocol status (mapped) | **`candidate`** |
| Pack | `research/reports/factors/FlowDensity20/` |
| Long-form | `research/reports/l2_flow_density_v1/` |

| Signal | ICIR (indicative) |
|--------|------------------:|
| FlowDensity (size+ind) | **+4.85** |
| Amount / gross / buy / sell style | ≈ **−8.6** |
| Flow ⊥ Amount | **−2.49** (sign flip) |

**Conclusion:** not pure smart-money flow → **flow × liquidity interaction**.  
Net Sharpe (execution best) ≈ **2.88**; turnover can compress sharply under buffers.

**Orthogonality (probe):** TGD vs Flow raw corr ≈ 0.22; equal-rank 50/50 **worse** than TGD alone → no naive blend. Report: `research/reports/factor_orthogonality/TGD20_FlowDensity20/`.

---

### 4.3 Factor Cutting framework — discovery method proven

| Field | Value |
|-------|--------|
| Method | Object → Knife → Output |
| Code / reports | `factor_cutting/`, `research/reports/factor_cutting_v1/` |
| Protocol status | **`testing`** (not yet standard pack via harness) |

**Ideal Reversal (indicative, pre-Protocol pack):** RankIC ≈ −3.38%, ICIR ≈ −6.48; \(M_{high}\) carries alpha, \(M_{low}\) closer to noise.  
**Knife search:** ATS more independent than amount; amount+ATS residual add can beat ATS alone (e.g. ICIR ≈ −6.88 vs −6.48) — early Alpha Factory style discovery, still **outside** unified pack admission.

**Ideal Amplitude / Active Trade / Smart Money / DBD-GRU:** research exists or planned; **not** Phase I deliverables for OS admission.

---

### 4.4 Other inventory (not Protocol-admitted)

| Asset | Status | Note |
|-------|--------|------|
| D1 / D4 / D5 | pending migration | EOD style library; needs Registry + Production Track |
| Extreme Return study | research satellite | `research/extreme_return_study/` |
| L2 Flow Density validation grids | done as study | bartime / horizon / neut / cost; stamp discipline |
| Report Generator v1 | exists | `factor_report_generator.py` — Template v1 packs |
| Harness 1A | skeleton | TGD20 read-only adapter; no Dual Benchmark recompute |

---

## 5. Research philosophy (frozen wording)

**Do not** optimize for “max IC on any slice.”

**Do:**

```
discover → validate → explain (mechanism) → combine (orthogonal) → trade (execution)
```

Composite eligibility (Phase II rule of thumb, not coded yet):

- Protocol status ≥ **`validated`**
- **`mechanism/`** complete
- Production Track metrics present  
→ **not** every `candidate` (e.g. interaction-only Flow) auto-enters equal-weight blend.

---

## 6. Quality-bar inventory (Phase I) — not only TGD / Flow

**Inventory screen (soft, for Registry backlog — not a Protocol numeric admission bar):**

- \|H–L Sharpe\| **> 2**
- IC / ICIR OK ractical: \|RankIC\| ≳ 2% **or** \|ICIR\| ≳ 3)
- Monotonicity OK (≳ 0.8 when measured; else flag `mono_missing`)

Protocol still requires Dual Benchmark + mechanism for **`validated`**. Soft screen only decides **who enters the Registry backlog** as `testing` / `candidate`.

### 6.1 Already in standard Template v1 packs

| Factor | Family | Protocol status | Key result | Pack |
|--------|--------|-----------------|------------|------|
| TGD20 | temporal_information | validated* | ICIR 11.29 (size+ind); Net Sharpe 2.32; mono 0.99 | ✅ |
| FlowDensity20 | microstructure + liquidity | candidate | ICIR ~4.85; Net Sharpe ~2.88 | ✅ |

### 6.2 Frozen OHLCV library (must register in Phase II)

Source: `research/frozen_candidate_pool_v1.json` (2026-07-09).

| Factor | Role | Soft bar (confirmation) | Notes |
|--------|------|-------------------------|-------|
| D1 `low_vol_liquidity_quality_60d` | production_base | IC 0.057 / ICIR 6.01 / Sharpe **2.26** / mono **0.8** | ✅ clear pass |
| D4 `winner_sentiment_reversal_5d` | production_base | (Base3 stack ICIR 5.69 / Sharpe **3.76**) | Register; fill solo Production metrics |
| D5 `upside_fragility_20d` | production_base | in Base3 | **Mono historically weak (~0.2)** — register with flag, not silent pass |
| Base3 equal-z(D1,D4,D5) | stack | Sharpe **3.76** / ICIR **5.69** / mono **1.0** (951d) | Registry as composite *candidate stack*, not a single factor |
| `amihud_shock_reversal_5d` | enhancer | Sharpe **2.06** / mono 0.9 | Register as enhancer, not base |
| `cn_cancel_shock` | L2 enhancer | strong Base3 uplift | Register as enhancer |

### 6.3 `result/` EOD singles meeting soft bar (Research / ALL-heavy)

Indicative; many lack mono / CSI1000 Production. Phase II: register as `testing` + `needs_rerun`.

| Factor | Univ (best hit) | Sharpe | ICIR | RankIC |
|--------|-----------------|-------:|-----:|-------:|
| amount_stability_20d | ALL | 4.58 | 8.64 | 0.041 |
| range_contraction_20d | ALL | 2.01 | 5.68 | 0.060 |
| reversal_20d | ALL | 2.44 | 4.13 | 0.044 |
| low_attention_reversal_20d | ALL | 2.37 | 4.06 | 0.043 |
| rsi_14_reversal | ALL | 2.26 | 4.10 | 0.039 |
| overheated_turnover_proxy_20d | CSI1000 | 2.13 | 3.58 | 0.038 |
| price_volume_divergence_20d | ALL | 2.92 | 3.01 | 0.016 |
| volume_price_efficiency_20d | ALL | 2.45 | −4.11 | −0.044 |
| amount_20d_mean / volume_20d_mean / … | ALL | >2 | \|ICIR\|>3 | … |

**Redundancy caution (do not treat as new independent alpha):**  
`amount_stability_20d` was **rejected** in frozen pool as latent D1 (corr ~0.79). High Sharpe alone ≠ new Registry “validated” source — still **list** it, tag `likely_redundant_to_D1`.

### 6.4 Cutting / microstructure (pre-pack)

| Factor | Soft bar | Protocol status |
|--------|----------|-----------------|
| Ideal Reversal | ICIR ~−6.48 (cutting study) | testing → standard pack in Phase II |
| Ideal Amplitude | cutting studies | testing |
| amount+ATS dual knife | ICIR ~−6.88 vs ATS −6.48 | discovery note |

### 6.5 Explicitly not “only two factors”

Phase I **deep case studies** = TGD + Flow.  
Phase I **quality backlog for Registry** = Base3 (D1/D4/D5) + enhancers + soft-bar EOD list + Cutting — **dozen-scale**, not two.

\* TGD mapped from legacy `validated_single_factor`; Dual Benchmark Production re-run still open for Phase II.

---

## 7. Infrastructure snapshot

| Component | Status | Resume note |
|-----------|--------|-------------|
| Protocol | Frozen | Treat as constitution; patch only with explicit review |
| Schemas | Frozen | `docs/schemas/*` |
| Harness | Skeleton (1A) | Next: wire compute/evaluate **or** 1B Registry — choose at Phase II kickoff |
| Report Generator | Exists | Legacy Template v1; migrate via metadata, no bulk file moves |
| Execution Layer | Exists | Keep grids under `execution/` / diagnostics |
| Registry | Next | Upgrade `index.csv` → `factor_registry.csv` |
| Factor Matrix | Later | After ≥5 Production-comparable rows |
| Composite | Later | After Matrix; IC-weighted; no 50/50 default |

---

## 8. Why pause now (explicit)

Continuing Ideal Amplitude / Active Trade / Smart Money / Composite **without** Registry + harness admission would recreate:

```
new factor → new script → new report island
```

Phase I’s job was to **stop that failure mode**. Checkpoint before expansion.

---

## 9. Three assets worth keeping above all else

1. **TGD20** — proof that a **temporal information layer** works end-to-end (mechanism → execution → pack).  
2. **Factor Cutting** — mechanism-driven discovery (Object / Knife / Output), not blind formula search.  
3. **Protocol + Harness 1A** — the extension surface for every future paper / search / ML factor.

---

## 10. Phase II kickoff roadmap (do not start until resume)

**Revised order (assetization before Registry):**

```
Protocol → Harness → Research Report Template v2 → Factor Pack Migration
    → Registry → Orthogonality Matrix → Composite → Production
```

| Step | Name | Goal |
|------|------|------|
| 1B | **Report Schema** | ✅ |
| 1C | **TGD20 Golden Pack** | ✅ `factor_report_generator_v2.py` + `research/reports/factors/TGD20/` |
| 1D | **Representative Packs** | ✅ Flow / D1 / IdealReversal — `docs/milestone_1D_pack_validation.md` |
| 1D.5 | **Research Pack Audit** | ✅ `docs/research_pack_audit_v1.md` — not Registry-ready en masse |
| 1D.6 | **Generator schema cleanup** | ✅ `docs/milestone_1D6_generator_cleanup.md` — artifacts/code_map in YAML |
| 1D.7 | **Pre-Registry completion** | ✅ Flow protocol charts + D1 execution — `docs/milestone_1D7_pre_registry_completion.md` |
| 1E | **Registry v1** | ✅ `research/registry/` — `docs/milestone_1E_registry_v1.md` |
| 1F | **Factor Similarity Matrix** | ✅ `research/reports/factor_similarity_matrix/` — Flow⊥D1 absorbed |
| 1F.5 | **Attribution Review** | ✅ `docs/factor_attribution_review_v1.md` — cores TGD+D1 |
| 2.0 | **Composite Alpha Engine v1** | ✅ A/B/C incremental — `docs/milestone_2_0_composite_alpha_v1.md` · baseline **TGD+D1**; Flow = enhancer |
| 2.1 | **Production Stress** | ✅ `docs/milestone_2_1_production_stress.md` — ALL/CSI1000 OK; CSI300/500 fail; cost survives ~30bp |
| 2.2 | **Portfolio Construction capability** | ✅ FROZEN — [`docs/milestone_2_2_portfolio_construction.md`](milestone_2_2_portfolio_construction.md) · not production |
| 2.2.x | Research track notes | [`docs/milestone_2_2_research_track_expansion.md`](milestone_2_2_research_track_expansion.md) |
| **3.0** | **Alpha Library Expansion** | ✅ ACTIVE — [`docs/milestone_3_0_alpha_library_expansion.md`](milestone_3_0_alpha_library_expansion.md) |
| Next | III-B SUE / Fundamental (SmartMoney when minute ready) | Library depth — Portfolio v2 frozen |
| 4 | **Ideal Reversal / Amplitude** greenfield via harness + v2 pack | |
| 5 | **Factor Orthogonality Matrix** | (superseded by 1F) |
| 6 | **Composite Alpha Engine v1** | (done as 2.0) |
| 7 | Production readiness | |

**Out of scope until classic packs are admitted:** DBD-GRU / deep enhancement.

---

## 11. Key path index

| Item | Path |
|------|------|
| This report | `docs/Alpha_Research_OS_v1_summary.md` |
| Protocol | `docs/factor_research_protocol_v1.md` |
| Schemas | `docs/schemas/` |
| Harness CLI | `run_factor_research.py` |
| Harness lib | `factor_research_harness.py` |
| Specs | `factor_specs/` |
| Factor packs | `research/reports/factors/` |
| Prior research milestone | `research/reports/factors/MILESTONE_SUMMARY.md` |
| Roadmap (pre-OS; superseded for next steps) | `research/reports/factors/ROADMAP.md` |
| TGD long-form | `research/reports/tgd_v1/` |
| Flow long-form | `research/reports/l2_flow_density_v1/` |
| Orthogonality | `research/reports/factor_orthogonality/TGD20_FlowDensity20/` |
| Cutting | `research/reports/factor_cutting_v1/` |

---

## 12. Resume checklist (Phase II day-1)

When restarting, do **not** invent a new framework. Confirm:

1. [ ] Protocol still frozen (`docs/factor_research_protocol_v1.md`)  
2. [ ] `python run_factor_research.py --factor TGD20 --mode production` still OK  
3. [ ] Start **Milestone 1B Registry** (or Ideal Reversal pack **only after** Registry row schema exists)  
4. [ ] No new `run_*_v1.py` islands without harness entry  

---

## 13. Closing statement

Phase I answer to the organization problem:

> Stop uncontrolled factor generation → freeze research law → skeleton factory entry → keep TGD/Flow/Cutting as proof assets.

The next question is no longer *“can we find another alpha?”* but:

> *Can we systematically turn many research ideas into a few investable, orthogonal information sources?*

That is **Alpha Factory Expansion v2** — start only after this checkpoint is accepted.
