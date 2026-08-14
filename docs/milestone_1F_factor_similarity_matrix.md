# Milestone 1F — Factor Similarity Matrix v1

**Date:** 2026-07-20  
**Status:** PASS (analysis only — **no Composite**)  
**Script:** `run_factor_similarity_matrix_v1.py`  
**Outputs:** `research/reports/factor_similarity_matrix/`

---

## Scope honored

| Constraint | Result |
|------------|--------|
| Registry factors only | TGD20, FlowDensity20, D1, D4, D5 |
| No Composite | ✅ |
| No Registry schema changes | ✅ |
| No formula changes | ✅ |

**Signal book (single, documented):** confirmation 951d · size+industry CS-z · `signal_shift=1`  
(Do not mix with D1 pack’s raw-execution book when reading Net Sharpe.)

---

## Artifacts

| File | Content |
|------|---------|
| `factor_ic_corr.csv` | Daily RankIC series correlation |
| `factor_return_corr.csv` | Daily H–L return correlation |
| `cs_corr_matrix.csv` | Mean CS Spearman between panels |
| `residual_ic_matrix.csv` | Residual ICIR matrix (Y ⊥ X) |
| `residual_ic_long.csv` | Pairwise residual detail + role |
| `factor_clusters.yaml` | Clusters + `alpha_role_hint` |
| `similarity_report.md` | Human report |
| `similarity_verdict.json` | Machine summary |
| `figures/*.png` | Heatmaps |

---

## Q1 — Same alpha source? (IC corr)

|  | TGD | Flow | D1 | D4 | D5 |
|--|-----|------|----|----|-----|
| **TGD** | 1 | 0.32 | **0.66** | 0.02 | −0.34 |
| **Flow** | | 1 | 0.29 | 0.11 | −0.22 |
| **D1** | | | 1 | −0.05 | −0.54 |
| **D4** | | | | 1 | −0.31 |
| **D5** | | | | | 1 |

High overlap risk: **TGD ↔ D1** (IC corr 0.66). Flow–D1 IC series only moderately correlated (0.29) — residual test is decisive.

---

## Q2 — Incremental alpha? (residual ICIR)

Critical pairs:

| Y ⊥ X | Resid ICIR | Retention | Role |
|-------|-----------:|----------:|------|
| **TGD ⊥ Flow** | 9.12 | 0.81 | independent_source |
| **Flow ⊥ TGD** | 1.68 | 0.35 | mostly_redundant |
| **Flow ⊥ D1** | **−0.62** | −0.13 | **redundant_or_absorbed** |
| **D1 ⊥ Flow** | 8.71 | 0.90 | independent_source |
| **D1 ⊥ TGD** | 7.21 | 0.74 | independent_source |
| **TGD ⊥ D1** | 6.06 | 0.54 | partial_overlap_enhancer |
| **D4 ⊥ D1** | −0.55 | −0.10 | redundant_or_absorbed |
| **D5 ⊥ TGD** | −7.09 | 0.74 | independent_source (signed) |

---

## Q3 — Answers for Composite readiness

### Independent alpha sources (keep)

1. **TGD20** — temporal core. Survives Flow residualization. Partially overlaps D1 but retains ICIR≈6 after ⊥D1.  
2. **D1_LiquidityQuality60d** — liquidity quality core/satellite. Survives ⊥TGD and ⊥Flow.  
3. **D5_UpsideFragility20d** — signed independent vs TGD/Flow (negative ICIR book); still **library inventory / mono-weak**; needs pack before Composite weight.

### Redundant / enhancer-only (do not treat as new cores)

1. **FlowDensity20** — **absorbed by D1** (Flow⊥D1 resid ICIR≈−0.62). Confirms prior mechanism story: liquidity-conditioned interaction, not a second liquidity alpha. Role hint: `redundant_risk` / satellite enhancer at best — **not** a second core beside D1.  
2. **D4_WinnerSentimentReversal5d** — absorbed by D1; mostly redundant vs TGD. Role hint: `redundant_risk`. Prefer Base3 component documentation, not standalone Composite leg until pack + residual redesign.

### Combination implication (no weights yet)

```
Do NOT:  equal-weight {TGD, Flow, D1, D4, D5}

Prefer review set:
  core candidates:   TGD20 + D1
  satellite/review:  D5 (sign + pack + mono)
  do not add as cores: Flow, D4   (D1-redundant on this book)
```

Equal-rank TGD+Flow already known to underperform TGD alone; 1F strengthens: Flow adds little once D1 exists.

---

## D4/D5 pack note (no schema change)

Registry still lists D4/D5 as `candidate` (1E).  
1F recommends human label mentally as **`candidate_pending_pack`** until Template v2 packs exist — without editing Registry schema in this milestone.

---

## Explicit non-goals

- Composite Alpha Engine  
- Registry field adds (`information_layer`, `alpha_role`) — deferred optional patch  
- Formula retune  

---

## Next

```
1F Similarity Matrix   ✅
        ↓
Human review of Flow/D4 redundancy vs D1
        ↓
Composite Alpha Engine v1
  (only TGD + D1 [+ optional D5 after pack], IC-weighted — not equal-rank kitchen sink)
```
