# Factor Attribution Review v1 — Milestone 1F.5

**Date:** 2026-07-20  
**Status:** PASS (documentation + classification only)  
**Inputs:** Milestone 1F Similarity Matrix (`research/reports/factor_similarity_matrix/`)  
**Constraints:** No Composite · No Registry schema changes · No formula changes

---

## 0. One-line verdict

The Registry listed five “high Sharpe” names. After residual IC, the library collapses to roughly **two independent core sources** (TGD + D1), plus satellites and one signed pending case (D5). Composite must not treat five ICIR rankings as five information sources.

```
Surface inventory:  5 named factors
Effective cores:    ~2 (TGD20, D1)
Satellites:         Flow, D4  (D1-overlapping)
Pending:            D5 (direction + pack + mono)
```

---

## 1. Signal book (must not mix)

All 1F numbers below use:

| Setting | Value |
|---------|-------|
| Window | confirmation 2022-01-28 → 2025-12-31 (951d) |
| Neutralization | size + industry |
| Transform | CS z-score |
| Shift | `signal_shift=1` |

Do **not** mix with D1 pack Net Sharpe from **raw** execution (1D.7). Attribution = information structure; execution Net Sharpe = tradeability of a chosen book.

---

## 2. IC correlation interpretation

| Pair | IC corr | Reading |
|------|--------:|---------|
| TGD ↔ D1 | **0.66** | Strong co-movement of predictive ranks — shared regime / style exposure risk, not proof they are the same factor |
| TGD ↔ Flow | 0.32 | Mild overlap; residual test more informative |
| Flow ↔ D1 | 0.29 | Moderate IC-series corr — **insufficient** to declare independence (see §3) |
| TGD ↔ D4 | 0.02 | Near-orthogonal IC series; still check residual vs D1 |
| D1 ↔ D5 | −0.54 | Strong negative co-movement — D5 often flips relative to liquidity quality |
| TGD ↔ D5 | −0.34 | Signed opposition to temporal core |

**Takeaway:** High IC corr (TGD–D1) means *overlapping forecasts*, not automatic redundancy. Low/moderate IC corr (Flow–D1) does **not** guarantee new alpha — residualization can still wipe Flow.

---

## 3. Residual IC interpretation

| Y ⊥ X | Resid ICIR | Retention | Classification |
|-------|-----------:|----------:|----------------|
| TGD ⊥ Flow | 9.12 | 0.81 | TGD keeps almost all power → **independent_core** |
| D1 ⊥ TGD | 7.21 | 0.74 | D1 keeps most power → **independent_core** |
| TGD ⊥ D1 | 6.06 | 0.54 | TGD partially overlaps D1 but still strong → core with monitored overlap |
| D1 ⊥ Flow | 8.71 | 0.90 | D1 not a Flow wrapper |
| **Flow ⊥ D1** | **−0.62** | −0.13 | Flow predictive content **absorbed by D1** → satellite / liquidity repackaging |
| Flow ⊥ TGD | 1.68 | 0.35 | Weak leftover vs TGD alone |
| **D4 ⊥ D1** | **−0.55** | −0.10 | D4 absorbed by D1 → satellite enhancer, not core |
| D4 ⊥ TGD | 1.92 | 0.34 | Mostly redundant vs TGD |
| D5 ⊥ TGD | −7.09 | 0.74 | Large residual but **negative** book → pending direction check |
| D5 ⊥ Flow | −8.99 | 0.94 | Same signed story |

**Rule used here**

```
independent_core     := survives residualization vs other cores (|resid ICIR| large, retention high)
satellite_enhancer   := standalone looks fine; ⊥ primary liquidity/temporal core collapses
redundant            := absorbed and no clear overlay economic use (flag for drop/demote)
pending_validation   := residual looks “independent” but sign/pack/mono unresolved
```

---

## 4. Economic mechanism explanation

### TGD20 — temporal information (independent_core)

```
minute returns → Gu / Gd → residualize amplitude/open → εd ⊥ εu → MA20 → TGD20
```

- Hypothesis: **when** downside timing is abnormal (not raw τ = Gd−Gu).  
- Evidence: mechanism ladder rejected τ/υ; εd / TGD accepted.  
- vs D1: different data layer (minute timing vs EOD liquidity state). Residual survival confirms **not** a liquidity clone.

### D1_LiquidityQuality60d — liquidity / quality (independent_core)

```
low vol + liquidity stability (60d library) → quality score
```

- Hypothesis: stable, low-churn liquidity states earn a slower premium.  
- Evidence: soft-bar pass; execution Net Sharpe modest (~1.38 on **raw** book) but positive.  
- vs TGD: EOD quality vs intraday timing — correlated ranks, **orthogonal residual**.

### FlowDensity20 — liquidity-conditioned microstructure (satellite_enhancer)

```
net_active_flow / mktcap → MA20
+ mechanism: Amount / Flow⊥Amount flips edge
```

- Prior story (mechanism pack): interaction, **not** pure smart-money flow.  
- 1F: **Flow ⊥ D1 absorbed** — tradable ICIR largely liquidity-state content already in D1.  
- Reposition: test as **D1 overlay / enhancer**, not second core; **do not** default TGD+Flow composite.

### D4_WinnerSentimentReversal5d — behavioral short-horizon (satellite_enhancer)

```
winner sentiment / short-horizon reversal (library Base3 leg)
```

- Standalone ICIR exists on SI book, but **D4 ⊥ D1 absorbed**.  
- Role: possible **behavioral timing overlay on D1**, not independent inventory alpha.  
- Pack incomplete → keep as satellite until Template v2 + overlay tests.

### D5_UpsideFragility20d — fragility / tail (pending_validation)

```
upside fragility 20d (library Base3 leg)
```

- Residual vs TGD/Flow is large in magnitude but **negative ICIR** → likely **direction convention** (short high-fragility).  
- Mono historically weak; no v2 pack.  
- Status: `pending_validation` = direction fix + pack + soft-bar review before any core claim.

---

## 5. Classification table

| Factor | Class | Alpha role (verbal) | Registry status (unchanged) |
|--------|-------|---------------------|-----------------------------|
| **TGD20** | `independent_core` | core_temporal | validated |
| **D1_LiquidityQuality60d** | `independent_core` | core_liquidity | candidate |
| **FlowDensity20** | `satellite_enhancer` | liquidity_enhancer (vs D1) | candidate |
| **D4_WinnerSentimentReversal5d** | `satellite_enhancer` | behavioral_enhancer (vs D1) | candidate |
| **D5_UpsideFragility20d** | `pending_validation` | candidate_core_after_direction_fix | candidate |

No factor classified pure `redundant` for deletion — Flow/D4 retain **enhancer research value**. They are **redundant as cores**.

---

## 6. Final alpha topology

```
                    Alpha Portfolio (post Composite 2.0)

                          |
        ---------------------------------
        |                               |
    Alpha Source                  Portfolio Enhancer
        |                               |
   -------------                   FlowDensity20
   |           |                   (implementation /
  TGD20       D1                    conditioning —
  primary   independent             not new IC source)
            source

        Research Satellite: D4
        Pending:            D5
        Research:           IdealReversal
```

| Role | Factor | Evidence |
|------|--------|----------|
| Primary Alpha Source | TGD20 | RankICIR ≈ 11.3; high TO → Net drag alone |
| Independent Alpha Source | D1 | Resid ⊥ TGD; Net Sharpe B−A ≈ +1.0 via trading complementarity |
| Combination Enhancer | FlowDensity20 | C−B Net ≈ +0.60; resid ⊥ (TGD,D1) ICIR ≈ −1.74 → not Core |
| Research Satellite | D4 | Absorbed by D1; overlay research only |
| Pending | D5 | Direction + pack unfinished |

**Do not read Net Sharpe stack as “three cores.”**  
Composite maximizes Return / (Risk + Cost), not RankICIR.

**Effective portfolio engine (v1)**

```
TGD  → signal generation
D1   → signal stabilization (liquidity anchor)
Flow → implementation improvement (optional)
```

Not: TGD+D1+D4+D5+Flow kitchen-sink.

---

## 7. Composite policy — executed in Milestone 2.0

| Model | Spec | Result |
|-------|------|--------|
| A | TGD alone | Net ≈ 1.28 · TO high |
| B | TGD + D1 (IC-weighted) | Net ≈ 2.28 · **baseline** |
| C | TGD + D1 + Flow | Net ≈ 2.88 · enhancer OK, not Core |

See `docs/milestone_2_0_composite_alpha_v1.md`.

**Deprecated default:** TGD + Flow as primary pair (superseded by TGD + D1).

---

## 8. Explicit non-goals (this milestone)

- Composite engine / weights  
- Registry CSV/YAML schema edits (`alpha_role` field deferred)  
- Formula or direction flips on D5  
- New evaluations  

---

## 9. Next

```
1F Similarity Matrix     ✅
1F.5 Attribution Review  ✅
2.0 Composite Incremental ✅
        ↓
2.1 Production Stress Validation
  universe · cost · weight · calendar OOS
        ↓
2.2 Fundamental Alpha Layer (not more microstructure)
```
