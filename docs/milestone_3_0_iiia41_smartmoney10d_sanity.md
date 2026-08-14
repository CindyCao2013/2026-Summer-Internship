# III-A4.1 Phase 1.5 — SmartMoney10d Sanity Gate

**Date:** 2026-07-20  
**Status:** hard PASS · soft PASS  
**Script:** `run_milestone_3_0_smart_money10d_sanity.py`  
**Artifacts:** `research/reports/smart_money_v1/sanity/`

---

## Question

Is SmartMoney10d a **rankable low-amplitude factor**, or a **near-constant** with no CS information?

Smoke showed mean(Q)≈1 and ~87% within |Q−1|<0.01 — expected for a VWAP ratio, but insufficient alone.

---

## Window

| Item | Value |
|------|-------|
| Dates | 2024-06 (19d) |
| Names | 795 (800 subsample) |
| Signal | **raw Q** (no z-score book flip) |
| Sign flip | **false** |

---

## Gate results

### 1. Cross-sectional dispersion

| Metric | Value |
|--------|------:|
| mean daily σ(Q) | **0.0120** |
| median daily σ(Q) | 0.0120 |
| min / max σ | 0.0080 / 0.0163 |

→ Not near-constant (`≪1e-4`). Band ~0.001–0.01: **useful for ranking**.

### 2. Rank uniqueness

| Metric | Value |
|--------|------:|
| mean unique ranks | **792.1** / ~792 names |
| mean tie ratio | **0.0** |

→ Essentially no ties; fully rankable.

### 3. Raw direction (no inversion)

| Metric | Value |
|--------|------:|
| RankIC(Q, r_{t+1}) | **−0.0249** (18 days) |
| Paper expected | RankIC < 0 |
| Empirical | `negative_matches_paper` |

→ Short window; ICIR not trusted. Direction matches paper — **do not flip formula**.

---

## Verdict

```
near-constant?     NO
rankable?          YES
direction sanity?  matches paper (negative IC)
```

**Phase 1.5 PASS** → eligible for **Phase 2A** (CSI1000 scout), not yet 252d ALL / Registry.

---

## Next

```
Phase 2A: CSI1000 · 2023-01 → 2025-12 scout
  RankIC / ICIR / decile / H-L / turnover
  soft bar: |IC|>0.02, ICIR>2, mono acceptable

Phase 2B: only if 2A passes → longer ALL window
Phase 2C: pack + Registry testing
```

Still forbidden: formula change · Active_* · APM · sign-flip “fix” · Registry before pack.
