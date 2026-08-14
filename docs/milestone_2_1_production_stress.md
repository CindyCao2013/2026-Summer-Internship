# Milestone 2.1 — Composite Production Stress Validation

**Date:** 2026-07-20  
**Status:** PASS with universe caveat  
**Script:** `run_composite_production_stress_v1.py`  
**Outputs:** `research/reports/composite_production_stress_v1/`

---

## Scope

Stress **Composite v1** (A/B/C) only. No Registry writes. No D4/D5. No new factors.

| Dimension | Grid |
|-----------|------|
| Universe | CSI300 / CSI500 / CSI1000 / ALL |
| Cost RT | 10 / 15 / 20 / 30 / 50 bp |
| Weights | static 50/50 · rolling IC 60/120 · vol-adj IC 60 |
| Calendar OOS | discovery →2022 · validation 2023–24 · test 2025 |

**Window:** 2020-01-02 → 2025-12-31 (1455d). Discovery starts at `cfg.START_DAY` (2020), not 2018.

**Alpha roles (locked — Net stack ≠ three cores):**

| Role | Factor |
|------|--------|
| Primary alpha source | TGD20 |
| Independent alpha source | D1 |
| Combination enhancer | FlowDensity20 |

---

## A. Universe — Net Sharpe (15bp · rolling IC 60)

| Universe | A | B | C |
|----------|--:|--:|--:|
| ALL | 1.38 | **2.16** | **2.62** |
| CSI1000 | 0.69 | **0.99** | **1.33** |
| CSI500 | −0.65 | −0.58 | −0.37 |
| CSI300 | −0.38 | −0.64 | −0.83 |

**Verdict:** Stack is **not mega-cap**. Works on ALL and CSI1000; **fails CSI300/500**.  
→ Production Track must target mid/small (CSI1000+) or ALL — not CSI300.

---

## B. Cost sensitivity (ALL)

| Cost bp | A | B | C |
|--------:|--:|--:|--:|
| 10 | 2.37 | 2.66 | 3.14 |
| 15 | 1.38 | 2.16 | 2.62 |
| 20 | 0.38 | 1.65 | 2.10 |
| 30 | −1.61 | **0.64** | **1.06** |
| 50 | −5.58 | −1.37 | −1.02 |

**Verdict:** D1 (and Flow) buy cost resilience vs TGD alone. B/C survive ~30bp; break at 50bp.  
TGD-alone is not production-viable above ~20bp on this book.

---

## C. Weight robustness (ALL · 15bp)

| Scheme | A | B | C |
|--------|--:|--:|--:|
| static_50_50 | 1.38 | 2.21 | 2.92 |
| rolling_ic_60 | 1.38 | 2.16 | 2.62 |
| rolling_ic_120 | 1.38 | 2.18 | 2.75 |
| vol_adj_ic_60 | 1.38 | 2.15 | 2.55 |

**Verdict:** Weight method is **not** the fragile link. B≈2.1–2.2, C≈2.5–2.9 across schemes.

---

## D. Calendar OOS (ALL · 15bp)

| Period | Window | A | B | C |
|--------|--------|--:|--:|--:|
| discovery | 2020–2022 | 1.64 | 2.16 | 2.54 |
| validation | 2023–2024 | 1.78 | 2.85 | 3.19 |
| test | 2025 | 0.13 | **0.92** | **1.95** |

**Verdict:** B and C remain **positive in 2025 test**. A nearly flat. Enhancer value of Flow is largest in test (ΔNet C−B ≈ +1.0).

---

## Overall

| Gate | Result |
|------|--------|
| B ≫ A on Net (trading complementarity) | ✅ |
| C enhancer without claiming Core | ✅ |
| Weight scheme robust | ✅ |
| Cost: survives 30bp, not 50bp | ✅ / caveat |
| CSI1000 positive | ✅ |
| CSI300/500 | ❌ size/liquidity regime limit |
| 2025 OOS B/C > 0 | ✅ |

**Production baseline candidate:** Model **B (TGD+D1)** on ALL / CSI1000.  
**Optional overlay:** Model **C** (Flow enhancer).  
**Do not** kitchen-sink D4/D5.  
**Do not** sell as CSI300 product.

---

## Artifacts

| File | Content |
|------|---------|
| `universe_stress.csv` | A/B/C × universe |
| `cost_stress.csv` | cost grid |
| `weight_stress.csv` | weight schemes |
| `period_stress.csv` | calendar OOS |
| `stress_report.md` | human report |
| `stress_verdict.json` | machine summary |
| `charts/*.png` | cost curve · universe bars |

---

## Next (Phase route)

```
2.1 Production Stress     ✅
        ↓
Optional: execution buffer / investability mask on B/C
        ↓
2.2 Fundamental Alpha Layer   ← information expansion (not more microstructure)
        ↓
2.3 Expand Composite (only after new independent source)
```
