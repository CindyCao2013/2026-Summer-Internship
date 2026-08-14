# III-A4.3 / Phase 2A.1 — SmartMoney Horizon + Turnover Diagnosis

**Date:** 2026-07-20  
**Status:** complete — recommend **`research_candidate`** (mechanism validated; not daily production)  
**Script:** `run_milestone_3_0_smart_money10d_horizon.py`  
**Artifacts:** `research/reports/smart_money_v1/phase2a1_horizon/`  
**Panel:** frozen CSI1000 Phase2A Q (2023–2025)

---

## Question

Is SmartMoney “dead at H>1”, or is the Phase2A failure **horizon mismatch** (daily rebalance vs slower alpha)?

---

## A/C — IC decay curve

RankIC\((Q_t,\sum_{k=1}^{H} r_{t+k})\):

| H | RankIC | \|ICIR\| |
|--:|-------:|---------:|
| 1 | −4.53% | 6.09 |
| 3 | −4.95% | 6.74 |
| 5 | −5.12% | 6.82 |
| 10 | −5.49% | 7.36 |
| 20 | −5.75% | 7.84 |

**Finding:** IC does **not** die after day 1. It **strengthens** through H=20.

→ Half-life proxy: **none within 20d** (never falls to ½ of \|IC₁\|).  
→ Class: **`medium_or_longer_persistent`** (cumulative multi-day information), **not** ultra-short daily-only alpha.

---

## A — Holding proxies (long low-Q)

Overlapping H-return / H turnover proxy + non-overlap rebalance every H:

| H | Gross~ | Net proxy~ | Net non-overlap | TO proxy |
|--:|-------:|-----------:|----------------:|---------:|
| 1 | 1.67 | −0.95 | −0.95 | 1.14 |
| 5 | 2.82 | 1.55 | 0.78 | 0.23 |
| 10 | 3.70 | 2.80 | 0.32 | 0.11 |
| 20 | 4.43 | 3.75 | 0.21 | 0.06 |

Overlapping proxies look strong at H≥5; non-overlap is weaker (sample / path). Treat as **directional** evidence that slowing the book helps, not as production Sharpe.

---

## B — Execution grid (authoritative for investability)

`evaluate_execution` on long-lowQ book @15bp:

| Scheme | Net Sharpe | Daily TO |
|--------|----------:|---------:|
| daily plain | −0.93 | 1.14 |
| every_5d | −0.10 | 0.55 |
| every_10d | −0.16 | 0.33 |
| daily \| buffer_5_15 | −0.07 | 0.80 |
| **every_5d \| buffer_10_30** | **+0.31** | **0.35** |
| every_5d \| buffer_5_15 | +0.28 | 0.47 |

**Best net ≈ 0.31** — improves vs daily, **does not** clear a comfortable production bar (e.g. Net>1 or even >0.5 soft).

---

## Synthesis

```
Phase2A:     IC strong · daily TO kills net
Phase2A.1:   IC persists/compounds to H=20
             slowing + buffer recovers net to ~0.3
             still not production-grade
```

| Gate | Status |
|------|--------|
| Mechanism / formula | PASS |
| IC / direction | PASS |
| Horizon understanding | PASS — persistent, not 1-day flash |
| Daily production | FAIL |
| Slowed execution | PARTIAL (net~0.3) |
| Registry / pack | **NOT YET** |

**Recommended status:** `research_candidate` (**PARKED** — III-A closed)  
(= mechanism validated + alpha exists; investability open / not daily)

Checkpoint: [`docs/checkpoint_2026_07_iiia_complete.md`](checkpoint_2026_07_iiia_complete.md)  
Mainline next: **III-B SUE** — [`docs/milestone_3_0_iiib_fundamental_entry.md`](milestone_3_0_iiib_fundamental_entry.md)

---

## What not to do

- ❌ Formula retune / Active_*  
- ❌ Registry / Composite  
- ❌ Call Phase2A a “failed factor”

## What to do next

1. **Park SmartMoney** as research_candidate; optional later testing pack with documented `every_5d|buffer_*` recipe if you want library completeness.  
2. Or proceed **III-A4.2 APM / III-B SUE** as mainline — SmartMoney already answered “is there an efficiency layer?” → **yes**.

---

## One-line verdict

> SmartMoney10d is a real, persistent microstructure alpha whose natural book is slower than daily; Phase2A net failure was horizon/turnover mismatch, not absence of signal.
