# III-A4.1 Phase 2A — SmartMoney10d CSI1000 Scout

**Date:** 2026-07-20  
**Status:** Scout complete — **IC PASS · investability FAIL at daily 15bp**  
**Script:** `run_milestone_3_0_smart_money10d_phase2a.py`  
**Artifacts:** `research/reports/smart_money_v1/phase2a/`  
**Identity:** raw \(Q\), Option B, β=0.25 — **no sign flip · no Registry · no formula change**

---

## Setup

| Item | Value |
|------|-------|
| Universe | CSI1000 |
| Period | 2023-01-01 → 2025-12-31 (727d) |
| Signal | raw `SmartMoney10d` Q |
| H-L book | **long low-Q / short high-Q** (economic direction; IC stays on raw Q) |
| Cost | 15bp RT |
| Neutralization | raw + size+industry |

---

## Headline results

### IC (raw Q — no flip)

| Mode | RankIC | \|ICIR\| | IC>0 frac | n |
|------|-------:|--------:|----------:|--:|
| raw | **−4.53%** | **6.09** | 0.32 | 726 |
| size+industry | **−3.65%** | **8.37** | 0.29 | 726 |

Paper direction confirmed every year (RankIC < 0). Soft bar \|IC\|>2% and \|ICIR\|>1.5: **PASS**.

### Portfolio (lowQ − highQ)

| Mode | Gross Sharpe | Net Sharpe @15bp | Daily TO |
|------|-------------:|-----------------:|--------:|
| raw | **1.68** | **−0.93** | **1.14** |
| SI z | **2.31** | **−1.33** | **1.11** |

Decile low−high mean daily ≈ **+11bp** (gross edge exists).

### Yearly stability

| Year | RankIC | \|ICIR\| | Gross Sharpe | Net Sharpe | year_works* |
|------|-------:|--------:|-------------:|-----------:|:-----------:|
| 2023 | −4.1% | 7.7 | 0.75 | −2.92 | no |
| 2024 | −4.1% | 4.9 | 1.78 | −0.24 | no |
| 2025 | −5.4% | 6.5 | 2.45 | −0.46 | no |

\*Coded gate required net Sharpe>0 **and** RankIC<0 → 0/3.  
If “year works” = RankIC matches paper: **3/3**.  
If = gross Sharpe>0: **3/3**. Failure is **cost × turnover**, not alpha death.

### Peer IC-series correlation (not residual)

| vs | corr(IC_SM, IC_peer) |
|----|---------------------:|
| TGD20 | −0.29 |
| D1 | −0.52 |
| FlowDensity20 | −0.13 |

Not a Flow clone. Moderate IC-timing overlap with D1 — residual IC deferred to Similarity v2.

---

## Soft-bar table (as pre-agreed)

| Metric | Threshold | Result |
|--------|-----------|--------|
| \|RankIC\| | >2% | ✅ 4.5% |
| \|ICIR\| | >1.5 | ✅ 6.1 |
| Years stable | ≥2/3 “positive” | ⚠️ IC yes / **net no** |
| H-L Net Sharpe | >1 | ❌ −0.93 |
| Turnover | explainable | ✅ explained: **~full daily book** |

**Coded verdict:** `FAIL_scout` (net investability).  
**Research verdict:** **mechanism + CS alpha PASS**; **daily rebalance at 15bp does not clear cost**.

---

## Interpretation (do not retune formula)

```
SmartMoney10d
  = strong negative RankIC / ICIR
  + gross H-L works
  + turnover ≈ 1.1/day
  → net destroyed at 15bp RT
```

Same class of problem as many minute microstructure factors: **information is real; capacity/holding is the question**.

**Do not:**
- flip to −Q for IC cosmetics  
- invent Active_* knife  
- pack / Registry as production-ready  

**Do (Phase 2A.1 / execution, optional):**
- holding 2–5d / buffer / lower top-frac  
- cost grid  
- then decide pack as `testing` with high-TO note  

---

## Cache / eng notes

- L2: `research/cache/smart_money/minute_feature/smart_score_YYYYMM.parquet` (2022-12→2025-12)  
- Panel: `.../factor_panel/SmartMoney10d_CSI1000_20230101_20251231.parquet`  
- Prefetch uses `ensure_minute_feature_months` (no full-history RAM concat)

---

## Next

```
NOT yet: Phase2B ALL / Phase2C Registry pack

Options:
  A) Phase2A.1 execution holding/turnover diagnosis (preferred before pack)
  B) Admit research note only — IC layer validated, investability open
  C) Move to III-A4.2 APM / III-B SUE while SmartMoney sits as "IC-proven, TO-open"
```
