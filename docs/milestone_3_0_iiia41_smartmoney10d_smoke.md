# III-A4.1 Phase 1 — SmartMoney10d Smoke

**Date:** 2026-07-20  
**Status:** PASS  
**Script:** `run_milestone_3_0_smart_money10d_smoke.py`  
**Report:** `research/reports/smart_money_v1/smoke/smoke_report.json`

---

## Paper window

| Check | Result |
|-------|--------|
| Option A (single-day) | rejected |
| Option B (rolling 10 trading days) | **confirmed** (Kaiyuan 步骤1) |
| β | **0.25** (locked Stage-0; not original 0.5; not blog −0.5) |

---

## Smoke (2024-06 + May preheat)

| Layer | Result |
|-------|--------|
| L1 `minute_raw` | 46.9M rows; cols = symbol/date/bartime/close/volume/amount only |
| L2 `minute_feature` | + `ret_1m`, `smart_score`; **no Active_*** |
| L3 panel (400-symbol subsample) | 19×399; coverage_cell ≈ **0.998** |
| Q distribution | mean≈1.000, p50≈0.999, frac\|Q−1\|<0.01 ≈ **0.87** (near-1 mass expected) |

Unit tests: `core/l2_features/test_smart_money.py` — 5 passed.

---

## Gates

All smoke gates **PASS**. No Registry. No 252d pack yet.

---

## Next (Phase 2 — needs explicit go)

1. Full-universe L3 for eval window (or month-chunk Q in DDB if Python too slow)  
2. 252d RankIC / direction / turnover / execution  
3. Pack with **information efficiency** mechanism narrative  
4. Registry only after pack review  

Still forbidden: APM · Active_* · ActiveTradeProxy rename · production status.
