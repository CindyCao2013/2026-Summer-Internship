# Factor Cutting — Internal Notes (v2.1+)

Date: 2026-07-13  
See also: `research/reports/factor_cutting_v1/VALIDATION_SUMMARY.md`

## Board

| Module | Status |
|--------|--------|
| Daily information loop (IC/mech/viz/neut/proxy/family/limit) | ✅ closed |
| Operability (long excess / TO / size) | 🟡 measured — TO too high |
| Production / long-only ready | ❌ not yet |
| Minute layer | ⏸ after TO engineering |

## Operability (2018–2025 DDB) — binding

| Factor | Long excess raw → filter | Month-end long TO |
|--------|--------------------------|-------------------|
| ATS | 1.4% → 8.0% | ~88% |
| amount+ATS | 3.8% → 8.8% | ~89% |
| ideal_amplitude | 9.3% → 11.6% | ~86% |

- Raw dual/ATS: alpha mostly short-side; long-only weak without limit mask  
- `filter_signal` mandatory for long-biased books  
- Size pctile ~0.43 (not pure microcap)  
- Best long-biased today: **amplitude + not-limit**

## Next

1. Turnover engineering (month-end / band; fee-net)  
2. Optional: wire operability into v21 default gate  
3. Minute only after TO is acceptable
