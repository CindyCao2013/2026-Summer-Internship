# P1 Event Density — Major Holder / Insider

**Sample (discovery):** 2023-12-05 → 2025-12-31 (504d)  
**Gate:** size+industry neutralized → residual IC \(t\ge2\) vs Base3 → stack ICIR uplift

## Data

| Source | Status |
|--------|--------|
| `ASHAREMJRHOLDERTRADE` | OK (~100k rows in window) |
| `ASHAREINSIDERTRADE` | OK (~83k rows) |
| `ASHARESTOCKINCENTIVEIMPLEMENT` | **BLOCKED** — only ~444 rows in Wind feed |

## Results (size+industry neutralized)

| Factor | Mode | ICIR | resid_t vs Base3 | Stack uplift | Notes |
|--------|------|------|------------------|--------------|-------|
| `major_holder_net_increase` | **hold** | **2.93** | **3.63** | 0 | Independent residual; stack λ*=0 |
| `major_holder_net_increase` | decay | 2.29 | 2.29 | 0 | Same story, weaker |
| `major_holder_increase_only` | hold | −1.08 | −0.97 | 0 | Sign/coverage weak alone |
| `insider_net_buy` | hold | 0.50 | 1.41 | 0 | Below residual gate |

## Long-term observation (research satellite)

**`major_holder_net_increase` (hold 60d)** is on the **long-term observation list**, not production weight.

| Item | Value |
|------|-------|
| Discovery residual_t vs Base3 | **3.63** (clears independence) |
| Discovery turnover (ann proxy) | **~21%** (very low) |
| Stack ICIR uplift | **0** (best λ = 0) |
| Current verdict | **研究卫星** — independent but does not lift the daily Base3 blend |

**Why stack uplift is 0 despite independence:** signal frequency is ~60d event-hold; Base3 is daily. Orthogonal information at a mismatched horizon often fails to raise equal-weight daily ICIR.

**Re-evaluate when:** building a weekly / multi-horizon combination layer where a low-frequency enhancer can sit on a separate clock. Until then: observe on confirmation windows; do **not** add to production weights.

## Interpretation

- Net (增持−减持) beats increase-only — selling pressure from 减持 carries signal.
- Equity incentive deferred until vendor table is complete.
- Do not expand to 100 event dummies.

## Next (deferred)

- Optional confirmation window for satellite tracking only (not blocking P2)
- Optional: industry-conditional or float-adjusted ratio variants (≤3 formulas)
