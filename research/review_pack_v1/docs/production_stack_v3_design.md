# Production Stack v3 — Design Spec (Research Freeze)

**Status:** design-only (no production code in this document)  
**Date:** 2026-07-13 (P2/C7 update)  
**Inputs:** `alpha_library_v1.0-frozen.json`, `alpha_combination_v1.json`, `l2_flow_density_v1/combination_c7/`  
**Audience:** production / execution engineers implementing daily signal generation

---

## 1. Purpose

Define the **single daily composite alpha** that production may implement after Alpha Research Phase 3 freeze.

Research recommendation (ICIR-primary), **with confirmed P2 enhancer (C7, 2026-07-13)**:

```
c2_t = 0.60 * z(D1)_t + 0.20 * z(D4)_t + 0.20 * z(D5)_t
raw_t = 0.70 * z(c2)_t + 0.30 * z(net_active_flow_mktcap_20d)_t
signal_t = z( neutralize_size_industry(raw_t) )
```

where `z(·)` is **cross-sectional z-score** on the tradeable universe at date `t`, and
`neutralize_size_industry` = CITICS industry demean + residualize vs ln(float mktcap).

Base-only fallback (no L2 / no size residual):

```
signal_t = 0.60 * z(D1)_t + 0.20 * z(D4)_t + 0.20 * z(D5)_t
```

---

## 2. Base sources (frozen)


| Slot | Factor                          | Weight   | Role                               |
| ---- | ------------------------------- | -------- | ---------------------------------- |
| D1   | `low_vol_liquidity_quality_60d` | **0.60** | Stable liquidity / low-vol quality |
| D4   | `winner_sentiment_reversal_5d`  | **0.20** | Behavioral reversal                |
| D5   | `upside_fragility_20d`          | **0.20** | Tail / upside fragility            |


**Dropped from base:** D2 `volatility_60d` (absorbed by D1), D3 `lower_shadow_support_20d` (research satellite only).

**Production alternatives (not research-optimal):**


| Blend      | Weights            | Intent                     |
| ---------- | ------------------ | -------------------------- |
| C2_D1_0.70 | 0.70 / 0.15 / 0.15 | Lower turnover compromise  |
| C2_D1_0.80 | 0.80 / 0.10 / 0.10 | Lowest-turnover compromise |


Research team recommendation remains **0.60 / 0.20 / 0.20**. Production may choose a compromise only with explicit cost/capacity justification.

---

## 3. Daily signal construction

1. Load EOD panels for D1, D4, D5 (same formulas as research freeze).
2. Apply tradability mask (see §5) → NaN non-tradable names.
3. Cross-sectional z-score each panel (ignore NaNs).
4. Form weighted sum with fixed weights above.
5. Optional: apply state modifier (§4) — **off by default** until production re-validates.
6. Build long/short book from composite signal (§6).

**Timing:** signal computed after close on day `t`; returns attributed to day `t+1` (standard 1-day shift).

**Rebalance frequency:** daily (research default). Lower frequency is a production optimization, not part of research freeze.

---

## 4. Optional enhancer: `cn_cancel_shock` as state_modifier

**Not additive in production default.** Additive λ=0.2 raises annual one-way turnover above the 120% research fence.

### Spec (research C6; default OFF)

- Compute cancel cross-sectional mean of z-scored `cn_cancel_shock`.
- **Stress day** if that mean is in the bottom 20% of the confirmation-period distribution (or rolling equivalent in live).
- On stress days only:
  - D4 weight: `0.20 → 0.10`
  - Redistribute `0.05` to D1 and `0.05` to D5  
  - Effective: `{D1: 0.65, D4: 0.10, D5: 0.25}` on stress days
- On non-stress days: keep `{0.60, 0.20, 0.20}`

### Research decision (2026-07-10 head-to-head)

Artifact: `research/results/alpha_combination_v1/c6_vs_c2_060_decision.json`


| Scheme               | ICIR      | Gross Sharpe | TO   | Net Sharpe |
| -------------------- | --------- | ------------ | ---- | ---------- |
| C2_D1_0.60           | **5.733** | 1.983        | 66.2 | 0.672      |
| C6_D1_0.60_state_mod | 5.571     | 1.896        | 67.5 | 0.555      |


**Decision: do NOT upgrade.** Keep state_modifier **disabled** in production default. C6 does not improve ICIR at w1=0.60.

---

## 4b. Confirmed L2 enhancer: `net_active_flow_mktcap_20d` (P2) — C7

**Status (2026-07-13):** confirmation + investability passed; **C7 combo gate passed**.  
Artifacts: `research/reports/l2_flow_density_v1/combination_c7/`

### Construction (research recommended)

```
raw_t = (1-λ) * z(C2_D1_0.60)_t + λ * z(P2)_t
signal_t = z( neutralize_size_industry(raw_t) )   # ALL book + tight size/ind residual
```

with **λ = 0.3** (ICIR-max among size|exposure|≤0.2σ feasible points).

| Scheme | ICIR | Net Sharpe | Annu TO 1-way | size \|μ\| | Notes |
|--------|------|------------|---------------|-----------|-------|
| C2_D1_0.60 (baseline) | 5.73 | 0.67 | 66.2 | 0.60 | frozen Base3 |
| C7 additive λ=0.5 | 6.94 | 1.74 | 57.5 | 0.25 | clean additive uplift; size soft-miss |
| **C7 size_tight λ=0.3** | **9.63** | **2.25** | **62.5** | **0.07** | **recommended** |

Universe: ALL (tradability masks). CSI300/500 standalone IC of P2 is negative — do not deploy as large-cap-only book.

**vs cancel:** Prefer P2 as additive/linear enhancer. Keep `cn_cancel_shock` OFF as additive; optional state_modifier remains research-only.

**Soft fence:** aspirational annu TO ≤50% not met (62.5%); hard fence ≤100% passes.

---

## 5. Tradability filters (mandatory)

Exclude from signal / book:

- Limit-up names when going long; limit-down when going short (`get_EOD_Not_Limit`)
- ST / *ST (`get_EOD_Not_ST`)
- Suspended / non-trading (`get_TradeStatus`)
- IPO seasoning: fewer than 60 trading days of valid closes (research proxy)

Report daily **coverage** = tradable / candidates.

---

## 6. Portfolio construction (research-aligned)


| Parameter                   | Value                                                         |
| --------------------------- | ------------------------------------------------------------- |
| Long                        | top 20% by signal rank (equal weight)                         |
| Short                       | bottom 20% by signal rank (equal weight)                      |
| Neutral                     | middle 60% not held                                           |
| Cost model (research fence) | 15 bp round-trip on combined long+short L1 turnover           |
| Net Sharpe role             | production feasibility gate only (must be >0 on confirmation) |


---

## 7. Research metrics to monitor in production

**Primary (signal quality):**

- Daily rank IC / ICIR of composite vs next-day return
- Decile monotonicity / H-L gross Sharpe

**Feasibility fence:**

- Annualized one-way turnover ≤ 120% (research gate)
- Net Sharpe after 15 bp > 0

**Do not** re-optimize weights on net Sharpe alone in research; that was corrected in Phase 3.

---

## 8. Explicit non-goals

- Do not reintroduce D2/D3 into production weights.
- Do not use inactive enhancers (`amihud_shock`, `value_composite`, `quality_composite`) without a new confirmation study.
- Do not treat research satellites as production alpha weights.
- Do not chase single-factor gross Sharpe > 3 as a gate.

---

## 9. Handoff checklist

- [ ] Implement §3 with frozen factor formulas + §4b P2 enhancer (λ=0.3, size_tight)
- [ ] Wire §5 masks
- [ ] Match confirmation metrics within tolerance (C2 ICIR ~5.73; C7 size_tight λ0.3 ICIR ~9.63 / net ~2.25)
- [ ] Decide ON/OFF for §4 cancel using `c6_vs_c2_060_decision.json` (default OFF)
- [ ] Log daily coverage, turnover, IC, size exposure for monitoring

**Artifact pointers**

- Library: `research/alpha_library_v1/alpha_library_v1.0-frozen.json`
- Combination C1–C6: `research/results/alpha_combination_v1/alpha_combination_v1.json`
- Combination C7 (P2): `research/reports/l2_flow_density_v1/combination_c7/`
- Figures: `research/alpha_library_v1/figures/`
- Review pack: `research/review_pack_v1/`

