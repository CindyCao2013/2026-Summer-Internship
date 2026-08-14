# Alpha Research Review Pack v1.0

**Tag (when git available):** `v1.0-research-freeze`  
**Freeze date:** 2026-07-10  
**Status:** Research Phase 3 complete — hand off to production validation

---

## One-page summary

### Goal
Discover independent OHLCV return mechanisms, freeze an Alpha Library, and produce a cost-aware but **ICIR-primary** combination for A-share long/short research.

### Method
```
Factor mining → Density / residual IC → Role assignment
→ Base3 freeze (D1+D4+D5) → Investability fences
→ Combination C1–C6 → ICIR re-rank → C6@0.60 head-to-head
```

### Core findings
1. **OHLCV independent return alphas = 3:** D1 liquidity-quality, D4 behavioral reversal, D5 upside fragility. D2 absorbed by D1; D3 = research satellite only.
2. **Base3 equal-weight** has strong pre-cost quality (gross Sharpe ~3.8) but fails the net-Sharpe fence under 15 bp + daily top/bottom 20% (high turnover).
3. **Research recommendation:** `C2_D1_0.60` = **0.60·D1 + 0.20·D4 + 0.20·D5**  
   - IC 5.48%, **ICIR 5.73**, gross Sharpe 1.98, TO 66%, net Sharpe 0.67 (feasible)
4. **C6 state_modifier** (cancel attenuates D4 on stress days) does **not** beat C2@0.60 on ICIR → **not upgraded**.
5. **Metric paradigm:** ICIR / gross Sharpe for research ranking; turnover ≤120% and net Sharpe >0 as feasibility fences only.

### Recommended production signal
```
signal = 0.60 * z(D1) + 0.20 * z(D4) + 0.20 * z(D5)
```
See `docs/production_stack_v3_design.md`. State_modifier default **OFF**.

### Production compromises (not research-optimal)
- `0.70 / 0.15 / 0.15` or `0.80 / 0.10 / 0.10` if execution needs lower turnover — document as cost compromise, not ICIR optimum.

---

## Pack contents

| Path | What |
|------|------|
| `alpha_library_v1.0-frozen.json` | Frozen library + investability + combination block |
| `frozen_candidate_pool_v1.json` | Role taxonomy + Base3 + enhancers |
| `alpha_combination_v1.json` | Combination verdict (ICIR-primary) |
| `tables/combination_results_research_rank.csv` | Full scheme table |
| `tables/c2_d1_tilt_icir_curve.csv` | D1 weight vs ICIR |
| `tables/c6_vs_c2_060_head_to_head.csv` | C6 decision evidence |
| `figures/` | Decile + IC plots for D1/D4/D5/C1/C2_0.60 + ICIR curve |
| `docs/production_stack_v3_design.md` | Production handoff (no code) |

---

## How to read the figures (30 seconds)

1. **D1 / D4 deciles** — textbook separation; top/bottom clean → linear return alphas.  
2. **D5 deciles** — weaker mono (tail mechanism); still independent residual IC.  
3. **C1 vs C2_0.60** — combination preserves mono; C2 improves ICIR vs equal-weight under cost fence.  
4. **c2_d1_weight_vs_icir.png** — why 0.60 wins on information efficiency (not net Sharpe).

---

## Consistency check (freeze)

| Field | Value |
|-------|-------|
| Recommended label | `C2_D1_0.60` |
| Weights | D1=0.6, D4=0.2, D5=0.2 |
| ICIR | 5.7325 |
| Gross Sharpe | 1.9830 |
| Annu one-way TO | 66.22 |
| Net Sharpe | 0.6724 |
| C6 upgrade | **false** |

---

## What not to do next

- Do not run more combination experiments without a new research question.
- Do not re-rank by net Sharpe alone.
- Do not put D2/D3 or inactive enhancers into production weights without a new study.

**Next owner:** production validation against `docs/production_stack_v3_design.md`.
