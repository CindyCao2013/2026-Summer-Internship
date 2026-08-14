# P0 SUE Density — Useful Fail Report

**Status:** SEALED  
**Sample:** 2023-12-05 → 2025-12-31 (504 trading days)  
**Audience:** mentor / research review  
**Gate:** residual IC \(t \ge 2\) vs Base3 **and** stack ICIR uplift \(\ge 0\) at \(\lambda \le 0.3\) **and** turnover fence

---

## What was validated

Five SUE / earnings-event factors × two signal modes (event-hold 20d, daily exponential decay HL=5):

| Factor | Role |
|--------|------|
| `unexpected_profit_notice_surprise_20d` | Ace candidate — notice mid vs past 4Q NP |
| `sue_np_yoy_z` | YoY NP surprise on earliest-known date |
| `sue_eps_consensus` | Actual EPS vs pre-ann consensus |
| `analyst_np_revision_20d` | 20d consensus NP revision |
| `profit_notice_mid_surprise` | Notice mid vs YoY NP |

---

## Hard requirements (all met)

1. **Earliest known announcement date** = min(业绩预告, 业绩快报, 正式报告); NP units unified to **yuan**  
2. **Event-hold** and **daily-decay** both tested  
3. **Industry + ln(mktcap) neutralization** before residual IC vs Base3  
4. **`unexpected_profit_notice_surprise_20d` included**

---

## Result

| Factor | Best mode | ICIR | resid_t vs Base3 | Stack uplift | Verdict |
|--------|-----------|------|------------------|--------------|---------|
| `sue_eps_consensus` | decay | **3.71** | 1.76 | 0 | raw_signal_only (near miss) |
| `sue_np_yoy_z` | hold | 0.54 | 0.32 | 0 | raw_signal_only |
| `unexpected_profit_notice_surprise_20d` | decay* | 1.08* | ≈0 | 0 | drop — not independent |
| `analyst_np_revision_20d` | decay | 0.47 | 1.12 | 0 | drop |
| `profit_notice_mid_surprise` | hold | 0.03 | 0.24 | 0 | drop |

\*decay ICIR for notice surprise; hold residual shown in summary as near zero.

**All stack λ-grids prefer λ = 0.** No SUE factor improves Base3 ICIR.

**Gate outcome: ALL FAIL.** Do not promote any SUE factor into the library / production stack.

---

## Why (structural, not parametric)

1. **A-share digestion is fast.** Pre-announcements leak information; PEAD-style drift after formal report is thin at daily L/S horizon.  
2. **Size & industry absorb most of the signal.** Consensus coverage skews mid/large-cap; after size+industry residual, \(t\) collapses (e.g. consensus 3.71 ICIR → resid_t 1.76).  
3. **Overlap with Base3.** Remaining price behavior after earnings news is partly already in D1 (quality) / D4 (sentiment repair). Stack finds no incremental ICIR.

**Market takeaway:** In this sample, classic SUE is a **weak / non-independent** A-share factor once size, industry, and OHLCV Base3 are controlled — not a missing fourth pillar.

---

## What we reuse (process win)

| Asset | Path |
|-------|------|
| Earliest-known earnings loader | `sue_data.py` |
| Event hold / decay transforms | `factor_formulas_sue.py` |
| Density gate (size+ind → residual vs Base3 → stack λ) | `run_sue_density_v1.py` |
| Artifacts | `sue_density_summary.csv`, `sue_cross_dimension_independence.csv`, `sue_stack_lambda_grid.csv`, `sue_density_verdict.json` |

**Do not retune SUE windows or add analyst variants.** Marginal return is zero; the failure is structural.

---

## Implications for P1 / P2

- Any event factor must pass the **same** size+industry residual + Base3 stack gate.  
- Prefer **incentive / holder-trade / buyback** (behavior change, not PEAD).  
- P0 pipeline is ready: **P1 data source is plug-and-play** into the same density harness.

**Next:** P1 corporate events · P2 L2 net active flow (parallel). No further SUE work.
