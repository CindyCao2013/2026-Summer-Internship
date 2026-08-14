# FlowDensity20 — Factor Research Report

**Status:** `candidate` · **Family:** flow · **Source:** paper/internal  
**Delivery card:** `research_delivery/factors/FlowDensity20/`  
**Experiment provenance:** [`research/reports/factors/FlowDensity20/`](../../../research/reports/factors/FlowDensity20/)

---

## 1. Research Motivation

主动资金净流入相对市值的密度刻画“资金到达强度”，而非单纯聪明钱方向标签。

---

## 2. Economic Hypothesis

\(\mathrm{net\_active\_flow}/\mathrm{mktcap}\) smoothed over 20d predicts returns as a **Flow × Liquidity** interaction. Amount-orthogonal tests can flip sign — not pure smart-money.

---

## 3. Formula

\[
\mathrm{FlowDensity20}=\mathrm{cs\_zscore}\Big(\mathrm{MA}_{20}\sum(\mathrm{net\_active\_flow}/\mathrm{mktcap})\Big)
\]

`shift(1)`.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Data | L2 active flow |
| Universe | ALL |
| Period | 2022-01-28 → 2025-12-31 (951d) |
| Modules | `factor_formulas_l2_flow_p2.py`, `l2_data_loaders.py` |
| Spec | `factor_specs/FlowDensity20.yaml` |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC raw / SI | 0.0178 / **0.0236** |
| ICIR raw / SI | 2.07 / **4.85** |
| **Group10 excess Sharpe (exact universe EW), raw / SI** | −0.05 / **0.50** |
| Group10 excess annual return, SI | 3.98% |
| Group10 excess MDD, SI | −16.77% |
| Yearly RankIC + | 4/4 |

![IC](plots/ic_curve.png)

![Decile](plots/decile_return.png)

![Long-short](plots/cumulative_long_short.png)

![Yearly](plots/stability_yearly.png)

---

## 6. Portfolio and trading

| Metric | Value |
|--------|-------|
| **Headline: Group10 excess Sharpe (SI, exact ALL EW)** | **0.50** |
| H–L Net Sharpe SI @15bp | 1.85 (execution diagnostic) |
| Best optimized H–L Net Sharpe @15bp | 2.88 (execution diagnostic) |
| Daily turnover | 0.165 |

The raw Group10 has no market-relative alpha (excess Sharpe ≈ −0.05).
The positive result depends on size/industry neutralization. This materially
weakens the earlier claim that raw FlowDensity20 is one of the two strongest
standalone long-book factors.

![Turnover](plots/turnover.png)

---

## 7. Delivery verdict

**Candidate, not validated.** Predictive IC and H–L diagnostics are strong, but
the exact long-book market-relative Sharpe is only 0.50 after neutralization.
Keep orth/amount mechanism caution in any future combination.
