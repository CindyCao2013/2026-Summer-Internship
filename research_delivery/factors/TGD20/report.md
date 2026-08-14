# TGD20 — Factor Research Report

**Status:** `validated` · **Family:** temporal · **Source:** paper  
**Delivery card:** `research_delivery/factors/TGD20/`  
**Experiment provenance:** [`research/reports/factors/TGD20/`](../../../research/reports/factors/TGD20/)

---

## 1. Research Motivation

收益并非均匀分布于交易时间；上涨与下跌发生的**时刻结构**本身携带信息。研报从日内分钟收益的时序重心出发，构造可交易的时间结构 alpha。

---

## 2. Economic Hypothesis

Downside vs upside return **timing** (after return-structure controls) is predictive cross-sectionally. Late/informed participation in the day leaves a residual timing footprint that is not spanned by simple return or volatility.

---

## 3. Formula

\[
G_u=\frac{\sum_{r_t>0}t\cdot r_t}{\sum_{r_t>0}r_t},\quad
G_d=\frac{\sum_{r_t<0}t\cdot|r_t|}{\sum_{r_t<0}|r_t|}
\]

Residualize → \(\varepsilon_u,\varepsilon_d\). Frozen signal:

\[
\mathrm{TGD20}=\mathrm{MA}_{20}\big(\mathrm{CS\text{-}residual}(\varepsilon_d\mid\varepsilon_u)\big)
\]

**Not** raw \(G_d-G_u\). See [`formula.md`](formula.md).

---

## 4. Implementation

| Item | Value |
|------|-------|
| Data | Minute L2 → Gu/Gd lineage (DolphinDB) |
| Universe | ALL (research) |
| Period | 2022-01-28 → 2025-12-31 (951d; coverage exception) |
| Modules | `core/l2_features/return_timing.py`, `timing_residual.py`, `tgd.py`, `tgd_panel_builder.py` |
| Spec | `factor_specs/TGD20.yaml` |
| Signal shift | 1 |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC raw / SI | 0.0430 / **0.0415** |
| ICIR raw / SI | 6.98 / **11.29** |
| **Group10 excess Sharpe (exact universe EW), raw / SI** | 1.24 / **2.16** |
| Group10 excess annual return, SI | 8.81% |
| Group10 excess MDD, SI | −5.04% |
| Monotonicity | 0.988 |
| Yearly RankIC + | 6/6 |

### IC curve

![IC](plots/ic_curve.png)

### Quantile (Q1–Q10)

![Decile](plots/decile_return.png)

### Long-short cumulative

![Long-short](plots/cumulative_long_short.png)

### Yearly stability

![Yearly](plots/stability_yearly.png)

Source: Pack `ic_analysis/` · `quantile_analysis/` · `stability/`.

---

## 6. Portfolio and trading

| Metric | Value |
|--------|-------|
| **Headline: Group10 excess Sharpe (SI, exact ALL EW)** | **2.16** |
| H–L Net Sharpe SI @15bp | 1.72 (execution diagnostic) |
| Best optimized H–L Net Sharpe @15bp | 2.32 (execution diagnostic) |
| Daily turnover (best recipe) | 0.297 |

Excess benchmark is the daily mean return of every valid stock in the ALL
test universe, not the mean of decile returns.

![Turnover](plots/turnover.png)

---

## 7. Delivery verdict

**Library validated.** First complete paper-replication delivery card. Template for subsequent cards.
