# IdealAmplitude — Factor Research Report

**Status:** `testing` · **Family:** behavior · **Source:** paper  
**Delivery card:** `research_delivery/factors/IdealAmplitude/`  
**Experiment provenance:** [`research/reports/factors/IdealAmplitude/`](../../../research/reports/factors/IdealAmplitude/)

---

## 1. Research Motivation

振幅因子隐藏结构：按收盘价状态切割后，高收盘状态振幅携带更强负向 alpha。

---

## 2. Economic Hypothesis

High close-state amplitude embeds fragile upside / overreaction; spread \(V=V_{\mathrm{high}}-V_{\mathrm{low}}\) purifies vs raw Amp20.

---

## 3. Formula

\[
\mathrm{Amp}=\mathrm{high}/\mathrm{low}-1,\quad
V=\mathrm{mean}(\mathrm{amp}\mid\mathrm{high\text{-}close})-\mathrm{mean}(\mathrm{amp}\mid\mathrm{low\text{-}close})
\]

Direction: **negative** RankIC.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Data | EOD |
| Universe | ALL |
| Period | long harvest + exec last 252d |
| Modules | `factor_cutting/ideal_amplitude.py` |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC SI | −0.0475 |
| ICIR SI | **−9.97** |
| Monotonicity | **0.111** (very weak) |
| Yearly RankIC − | 16/16 |

![IC](plots/ic_curve.png)

![Decile](plots/decile_return.png)

![Long-short](plots/cumulative_long_short.png)

![Yearly](plots/stability_yearly.png)

---

## 6. Trading

| Metric | Value |
|--------|-------|
| Best Net Sharpe @15bp | **3.40** |
| Daily turnover | 0.446 |

![Turnover](plots/turnover.png)

---

## 7. Delivery verdict

**Testing only** — Net Sharpe high but mono fails. Do not promote; related to IdealReversal family.
