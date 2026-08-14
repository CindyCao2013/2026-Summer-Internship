# IdealReversal — Factor Research Report

**Status:** `testing` · **Family:** behavior · **Source:** paper (cutting)  
**Delivery card:** `research_delivery/factors/IdealReversal/`  
**Experiment provenance:** [`research/reports/factors/IdealReversal/`](../../../research/reports/factors/IdealReversal/)

---

## 1. Research Motivation

A股反转并非同质：按平均单笔成交金额（ATS）切割后，高 ATS 日的反转更强，低 ATS 日更接近噪声。

---

## 2. Economic Hypothesis

Institutional-sized trades concentrate reversible overreaction; cutting Ret20 by ATS isolates a cleaner reversal alpha.

---

## 3. Formula

\[
M=\sum(r\mid\mathrm{high\,ATS})-\sum(r\mid\mathrm{low\,ATS})
\]

Direction: **negative** RankIC (short high \(M\)). Window 20d.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Data | EOD |
| Universe | ALL |
| Period | cutting harvest (~1703d) |
| Modules | `factor_cutting/ideal_reversal.py`, `w_cut.py` |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC SI | −0.0331 |
| ICIR SI | **−9.46** |
| Monotonicity | **0.444** (soft bar fail) |
| Yearly RankIC − | 7/7 |

![IC](plots/ic_curve.png)

![Decile](plots/decile_return.png)

![Long-short](plots/cumulative_long_short.png)

![Yearly](plots/stability_yearly.png)

---

## 6. Trading

| Metric | Value |
|--------|-------|
| Best Net Sharpe @15bp | 1.70 |
| Daily turnover | 0.153 |

![Turnover](plots/turnover.png)

---

## 7. Delivery verdict

**Testing only** — strong ICIR, weak mono. Same cutting family as IdealAmplitude; not an independent library slot without residual analysis.
