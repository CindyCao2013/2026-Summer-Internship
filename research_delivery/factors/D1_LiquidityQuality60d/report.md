# D1_LiquidityQuality60d — Factor Research Report

**Status:** `candidate` · **Family:** liquidity · **Source:** internal  
**Delivery card:** `research_delivery/factors/D1_LiquidityQuality60d/`  
**Experiment provenance:** [`research/reports/factors/D1_LiquidityQuality60d/`](../../../research/reports/factors/D1_LiquidityQuality60d/)

---

## 1. Research Motivation

低波动与稳定成交参与度在横截面上常伴随流动性溢价；D1 将二者合成为可交易的流动性质量信号。

---

## 2. Economic Hypothesis

Stocks with **low volatility** and **stable liquidity participation** earn a cross-sectional premium beyond raw size/liquidity levels.

---

## 3. Formula

\[
\mathrm{D1}=\mathrm{CS\text{-}rank\text{-}mean}\big(-\mathrm{vol}_{60d},\;-\mathrm{amount\_cv}_{20d}\big)
\]

`shift(1)`. See Pack `formula.md`.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Data | EOD OHLCV |
| Universe | ALL |
| Period | confirmation_1455d harvest |
| Modules | `factor_formulas_liquidity_d1.py`, `factor_formulas_eod_engine.py` |
| Spec | `factor_specs/D1_LiquidityQuality60d.yaml` |
| Signal shift | 1 |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC raw | **0.0573** |
| ICIR raw | **6.01** |
| SI RankIC / ICIR | pending protocol |
| Monotonicity | 0.8 |

![IC](plots/ic_curve.png)

![Decile](plots/decile_return.png)

![Long-short](plots/cumulative_long_short.png)

![Yearly](plots/stability_yearly.png)

---

## 6. Trading

| Metric | Value |
|--------|-------|
| Best Net Sharpe @15bp | **1.38** |
| Daily turnover | 0.234 |

![Turnover](plots/turnover.png)

---

## 7. Delivery verdict

**Candidate delivery card.** Independent liquidity family vs TGD/Flow/APM. SI neutralization pending — do not claim SI metrics yet.
