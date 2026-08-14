# SmartMoney10d — Factor Research Report (parked)

**Status:** `research` (parked) · **Family:** microstructure · **Source:** paper  
**Delivery card:** `research_delivery/factors/SmartMoney10d/`  
**Experiment provenance:** [`research/reports/smart_money_v1/`](../../../research/reports/smart_money_v1/)

---

## 1. Research Motivation

刻画“聪明资金效率 / 微观结构刀切”类信号；IC 强但日频换手下净收益难以存活。

---

## 2. Economic Hypothesis

Cross-sectional smart-money / efficiency measures predict returns, but **execution cost + turnover** may erase Net Sharpe under daily rebalance.

---

## 3. Formula

See research notes under `smart_money_v1/` (no Pack `formula.md` yet). Card documents **verdict**, not a production freeze.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Artifacts | `research/reports/smart_money_v1/phase2a/` |
| Pack v1 | **none** (intentionally not library-ingested) |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC raw | −0.0453 |
| ICIR raw (abs) | 6.09 |
| RankIC SI | −0.0365 |
| ICIR SI (abs) | 8.37 |

![Decile](plots/decile_return.png)

![Yearly](plots/stability_yearly.png)

*(Native ic_curve / cumulative LS PNGs not exported by runner.)*

---

## 6. Trading

| Metric | Value |
|--------|-------|
| Best Net Sharpe @15bp | **~0.31** (`lowQ\|every_5d\|buffer_10_30`) |
| Daily plain Net | negative (~−0.93) |

---

## 7. Delivery verdict

**Parked research card.** Contrast with APM (Net≈1.50): same broad “micro” theme, different investability. Not counted toward the 20 production-delivery goal until an investable recipe exists.
