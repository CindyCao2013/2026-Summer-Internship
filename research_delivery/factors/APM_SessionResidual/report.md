# APM_SessionResidual — Factor Research Report

**Status:** `testing_candidate` · **Family:** session · **Source:** paper_adapted  
**Delivery card:** `research_delivery/factors/APM_SessionResidual/`  
**Experiment provenance:** [`research/reports/factors/APM_SessionResidual/`](../../../research/reports/factors/APM_SessionResidual/) · scout [`apm_session_v1/`](../../../research/reports/apm_session_v1/)

---

## 1. Research Motivation

主动买卖研报强调隔夜与午后时段相对指数的残差 α；session 结构反映交易者构成与信息到达节奏。

---

## 2. Economic Hypothesis

Overnight vs PM residual α (then CS-residual vs Ret20) predicts next-day cross-section. **Adapted:** index PM uses EOD daytime proxy (no index minute in Stock_one_minute).

---

## 3. Formula

\[
\Delta\alpha=\alpha_{\mathrm{on}}-\alpha_{\mathrm{pm}},\quad
\mathrm{APM\_stat}=t\text{-stat}_{20}(\Delta\alpha),\quad
\mathrm{apm\_cs}=\mathrm{CS\text{-}residual}(\mathrm{APM\_stat}\mid\mathrm{Ret20})
\]

Positive direction; no sign flip; `shift(1)`.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Data | Minute stock PM + EOD index |
| Universe | CSI1000 |
| Period | 2021–2025 (1212d) |
| Modules | `apm_session_panel_builder.py`, `apm_session_signal.py` |
| Runners | `run_milestone_c1_apm_session_*.py` |

---

## 5. Basic Backtest

| Metric | Value |
|--------|-------|
| RankIC raw / SI | 0.0239 / **0.0225** |
| ICIR raw / SI | 4.10 / **6.55** |
| Monotonicity | 0.778 |
| Yearly RankIC + | 5/5 |

![IC](plots/ic_curve.png)

![Decile](plots/decile_return.png)

![Yearly](plots/stability_yearly.png)

*(Cumulative LS PNG not exported by scout runner — see Pack quantile CSV.)*

---

## 6. Trading

| Metric | Value |
|--------|-------|
| Frozen recipe | `highAPM\|daily\|buffer_10_30` |
| Best Net Sharpe @15bp | **1.50** |
| Daily turnover | 0.280 |
| Net daily plain | 0.92 |

![Turnover](plots/turnover.png)

---

## 7. Delivery verdict

**Testing candidate** (not validated / not Registry). Distinct from ActiveTradeProxy and SmartMoney.
