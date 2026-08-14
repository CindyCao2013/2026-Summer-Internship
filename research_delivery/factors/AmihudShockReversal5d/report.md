# AmihudShockReversal5d — Factor Research Report

**Status:** `testing_candidate` · **Tier:** A · **Family:** liquidity_shock  
**Delivery card:** `research_delivery/factors/AmihudShockReversal5d/`  
**Engine id:** `amihud_shock_reversal_5d`  
**Experiment provenance:** [`confirmation_1455d/amihud_shock_reversal_5d/`](../../../research/reports/d1_liquidity_density_v1/confirmation_1455d/amihud_shock_reversal_5d/)

> Promotion only — **no new factor logic**. Formula and plots come from existing EOD engine + D1 density confirmation artifacts.

---

## 1. Research Motivation

普通 Amihud 描述的是流动性**水平**溢价。本因子研究的是：

> 暂时的流动性压力（Amihud spike）叠加短期收益后，存在均值回复机会。

它回答的不是 “illiquid stocks earn premium”，而是 “liquidity **stress** fades”。

---

## 2. Economic Hypothesis

Temporary liquidity stress amplifies short-horizon over-/under-reaction; when stress is high, recent 5d return tends to reverse.  
Hence the interaction \(-\mathrm{Shock}\times R_{5d}\) is a **liquidity-shock reversal**, distinct from D1’s *stable* liquidity-quality mechanism.

Density evidence (existing study):

| Metric | Value | Source |
|--------|-------|--------|
| Corr with D1 rep | **0.0159** | `d1_liquidity_density_summary.csv` |
| IC after D1 | **0.0172** | same |
| Classification | `independent_alpha` | same |

---

## 3. Formula

See [`formula.md`](formula.md). Exact code freeze:

\[
\mathrm{ILLIQ}_t=\frac{|r_t|}{\mathrm{Amount}_t},\quad
\mathrm{Shock}_t=\frac{\mathrm{ILLIQ}_t}{\mathrm{MA}_{20}(\mathrm{ILLIQ})},\quad
\mathrm{ASR5}_t=-\mathrm{Shock}_t\cdot R_{t-5:t}
\]

**Not** plain Amihud; **not** \((\mathrm{ILLIQ}-\mathrm{MA})/\mathrm{STD}\) unless a new id is opened.

---

## 4. Implementation

| Item | Value |
|------|-------|
| Code | `factor_formulas_eod_engine.py::f_amihud_shock_reversal_5d` |
| Helpers | `_amihud_daily`, `_amihud_mean_20d` |
| Data | EOD `ret_1d`, `amount`, `ret_5d` |
| Runner / harvest | `run_d1_confirmation_reports.py --factors amihud_shock_reversal_5d` |
| Window | confirmation_1455d |
| Signal shift | evaluation pipeline standard (confirmation report) |

---

## 5. Basic Backtest (confirmation_1455d, ALL)

Source: confirmation `report/report.md` / `universe_stats.csv` — **not invented**.

| Metric | Value |
|--------|-------|
| RankIC mean | **0.0399** |
| ICIR | **4.36** |
| IC positive ratio | 0.615 |
| Monotonicity | **0.90** |
| HL Sharpe (gross) | **2.06** |
| HL ann. return | 41.4% |
| HL MDD | −25.6% |
| HL avg turnover | **2.55** (high) |

By universe (RankIC / ICIR / HL Sharpe):

| Universe | RankIC | ICIR | HL Sharpe |
|----------|--------|------|-----------|
| CSI300 | 0.023 | 1.96 | 0.35 |
| CSI500 | 0.028 | 2.69 | 0.71 |
| CSI1000 | 0.037 | 3.90 | 1.70 |
| ALL | 0.040 | 4.36 | 2.06 |

### IC curve

![IC](plots/ic_curve.png)

### Quantile (Q1–Q10)

![Decile](plots/decile_return.png)

### Long-short cumulative

![Long-short](plots/cumulative_long_short.png)

### IC decay

![IC decay](plots/ic_decay.png)

---

## 6. Trading

| Item | Status |
|------|--------|
| Gross HL Sharpe | 2.06 (confirmation) |
| Daily / HL turnover | **~2.55** — investability concern |
| Net Sharpe @15bp / buffer recipe | **not in artifacts** (do not invent) |
| Base3 enhancer role | `review_or_drop` in `base3_enhancer_verdicts.json` (additive uplift failed) |

**Interpretation:** Strong as **standalone** confirmation alpha and D1-orthogonal; **not** yet an execution-frozen Pack v1. High turnover must be addressed before validated / investable promotion.

---

## 7. Delivery verdict

| Question | Answer |
|----------|--------|
| Tier A mechanism? | **Yes** — liquidity shock reversal |
| Independent of D1? | **Yes** (corr≈0.02, residual IC≈1.7%) |
| Counts toward library? | **yes** (testing_candidate) |
| Validated / Pack v1? | **No** — needs execution recipe + optional Pack normalize |
| New code written? | **No** — promotion only |

**Next for this id:** execution / buffer study (or holding-period lengthening) using existing signal — still no formula change.
