# Factor Operability Report

**Period:** 2018-01-02 → 2025-12-31 · source=`ddb`  
IC/ICIR = information purity. Below = **tradability** (daily rebalance + month-end TO check).

## Headline table

| Factor | Mode | RankIC | ICIR | Long excess ann | Long excess Sharpe | HL ann | HL Sharpe | HL MDD | Daily TO long | Month-end long TO (1-way) | Long size pctile | Frac size≤20% |
|--------|------|--------|------|-----------------|--------------------|--------|-----------|--------|---------------|---------------------------|------------------|---------------|
| `ats_trade_count` | raw | −3.38% | −6.48 | **1.4%** | 0.29 | 14.2% | 1.19 | −16% | 0.52 | 88% | 0.43 | 25% |
| `ats_trade_count` | filter_signal | −3.82% | −7.20 | **8.0%** | 1.69 | 37.9% | 3.23 | −15% | 0.53 | 89% | 0.43 | 25% |
| `amount+ATS` | raw | −4.08% | −6.88 | **3.8%** | 0.69 | 18.4% | 1.40 | −15% | 0.50 | 89% | 0.43 | 25% |
| `amount+ATS` | filter_signal | −4.49% | −7.42 | **8.8%** | 1.62 | 40.3% | 3.09 | −12% | 0.52 | 89% | 0.43 | 25% |
| `ideal_amplitude` | raw | −4.81% | −7.91 | **9.3%** | 1.23 | 49.4% | 3.28 | −19% | 0.50 | 86% | 0.42 | 27% |
| `ideal_amplitude` | filter_signal | −5.02% | −8.10 | **11.6%** | 1.51 | 58.9% | 4.00 | −22% | 0.52 | 86% | 0.42 | 27% |

## Critical findings

### 1. Long-only is the binding constraint (not ICIR)

Without limit filter, **ATS / dual long excess is weak** (1.4% / 3.8%). Most H-L comes from the **short leg**. In A-share long-biased books this is a hard operability hit.

`filter_signal` (mask 涨跌停 on finished factor) is not optional cosmetics — it lifts dual long excess **3.8% → 8.8%**.

### 2. Turnover is high

| Cadence | Metric | Dual raw | Amplitude raw |
|---------|--------|----------|---------------|
| Daily rebalance | long-book \|Δw\| mean | **~0.50 / day** | ~0.50 / day |
| Month-end rebalance | long one-way name replace | **~89%** | ~86% |

Far above the research-note style “单笔换仓 ~40%”. Fee drag will be material; need lower-turnover variants (monthly signal, band rebalance) before production.

### 3. Size exposure — not pure microcap

Long-book median size percentile ≈ **0.43** (slightly below median). Bottom-20% size share ≈ 25%. Small-cap tilt exists but is not “all microcaps”.

### 4. Ranking by operability (long excess + limit filter)

1. **`ideal_amplitude` + filter_signal** — best long excess (11.6%), best ICIR  
2. **`amount+ATS` + filter_signal** — 8.8% long excess; still high TO  
3. ATS / dual **raw** — fail long-only bar without limit mask  

## Recipe (operability-safe)

1. Build W-cut on **full** sample (do not `filter_cut`).  
2. Evaluate / trade with **`get_EOD_Not_Limit`** (`filter_signal`).  
3. Prefer **amplitude** for long-biased books; dual ATS for information / short-capable books.  
4. Do **not** claim production readiness until turnover is cut (monthly / banded).

## Artifacts

- `operability_summary.csv` — flat metrics  
- `monthly_rebalance_turnover.csv` — month-end one-way TO  
- `{factor}/{mode}/group_pnl.csv`, `group_turnover.csv`, `long_excess.csv`, `hl_signed.csv`

## Definitions

- Long group = profitable decile given sign (neg IC → group 1 / low factor).  
- Long excess = long EW − universe EW (same coverage).  
- Daily TO from `Factor_Dev_Lib.groupTest` (|Δw|).  
- Month-end TO = 1 − |names kept| / |long book| between consecutive month-ends.
