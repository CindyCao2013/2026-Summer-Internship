# IdealAmplitude — Validation

**Sample:** cutting_v1 harvest · 3885 days · ALL A-shares · 10 groups · `signal_shift=1`  
**Execution window:** last 252d · **Source:** `ic_analysis/factor_summary.csv` · `stability/` · `execution/execution_summary.csv`

---

## 1. IC / ICIR

| Mode | RankIC | ICIR | H-L Sharpe | H-L ann. ret | Mono |
|------|-------:|-----:|-----------:|-------------:|-----:|
| raw | −0.0378 | −7.66 | 3.44 | 40.1% | **0.111** |
| size | −0.0523 | −8.21 | 3.44 | 40.1% | 0.111 |
| industry | −0.0499 | −8.62 | 3.44 | 40.1% | 0.111 |
| **size+industry** | **−0.0475** | **−9.97** | **3.44** | **40.1%** | 0.111 |

Pearson IC: not computed on this harvest.

---

## 2. Quantile / monotonicity

| Metric | Value |
|--------|------:|
| Monotonicity (Spearman of decile means) | **0.111** — **very weak** |
| H-L ann. return (raw) | 40.1% |
| H-L Sharpe (raw) | 3.44 |
| H-L MDD (execution best) | −5.0% |

Decile means: `quantile_analysis/decile_means.csv`  
Mechanism legs: `quantile_analysis/mechanism_legs.csv`

**Monotonicity note:** decile structure is flat/noisy despite strong aggregate ICIR and Sharpe.
High leg RankIC −0.0507 vs low leg −0.0227 — knife separation passes; decile rank-order **fails** soft bar (>0.8).

---

## 3. Stability (yearly RankIC)

Source: `stability/yearly_by_year.csv` (factor_cutting_v1 lineage)

| Year | RankIC | ICIR |
|------|-------:|-----:|
| 2010 | −0.0207 | −3.28 |
| 2011 | −0.0203 | −4.34 |
| 2012 | −0.0337 | −6.03 |
| 2013 | −0.0325 | −6.36 |
| 2014 | −0.0312 | −5.76 |
| 2015 | −0.0464 | −8.42 |
| 2016 | −0.0409 | −8.06 |
| 2017 | −0.0317 | −5.86 |
| 2018 | −0.0407 | −8.66 |
| 2019 | −0.0547 | −12.07 |
| 2020 | −0.0331 | −8.11 |
| 2021 | −0.0345 | −9.21 |
| 2022 | −0.0425 | −12.99 |
| 2023 | −0.0417 | −11.60 |
| 2024 | −0.0437 | −7.32 |
| 2025 | −0.0568 | −12.98 |

**16/16 years RankIC < 0.**  
Block summary: `stability/stability.csv` (3885d, ICIR −7.66, pos_ic_frac 29.0%).

---

## 4. Execution (15 bp RT, last 252d)

| Recipe | Daily TO | Gross Sharpe | Net Sharpe | MDD net |
|--------|---------:|-------------:|-----------:|--------:|
| signed_cs_z\|daily\|buffer_5_15 (**best**) | **0.446** | 4.66 | **3.40** | −5.0% |
| signed_cs_z\|weekly_friday | 0.390 | 4.11 | 2.76 | −3.7% |
| signed_cs_z\|every_20d | 0.165 | 2.83 | 2.32 | −4.8% |
| signed_cs_z\|daily (naive) | 0.776 | 4.70 | 2.41 | −5.4% |

Full grid: `execution/execution_summary.csv`

---

## 5. Verdict

```text
Formula reproduced (cutting)     ✓
IC / ICIR strong               ✓  (SI ICIR −9.97)
H-L Sharpe strong              ✓  (3.44 gross)
Mechanism legs separate          ✓  (V_high >> V_low)
Decile mono                    ✗  (0.111 — very weak)
Net after cost (best)          ✓  (Net Sharpe 3.40)
Status                         testing
```

Strong headline metrics but **weak monotonicity blocks validated admission**.
