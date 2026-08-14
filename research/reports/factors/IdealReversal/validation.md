# IdealReversal — Validation

**Sample:** cutting_v1 harvest · 1703 days · ALL A-shares · 10 groups · `signal_shift=1`  
**Source tables:** `ic_analysis/factor_summary.csv` · `stability/` · `execution/execution_summary.csv`

---

## 1. IC / ICIR

| Mode | RankIC | ICIR | H-L Sharpe | H-L ann. ret | Mono |
|------|-------:|-----:|-----------:|-------------:|-----:|
| raw | −0.0311 | −8.60 | 1.70 | 16.6% | 0.444 |
| size | −0.0374 | −7.21 | 1.70 | 16.6% | 0.444 |
| industry | −0.0334 | −8.17 | 1.70 | 16.6% | 0.444 |
| **size+industry** | **−0.0331** | **−9.46** | **1.70** | **16.6%** | 0.444 |

Pearson IC: not computed on this harvest.

---

## 2. Quantile / monotonicity

| Metric | Value |
|--------|------:|
| Monotonicity (Spearman of decile means) | **0.444** — weak |
| H-L ann. return (raw) | 16.6% |
| H-L Sharpe (raw) | 1.70 |

Decile means: `quantile_analysis/decile_means.csv`  
Mechanism legs: `quantile_analysis/mechanism_legs.csv`

Mechanism verdict: high-ATS leg ICIR ≈ −5.73 vs low-ATS ≈ +0.33 — cutting claim **passes**; decile shape **fails** soft bar.

---

## 3. Stability (yearly RankIC)

Source: `stability/yearly_by_year.csv` (factor_cutting_v1 lineage)

| Year | RankIC | ICIR | n |
|------|-------:|-----:|--:|
| 2019 | −0.0406 | −8.94 | 244 |
| 2020 | −0.0230 | −6.41 | 243 |
| 2021 | −0.0269 | −7.59 | 243 |
| 2022 | −0.0267 | −9.59 | 242 |
| 2023 | −0.0373 | −14.70 | 242 |
| 2024 | −0.0238 | −5.74 | 242 |
| 2025 | −0.0382 | −10.74 | 243 |

**7/7 years RankIC < 0** (consistent with negative-IC reversal signal).  
Block summary: `stability/stability.csv` (1703d, ICIR −8.60, pos_ic_frac 27.2%).

---

## 4. Execution (15 bp RT, last 252d)

| Recipe | Daily TO | Gross Sharpe | Net Sharpe | MDD net |
|--------|---------:|-------------:|-----------:|--------:|
| raw_cs_z\|best_e1_buffer_5_15 (**best**) | **0.153** | 2.09 | **1.70** | −6.9% |
| raw_cs_z\|every_20d | 0.168 | 1.97 | 1.47 | −8.7% |
| raw_cs_z\|daily | 1.025 | 1.71 | −1.47 | −18.9% |

Full grid: `execution/execution_summary.csv`

---

## 5. Verdict

```text
Formula reproduced (cutting)     ✓
IC / ICIR strong               ✓  (SI ICIR −9.46)
Mechanism legs separate          ✓  (M_high >> M_low)
Decile mono                    ✗  (0.444 — weak)
Sharpe soft bar (>2)           ✗  (1.70)
Net after cost (best)          ✓  (Net Sharpe 1.70)
Status                         testing
```

Paper-replication pack — mechanism validated, admission soft bars not met.
