# TGD20 — Validation

**Sample:** 2022-01-28 → 2025-12-31 · ALL A-shares · 10 groups · `signal_shift=1`  
**Source tables:** `ic_analysis/factor_summary.csv` · `stability/yearly_stability.csv` · `execution/execution_summary.csv`

---

## 1. IC / ICIR

| Mode | RankIC | ICIR | H-L Sharpe | Net Sharpe @15bp | Mono |
|------|-------:|-----:|-----------:|-----------------:|-----:|
| raw | 0.0430 | 6.98 | 2.77 | 1.00 | 0.988 |
| size | 0.0443 | 8.67 | 3.52 | 1.51 | 1.000 |
| industry | 0.0408 | 8.90 | 3.19 | 1.16 | 0.988 |
| **size+industry** | **0.0415** | **11.29** | **4.06** | **1.72** | 0.988 |

Charts: `ic_analysis/ic_curve.png` · `charts/neutralization_compare.png`

---

## 2. Quantile / monotonicity

| Metric | Value |
|--------|------:|
| Monotonicity (Spearman of decile means) | **0.988** |
| H-L ann. return (raw) | 36.9% |
| H-L MDD (raw / SI) | −19.0% / −6.0% |

Charts: `quantile_analysis/decile_return.png` · `quantile_analysis/cumulative_long_short.png`

---

## 3. Stability (yearly RankIC)

| Year | RankIC | ICIR | IC>0 frac |
|------|-------:|-----:|----------:|
| 2020 | 0.036 | 8.01 | 0.70 |
| 2021 | 0.036 | 6.69 | 0.65 |
| 2022 | 0.047 | 9.12 | 0.73 |
| 2023 | 0.052 | 8.95 | 0.73 |
| 2024 | 0.032 | 4.25 | 0.64 |
| 2025 | 0.043 | 7.06 | 0.68 |

**6/6 years RankIC > 0.** Charts: `stability/stability_yearly.png`

---

## 4. Execution (15 bp RT)

| Recipe | Daily TO | Net Sharpe |
|--------|---------:|-----------:|
| SI daily (naive) | 0.65 | 1.28 |
| SI daily + buffer 5/15 (**best**) | **0.30** | **2.32** |
| SI every 5d | 0.31 | 2.06 |

Full grid: `execution/execution_summary.csv` · chart `execution/turnover.png`

---

## 5. Verdict

```text
Formula reproduced     ✓
IC / ICIR strong       ✓  (SI ICIR 11.29)
Decile mono            ✓  (0.988)
Yearly stable          ✓  (6/6)
Net after cost         ✓  (best Net Sharpe 2.32)
Status                 validated → Factor Library
```

This is the **first complete paper-replication asset** in the library.
