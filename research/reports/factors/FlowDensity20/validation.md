# FlowDensity20 — Validation

**Sample:** 2022-01-28 → 2025-12-31 · ALL A-shares · 10 groups · `signal_shift=1`  
**Source tables:** `ic_analysis/factor_summary.csv` · `stability/yearly_stability.csv` · `execution/execution_summary.csv`

---

## 1. IC / ICIR

| Mode | RankIC | ICIR | H-L Sharpe | Net Sharpe @15bp | Mono |
|------|-------:|-----:|-----------:|-----------------:|-----:|
| raw | 0.0178 | 2.07 | 1.52 | −0.18 | N/A |
| size | 0.0271 | 4.41 | 2.95 | 1.60 | N/A |
| industry | 0.0152 | 2.69 | 1.90 | 0.08 | N/A |
| **size+industry** | **0.0236** | **4.85** | **3.38** | **1.85** | N/A |

Neutralization **improves** measured RankICIR versus raw. Monotonicity not in factor_summary harvest (see mechanism table in legacy report).

---

## 2. Quantile / monotonicity

| Metric | Value |
|--------|------:|
| Monotonicity | N/A (not_in_artifacts) |
| H-L ann. return (size+industry) | 29.2% |
| H-L MDD (size+industry) | −9.5% |
| Daily turnover (size+industry H-L) | 0.463 |

---

## 3. Stability (yearly RankIC)

| Year | RankIC | ICIR | n_days |
|------|-------:|-----:|-------:|
| 2022 | 0.0206 | 4.53 | 223 |
| 2023 | 0.0231 | 4.81 | 242 |
| 2024 | 0.0234 | 4.21 | 242 |
| 2025 | 0.0268 | 6.05 | 243 |

**4/4 years RankIC > 0** on confirmation window.

---

## 4. Execution (15 bp RT)

Best row on size+industry grid:

| Recipe | Daily TO | Net Sharpe |
|--------|---------:|-----------:|
| size_industry\|daily\|buffer_10_30 (**best**) | **0.165** | **2.88** |
| size_industry\|every_10d | 0.179 | 2.69 |
| size_industry\|daily\|buffer_10_20 | 0.220 | 2.69 |

Full grid: `execution/execution_summary.csv`

---

## 5. Verdict

```text
Formula reproduced       ✓  (harvest pipeline)
IC / ICIR (size+ind)     ✓  (ICIR 4.85)
Decile mono              ?  (not in artifacts)
Yearly stable            ✓  (4/4)
Net after cost           ✓  (best Net Sharpe 2.88)
Pure-flow story          ✗  (Flow⊥Amount sign flip)
Status                   candidate → not yet validated
```

Interaction factor with strong execution after buffers; formula not frozen pending amount-orth admission review.
