# D1_LiquidityQuality60d — Validation

**Sample:** confirmation_1455d · ALL A-shares · 10 groups · raw mode  
**Source tables:** `ic_analysis/factor_summary.csv` · `stability/stability.csv` · `execution/execution_summary.csv`

---

## 1. IC / ICIR

| Mode | RankIC | ICIR | H-L Sharpe | Net Sharpe @15bp | Mono |
|------|-------:|-----:|-----------:|-----------------:|-----:|
| **raw** | **0.0573** | **6.01** | **2.26** | N/A | **0.800** |
| size | N/A | N/A | N/A | N/A | N/A |
| industry | N/A | N/A | N/A | N/A | N/A |
| size+industry | N/A | N/A | N/A | N/A | N/A |

Neutralization ladder was **not** in confirmation pack artifacts. Universe ladder (CSI300/500/1000/ALL) is in `diagnostics/universe_ladder.csv`.

---

## 2. Quantile / monotonicity

| Metric | Value |
|--------|------:|
| Monotonicity (Spearman of decile means) | **0.800** |
| H-L ann. return (raw) | 43.5% |
| H-L MDD (raw) | −19.7% |
| Daily turnover (raw H-L) | 0.482 |

Decile pattern is orderly but not perfect (80% mono).

---

## 3. Stability

Only a **confirmation block** summary is harvested (not a full yearly panel):

| Period | RankIC | ICIR | IC>0 frac | n_days |
|--------|-------:|-----:|----------:|-------:|
| confirmation_1455d | 0.0573 | 6.01 | 0.653 | 1455 |

Marked as **limited stability evidence** pending Protocol re-run.

---

## 4. Execution (15 bp RT)

Best row on 1D.7 **raw** grid:

| Recipe | Daily TO | Net Sharpe |
|--------|---------:|-----------:|
| raw\|daily\|buffer_5_15 (**best**) | **0.234** | **1.38** |
| raw\|daily | 0.397 | 1.01 |
| raw\|every_5d | 0.268 | 1.09 |

Full grid: `execution/execution_summary.csv`

Net Sharpe remains below TGD/Flow execution leaders; treat as candidate investability evidence, not validated production admission.

---

## 5. Verdict

```text
Formula frozen           ✓  (library constructor)
IC / ICIR strong (raw)   ✓  (RankICIR 6.01)
Decile mono              ~  (0.800)
Yearly stable            ✗  (block only — no yearly panel)
Net after cost           ~  (best Net Sharpe 1.38)
Neutralization ladder    ✗  (not harvested)
Status                   candidate → not yet validated
```

Soft quality screen passed on ALL confirmation harvest; Dual Benchmark Production re-run still pending.
