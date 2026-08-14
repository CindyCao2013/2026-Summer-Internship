# TGD Execution Optimization v1

**Factor layer:** TGD20 frozen — no window / Gu/Gd / residual changes.
**Objective:** maximize **Net Sharpe** (15bp RT), not gross.

## E0 Baseline (Stage-4 confirmation, frozen)

| Mode | RankIC | ICIR | H-L Sharpe | Daily TO | Net@15bp |
|------|--------|------|------------|----------|----------|
| raw | 0.0430 | 6.98 | 2.77 | 0.65 | 1.00 |
| size+industry | 0.0415 | 11.29 | 4.06 | 0.65 | 1.72 |

Execution experiments use **`size_industry`** signal + top/bottom **10%** LS book.

## Leaderboard (by Net Sharpe)

| Rank | Label | Stage | Gross | Net | Daily TO | ICIR |
|------|-------|-------|------:|----:|---------:|-----:|
| 1 | `size_industry|daily|buffer_5_15` | E2_buffer | 3.51 | 2.32 | 0.297 | 11.28 |
| 2 | `size_industry|best_e1|zscore` | E3 | 3.50 | 2.24 | 0.324 | 9.28 |
| 3 | `size_industry|daily|buffer_10_30` | E2_buffer | 3.36 | 2.21 | 0.217 | 11.28 |
| 4 | `size_industry|best_e1|hold_10d` | E2_hold | 3.41 | 2.20 | 0.257 | 9.28 |
| 5 | `size_industry|best_e1|buffer_5_15` | E2_buffer | 3.04 | 2.20 | 0.216 | 9.28 |
| 6 | `size_industry|best_e1|buffer_10_30` | E2_buffer | 2.94 | 2.09 | 0.165 | 9.28 |
| 7 | `size_industry|daily_buffer_10_20` | E4_combo | 3.55 | 2.09 | 0.298 | 11.28 |
| 8 | `size_industry|daily|buffer_10_20` | E2_buffer | 3.55 | 2.09 | 0.298 | 11.28 |
| 9 | `size_industry|best_e1|rank` | E3 | 3.44 | 2.08 | 0.311 | 9.28 |
| 10 | `size_industry|best_e1_buffer_10_20_hold5` | E4_combo | 3.09 | 2.07 | 0.209 | 9.28 |
| 11 | `size_industry|best_e1_buffer_10_20` | E4_combo | 3.09 | 2.07 | 0.209 | 9.28 |
| 12 | `size_industry|best_e1|buffer_10_20` | E2_buffer | 3.09 | 2.07 | 0.209 | 9.28 |

## Recommended investable config

- **Best Net Sharpe:** `size_industry|daily|buffer_5_15` → net **2.32** (gross 3.51, TO 0.297)
- E1 best rebalance: `size_industry|every_5d`
- E2 best buffer: `size_industry|daily|buffer_5_15`

## Artifacts

- `baseline_metrics.json`
- `rebalance_frequency.csv`
- `buffer_test.csv`
- `holding_period.csv`
- `weight_method.csv`
- `combo_test.csv`
- `all_experiments.csv`

## Principle

Do not retune TGD20. Execution only: fewer / smarter trades to harvest the same alpha.

