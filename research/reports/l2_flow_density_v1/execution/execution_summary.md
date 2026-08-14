# FlowDensity20 Execution Optimization v1

**Factor column:** `net_active_flow_mktcap_20d` — formula not retuned in this run.
**Objective:** maximize **Net Sharpe** (15bp RT), same grid as TGD20.

## E0 Baseline (neut ladder, frozen for execution)

| Mode | RankIC | ICIR | H-L Sharpe | Daily TO | Net@15bp |
|------|--------|------|------------|----------|----------|
| raw | 0.0178 | 2.07 | 1.52 | 0.51 | -0.18 |
| size+industry | 0.0236 | 4.85 | 3.38 | 0.46 | 1.85 |

Execution experiments use **`size_industry`** signal + top/bottom **10%** LS book.

## Leaderboard (by Net Sharpe)

| Rank | Label | Stage | Gross | Net | Daily TO | ICIR |
|------|-------|-------|------:|----:|---------:|-----:|
| 1 | `size_industry|daily|buffer_10_30` | E2_buffer | 3.71 | 2.88 | 0.165 | 4.85 |
| 2 | `size_industry|best_e1|buffer_10_30` | E2_buffer | 3.38 | 2.86 | 0.102 | 4.47 |
| 3 | `size_industry|best_e1_buffer_10_20` | E4_combo | 3.40 | 2.76 | 0.128 | 4.47 |
| 4 | `size_industry|best_e1_buffer_10_20_hold5` | E4_combo | 3.40 | 2.76 | 0.128 | 4.47 |
| 5 | `size_industry|best_e1|buffer_10_20` | E2_buffer | 3.40 | 2.76 | 0.128 | 4.47 |
| 6 | `size_industry|daily_buffer_10_20` | E4_combo | 3.75 | 2.69 | 0.220 | 4.85 |
| 7 | `size_industry|daily|buffer_10_20` | E2_buffer | 3.75 | 2.69 | 0.220 | 4.85 |
| 8 | `size_industry|best_e1|hold_10d` | E2_hold | 3.53 | 2.69 | 0.179 | 4.47 |
| 9 | `size_industry|best_e1|hold_5d` | E2_hold | 3.53 | 2.69 | 0.179 | 4.47 |
| 10 | `size_industry|best_e1_plain` | E4_combo | 3.53 | 2.69 | 0.179 | 4.47 |
| 11 | `size_industry|best_e1|hold_1d` | E2_hold | 3.53 | 2.69 | 0.179 | 4.47 |
| 12 | `size_industry|every_10d` | E1 | 3.53 | 2.69 | 0.179 | 4.47 |

## Recommended investable config

- **Best Net Sharpe:** `size_industry|daily|buffer_10_30` → net **2.88** (gross 3.71, TO 0.165)
- E1 best rebalance: `size_industry|every_10d`
- E2 best buffer: `size_industry|daily|buffer_10_30`

## Artifacts

- `baseline_metrics.json`
- `rebalance_frequency.csv`
- `buffer_test.csv`
- `holding_period.csv`
- `weight_method.csv`
- `combo_test.csv`
- `all_experiments.csv`
- `execution_summary.csv` (top by Net Sharpe)

## Next

After this closes: deepen FlowDensity mechanism (buy/sell components), then TGD20 ⟂ FlowDensity20 orthogonality — before Composite Alpha Engine v1.

