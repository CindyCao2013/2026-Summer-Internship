# Alpha Report: `range_contraction_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0414 | 2.9883 | 0.5812 | 0.2355 | 0.0606 | 0.4433 |
| CSI500 | 0.0498 | 4.3182 | 0.6100 | 0.9341 | 0.2066 | 0.4657 |
| CSI1000 | 0.0563 | 5.1763 | 0.6403 | 1.7498 | 0.3877 | 0.4630 |
| ALL | 0.0602 | 5.6835 | 0.6671 | 2.0071 | 0.4733 | 0.4341 |

## ALL-universe summary

- **factor_name**: range_contraction_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0602
- **abs_rank_ic_mean**: 0.0602
- **icir**: 5.6835
- **ic_positive_ratio**: 0.6671
- **hl_sharpe**: 2.0071
- **hl_annu_ret**: 0.4733
- **hl_mdd**: -0.3427
- **hl_avg_turnover**: 0.4341
- **direction**: 1
- **monotonicity_score**: 0.8000
- **stats_title**: H-L, Direction: 1, AnnuRet: 47.33%, Sharpe: 2.01, MDD: -34.27%, Turnover: 0.43, RankIC: 0.0602, ICIR: 5.68

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0525 |
| 5.0000 | 0.0724 |
| 10.0000 | 0.0840 |
| 20.0000 | 0.0990 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.001182 |
| 2 | 0.000017 |
| 3 | 0.000397 |
| 4 | 0.000683 |
| 5 | 0.000792 |
| 6 | 0.000869 |
| 7 | 0.000887 |
| 8 | 0.000917 |
| 9 | 0.000887 |
| 10 | 0.000711 |
| H-L | 0.001893 |

Monotonicity score: **80.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
