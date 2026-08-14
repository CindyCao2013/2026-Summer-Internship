# Alpha Report: `amount_stability_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0206 | 2.4979 | 0.5715 | 0.9921 | 0.1614 | 0.6570 |
| CSI500 | 0.0260 | 4.0592 | 0.6149 | 1.6536 | 0.2237 | 0.6517 |
| CSI1000 | 0.0343 | 6.4558 | 0.6857 | 2.9517 | 0.3458 | 0.6512 |
| ALL | 0.0405 | 8.6447 | 0.7228 | 4.5777 | 0.4654 | 0.6363 |

## ALL-universe summary

- **factor_name**: amount_stability_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0405
- **abs_rank_ic_mean**: 0.0405
- **icir**: 8.6447
- **ic_positive_ratio**: 0.7228
- **hl_sharpe**: 4.5777
- **hl_annu_ret**: 0.4654
- **hl_mdd**: -0.0857
- **hl_avg_turnover**: 0.6363
- **direction**: 1
- **monotonicity_score**: 1.0000
- **stats_title**: H-L, Direction: 1, AnnuRet: 46.54%, Sharpe: 4.58, MDD: -8.57%, Turnover: 0.64, RankIC: 0.0405, ICIR: 8.64

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0332 |
| 5.0000 | 0.0451 |
| 10.0000 | 0.0492 |
| 20.0000 | 0.0503 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000919 |
| 2 | 0.000185 |
| 3 | 0.000421 |
| 4 | 0.000486 |
| 5 | 0.000574 |
| 6 | 0.000753 |
| 7 | 0.000761 |
| 8 | 0.000864 |
| 9 | 0.000911 |
| 10 | 0.000942 |
| H-L | 0.001861 |

Monotonicity score: **100.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
