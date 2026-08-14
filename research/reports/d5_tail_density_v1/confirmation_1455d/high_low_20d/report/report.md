# Alpha Report: `high_low_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0455 | -2.8802 | 0.4340 | 0.1708 | 0.0511 | 0.2906 |
| CSI500 | -0.0519 | -3.6134 | 0.4175 | 0.4501 | 0.1221 | 0.3170 |
| CSI1000 | -0.0579 | -4.1684 | 0.3982 | 1.2006 | 0.3334 | 0.3265 |
| ALL | -0.0610 | -4.7153 | 0.3913 | 1.3650 | 0.3761 | 0.3176 |

## ALL-universe summary

- **factor_name**: high_low_20d
- **universe**: ALL
- **rank_ic_mean**: -0.0610
- **abs_rank_ic_mean**: 0.0610
- **icir**: -4.7153
- **ic_positive_ratio**: 0.3913
- **hl_sharpe**: 1.3650
- **hl_annu_ret**: 0.3761
- **hl_mdd**: -0.3291
- **hl_avg_turnover**: 0.3176
- **direction**: -1
- **monotonicity_score**: 0.3000
- **stats_title**: H-L, Direction: -1, AnnuRet: 37.61%, Sharpe: 1.36, MDD: -32.91%, Turnover: 0.32, RankIC: -0.0610, ICIR: -4.72

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0539 |
| 5.0000 | -0.0739 |
| 10.0000 | -0.0855 |
| 20.0000 | -0.1020 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000577 |
| 2 | 0.000760 |
| 3 | 0.000838 |
| 4 | 0.000824 |
| 5 | 0.000836 |
| 6 | 0.000757 |
| 7 | 0.000656 |
| 8 | 0.000450 |
| 9 | 0.000207 |
| 10 | -0.000928 |
| H-L | -0.001505 |

Monotonicity score: **30.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
