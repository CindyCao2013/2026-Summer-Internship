# Alpha Report: `high_low_60d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0434 | -2.6133 | 0.4360 | 0.1263 | 0.0385 | 0.1566 |
| CSI500 | -0.0458 | -3.0419 | 0.4333 | 0.2883 | 0.0790 | 0.1781 |
| CSI1000 | -0.0485 | -3.3266 | 0.4202 | 0.7319 | 0.2066 | 0.1809 |
| ALL | -0.0497 | -3.6820 | 0.4092 | 0.8372 | 0.2307 | 0.1705 |

## ALL-universe summary

- **factor_name**: high_low_60d
- **universe**: ALL
- **rank_ic_mean**: -0.0497
- **abs_rank_ic_mean**: 0.0497
- **icir**: -3.6820
- **ic_positive_ratio**: 0.4092
- **hl_sharpe**: 0.8372
- **hl_annu_ret**: 0.2307
- **hl_mdd**: -0.3228
- **hl_avg_turnover**: 0.1705
- **direction**: -1
- **monotonicity_score**: 0.2000
- **stats_title**: H-L, Direction: -1, AnnuRet: 23.07%, Sharpe: 0.84, MDD: -32.28%, Turnover: 0.17, RankIC: -0.0497, ICIR: -3.68

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0455 |
| 5.0000 | -0.0641 |
| 10.0000 | -0.0758 |
| 20.0000 | -0.0935 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000544 |
| 2 | 0.000680 |
| 3 | 0.000757 |
| 4 | 0.000743 |
| 5 | 0.000733 |
| 6 | 0.000701 |
| 7 | 0.000544 |
| 8 | 0.000484 |
| 9 | 0.000214 |
| 10 | -0.000379 |
| H-L | -0.000923 |

Monotonicity score: **20.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
