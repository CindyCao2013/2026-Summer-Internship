# Alpha Report: `max_daily_return_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0380 | 2.9879 | 0.5860 | 0.0978 | 0.0246 | 0.4051 |
| CSI500 | 0.0453 | 4.1522 | 0.6045 | 0.5141 | 0.1143 | 0.4214 |
| CSI1000 | 0.0525 | 5.1021 | 0.6424 | 1.3698 | 0.2915 | 0.4147 |
| ALL | 0.0585 | 6.1348 | 0.6637 | 1.8620 | 0.3892 | 0.3964 |

## ALL-universe summary

- **factor_name**: max_daily_return_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0585
- **abs_rank_ic_mean**: 0.0585
- **icir**: 6.1348
- **ic_positive_ratio**: 0.6637
- **hl_sharpe**: 1.8620
- **hl_annu_ret**: 0.3892
- **hl_mdd**: -0.2100
- **hl_avg_turnover**: 0.3964
- **direction**: 1
- **monotonicity_score**: 0.8000
- **stats_title**: H-L, Direction: 1, AnnuRet: 38.92%, Sharpe: 1.86, MDD: -21.00%, Turnover: 0.40, RankIC: 0.0585, ICIR: 6.13

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0521 |
| 5.0000 | 0.0714 |
| 10.0000 | 0.0820 |
| 20.0000 | 0.0954 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000871 |
| 2 | -0.000101 |
| 3 | 0.000442 |
| 4 | 0.000644 |
| 5 | 0.000764 |
| 6 | 0.000798 |
| 7 | 0.000827 |
| 8 | 0.000905 |
| 9 | 0.000887 |
| 10 | 0.000686 |
| H-L | 0.001557 |

Monotonicity score: **80.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
