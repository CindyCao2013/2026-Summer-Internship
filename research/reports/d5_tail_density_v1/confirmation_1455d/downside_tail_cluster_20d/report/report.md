# Alpha Report: `downside_tail_cluster_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0061 | 0.8448 | 0.5124 | 0.5075 | 0.0653 | 0.3284 |
| CSI500 | 0.0105 | 1.7573 | 0.5406 | 0.7518 | 0.0920 | 0.3458 |
| CSI1000 | 0.0100 | 1.8460 | 0.5447 | 0.4461 | 0.0528 | 0.3586 |
| ALL | 0.0129 | 2.4761 | 0.5468 | 0.7897 | 0.0933 | 0.3533 |

## ALL-universe summary

- **factor_name**: downside_tail_cluster_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0129
- **abs_rank_ic_mean**: 0.0129
- **icir**: 2.4761
- **ic_positive_ratio**: 0.5468
- **hl_sharpe**: 0.7897
- **hl_annu_ret**: 0.0933
- **hl_mdd**: -0.1431
- **hl_avg_turnover**: 0.3533
- **direction**: 1
- **monotonicity_score**: 0.7000
- **stats_title**: H-L, Direction: 1, AnnuRet: 9.33%, Sharpe: 0.79, MDD: -14.31%, Turnover: 0.35, RankIC: 0.0129, ICIR: 2.48

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0118 |
| 5.0000 | 0.0192 |
| 10.0000 | 0.0250 |
| 20.0000 | 0.0304 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000228 |
| 2 | 0.000263 |
| 3 | 0.000428 |
| 4 | 0.000460 |
| 5 | 0.000650 |
| 6 | 0.000598 |
| 7 | 0.000679 |
| 8 | 0.000416 |
| 9 | 0.000580 |
| 10 | 0.000602 |
| H-L | 0.000373 |

Monotonicity score: **70.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
