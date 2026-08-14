# Alpha Report: `return_skew_shift_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0057 | 0.7872 | 0.5151 | 0.8315 | 0.1116 | 0.6735 |
| CSI500 | 0.0019 | 0.3298 | 0.5014 | 0.4696 | 0.0475 | 0.6891 |
| CSI1000 | 0.0008 | 0.1614 | 0.4904 | 0.0193 | 0.0018 | 0.6639 |
| ALL | 0.0009 | 0.2048 | 0.4931 | 0.2325 | 0.0197 | 0.6585 |

## ALL-universe summary

- **factor_name**: return_skew_shift_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0009
- **abs_rank_ic_mean**: 0.0009
- **icir**: 0.2048
- **ic_positive_ratio**: 0.4931
- **hl_sharpe**: 0.2325
- **hl_annu_ret**: 0.0197
- **hl_mdd**: -0.2116
- **hl_avg_turnover**: 0.6585
- **direction**: -1
- **monotonicity_score**: 0.5000
- **stats_title**: H-L, Direction: -1, AnnuRet: 1.97%, Sharpe: 0.23, MDD: -21.16%, Turnover: 0.66, RankIC: 0.0009, ICIR: 0.20

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0003 |
| 5.0000 | 0.0029 |
| 10.0000 | 0.0039 |
| 20.0000 | 0.0103 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000579 |
| 2 | 0.000345 |
| 3 | 0.000348 |
| 4 | 0.000459 |
| 5 | 0.000482 |
| 6 | 0.000557 |
| 7 | 0.000551 |
| 8 | 0.000646 |
| 9 | 0.000575 |
| 10 | 0.000500 |
| H-L | -0.000079 |

Monotonicity score: **50.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
