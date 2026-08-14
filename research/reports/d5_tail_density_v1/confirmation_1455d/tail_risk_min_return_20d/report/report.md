# Alpha Report: `tail_risk_min_return_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0244 | 1.8096 | 0.5461 | 0.0275 | 0.0065 | 0.3818 |
| CSI500 | 0.0283 | 2.3515 | 0.5530 | 0.0585 | 0.0121 | 0.4121 |
| CSI1000 | 0.0306 | 2.6304 | 0.5681 | 0.3081 | 0.0664 | 0.4213 |
| ALL | 0.0329 | 2.9080 | 0.5791 | 0.5393 | 0.1239 | 0.4064 |

## ALL-universe summary

- **factor_name**: tail_risk_min_return_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0329
- **abs_rank_ic_mean**: 0.0329
- **icir**: 2.9080
- **ic_positive_ratio**: 0.5791
- **hl_sharpe**: 0.5393
- **hl_annu_ret**: 0.1239
- **hl_mdd**: -0.4010
- **hl_avg_turnover**: 0.4064
- **direction**: 1
- **monotonicity_score**: 0.7000
- **stats_title**: H-L, Direction: 1, AnnuRet: 12.39%, Sharpe: 0.54, MDD: -40.10%, Turnover: 0.41, RankIC: 0.0329, ICIR: 2.91

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0313 |
| 5.0000 | 0.0440 |
| 10.0000 | 0.0518 |
| 20.0000 | 0.0634 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000085 |
| 2 | 0.000203 |
| 3 | 0.000431 |
| 4 | 0.000554 |
| 5 | 0.000662 |
| 6 | 0.000727 |
| 7 | 0.000734 |
| 8 | 0.000724 |
| 9 | 0.000619 |
| 10 | 0.000411 |
| H-L | 0.000496 |

Monotonicity score: **70.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
