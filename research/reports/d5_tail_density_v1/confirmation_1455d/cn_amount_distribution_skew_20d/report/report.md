# Alpha Report: `cn_amount_distribution_skew_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0086 | -1.3726 | 0.4601 | 1.4762 | 0.1941 | 0.9299 |
| CSI500 | -0.0070 | -1.3975 | 0.4635 | 1.0416 | 0.1200 | 0.9581 |
| CSI1000 | -0.0058 | -1.4009 | 0.4587 | 1.2055 | 0.1264 | 0.9779 |
| ALL | -0.0076 | -2.2609 | 0.4298 | 2.0366 | 0.1633 | 0.9739 |

## ALL-universe summary

- **factor_name**: cn_amount_distribution_skew_20d
- **universe**: ALL
- **rank_ic_mean**: -0.0076
- **abs_rank_ic_mean**: 0.0076
- **icir**: -2.2609
- **ic_positive_ratio**: 0.4298
- **hl_sharpe**: 2.0366
- **hl_annu_ret**: 0.1633
- **hl_mdd**: -0.0891
- **hl_avg_turnover**: 0.9739
- **direction**: -1
- **monotonicity_score**: 0.3000
- **stats_title**: H-L, Direction: -1, AnnuRet: 16.33%, Sharpe: 2.04, MDD: -8.91%, Turnover: 0.97, RankIC: -0.0076, ICIR: -2.26

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0037 |
| 5.0000 | -0.0010 |
| 10.0000 | 0.0014 |
| 20.0000 | 0.0052 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000557 |
| 2 | 0.000681 |
| 3 | 0.000692 |
| 4 | 0.000574 |
| 5 | 0.000525 |
| 6 | 0.000574 |
| 7 | 0.000526 |
| 8 | 0.000507 |
| 9 | 0.000438 |
| 10 | -0.000096 |
| H-L | -0.000653 |

Monotonicity score: **30.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
