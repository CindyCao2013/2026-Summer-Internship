# Alpha Report: `tail_adjusted_momentum_60d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0057 | -0.3845 | 0.5000 | 0.2377 | 0.0674 | 0.6013 |
| CSI500 | -0.0200 | -1.5638 | 0.4567 | 0.5295 | 0.1369 | 0.6338 |
| CSI1000 | -0.0313 | -2.7593 | 0.4326 | 1.2284 | 0.3003 | 0.6556 |
| ALL | -0.0383 | -3.7876 | 0.4037 | 1.9934 | 0.4480 | 0.6645 |

## ALL-universe summary

- **factor_name**: tail_adjusted_momentum_60d
- **universe**: ALL
- **rank_ic_mean**: -0.0383
- **abs_rank_ic_mean**: 0.0383
- **icir**: -3.7876
- **ic_positive_ratio**: 0.4037
- **hl_sharpe**: 1.9934
- **hl_annu_ret**: 0.4480
- **hl_mdd**: -0.1868
- **hl_avg_turnover**: 0.6645
- **direction**: -1
- **monotonicity_score**: 0.1000
- **stats_title**: H-L, Direction: -1, AnnuRet: 44.80%, Sharpe: 1.99, MDD: -18.68%, Turnover: 0.66, RankIC: -0.0383, ICIR: -3.79

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0294 |
| 5.0000 | -0.0470 |
| 10.0000 | -0.0548 |
| 20.0000 | -0.0638 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000870 |
| 2 | 0.000867 |
| 3 | 0.000816 |
| 4 | 0.000749 |
| 5 | 0.000763 |
| 6 | 0.000692 |
| 7 | 0.000639 |
| 8 | 0.000505 |
| 9 | 0.000092 |
| 10 | -0.000922 |
| H-L | -0.001792 |

Monotonicity score: **10.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
