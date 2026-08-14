# Alpha Report: `low_vol_liquidity_quality_60d`

- **Dimension**: D1
- **Role**: base
- **Hypothesis**: Low vol × liquidity stability — quality liquidity state

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0406 | 3.1960 | 0.5901 | 0.6249 | 0.1486 | 0.5142 |
| CSI500 | 0.0468 | 4.3643 | 0.6149 | 0.8982 | 0.1833 | 0.5127 |
| CSI1000 | 0.0534 | 5.3675 | 0.6431 | 1.8444 | 0.3641 | 0.5001 |
| ALL | 0.0573 | 6.0135 | 0.6534 | 2.2614 | 0.4351 | 0.4822 |

## ALL-universe summary

- **factor_name**: low_vol_liquidity_quality_60d
- **universe**: ALL
- **rank_ic_mean**: 0.0573
- **abs_rank_ic_mean**: 0.0573
- **icir**: 6.0135
- **ic_positive_ratio**: 0.6534
- **hl_sharpe**: 2.2614
- **hl_annu_ret**: 0.4351
- **hl_mdd**: -0.1973
- **hl_avg_turnover**: 0.4822
- **direction**: 1
- **monotonicity_score**: 0.8000
- **stats_title**: H-L, Direction: 1, AnnuRet: 43.51%, Sharpe: 2.26, MDD: -19.73%, Turnover: 0.48, RankIC: 0.0573, ICIR: 6.01

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0504 |
| 5.0000 | 0.0708 |
| 10.0000 | 0.0811 |
| 20.0000 | 0.0929 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000869 |
| 2 | 0.000006 |
| 3 | 0.000326 |
| 4 | 0.000509 |
| 5 | 0.000684 |
| 6 | 0.000758 |
| 7 | 0.000874 |
| 8 | 0.000952 |
| 9 | 0.000934 |
| 10 | 0.000872 |
| H-L | 0.001740 |

Monotonicity score: **80.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
