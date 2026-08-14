# Alpha Report: `cn_cancel_shock`

- **Dimension**: L2
- **Role**: enhancer
- **Hypothesis**: Cancel withdrawal shock — trade-flow state

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0087 | -1.3450 | 0.4553 | 1.3030 | 0.1680 | 3.3117 |
| CSI500 | -0.0039 | -0.7349 | 0.4546 | 0.7363 | 0.0764 | 3.2921 |
| CSI1000 | -0.0070 | -1.5673 | 0.4381 | 1.2796 | 0.1182 | 3.2950 |
| ALL | -0.0069 | -1.7737 | 0.4223 | 1.8882 | 0.1641 | 3.2778 |

## ALL-universe summary

- **factor_name**: cn_cancel_shock
- **universe**: ALL
- **rank_ic_mean**: -0.0069
- **abs_rank_ic_mean**: 0.0069
- **icir**: -1.7737
- **ic_positive_ratio**: 0.4223
- **hl_sharpe**: 1.8882
- **hl_annu_ret**: 0.1641
- **hl_mdd**: -0.1766
- **hl_avg_turnover**: 3.2778
- **direction**: -1
- **monotonicity_score**: 0.3000
- **stats_title**: H-L, Direction: -1, AnnuRet: 16.41%, Sharpe: 1.89, MDD: -17.66%, Turnover: 3.28, RankIC: -0.0069, ICIR: -1.77

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0018 |
| 5.0000 | 0.0018 |
| 10.0000 | 0.0038 |
| 20.0000 | 0.0062 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000774 |
| 2 | 0.000624 |
| 3 | 0.000575 |
| 4 | 0.000505 |
| 5 | 0.000422 |
| 6 | 0.000454 |
| 7 | 0.000582 |
| 8 | 0.000502 |
| 9 | 0.000517 |
| 10 | 0.000118 |
| H-L | -0.000656 |

Monotonicity score: **30.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
