# Alpha Report: `AlphaSKEW20`

- **Dimension**: HM
- **Role**: base
- **Hypothesis**: Alpha=-SKEW20 — lottery skewness anomaly (daily)

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0106 | 1.3481 | 0.5447 | 0.1147 | 0.0180 | 0.7154 |
| CSI500 | 0.0156 | 2.5373 | 0.5564 | 0.1828 | 0.0236 | 0.7254 |
| CSI1000 | 0.0187 | 3.2635 | 0.5867 | 0.7392 | 0.0884 | 0.7165 |
| ALL | 0.0247 | 4.7783 | 0.6348 | 1.4224 | 0.1684 | 0.7118 |

## ALL-universe summary

- **factor_name**: AlphaSKEW20
- **universe**: ALL
- **rank_ic_mean**: 0.0247
- **abs_rank_ic_mean**: 0.0247
- **icir**: 4.7783
- **ic_positive_ratio**: 0.6348
- **hl_sharpe**: 1.4224
- **hl_annu_ret**: 0.1684
- **hl_mdd**: -0.1506
- **hl_avg_turnover**: 0.7118
- **implied_annu_fee**: 0.1335
- **direction**: 1
- **monotonicity_score**: 0.7000
- **stats_title**: H-L, Direction: 1, AnnuRet: 16.84%,Sharpe_Ratio: 1.42, MDD: -15.06%, Daily Turnover: 0.71,
 Implied AnnuFee(7.5%): 13.35%, Daily IC: 0.0247, Annu ICIR: 4.78

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0235 |
| 5.0000 | 0.0307 |
| 10.0000 | 0.0338 |
| 20.0000 | 0.0345 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000086 |
| 2 | 0.000305 |
| 3 | 0.000442 |
| 4 | 0.000475 |
| 5 | 0.000612 |
| 6 | 0.000645 |
| 7 | 0.000641 |
| 8 | 0.000680 |
| 9 | 0.000677 |
| 10 | 0.000588 |
| H-L | 0.000674 |

Monotonicity score: **70.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
