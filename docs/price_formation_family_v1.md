# Price Formation Family v1 — frozen specification

## Scope

Sprint 6 studies how trades and information enter the intraday price path. It
contains 32 level formulas and no parameter search, weekly/biweekly tuning,
secondary neutralization, factor combination, machine learning, or KEEP/DROP.
All daily signals are consumed by the shared baseline with
`signal.shift(1)`.

## Canonical source

- DolphinDB `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Target trading-day coverage: 2019-01-02 through 2026-07-31
- Price adjustment: `raw price * Adjfactor`
- Amount: CNY; volume: shares
- ClickHouse `SSE_AL_KLIN_EXG` and `SZSE_AL_KLIN_CMD` are parity references
  only. They are never concatenated with DolphinDB.

The source decision and the 2024-06 fixed-sample parity are under
`research/results/l2_reproduction/primitives/price_formation_daily/`.

## Minute grid

- Continuous morning: `[09:30, 11:30)` (120 labels)
- Continuous afternoon: `[13:00, 15:00)` (120 labels)
- Close auction: 15:00, stored separately
- Opening auction: 09:25, excluded from the continuous path

The DDB source structurally omits 14:57–14:59 and consolidates the closing
auction at 15:00. Price state may be carried from 14:56 through those three
labels, but amount and volume are never filled. Price fill is limited to three
consecutive labels inside one session and never crosses lunch.

`valid_minute_count` counts observed valid continuous prices.
`imputed_price_minute_count` counts price-state fills. A symbol-day is eligible
only when `coverage_ratio >= 0.80` and `daily_amount > 0`.

Realized moments use observed-to-observed minute returns. The lunch transition
is included when both 11:29 and 13:00 are observed. Imputed returns are not
treated as observations.

## Primitive formulas

For adjusted continuous close `p_t`:

- `r_t = log(p_t / p_{t-1})`
- `daily_vwap = sum(amount_t * Adjfactor_t) / sum(volume_t)`
- `CLV = (2*continuous_close-high-low)/(high-low)`
- `path_efficiency = abs(continuous_close-open) /
  (abs(first_close-open)+sum(abs(p_t-p_{t-1})))`
- `variance_ratio_5m = mean(log(p_t/p_{t-5})^2) /
  (5*mean(r_t^2))`
- `realized_variance = sum(r_t^2)`
- `realized_skewness = sqrt(N)*sum(r_t^3)/realized_variance^(3/2)`
- `realized_kurtosis = N*sum(r_t^4)/realized_variance^2`
- `bipower_variation = (pi/2)*sum(abs(r_t)*abs(r_{t-1}))`
- `jump_variation = max(realized_variance-bipower_variation, 0)`
- `tail_return_share` is the variance share from minute returns whose absolute
  value is at or above the symbol-day 95th percentile
- `volume_concentration_hhi = sum((amount_t/daily_amount)^2)`
- `amount_time_center = sum(minute_index_t*amount_t) /
  (239*sum(amount_t))`
- `intraday_amihud = mean(abs(r_t)/amount_t)` on `amount_t > 0`

`overnight_gap` uses the previous available adjusted continuous close because
the canonical minute table has no `PreClose` field.

## Frozen candidates

### Intraday Path (13)

`overnight_gap`, `open_to_30m_return`, `morning_return`,
`afternoon_return`, `closing_30m_return`, `lunch_gap_return`,
`close_auction_return`, `vwap_close_deviation`, `close_location_value`,
`path_efficiency`, `intraday_return_sign_persistence`,
`minute_return_autocorr1`, `variance_ratio_5m`.

### Realized Distribution (9)

`realized_volatility`, `downside_semivariance_share`,
`realized_skewness`, `realized_kurtosis`, `jump_share`,
`max_abs_minute_return`, `tail_return_share`,
`intraday_max_drawdown`, `intraday_max_drawup`.

### Volume and Timing (7)

`opening_amount_share`, `closing_amount_share`,
`morning_afternoon_amount_imbalance`, `volume_concentration_hhi`,
`amount_time_center`, `volume_return_corr`, `volume_abs_return_corr`.

### Price Impact / Efficiency (3)

`intraday_amihud`, `return_per_amount`, `range_per_amount`.

`realized_variance` remains a primitive but is not a candidate alias of
`realized_volatility`. Rank, z-score, winsorized, sign-flipped, and monotonic
rescaling clones are excluded. No rolling shock is part of v1.

Active-flow candidates 33–35 are excluded because the audited ClickHouse KLIN
tables do not expose a consistent SSE/SZSE minute active-flow definition.

## Evaluation and redundancy

- Benchmark: CSI1000 (`000852.SH`)
- Cost display: 7.5 bps implied fee
- Groups: ten
- Raw RankIC/ICIR: frozen formula direction
- Effective direction: grouping display only
- Production direction: not decided in Sprint 6
- Family redundancy: time mean of daily cross-sectional Spearman
- High-correlation threshold: `|rho| >= 0.80`
- Near-alias observation threshold: `|rho| >= 0.95`

Cross-family correlation is a read-only representative reference versus Trade
Flow, Order Size, and Order Book. It is not a selection or combination step.
