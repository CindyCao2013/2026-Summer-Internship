# Order Book Family v1 — Frozen Discovery Contract

## Scope

This Sprint builds one reusable daily primitive from ClickHouse SSL2 Snapshot
tables and screens frozen daily factors. It does not optimize parameters,
combine factors, neutralize a second time, or make KEEP/DROP decisions.

Source tables:

- `cmds.SSE_AL_SSL2_EXG`
- `cmds.SZSE_AL_SSL2_EXG`

Target sample is 2019-01-01 through 2026-07-31. Phase-0 inventory found
continuous SSE coverage from 2015-01-05 and SZSE coverage from 2008-01-02, so
the target period is supported by both exchanges.

## Source normalization

- Symbol policy: SSE code prefix `6`; SZSE prefixes
  `000/001/002/003/300/301/302`; append `.SH` or `.SZ`.
- ClickHouse arrays are one-indexed. Level 1 is `array[1]`.
- Require at least ten elements in all four price/volume arrays.
- Require `Bid1 > 0`, `Ask1 >= Bid1`, non-negative ten-level volumes, and
  positive combined ten-level depth.
- SZSE additionally requires `Type='010'`. Its paired empty-array rows are
  rejected by the ten-level filter.
- Continuous grid contains 240 minutes:
  `[09:30, 11:30)` and `[13:00, 15:00)`.
- The valid last state in each `symbol × TradeDate × minute` is selected with
  `argMax(metric, ExchTime)`.
- Valid 15:00-minute close-auction states are retained in separate audit
  columns and never mixed into continuous-session daily means.
- No forward fill is permitted.

## Snapshot formulas

All depth inputs are cast to signed Float64 before subtraction.

- `obi_1`, `obi_5`, `obi_10` use
  `(bid_depth - ask_depth) / (bid_depth + ask_depth)`.
- `weighted_obi` uses the single frozen weight `w_k = 1/k`, `k=1..10`.
- `relative_spread = (Ask1 - Bid1) / ((Ask1 + Bid1)/2)`.
- `microprice = (Ask1*BidVolume1 + Bid1*AskVolume1) /
  (BidVolume1 + AskVolume1)`.
- `near_far_imbalance` is bid near-depth share minus ask near-depth share,
  where near is levels 1-3 and far is levels 4-10.
- Bid/ask HHI is the sum of squared ten-level depth shares.
- Depth slope is OLS slope of cumulative depth share on relative price
  distance from mid. Cumulative depth is normalized by the same-side
  ten-level total before regression, making bid and ask slopes comparable.
- `book_vwap_gap` always uses a self-computed ten-level price-volume VWAP.
  Phase 0 found SZSE source `BidVWAP/AskVWAP` unusable on all valid rows.
- `log_total_depth = log1p(bid_depth_10 + ask_depth_10)`.

Every ratio uses a safe denominator. Invalid values become NULL, never
infinity or an imputed zero.

## Daily primitive contract

Keys and coverage:

- `symbol`, `TradeDate`
- `valid_minute_count`, `expected_minute_count=240`, `coverage_ratio`
- raw and valid source-row counts for audit

Continuous-session fields:

- Means: `obi_1_mean`, `obi_5_mean`, `obi_10_mean`,
  `weighted_obi_mean`, `relative_spread_mean`,
  `microprice_deviation_mean`, `near_far_imbalance_mean`,
  `bid_depth_hhi_mean`, `ask_depth_hhi_mean`,
  `depth_concentration_asymmetry_mean`, `bid_depth_slope_mean`,
  `ask_depth_slope_mean`, `depth_slope_asymmetry_mean`,
  `book_vwap_gap_mean`, `log_total_depth_mean`
- Volatility: `obi_1_std`, `obi_5_std`, `weighted_obi_std`,
  `relative_spread_std`, `microprice_deviation_std`,
  `log_total_depth_std`
- Segments: opening/closing 30-minute OBI5, spread, and log depth
- Trend/persistence: OBI5/spread/depth intraday slopes,
  `obi_5_sign_persistence`, `spread_widening_share`
- Extremes: OBI5 p10/p90, spread p90, log-depth p10
- Close auction: valid flag/count, OBI5, spread, and log depth

The primitive retains low-coverage symbol-days for audit. Factor values are
emitted only when `coverage_ratio >= 0.80`. This threshold is frozen for v1.

## Frozen factor registry

The registry contains 32 formulas:

### Level / imbalance (7)

1. `obi_l1_mean`
2. `obi_l5_mean`
3. `obi_l10_mean`
4. `weighted_obi_mean`
5. `obi_l1_volatility`
6. `obi_l5_volatility`
7. `weighted_obi_volatility`

### Book shape (7)

8. `near_far_imbalance`
9. `bid_depth_concentration`
10. `ask_depth_concentration`
11. `depth_concentration_asymmetry`
12. `bid_depth_slope`
13. `ask_depth_slope`
14. `depth_slope_asymmetry`

### Spread / price pressure (7)

15. `relative_spread_mean`
16. `relative_spread_volatility`
17. `microprice_deviation_mean`
18. `microprice_deviation_volatility`
19. `book_vwap_gap`
20. `total_depth_level`
21. `total_depth_volatility`

### Intraday timing (7)

22. `opening_obi_l5`
23. `closing_obi_l5`
24. `opening_closing_obi_change`
25. `opening_closing_spread_change`
26. `opening_closing_depth_change`
27. `obi_intraday_slope`
28. `obi_sign_persistence`

### Dynamic / shock (4)

29. `obi_shock_20d`
30. `spread_shock_20d`
31. `depth_shock_20d`
32. `microprice_shock_20d`

The shock denominator and mean use the current observation plus at most 19
historical observations, require 20 observations, and never use a future row.
The unified backtest subsequently applies `signal.shift(1)`.

Linear identities such as bid/ask/asymmetry and
opening/closing/change are explicitly tagged as expected redundancy. Formula
count is not interpreted as independent-alpha count; empirical daily
cross-sectional Spearman clusters provide that taxonomy.

## Evaluation contract

- Existing all-A investability filters and T+1 return pipeline
- Raw IC/ICIR in the frozen formula direction
- Group tests rerun after factor-level effective-direction flip
- 7.5 bps one-way cost convention
- Daily baseline only; no weekly optimization or second neutralization
- Family-local high-correlation threshold:
  `|mean daily cross-sectional Spearman| >= 0.80`
