# Order Size Family v1 — Frozen Discovery Contract

## Scope

Sprint 4 expands the L2 candidate universe. It does **not** optimize thresholds,
neutralize exposures, select survivors, combine factors, or construct a
portfolio.

```text
ClickHouse Tick
  -> order_size_distribution_daily primitive
  -> 20 threshold-explicit factor narrow files
  -> standard T+1 baseline backtest
  -> yearly stability + within-family redundancy
```

## Data contract

- Sample target: 2019-01-01 through 2026-07-31.
- Asset scope: A-share code prefixes only.
- Session: `09:30:00 <= ExchTime < 15:00:01` on every date.
- SSE executions: `Type='T'`; amount is
  `ifNull(Amount, Price * Volume)`; aggressor side uses `BSFlag`.
- SZSE executions: `Type='011'` with both order numbers positive; amount is
  `Price * Volume`; the later order number identifies the aggressor.
- Factor values use the complete trading day and enter the backtest with
  `signal.shift(1)`.

The primitive stores total amount/count and, at each fixed boundary, cumulative
amount, cumulative count, cumulative active-buy amount, and cumulative
active-sell amount.

## Frozen boundaries

The canonical distribution uses five mutually exclusive amount buckets:

1. `(0, 1w]`
2. `(1w, 5w]`
3. `(5w, 20w]`
4. `(20w, 100w]`
5. `(100w, +inf)`

An additional 4w cumulative boundary preserves the existing research formula
`mid_order_ratio = amount(4w,20w] / total`.

Thresholds are encoded in factor names where ambiguity would otherwise exist.
The legacy `small_order_ratio` implementation means `<=4w`; Sprint 4 therefore
uses `small_order_ratio_1w` and `small_order_ratio_4w` rather than silently
redefining that name.

## Candidate registry (20)

### Distribution levels

- `small_order_ratio_1w`
- `small_order_ratio_4w`
- `mid_order_ratio_4w_20w`
- `mid_order_ratio_5w_20w`
- `large_order_ratio_20w`
- `super_large_order_ratio_100w`
- `large_small_spread`

### Distribution shape

- `order_size_entropy`: normalized Shannon entropy across the five amount
  shares.
- `order_size_concentration`: HHI across the same shares.
- `order_size_tail_share`: smallest-bucket plus super-large-bucket share.

### Size-conditioned direction

- `small_order_pressure`
- `mid_order_pressure`
- `large_order_pressure`
- `super_large_order_pressure`
- `buy_large_order_ratio`
- `sell_large_order_ratio`
- `small_order_direction`
- `large_order_direction`

`pressure` divides signed bucket flow by total market amount. `direction`
divides signed flow by classified buy+sell amount inside that bucket.

### Time-series shocks

- `large_order_shock_20d`
- `order_size_entropy_shock_20d`

Each compares today's value with the strictly preceding 20 observations.
Current-day information is only in the numerator; the unified backtest then
lags the completed signal by one day.

## Audit invariants

The build fails if any of the following occurs:

- duplicate symbol-day rows;
- non-monotonic cumulative amount/count/side buckets;
- cumulative amount or count above the daily total;
- classified buy+sell amount above bucket or daily total;
- negative mutually exclusive bucket amount;
- factor ratios outside their mathematical bounds.

UInt count columns are converted to signed floating-point before arithmetic to
prevent the unsigned-subtraction failure previously found in
`net_buy_count_ratio`.

## Expected redundancy (not a pruning decision)

- The two mid-order thresholds are expected to be near aliases.
- Entropy and HHI measure opposite aspects of the same distribution shape.
- Buy-side and sell-side large-order ratios may share a stock-scale component.
- `large_small_spread` is a deterministic linear combination of two level
  features.

These features remain in the candidate registry for taxonomy and audit. They
must not be counted as independent alphas solely because they have separate
names.

