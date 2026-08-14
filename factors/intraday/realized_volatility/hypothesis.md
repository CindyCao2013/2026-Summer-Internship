# realized_volatility

## Hypothesis

High session-to-current realized volatility identifies stocks experiencing
unusually large intraday price moves. It measures the magnitude, rather than
the direction, of the price path available at each signal time and may capture
temporary uncertainty, information arrival, and volatility persistence.

## Data and signal times

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Input: split-adjusted minute close
- Signal times: 09:59, 10:29, 11:29, 13:29, 14:29
- Minimum observations: five valid simple minute returns per session
- Range: nonnegative

## No-look-ahead contract

- Bars are sorted by symbol, trading date, and bartime.
- Returns and squared-return sums reset at every trading-session boundary.
- The value at time \(t\) uses only closes timestamped at or before \(t\).
- Filtering to the five standard bartimes occurs after the ordered cumulative
  calculation.
- Changing, appending, or deleting bars after \(t\) cannot alter its signal.

## Production

The package delegates both reference and production computation to
`factors.intraday.discovery_v1`. The environment/configuration key
`INTRADAY_REALIZED_VOLATILITY_USE_DDB` selects DolphinDB; pandas remains the
golden reference.
