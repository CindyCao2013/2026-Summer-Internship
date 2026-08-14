# large_active_buy_ratio

## Research hypothesis

Bars whose average active-buy transaction size is unusually high relative to
their own recent session history may describe a distinct short-horizon buying
regime. The factor measures how much of the recent active-buy amount occurred
in bars classified into that regime.

## Critical interpretation limit

This is a **BAR-LEVEL PROXY**. The live minute-bar schema does not contain true
large-order amount buckets or order-level size labels. Consequently:

- it does not identify individual large orders;
- it is not an individual large-order ratio;
- it must not be described as institutional flow;
- a whole bar is classified from its average active-buy transaction size, and
  all active-buy amount in a classified bar enters the numerator.

The name is retained for factor-catalog compatibility, but reports and research
must preserve the proxy qualification.

## Data and timing

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Required live fields: active-buy amount, active-buy count, adjustment factor
- Signal times: 09:59, 10:29, 11:29, 13:29, 14:29
- History resets for each symbol and trading session
- Output range: `[0, 1]`

## No-look-ahead contract

The classification threshold at bar `t` uses only valid bars from the preceding
20-row rolling window, with at least 10 valid observations. Both the mean and
sample standard deviation are shifted by one bar, so bar `t` cannot enter its
own threshold. The trailing ratio at `t` uses bars through `t`; later bars
cannot alter an earlier signal.
