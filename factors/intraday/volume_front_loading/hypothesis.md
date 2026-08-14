# volume_front_loading

## Hypothesis

Unusually high early-session trading activity may measure information arrival and
attention. The factor is:

\[
\frac{V_{09:30\text{–}10:00,t}}
{\operatorname{mean}(V_{09:30\text{–}10:00,t-i}, i=1,\ldots,N)}
\]

where the denominator uses prior observed trading sessions only. The signal is
known after 10:00 and stamped at 10:29 for the standard return matrix.

## Data and implementation

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Python reference: shifted rolling mean in
  `core.intraday_alphas._compute_volume_front_loading_python`
- DDB-native: daily `sum(Volume)`, then `context by Symbol csort Date` with
  shifted `msum/mcount`
- Output: `bartime, symbol, factorname, value`

## No-look-ahead contract

The denominator must not include day T:

```dolphindb
move(msum(morning_vol, lookback, minP), 1)
\
move(mcount(morning_vol, lookback, minP), 1)
```

This is equivalent to pandas `shift(1).rolling(...).mean()`. A positive shift
uses the rolling state ending at T-1. Negative `move` is forbidden.

## Migration gates

1. Numeric max absolute difference `< 1e-10`
2. Cross-sectional Spearman `> 0.999`
3. Signal timestamp exactly `10:29`
4. IC, ICIR, decile excess Sharpe, H-L Sharpe and turnover parity

## Production flag

`factor_config.INTRADAY_VOLUME_FRONT_USE_DDB` selects DDB-native computation.
The Python implementation remains the reference and fallback.
