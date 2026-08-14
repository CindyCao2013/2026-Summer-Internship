# volume_back_loading

## Hypothesis

Unusually high closing-session trading activity may measure late information
arrival, institutional execution and liquidity demand:

\[
\frac{V_{14:30\text{–}15:00,t}}
{\operatorname{mean}(V_{14:30\text{–}15:00,t-i}, i=1,\ldots,N)}
\]

The numerator is known only after day-T close, so the signal is stamped at
T+1 09:59.

## Implementation

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- DDB: `sum(Volume)` by Symbol/Date, then shifted `msum/mcount`
- Python reference: `shift(1).rolling(...).mean()`
- Timestamp: existing pandas `BDay(T)+09:59` contract
- Output: `bartime, symbol, factorname, value`

Only the narrow daily factor state crosses from DDB to Python. Python applies
the existing next-business-day timestamp convention; it does not recompute the
factor.

## No-look-ahead contract

1. Historical denominator uses `move(..., 1)` and therefore ends at T-1.
2. Day-T closing volume is the numerator only.
3. The signal becomes tradable at T+1 09:59.
4. Negative/forward `move` is forbidden.

## Migration gates

1. Numeric max absolute difference `< 1e-10`
2. Cross-sectional Spearman `> 0.999`
3. Signal timestamp exactly next-business-day `09:59`
4. IC, ICIR, decile excess Sharpe, H-L Sharpe and turnover parity

## Production flag

`factor_config.INTRADAY_VOLUME_BACK_USE_DDB` selects DDB-native computation.
The Python implementation remains the reference and fallback.
