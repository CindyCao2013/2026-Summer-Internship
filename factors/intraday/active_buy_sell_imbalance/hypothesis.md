# active_buy_sell_imbalance

## Hypothesis

Aggressive buy/sell pressure accumulated through the session measures informed
directional participation:

\[
OFI_t =
\frac{\sum_{i \le t} ActiveBuyAmount_i-\sum_{i \le t} ActiveSellAmount_i}
{\sum_{i \le t} ActiveBuyAmount_i+\sum_{i \le t} ActiveSellAmount_i}
\]

Unlike `late_session_strength`, this factor uses session-to-date flow and emits
signals at five intraday bartimes. It is therefore not merely the affine
transformation `2 × late_session_strength − 1`.

## Data and signal times

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Fields: active buy/sell amount and adjustment factor
- Signal times: 09:59, 10:29, 11:29, 13:29, 14:29
- Range: `[-1, 1]`

## No-look-ahead contract

- Rows are sorted by `Symbol, Date, Bartime`.
- `cumsum` uses only current and earlier bars in the same session.
- Null active amounts contribute zero.
- A signal is emitted only when cumulative active amount is positive.
- No day-end or future-bar state may enter an earlier bartime.

## Migration gates

1. Numeric max absolute difference `< 1e-10`
2. Cross-sectional Spearman `> 0.999`
3. Exact bartime alignment
4. Values in `[-1, 1]`
5. IC, ICIR, G1–G10 excess Sharpe, H-L Sharpe and turnover parity

## Production flag

`factor_config.INTRADAY_ACTIVE_BUY_SELL_IMBALANCE_USE_DDB` selects the DDB
backend. Python remains the reference and explicit fallback.
