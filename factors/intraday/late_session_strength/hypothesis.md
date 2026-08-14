# late_session_strength

## Hypothesis

Closing-session active buying contains directional information beyond total
volume. The factor is:

\[
\frac{\sum_{14:30\text{–}15:00} ActiveBuyAmount}
{\sum_{14:30\text{–}15:00}(ActiveBuyAmount+ActiveSellAmount)}
\]

Higher values indicate stronger buyer-initiated flow. Because the complete
window is known only after the close, day-T information is stamped at T+1 09:59.

## Data and adjustment

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Fields: `Active_buy_amount`, `Active_sell_amount`, `Adjfactor`
- Both sides are adjusted using the canonical `_safe_adjust` semantics.
- All-null grouped amounts use zero, matching pandas `groupby.sum()`.

## No-look-ahead contract

- Only bars from 14:30 through 15:00 on source day T are aggregated.
- DDB returns source `Date`; it does not create a same-day tradable timestamp.
- Python applies the existing `BDay(T)+09:59` mapping to the narrow DDB result.
- Changing T+1 flow cannot change the T-derived 09:59 signal.

## Migration gates

1. Numeric max absolute difference `< 1e-10`
2. Cross-sectional Spearman `> 0.999`
3. Signal timestamp exactly T+1 09:59
4. Signal values remain in `[0, 1]`
5. IC, ICIR, G1–G10 excess Sharpe, H-L Sharpe and turnover parity

## Production flag

`factor_config.INTRADAY_LATE_SESSION_STRENGTH_USE_DDB` selects DDB-native
computation. Python remains the golden reference and explicit fallback.
