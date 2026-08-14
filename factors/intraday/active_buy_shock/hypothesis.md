# active_buy_shock

## Hypothesis

An unusually large adjusted active-buy amount relative to the stock's recent
within-session flow may identify abrupt aggressive demand. The factor is a
causal bar-level z-score: a larger positive value means the current active-buy
amount is farther above its immediately preceding intraday baseline.

## Data and signal times

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Inputs: `Active_buy_amount`, `Adjfactor`, `Symbol`, `Date`, `Bartime`
- Session: 09:30–11:30 and 13:00–15:00
- Signal times: 09:59, 10:29, 11:29, 13:29, 14:29
- Window: prior 20 bars, at least 10 prior observations
- Standard deviation: sample standard deviation (`ddof=1`)

## No-look-ahead contract

- Adjust active-buy amount before computing the baseline.
- Sort by `Symbol, Date, Bartime`; reset rolling state every session.
- Shift both rolling statistics by one positive bar, so the current bar is
  excluded and only the preceding 20 bars can enter the baseline.
- A future bar cannot change an already emitted signal.
- Emit no value when fewer than 10 prior observations exist or historical
  standard deviation is zero or invalid.

## Research expectation

The sign is intentionally not inverted. Positive shocks represent aggressive
buying pressure; whether that pressure continues or reverses is an empirical
question for the unchanged intraday evaluation layer.
