# average_active_trade_size

## Hypothesis

An unusually large average active-buy ticket can describe a change in the
current bar's aggressive-buy participation relative to that stock's recent
intraday baseline. The signal is a trade-size anomaly, not evidence about the
identity of the traders. In particular, it must not be interpreted as proving
institutional participation.

The comparison is local to each symbol and trading session. This reduces the
mechanical cross-stock scale difference in transaction amounts while retaining
changes in bar-level active-buy ticket size.

## Data and signal times

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Inputs: active-buy amount, active-buy count and adjustment factor
- Signal times: 09:59, 10:29, 11:29, 13:29 and 14:29
- Current ticket size is valid only when active-buy count is positive
- Baseline: prior 20 bars, at least 10 valid ticket-size observations

## Expected interpretation

Positive values mean that the current valid active-buy ticket is larger than
its shifted recent mean; negative values mean it is smaller. The factor is
unbounded and should be evaluated cross-sectionally rather than assigned a
participant identity.

## No-look-ahead contract

- Rows are ordered by `Symbol, Date, Bartime`.
- The rolling mean is shifted by one bar.
- The current bar never enters its own denominator.
- Negative/forward shifts are forbidden.
- Signals are emitted only at the five standard bartimes.
