# Hypothesis: Minute Skew

## Research question

Does the asymmetry of minute returns observed from the session open to the
current decision time contain information not captured by realized volatility
or price/VWAP state?

## Signal

At each standard bartime, compute the bias-corrected sample skewness of all
valid one-minute simple returns observed in the same stock-session so far.

Positive values indicate a right-tailed intraday return distribution; negative
values indicate downside-tail concentration. The sign is a research question,
not fixed ex ante.

## Contract

- Inputs: `Close`, `Adjfactor`, `Symbol`, `Date`, `Bartime`.
- Session state resets by `Symbol, Date`.
- Current and past bars are included; later bars are forbidden.
- Python expanding skew is the golden reference.
- DDB uses ordered cumulative first, second and third raw moments.
- Signals are emitted only at 09:59, 10:29, 11:29, 13:29 and 14:29.

Status: discovery candidate; signal parity is required before research use and
full backtest parity is required before production promotion.
