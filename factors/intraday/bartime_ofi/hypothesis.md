# bartime_ofi

## Hypothesis

Aggressive order flow in the current minute captures short-lived directional
pressure. A positive value means active buying dominates active selling in that
minute; a negative value means the reverse. Unlike a session-cumulative
imbalance, each observation deliberately forgets prior bars.

The signal is sampled at the five standard bartimes: 09:59, 10:29, 11:29,
13:29 and 14:29.

## Expected behavior

- Range: `[-1, 1]`.
- Larger values indicate stronger current-minute buying pressure.
- A zero total active amount produces no signal rather than an artificial zero.
- The factor may decay quickly because it contains no rolling or cumulative
  state.

## Research status

This is a discovery candidate. Formula, no-look-ahead, route and implementation
parity gates are defined here; promotion remains pending a full backtest.
