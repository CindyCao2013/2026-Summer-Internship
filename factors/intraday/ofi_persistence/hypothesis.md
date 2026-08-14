# ofi_persistence

## Hypothesis

Persistent positive order-flow imbalance (OFI) indicates repeated aggressive
buying rather than a single transient flow shock. The proportion of recent
valid minute bars with positive OFI should therefore rank stocks by the
consistency of short-horizon buying pressure.

The factor is bounded and intentionally ignores OFI magnitude. A value near
one means nearly every valid bar in the trailing window has positive OFI; a
value near zero means positive OFI is rare.

## Data and signal times

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Inputs: raw positive active buy and active sell amounts
- Window: trailing 20 observed trading-minute bars within `Symbol, Date`
- Minimum valid OFI observations: 5
- Current bar: included
- Signal times: 09:59, 10:29, 11:29, 13:29, 14:29
- Range: `[0, 1]`

## No-look-ahead contract

Bars are sorted by `Symbol, Date, Bartime` before the trailing calculation.
Only the current and preceding 19 bars in the same symbol-session may affect a
signal. Invalid bar OFI values are excluded from both numerator and
denominator, and no signal is emitted before five valid values are available.
