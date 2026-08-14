# intraday_amihud

## Hypothesis

Short-horizon price movement per unit of traded amount measures immediate
illiquidity. A larger value means that less traded amount is required to
move the price, identifying a thinner and potentially more fragile order book.

The factor is an intraday state variable rather than a directional return
forecast. Its sign is always non-negative; downstream tests determine whether
high illiquidity carries a premium or is more useful as a conditioning signal.

## Data and sampling

- Source: `dfs://QV_Trade_to_MinuteBar/Stock_one_minute`
- Inputs: raw minute `Close` and `Amount`
- Window: trailing five observed trading-minute rows within each symbol/session
- Minimum observations: three
- Signals: 09:59, 10:29, 11:29, 13:29, and 14:29
- Scale: unscaled canonical ratio; scaling is presentation-only.

## No-look-ahead contract

- Simple minute return uses only current and immediately preceding close in the
  same `Symbol, Date` session; a constant daily adjustment factor would cancel.
- Rolling return and amount sums use only the current and four preceding rows.
- Grouping by `Symbol, Date` resets both return lag and rolling windows at each
  session boundary.
- No future bar, next session, or day-end aggregate can alter an earlier signal.

## Validation gates

1. Synthetic formula and denominator checks.
2. Future-bar mutation and prior-session mutation checks.
3. Exact five-bartime contract.
4. Live Python/DolphinDB formula parity: max absolute difference `<= 1e-10`.
5. Live rank parity: per-bartime Spearman `>= 0.999`.
6. Backtest parity for IC, ICIR, grouped Sharpe, and turnover.
