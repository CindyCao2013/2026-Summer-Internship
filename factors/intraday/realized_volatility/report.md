# realized_volatility implementation report

## Scope

This package is a factor-local integration of the existing
`factors.intraday.discovery_v1` implementation. No shared source files are
modified.

## Contract

- Formula: session-to-current square root of summed squared simple minute
  returns
- Minimum history: five valid minute returns
- Reset key: symbol and trading date
- Signal times: 09:59, 10:29, 11:29, 13:29, 14:29
- Value domain: finite and nonnegative
- Backend gate: `INTRADAY_REALIZED_VOLATILITY_USE_DDB`

## Validation

`test_compare.py` covers:

1. Direct formula agreement on a deterministic session.
2. Invariance of the 09:59 signal to an arbitrarily changed 10:00 close.
3. Nonnegative values and reset at the next trading session.
4. Ordered, date-partitioned cumulative DolphinDB query construction.
5. Registry dispatch through the backend gate.
6. Python/DolphinDB numeric and bartime parity when `RUN_DDB_TESTS=1`.

Live signal and full-backtest parity are intentionally gated because they
require DolphinDB and market data. `run_backtest_parity.py` provides explicit
`--signal`, `--backtest`, and `--all` entry points for that environment.

## Test result

The non-live suite passed on 2026-07-29:

`python -m unittest factors.intraday.realized_volatility.test_compare -v`

Result: seven tests run, six passed, and the live DolphinDB parity test was
skipped by its `RUN_DDB_TESTS=1` gate.
