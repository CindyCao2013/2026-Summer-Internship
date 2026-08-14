# ofi_persistence implementation report

## Scope

This package adds the research record, package-level compute wrappers, a
representative DolphinDB script, causality/range tests, and a parity runner for
`ofi_persistence`. Shared modules are consumed unchanged.

## Implementation

- Python and production dispatch delegate to
  `factors.intraday.discovery_v1`.
- DolphinDB execution delegates to `discovery_v1_factor_script` through the
  shared implementation.
- The calculation uses the trailing 20 observed trading-minute bars within
  each `Symbol, Date`, includes the current bar, and requires five valid OFI
  observations.
- Signals are restricted to 09:59, 10:29, 11:29, 13:29, and 14:29.

## Validation gates

`test_compare.py` covers:

1. inclusion of the current bar at the five-observation threshold;
2. invariance of an earlier signal when a future bar is changed;
3. the trailing-20 behavior and the `[0, 1]` range;
4. ordered `Symbol, Date, Bartime` DDB rolling-window construction;
5. optional live Python/DDB numeric, rank, row-count, and bartime parity.

Run non-live validation with:

```bash
python -m unittest factors.intraday.ofi_persistence.test_compare -v
```

The live integration test and signal/backtest parity remain opt-in:

```bash
RUN_DDB_TESTS=1 python -m unittest \
  factors.intraday.ofi_persistence.test_compare -v
RUN_DDB_TESTS=1 python \
  factors/intraday/ofi_persistence/run_backtest_parity.py --all
```

## Validation result

The non-live suite passed on 2026-07-29: four formula/causality/SQL tests
passed, and the live DolphinDB integration test was skipped as designed because
`RUN_DDB_TESTS` was not set.
