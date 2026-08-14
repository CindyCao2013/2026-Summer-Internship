# intraday_amihud implementation report

## Implementation

The package binds `FACTOR_NAME = "intraday_amihud"` to the existing
`factors.intraday.discovery_v1` Python reference, DolphinDB builder, backend
selection, normalization, timestamp checks, and alignment helpers. No shared
files were changed.

The implemented value is the trailing five trading-minute sum of absolute
simple raw-close returns divided by the trailing sum of raw positive `Amount`.
The stored signal is the unscaled canonical ratio; display scaling
must not alter parity. Both windows use a minimum of three observations and
reset by symbol and trading session. Output is restricted to the five standard
bartimes.

## Non-live validation

Command:

```text
/opt/conda/anaconda3/envs/base_93/bin/python -m unittest \
  factors.intraday.intraday_amihud.test_compare -v
```

Result on 2026-07-29: `Ran 9 tests ... OK (skipped=2)`.

- Seven non-live tests passed.
- Formula, scale, minimum periods, session reset, future-bar isolation, exact
  bartimes, SQL leakage contract, and wrapper binding were covered.
- Both live DolphinDB tests were skipped by the intended `RUN_DDB_TESTS=1`
  gate.
- The package passed IDE diagnostics and Python bytecode compilation.
- The non-live production dispatch trace reached
  `factors.intraday.discovery_v1.ddb_version`.

## Live acceptance

Live signal and backtest parity are implemented but were not run as part of the
non-live request. Run:

```text
RUN_DDB_TESTS=1 /opt/conda/anaconda3/envs/base_93/bin/python \
  factors/intraday/intraday_amihud/run_backtest_parity.py --all
```

Acceptance thresholds are max absolute formula difference `<= 1e-10`,
per-bartime Spearman `>= 0.999`, exact signal keys, and exact IC, ICIR, grouped
Sharpe, signal-count, universe-count, and turnover parity.
