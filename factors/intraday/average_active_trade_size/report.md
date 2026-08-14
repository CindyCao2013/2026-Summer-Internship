# average_active_trade_size implementation report

Date: 2026-07-29

## Scope

The per-factor package delegates Python, DolphinDB, dispatch, normalization,
timestamp, alignment, and no-look-ahead behavior to
`factors.intraday.discovery_v1`. No shared implementation or configuration file
was changed.

## Contract implemented

- Current ticket: adjusted active-buy amount divided by active-buy count
- Baseline: prior 20 bars, at least 10 valid tickets, shifted by one bar
- Value: current ticket divided by the positive baseline, minus one
- Signal times: 09:59, 10:29, 11:29, 13:29, and 14:29
- Interpretation: a ticket-size anomaly only; no institutional identity claim

## Validation status

Non-live validation:

```text
/opt/conda/anaconda3/envs/base_93/bin/python -m unittest \
  factors.intraday.average_active_trade_size.test_compare -v

Ran 8 tests in 0.163s
OK (skipped=1)
```

Seven non-live tests passed. They cover the formula, minimum-valid-observation
gate, exact five timestamps, future-bar invariance, positive-shift SQL
contract, wrapper dispatch, and production-registry dispatch.

The skipped test is the gated live Python-versus-DDB parity test. Live signal
and full-backtest parity were not run and are not claimed as passing. Run them
explicitly with `RUN_DDB_TESTS=1` and `run_backtest_parity.py` when the DDB
environment is available.
