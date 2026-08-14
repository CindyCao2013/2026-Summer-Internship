# Implementation report

## Scope

Implemented only `factors/intraday/large_active_buy_ratio/`. The package wraps
`factors.intraday.discovery_v1`; no shared implementation or configuration file
was changed.

## Interpretation

`large_active_buy_ratio` is a **BAR-LEVEL PROXY** because the live schema lacks
true large-order amount buckets. It computes `buy_size = buy_amt / buy_count`,
classifies a bar when that value exceeds the shifted prior-20 mean plus shifted
prior-20 sample standard deviation (minimum 10 valid observations), and divides
trailing-20 classified-bar buy amount by trailing-20 total buy amount.

It is not an individual large-order ratio and is not evidence of institutional
flow.

## Included artifacts

- Python and DDB wrappers with `FACTOR_NAME = "large_active_buy_ratio"`
- DDB reference SQL
- Formula and hypothesis documentation
- Synthetic no-look-ahead, formula, bartime, and `[0,1]` tests
- Gated live Python/DDB parity test
- Gated signal and full-backtest parity runner

## Validation

Non-live validation used `/opt/conda/anaconda3/bin/python`:

```text
python -m unittest factors.intraday.large_active_buy_ratio.test_compare -v
Ran 5 tests in 0.139s
OK (skipped=1)
```

The skipped test is the live DDB parity check, gated by `RUN_DDB_TESTS=1`.
The non-live production registry trace also passed:

```text
TRACE OK: registry → discovery_v1.compute_factor → ddb_version
```

Live signal and backtest parity were not run. They require explicit opt-in:

```bash
RUN_DDB_TESTS=1 /opt/conda/anaconda3/bin/python \
  factors/intraday/large_active_buy_ratio/run_backtest_parity.py --all
```
