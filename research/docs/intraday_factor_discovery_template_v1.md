# Intraday Factor Discovery Template v1

This template separates rapid alpha expansion from production promotion. A
factor may use the DDB backend as a discovery candidate without being added to
the default production run list.

## Package contract

```text
factors/intraday/<factor_name>/
├── __init__.py
├── hypothesis.md
├── formula.md
├── compute.py
├── ddb.sql
├── test_compare.py
├── run_backtest_parity.py
└── report.md
```

Shared formulas and normalization may live in
`factors/intraday/discovery_v1.py`; each package remains the factor's explicit
research record and fallback entry point.

## Required interfaces

`compute.py` exposes:

- `python_version(...)`: golden reference using `MinuteBarStore`.
- `ddb_version(...)`: server-side narrow-table computation.
- `compute(...)`: backend-dispatched entry point.
- timestamp, alignment and no-look-ahead assertions.

Output schema:

```text
bartime | symbol | factorname | value
```

Only the existing `narrow_for_ddb()` adapter renames `bartime` to `tradetime`.

## Candidate gate

Before a factor enters research screening:

1. Formula test on a deterministic frame.
2. Explicit future-bar mutation test.
3. DDB SQL contract ordered by `Symbol, Date, Bartime`.
4. Python/DDB maximum absolute difference `<= 1e-10`.
5. Cross-sectional Spearman `>= 0.999`.
6. Exact signal timestamp alignment.
7. Registration in `INTRADAY_FACTOR_BACKEND`.

## Production gate

Candidate status does not imply production status. Promotion additionally
requires:

1. Full unchanged-layer backtest parity.
2. Multi-horizon IC/ICIR and alpha-decay analysis.
3. Measured position turnover and cost-adjusted performance.
4. Chronological out-of-sample validation.
5. Exposure checks appropriate to the hypothesis.

Candidates are not automatically added to `INTRADAY_CUSTOM_FACTOR_LIST`.

## Data-meaning rule

Do not infer unavailable microstructure fields. The current minute table has
active buy/sell amounts and counts but no true large-order amount bucket.
`large_active_buy_ratio` in Expansion v1 is therefore a documented bar-level
average-ticket-size proxy, not an individual large-order or institutional-flow
measurement.
