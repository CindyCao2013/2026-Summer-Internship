# bartime_ofi validation report

## Status

**Candidate — pending full backtest.**

The package defines local gates for:

- current-minute formula and all five standard bartimes;
- future-bar isolation and generated-query no-look-ahead structure;
- production dispatch through the shared DDB implementation;
- exact signal keys, numeric values, cross-sectional ranks and bounds;
- live Python/DDB parity when `RUN_DDB_TESTS=1`;
- full Python/DDB backtest parity through the established intraday harness.

## Promotion criteria

Promotion requires a successful live signal comparison and full unchanged
backtest comparison covering IC, ICIR, decile and H-L Sharpe, universe/row
counts and turnover. Until those gates run, this factor remains a research
candidate even though DDB is available.

Python remains the golden reference and explicit fallback through
`INTRADAY_BARTIME_OFI_USE_DDB=false` (or `False` in `factor_config`).
