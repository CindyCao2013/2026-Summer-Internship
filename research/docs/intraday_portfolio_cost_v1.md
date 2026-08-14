# Intraday Portfolio Cost v1

## Corrected turnover interpretation

`intraday_lib.intraday_turnover_b_hl()` is not measured constituent turnover.
It returns the fixed model value

`2 × (long_gross + short_gross) = 4 × side_gross`.

With the old default `side_gross=1`, this represents a 100% long plus 100%
short portfolio opened and closed every day. `Factor_Dev_Lib.py` also labels
the associated implied annual fee as display-only rather than an actual
deduction. The value is neither market turnover nor a portfolio-weight
measurement.

The v1 simulator instead freezes a 50% long / 50% short portfolio, opens at the
factor's frozen bartime, and closes at its frozen horizon. It records:

- `half_l1_turnover = 0.5 × sum(abs(all entry and exit trades))`;
- `traded_notional_turnover = sum(abs(all entry and exit trades))`;
- `transaction_cost = traded_notional_turnover × 7.5bps`.

Average half-L1 turnover is approximately 1.0 per day and average traded
notional is approximately 2.0 for all four factors. This is the expected
round-trip result for a gross-100% market-neutral portfolio that is flat after
each frozen horizon.

## Why the old and new net Sharpe are similar

Changing from 100%/100% legs to 50%/50% legs halves both H-L return and traded
notional. Under a linear one-way cost, it therefore leaves the return-to-cost
ratio nearly unchanged. The old fixed value `4` was not a measured turnover,
but its cost conclusion happened to approximate this specific round-trip
portfolio after gross scaling.

The simulator verifies daily zero-cost return against
`0.5 × frozen_direction × group_HML`; the maximum difference across every
factor-period is below `4e-17`.

## Tradeability results at 7.5bps one-way

### Realized volatility

`realized_volatility` is the only factor with positive net annualized return
and positive net Sharpe in all three periods.

- 2024H1: gross annualized return 49.36%, net 11.90%, net Sharpe 2.35.
- 2024H2: gross annualized return 62.89%, net 25.41%, net Sharpe 2.59.
- 2025 available: gross annualized return 42.58%, net 5.10%, net Sharpe 1.02.

Its one-way break-even cost is 9.88bps, 12.58bps, and 8.52bps. The 2025
headroom over the assumed 7.5bps is only 1.02bps, so execution quality is
critical.

### Close-to-VWAP deviation

Gross Sharpe remains positive, but one-way break-even cost is only 2.43bps in
train, 0.97bps in 2024H2, and 1.71bps in 2025 available. Net annualized return
is negative in every period at 7.5bps.

### Intraday Amihud

The factor has stable OOS IC and gross Sharpe, but its absolute return spread is
too small. One-way break-even cost is 0.77bps, 1.42bps, and 0.88bps. Net
annualized return is approximately -30% to -34%.

### Minute skew

One-way break-even cost is 3.29bps, 4.46bps, and 3.77bps. Gross performance is
stable, but net annualized return remains negative in all periods at 7.5bps.

## Research decision

- `realized_volatility`: retain as the only transaction-cost-qualified
  single-factor candidate.
- `close_vwap_deviation`, `intraday_amihud`, `minute_skew`: retain as alpha
  information sources, but do not trade as independent daily round-trip
  portfolios at 7.5bps.
- Portfolio construction should test whether these signals improve an RV-led
  portfolio without adding a second full round trip. Combining independently
  traded sleeves would preserve the cost failure.

This remains a synthetic market-neutral feasibility test. A-share T+1,
short-borrow availability, locate fees, capacity, impact, and auction/execution
slippage are not modeled. In particular, 15- and 30-minute cash-equity round
trips require pre-existing inventory or another executable instrument.

## Reproducible outputs

- Simulator: `research/intraday_portfolio_simulator_v1.py`
- Tests: `tests/test_intraday_portfolio_simulator_v1.py`
- Daily ledger:
  `research/results/intraday_portfolio_simulator_v1/daily_portfolio_ledger.csv`
- Factor-period metrics:
  `research/results/intraday_portfolio_simulator_v1/factor_period_summary.csv`
- Wide comparison:
  `research/results/intraday_portfolio_simulator_v1/intraday_portfolio_cost_v1.csv`
- Run contract:
  `research/results/intraday_portfolio_simulator_v1/summary.json`

No change was made to `Intraday_Factor_Test_Process.py`, `intraday_lib.py`, or
the portfolio engine.

## Sprint 4.2.3 execution namespace

The financial calculation is unchanged, but the versioned v2 schema uses
explicit long-short names:

- `gross_return` becomes `gross_ls_return`;
- `net_return` becomes `net_ls_return`;
- `gross_sharpe` becomes `gross_ls_sharpe`;
- `net_sharpe` becomes `net_ls_sharpe`.

The new files are `daily_portfolio_ledger_v2.csv`,
`factor_period_summary_v2.csv`, `intraday_portfolio_cost_v2.csv`, and
`summary_v2.json`. Legacy v1 artifacts remain unchanged.

These execution metrics answer tradeability after measured cost. They are not
group excess-return metrics and are not used in IC, decile, or H-L alpha
calculation. Canonical group and H-L definitions are documented in
`research/docs/intraday_evaluation_metrics_v2.md`.
