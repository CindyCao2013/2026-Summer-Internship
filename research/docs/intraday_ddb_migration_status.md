# Intraday DDB Factor Migration Status

Last updated: 2026-07-29

## Production registry

| Factor | Information family | Backend | Formula parity | Rank parity | Backtest parity | Status |
|---|---|---|---|---|---|---|
| `close_vwap_deviation` | Price / VWAP state | DDB-native | PASS | PASS | PASS | production |
| `active_buy_sell_imbalance` | Cumulative active order flow | DDB-native | PASS | PASS | PASS | production |
| `volume_front_loading` | Morning liquidity state | DDB-native | PASS | PASS | PASS | production |
| `volume_back_loading` | Closing liquidity state | DDB-native | PASS | PASS | PASS | production |
| `late_session_strength` | Active order flow | DDB-native | PASS | PASS | PASS | production |

Production flags default to DDB. Every migrated factor retains a Python
reference implementation and an independent per-factor fallback flag.

## Stable production contract

The Evaluation Layer is unchanged:

```text
DDB minute table
  → DDB-native factor compute
  → bartime | symbol | factorname | value
  → build_intraday_narrow_table()
  → tradetime | symbol | factorname | value
  → existing intraday backtest
```

`Intraday_Factor_Test_Process.py`, `intraday_lib.py`, index filtering, limit
filtering, return matrices and portfolio evaluation are outside factor migration
scope.

## Mandatory migration gates

Each factor package contains:

```text
factors/intraday/<factor>/
├── hypothesis.md
├── ddb.sql
├── compute.py
├── test_compare.py
└── run_backtest_parity.py
```

Required validation:

1. Formula parity: maximum absolute difference `< 1e-10`.
2. Ranking parity: cross-sectional Spearman `> 0.999`.
3. Signal parity: timestamps, row count and universe align.
4. Trading parity: IC, ICIR, G1–G10 excess Sharpe, H-L Sharpe and turnover.
5. Production trace: the enabled backend must reach `ddb_version()`.

## Timestamp policy

- Intraday cumulative state known before a standard evaluation point is stamped
  at that standard bartime.
- Information requiring the closing session is stamped at T+1 09:59.
- Timestamp alignment is validated independently from numeric alignment.
- Existing pandas `BDay` behavior remains the reference until an exchange
  calendar is introduced consistently across both paths.

## No-look-ahead policy

- Cumulative calculations use rows at or before signal time only.
- Historical rolling denominators end at T-1.
- DDB rolling state uses positive one-session shift:
  `move(msum(...), 1) / move(mcount(...), 1)`.
- Forward/negative `move`, future bars and day-end state stamped intraday on the
  same day are forbidden.
- Tests must explicitly prove that changing day-T input cannot change the
  historical denominator.

## Null and sparse-data policy

- Match pandas reference semantics for null, zero and all-null groups.
- `nullFill(sum(...), 0)` is required where pandas `groupby.sum()` maps an
  all-null session to zero.
- Missing bars, suspended stocks, symbol normalization and duplicate keys must
  be checked before accepting parity.

## Fallback policy

- DDB is the production default only after all migration gates pass.
- Python remains the golden reference and can be selected through the
  per-factor feature flag.
- A fallback is explicit; server or schema errors must not silently switch
  computation engines during a production research run.

## Recent migrations

Sprint 3.4 migrated `late_session_strength`, the existing active-buy share factor:

```text
sum(Active_buy_amount, 14:30–15:00)
────────────────────────────────────────
sum(Active_buy_amount + Active_sell_amount, 14:30–15:00)
```

Day-T closing flow is stamped at T+1 09:59. This extends the engine from price
and volume state into L2 active order flow without introducing a new backtest
contract.

Sprint 3.5 added `active_buy_sell_imbalance`, a session-cumulative L2 flow
factor at all five standard bartimes. This differs from closing-window strength:
it tracks the evolving aggressive buy/sell state using only bars available at
each signal time.

## Sprint 3.6: Intraday Alpha Library v1 audit

The first alpha-density audit covers ZZ1000 from 2024-01-01 through 2024-06-30.
It uses the unchanged index, limit-filter and return-matrix evaluation layer.
Correlations are computed only between signals sharing the same bartime.

Key findings:

- ABSI and `late_session_strength` are not duplicate flow signals: their mean
  daily cross-sectional Spearman at 09:59 is `0.057`.
- ABSI has a stable moderate relationship with `close_vwap_deviation`
  (Spearman `0.35–0.38` across the five common bartimes).
- Cross-sectionally residualizing 09:59 ABSI on LSS increases absolute Ret_15
  annualized ICIR from `1.81` to `2.41`; this is promising incremental
  information, not yet an out-of-sample production gate.
- No candidate passes the current cost gate. With fixed H-L turnover `4.0` and
  `7.5 bps` per turnover unit, the assumed daily cost is `30 bps`.

Reproducible outputs:

```text
research/run_intraday_alpha_library_v1.py
research/results/intraday_alpha_library_v1/
├── performance_by_slot.csv
├── spearman_by_slot.csv
├── absi_residual_lss_0959.parquet
└── summary.json
```

The next research gate is chronological OOS validation plus measured position
turnover. Migration parity and alpha investability remain separate decisions.

## Phase 4.1: Intraday Alpha Expansion Batch 1

Eight DDB-backed discovery candidates are registered without changing
`Intraday_Factor_Test_Process.py`, `intraday_lib.py`, return matrices or
portfolio logic:

```text
bartime_ofi
ofi_persistence
active_buy_shock
average_active_trade_size
large_active_buy_ratio
intraday_amihud
realized_volatility
minute_skew
```

The batch uses the shared implementation contract in
`factors/intraday/discovery_v1.py` and the per-factor research package contract
in `research/docs/intraday_factor_discovery_template_v1.md`.

Candidate-gate results:

- All eight emit the exact five standard bartimes through the production
  registry path.
- Deterministic formula, session-reset and future-bar mutation tests pass.
- Live Python/DDB checks pass the `1e-10` numeric gate on a cross-session
  sample; full-universe one-day maxima are also below `1e-10`.
- Python remains the explicit fallback for every factor.
- Candidates are deliberately excluded from the default
  `INTRADAY_CUSTOM_FACTOR_LIST` until full backtest and horizon gates pass.

The live minute schema contains active buy/sell amounts and counts, but no true
large-order amount bucket. Consequently `large_active_buy_ratio` is a
bar-level proxy based on average active-buy ticket size versus its shifted
prior-20-bar baseline. It must not be interpreted as an individual large-order
or institutional-flow measurement.

Raw positive amount/count fields are used for the new microstructure factors.
Negative correction rows are mapped to zero consistently in Python and DDB;
rolling denominators below one currency unit are treated as non-economic.

## Sprint 4.0 retrospective inventory gate

The live catalog and schema audit is recorded in
`research/docs/l2_feature_inventory_20260730.md`.

The current stock grant is a 22-field, one-minute bar dataset with active-flow,
trade-count and bid/ask cancellation labels. No readable snapshot, tick,
order-book, depth, spread, queue or OrderID source is available. Order-book
imbalance and microprice factors are therefore blocked rather than proxied.

The next P0 expansion is `cancel_imbalance`, `signed_price_impact`,
`active_trade_size_imbalance` and `volume_curve_deviation`. Batch 1 remains a
candidate library and is not promoted by the inventory result.

## Sprint 4.1 Intraday Alpha Discovery v1

The unified 2024H1 evaluation is recorded in
`research/docs/intraday_alpha_discovery_v1_2024h1.md`. It covers five production
DDB factors and seven primary discovery factors at every execution slot across
15, 30, 60 and 120 minute, EOD and next-day horizons.

The harness writes the required
`intraday_alpha_library_v2_candidates.csv`, full decay curves, same-slot
correlations, and production-control residual IC results. Discovery residuals
use daily cross-sectional rank-z OLS and only controls available at the same
bartime.

The discovery screen used a fixed display-model turnover for its provisional
cost column; that column is superseded by Sprint 4.2.2. `bartime_ofi`,
`active_buy_shock`, `intraday_amihud`, and
`minute_skew` retain strong residual evidence and are classified as research
satellites. `ofi_persistence` is downgraded to enhancer because its selected
residual ICIR retention is 45% and its 09:59 correlation with ABSI is 0.598.
`average_active_trade_size` is also an enhancer because it overlaps with active
buy shock and has weaker gross portfolio performance.

`large_active_buy_ratio` is explicitly excluded from the primary candidate
table as `proxy_only`. No production promotion follows from this in-sample
screen. The next required gate is a frozen-slot, frozen-horizon out-of-sample
evaluation.

## Sprint 4.2.1 frozen OOS validation

The immutable specification is stored in
`research/config/intraday_alpha_freeze_v1.json`. Its hash protects the complete
factor, bartime, horizon, direction, portfolio rule, residual-control and
residual-direction contract. The OOS runner contains no tuple search and never
re-infers direction from future IC or H-L returns.

Frozen validation covers 2024H2 and the available 2025 sample through
2025-08-18. Full results are documented in
`research/docs/intraday_alpha_oos_v1.md`.

`close_vwap_deviation`, `intraday_amihud`, `realized_volatility`, and
`minute_skew` retain their frozen signal in both periods. Only
`realized_volatility` also passes the 7.5 bps cost gate in both periods.
`late_session_strength` drops because its raw IC direction reverses in 2025.
`ofi_persistence` drops as incremental alpha because its frozen residual
direction reverses in 2024H2. The other six factors remain on watch.

The 2025 label is `test_2025_available`, not full-year 2025. Large DDB factor
queries are split into at most six-month chunks and complete period checkpoints
are validated against the freeze hash before reuse.

## Sprint 4.2.2 position-level portfolio cost

The corrected simulation is documented in
`research/docs/intraday_portfolio_cost_v1.md`.

The previous `turnover_b_hl=4` is a fixed model generated by
`4 × side_gross`, not measured constituent turnover. The OOS outputs now label
it and its associated Sharpe as legacy/model-implied.

The simulator builds actual extreme-decile weights at +50% long and -50% short,
closes at the frozen horizon, and charges 7.5 bps per unit of entry/exit traded
notional. Average half-L1 turnover is approximately 1.0 and traded-notional
turnover approximately 2.0 per day. Daily gross returns match one half of the
frozen-direction H-L spread within `4e-17`.

Only `realized_volatility` has positive net return and net Sharpe in train,
2024H2, and the available 2025 sample. Its net Sharpe is 2.35, 2.59, and 1.02;
the 2025 one-way break-even cost is 8.52 bps versus the assumed 7.5 bps.
`close_vwap_deviation`, `intraday_amihud`, and `minute_skew` remain valid alpha
information sources but fail as independently round-tripped sleeves.

The simulation is synthetic market-neutral research. Cash-equity T+1,
short-borrow availability, capacity, impact, and locate fees remain outside the
current model.

## Sprint 4.2.3 unified intraday evaluation metrics

The frozen metric contract is documented in
`research/docs/intraday_evaluation_metrics_v2.md`.

`core/evaluation/intraday_metrics.py` now owns the pure-Python definitions for
250-day ICIR, exact market-excess deciles, direction-adjusted H-L, benchmark
beta/correlation diagnostics, and explicit long-short execution naming.
`research/run_intraday_evaluation_v2.py` aggregates the benchmark directly from
filtered constituent returns in DDB; it does not approximate the market using
an unweighted mean of group means.

The regenerated evidence contains:

- 3,432 long-form metric rows across train, validation, and test;
- 12 frozen 2024H1 candidate rows;
- 24 frozen OOS diagnostics;
- zero missing signal or cross-sectional fields in the candidate table;
- zero frozen-direction mismatches in OOS.

The H-L beta result confirms the corrected interpretation. Maximum absolute
frozen OOS beta is 0.579, so algebraic benchmark cancellation is not treated as
a beta-neutrality gate. Four frozen factor-periods have negative
direction-adjusted H-L means and are reported as diagnostics rather than
silently flipped.

The position simulator's versioned schema now reports `gross_ls_*` and
`net_ls_*`. Its underlying weights, turnover, costs, and returns are unchanged.
Legacy v1 result files remain available; v2 artifacts are written to:

- `research/results/intraday_evaluation_v2/performance_all_v2.csv`;
- `research/results/intraday_evaluation_v2/intraday_alpha_library_v3_candidates.csv`;
- `research/results/intraday_evaluation_v2/frozen_oos_diagnostics_v2.csv`;
- `research/results/intraday_portfolio_simulator_v1/intraday_portfolio_cost_v2.csv`.

No change was made to `Intraday_Factor_Test_Process.py`, factor computation,
DDB-native factor packages, or the public `intraday_lib.py` API.
