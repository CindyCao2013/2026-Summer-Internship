# Intraday Evaluation Metrics v2

## Purpose

Sprint 4.2.3 freezes metric definitions without changing factor computation,
the DolphinDB factor backend, the backtest entry, or the public
`intraday_lib.py` API. It separates three questions:

1. Signal quality: does the factor predict the cross section?
2. Cross-sectional alpha: do ranked groups produce market-relative returns?
3. Execution feasibility: does a position-level long-short implementation
   survive measured turnover and cost?

The canonical pure-Python implementation is
`core/evaluation/intraday_metrics.py`; the DDB aggregation adapter is
`research/run_intraday_evaluation_v2.py`.

## Canonical definitions

All annualized means, Sharpes, and ICIRs use 250 trading days.

### Signal namespace

- `rank_ic`: mean daily Spearman rank IC of the raw factor and future return.
- `annualized_icir`: `mean(rank_ic_t) / std(rank_ic_t) * sqrt(250)`.
- `ic_win_rate`: fraction of dates on which frozen/discovery direction
  multiplied by raw IC is positive.

Direction is derived from training raw IC for discovery. Frozen OOS evaluation
uses the immutable direction in `intraday_alpha_freeze_v1.json`.

### Cross-sectional namespace

The benchmark is the exact equal-weight mean of all valid filtered constituent
returns at each `(Date, Bartime, horizon)`:

`market_return = mean(asset_return)`.

It is not the unweighted average of decile means. The DDB adapter aggregates
the benchmark directly from constituent returns, retaining `n_market_assets`
and group `n_assets`.

- `group_return_raw`: constituent-EW raw return for a decile.
- `group_return_excess = group_return_raw - market_return`.
- `group_excess_sharpe`: Sharpe of the daily group excess series.
- `raw_hl_return = G10_raw - G1_raw`.
- `hl_return = direction * raw_hl_return`.
- `hl_sharpe`: Sharpe of the direction-adjusted H-L series.
- `hl_market_beta = cov(hl_return, market_return) / var(market_return)`.
- `hl_market_corr = corr(hl_return, market_return)`.

The implementation asserts within `1e-12` that:

`(G10_raw - market) - (G1_raw - market) = G10_raw - G1_raw`.

This identity removes the benchmark return algebraically. It does not imply
zero benchmark beta. `hl_market_beta` and `hl_market_corr` are diagnostics, not
hard gates. A beta threshold is valid only after an explicit beta hedge.

### Execution namespace

Execution metrics remain position-level and are never mixed into IC or decile
metrics:

- `gross_ls_return`;
- `net_ls_return = gross_ls_return - transaction_cost`;
- `gross_ls_sharpe`;
- `net_ls_sharpe`;
- `half_l1_turnover`;
- `traded_notional_turnover`;
- `break_even_one_way_cost_bps`;
- `cost_headroom_bps`.

The current simulator constructs +50% long / -50% short extreme-decile
positions, closes at the frozen horizon, and charges 7.5 bps per unit of
one-way traded notional. The explicit `*_ls_*` names distinguish execution P&L
from group excess-return analytics.

## Output schema

`performance_all_v2.csv` is long-form. Each row belongs to either
`cross_sectional_group` or `cross_sectional_hl`. Signal fields are attached by
factor, bartime, and horizon. Group rows populate raw/excess group fields; H-L
rows populate direction-adjusted spread and benchmark diagnostic fields.

`intraday_alpha_library_v3_candidates.csv` contains each frozen 2024H1 tuple
with signal, G1/G10 excess Sharpe, H-L Sharpe, and H-L benchmark diagnostics.
Execution fields are merged only where a position-level simulator result
exists.

`frozen_oos_diagnostics_v2.csv` contains the unchanged frozen tuples for
2024H2 and the available 2025 sample. There is no OOS bartime, horizon, or
direction search.

Versioned execution output is
`research/results/intraday_portfolio_simulator_v1/intraday_portfolio_cost_v2.csv`.
Legacy v1 evidence is not overwritten.

## Compatibility

- `Intraday_Factor_Test_Process.py`: unchanged.
- `intraday_lib.py` public API: unchanged.
- DDB factor packages and `INTRADAY_FACTOR_BACKEND`: unchanged.
- `run_p2_intraday_heatmap.py` and `intraday_Factortest.py`: their legacy
  group panels are passed through the existing excess conversion before the
  existing analyzer.

The legacy `subtract_market_return` path remains an approximate compatibility
adapter because historical group tables do not carry constituent counts.
Canonical v2 research uses the exact constituent-level benchmark produced by
the new DDB adapter.

## Reproducibility

- Metric tests: `tests/test_intraday_metrics_v2.py`.
- Evaluation checkpoints:
  `research/results/intraday_evaluation_v2/checkpoints/`.
- Checkpoints are bound to the freeze SHA-256 and evaluation-spec SHA-256.
- The v2 specification fixes annualization, exact benchmark construction,
  H-L orientation, and diagnostic-only beta policy.

## Regenerated result checks

The completed run produced 3,432 performance rows, 12 candidate rows, and 24
frozen OOS rows. Candidate signal and cross-sectional fields contain no missing
values. Every OOS row uses the direction from the freeze specification.

The largest absolute H-L benchmark beta among frozen OOS rows is 0.579 and the
largest absolute correlation is 0.709. This is expected diagnostic evidence
that a dollar-neutral extreme-decile spread is not automatically beta-neutral.
Four frozen factor-periods have a negative direction-adjusted H-L mean; v2
reports these failures without changing the frozen direction.

Evaluation SHA-256:
`f847cfc274a164fdf9462a4f5a76a16d1a3d657d594a2e8c42421e00510d08af`.

## Qualified decile plots

`research/plot_intraday_evaluation_v2.py` reproduces the standard factor-mining
figures using exact-market daily group returns:

- `figures/qualified_decile_hl_cumulative.png`: G1-G10 market-excess
  cumulative returns plus frozen-direction H-L;
- `figures/qualified_decile_mean_bar.png`: mean market-excess return by raw
  factor decile.

The plotting gate requires:

1. frozen OOS conclusion `retain`;
2. 2024H1 frozen-tuple H-L Sharpe greater than 3;
3. direction-adjusted decile monotonicity Spearman at least 0.8;
4. at least six of nine adjacent decile steps in the expected direction.

The qualified set is `realized_volatility`, `close_vwap_deviation`, and
`intraday_amihud`. `minute_skew` passes the OOS and H-L gates but is excluded
because its decile monotonicity Spearman is only 0.321.
