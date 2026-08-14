# Intraday Alpha Discovery v1 — 2024H1 evaluation

## Scope

The unified harness evaluates five production DDB factors and seven primary
discovery factors on the `000852.SH` universe from 2024-01-01 through
2024-06-30. `large_active_buy_ratio` is excluded from primary role assignment:
without tick trade sizes or OrderID it is only a bar-level ticket-size proxy.

Every factor is evaluated at its emitted execution bartime for `Ret_15`,
`Ret_30`, `Ret_60`, `Ret_120`, `Ret_EOD` and `Ret_NDay`. Outputs include mean
rank IC, annualized ICIR, directional IC win rate, directional H-L Sharpe,
turnover and H-L Sharpe after 7.5 bps per one-way turnover unit.

Residual signals use daily cross-sectional rank-z OLS. Each discovery factor is
residualized against every production factor available at the same execution
bartime. No control from a later bartime is admitted.

## Research roles

- **Base:** `close_vwap_deviation`, `late_session_strength`, and
  `realized_volatility`.
- **Satellite:** `active_buy_sell_imbalance`, `bartime_ofi`,
  `active_buy_shock`, `intraday_amihud`, and `minute_skew`.
- **Enhancer:** `volume_front_loading`, `volume_back_loading`,
  `ofi_persistence`, and `average_active_trade_size`.
- **Drop:** none at this exploratory stage.

The roles are research-library classifications, not production approvals.
Discovery `Base` status requires positive cost-adjusted Sharpe under the current
fixed-turnover convention. `Satellite` requires strong standalone and
production-control residual evidence. `Enhancer` denotes gross-only, narrower,
or materially overlapping evidence.

## Main findings

`realized_volatility` is the only factor passing the current cost gate. Its
14:29 `Ret_30` result has IC -0.0989, annualized ICIR -11.53, directional H-L
Sharpe 9.75, cost-adjusted Sharpe 2.34, and residual ICIR -12.34.

The strongest new independent flow signals are `bartime_ofi` and
`active_buy_shock`. Their selected residual ICIR values are -7.92 and -6.98,
with residual retention of 96% and 102%, respectively.

`intraday_amihud` and `minute_skew` retain 89% and 130% of selected ICIR after
production controls. Both therefore add liquidity/distribution information
rather than simply reproducing the five production factors.

`ofi_persistence` is useful but not independent enough for a primary role. At
09:59 its mean Spearman correlation with `active_buy_sell_imbalance` is 0.598,
and selected residual ICIR retention falls to 45%.

`average_active_trade_size` retains its residual IC, but its gross H-L Sharpe
is only 1.96 and its strongest same-slot relationship is with
`active_buy_shock` (roughly 0.55–0.58 across the five slots). It remains an
enhancer until an out-of-sample test demonstrates incremental portfolio value.

The largest off-diagonal same-slot factor correlation is below 0.60 at every
slot. Batch 1 is therefore not a single duplicated flow cluster, although the
ABSI/OFI-persistence and active-shock/trade-size pairs require explicit
combination controls.

## Important limitation

The reported `best_bartime` and `best_horizon` are selected from up to 30
in-sample combinations per factor. Their absolute ICIR and Sharpe values are
diagnostic upper bounds, not unbiased production estimates. No discovery
factor should be promoted from this report alone.

The next gate is a frozen-specification out-of-sample evaluation: lock one
slot/horizon and sign per candidate from 2024H1, then test 2024H2 and 2025
without reselection. Portfolio-level cost tests should also replace the current
fixed turnover of 4 with the intended holding and rebalance schedule.

## Reproducible outputs

- Harness: `research/run_intraday_alpha_discovery_v1.py`
- Candidate table:
  `research/results/intraday_alpha_discovery_v1/intraday_alpha_library_v2_candidates.csv`
- Full decay: `research/results/intraday_alpha_discovery_v1/decay_by_slot.csv`
- Slot correlation:
  `research/results/intraday_alpha_discovery_v1/spearman_by_slot.csv`
- Full metrics:
  `research/results/intraday_alpha_discovery_v1/performance_all.csv`
- Proxy exclusions:
  `research/results/intraday_alpha_discovery_v1/excluded_proxies.csv`
