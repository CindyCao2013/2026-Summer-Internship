# Code Evidence Map

This map freezes the implementation evidence used by the normalized
single-factor study.  It distinguishes the authoritative strict construction
from deprecated legacy artifacts.

## Strict Tick construction

- `l2_factor_reproduction/python/ch_tick.py::_regular_session_filter_sql`
  applies the regular-session predicate independently to every Tick row.
- `l2_factor_reproduction/python/ch_tick.py::fetch_tick_bucketed`
  defines SSE executions as `Type='T'` with
  `amount=ifNull(Amount, Price*Volume)`.
- The same function defines SZSE executions as `Type='011'` with
  `BidOrderNo>0`, `AskOrderNo>0`, and `amount=Price*Volume`.
- Both numerator and denominator are traded amounts.  Tick rows are
  executions, not reconstructed parent orders.

## Authoritative A0

- Strict cache:
  `research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity/tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet`
- Strict-cache metadata:
  `research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity/tick_bucketed_strict_trade_2023-01-01_2024-06-30.metadata.json`
- Frozen checksum:
  `ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f`
- Formula:
  `(cum_200000-cum_40000)/TotalAmount`
- Precise factor id:
  `mid_trade_amount_share_abs_4w20w`
- Legacy alias:
  `mid_order_ratio`

The normalized package runs
`run_mid_trade_amount_normalized_parity.py` before any return evaluation.  On
the authoritative cache date grid it uses the canonical values exactly and
sets dynamic-only keys to missing.  After 2024-06-28 it extends A0 from the
same strict daily amount primitive only after full-period dynamic-versus-
primitive parity passes.

## Normalized dynamic construction

- `ch_mid_trade_amount_normalization.py::build_dynamic_factor_sql` joins
  lagged `ADV20_lag1` / `ATS20_lag1` and same-day Q20/Q80 through
  `clickhouse-connect` ExternalData.
- A0, all nine A1 candidates, all nine A2 candidates, and A3 are aggregated
  in one strict Tick scan per exchange and month.  Zero-selected-amount groups
  are represented by numeric zero; missing historical scales are restored to
  factor NaN by the scale-evidence gate in
  `build_mid_trade_amount_normalized_cache.py::build_factor_panels`.
- The cache builder enforces unique keys, exact scale joins, nonnegative
  selected amounts, selected-amount bounds, total-amount conservation, frozen
  config SHA256, resumable chunk request hashes, and a two-worker default with
  a hard maximum of ten.
- `finalize_mid_trade_amount_normalized_lineage.py` stores each chunk's exact
  SSE/SZSE SQL text, SQL SHA256, strict filters, ExternalData format, client
  version, server version, artifact hash, dates, row count, and parent lineage.

## Evaluation contract

- `generate_mid_order_ratio_report_artifacts.py::align_core_panels` applies
  factor-date PIT/tradability masks before the one-trading-day signal lag.
- `generate_mid_order_ratio_report_artifacts.py::evaluate_prepared` computes
  daily Spearman RankIC, ICIR with `ddof=1`, equal-weight deciles, H-L,
  turnover, and frozen-direction diagnostics.
- `Factor_Dev_Lib.py::groupTest` defines `H-L=G10-G1`.
- `Factor_Dev_Lib.py::implied_annu_fee` defines the display-only fee as
  daily H-L turnover times 7.5 bps times 250.
- The frozen effective direction is `-1`; no evaluated return window may
  infer or change it.
- `generate_mid_trade_amount_normalized_report.py` reuses these functions for
  all four PIT universes and all IS/validation/OOS segments.  Its A0 input is
  required to be the parity-gated authoritative panel; the dynamic A0 copy
  cannot silently become the headline source.

## PIT and exposure data

- `Factor_Dev_Lib.py::get_index_member_mask` supplies Wind point-in-time
  index membership.
- `Factor_Dev_Lib.py::get_EOD_Not_Limit`, `get_EOD_Not_ST`, and
  `get_TradeStatus` supply the common tradability mask.
- `Factor_Dev_Lib.py::panel_neutral_size_ind` and
  `cs_neutral_size_ind` implement daily industry-dummy and log-market-cap
  OLS residualization.
- Industry data are CITICS level-1 classifications; market cap is Wind
  `S_VAL_MV`, transformed with log, MAD treatment, and cross-sectional
  z-score.

## Legacy exclusion

`research/results/l2_reproduction/mid_order_ratio/LEGACY_QUERY_WARNING.md`
states that pre-2026-08-04 canonical/neutralization artifacts are not formal
research evidence.  This study never reads the legacy
`factor_narrow.parquet` as an A0 source.

