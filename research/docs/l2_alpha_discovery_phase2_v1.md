# Sprint 4.4 Phase 2 — SSL2 L2 Alpha Discovery Pipeline v1

## Goal

Validate whether ClickHouse SSL2 features add **independent alpha entropy**
versus the existing minute survivors (`realized_volatility`,
`close_vwap_deviation`, `intraday_amihud`), using the unchanged
`intraday_evaluation_v2` metric layer.

## Scope (first round)

Only four discovery factors:

| factor | aggregation | note |
|--------|-------------|------|
| `l2_weighted_oi_mean` | mean | ten-level WOI |
| `l2_microprice_bias_mean` | mean | top-of-book microprice bias |
| `l2_depth_imbalance_mean` | mean | unweighted depth OI |
| `l2_cancel_pressure_sum` | minute flow ratio | SSE withdraw only |

Deferred: spread / wall / skew / top_book.

## Aggregation upgrade (Phase 1.5 inside Phase 2)

ClickHouse server-side only:

- WOI → `avg` / `max` / `stddevPop` (max/std diagnostic)
- microprice / depth → `avg`
- cancel → `sum(signed withdraw) / sum(total withdraw)` (not sum of ratios)

Python never `groupby`s raw snapshots.

## Interface

Long panel (`research/results/l2_factor_panel/YYYYMMDD.parquet`):

```text
date | bartime | symbol | factor | value | source | aggregation
```

Converted to evaluation narrow signal:

```text
tradetime | symbol | factorname | value
```

## Evaluation grid

| Dimension | Value |
|-----------|-------|
| Universe | CSI1000 (`000852.SH`) |
| Bartimes | `09:59`, `10:29`, `10:59`, `11:29`, `13:29`, `13:59`, `14:29` (PREHEAT grid; not round `:00`) |
| Horizons | `Ret_15`, `Ret_30`, `Ret_60` (`Ret_5` absent from preheat matrix) |
| Metrics | RankIC, ICIR, IC win, G1/G10 excess Sharpe, H-L Sharpe, mono, beta diagnostic |
| Independence | daily IC Spearman vs RV/CVWAP/Amihud + residual IC after rank-z OLS |

## Layout

```text
research/l2_alpha/
  l2_factor_registry.py
  l2_factor_panel.py
  clickhouse_ssl2.py          # minute_agg_feature_sql
  export_l2_intraday_panel.py

research/run_l2_intraday_evaluation.py
research/docs/l2_alpha_discovery_phase2_v1.md
tests/test_l2_phase2_pipeline_v1.py
```

## Quick start

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

$PY -m unittest tests.test_l2_ssl2_formulas_v1 tests.test_l2_phase2_pipeline_v1 -v

# Smoke: 3 business days, 50 CSI1000 names, skip residual (faster)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_intraday_evaluation.py \
  --start 2024-06-03 --end 2024-06-05 \
  --limit-symbols 50 --skip-independence --rebuild-panel

# Full discovery → use Phase 2.1 runner (see l2_alpha_discovery_phase2_1_v1.md)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/export_l2_panel_csi1000.py \
  --start 2024-01-01 --end 2025-08-18 --workers 2
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_alpha_validation_v21.py
```

Smoke outputs under `research/results/l2_alpha_discovery_v1/`.
Full validation outputs under `research/results/l2_alpha_discovery_v2/`.

## Non-goals

- No new LOB formulas beyond the Phase-1 eight / Phase-2 four
- No edits to `intraday_evaluation_v2.py` / freeze JSON / DDB minute packages
- No tick / OrderID factors yet
