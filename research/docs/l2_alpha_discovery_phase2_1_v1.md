# Sprint 4.4 Phase 2.1 — SSL2 L2 Alpha Validation

## Research question

Does native ClickHouse SSL2 information provide **independent** alpha beyond
the OHLCV survivors:

- `realized_volatility`
- `close_vwap_deviation`
- `intraday_amihud`

Primary decision statistic: **Residual ICIR** (not raw Sharpe).

## Frozen contract

| Dimension | Value |
|-----------|-------|
| Universe | CSI1000 (`000852.SH`) |
| Period | 2024-01-01 → 2025-08-18 |
| Train | 2024H1 |
| Validation | 2024H2 |
| OOS | 2025-01-01 → 2025-08-18 |
| Bartimes | PREHEAT `:29/:59` grid |
| Horizons | `Ret_15`, `Ret_30`, `Ret_60` |
| Factors | four registered Phase-2 factors only |
| Cost | 7.5 bps one-way |

## Gates

| Gate | Rule |
|------|------|
| Standalone | \|ICIR\| > 2, **H-L Sharpe > 3**, decile mono (G1 lowest → G10 highest, Spearman ≥ 0.7) |
| OOS | **H-L Sharpe > 2** and same decile mono |
| Independence | \|Residual ICIR\| > 1.5 |
| Execution | net LS / break-even cost are diagnostics only |

## Commands

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# 1) Full CSI1000 minute panels (resume-safe)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/export_l2_panel_csi1000.py \
  --start 2024-01-01 --end 2025-08-18 --workers 4

# 2) Validation + residual + execution
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_alpha_validation_v21.py
```

## Outputs

`research/results/l2_alpha_discovery_v2/`

- `l2_factor_metrics.csv`
- `l2_decay_matrix.csv`
- `l2_decile_returns.csv`
- `l2_ic_series.csv`
- `l2_ic_correlation.csv`
- `l2_residual_alpha.csv`
- `l2_execution.csv`
- `l2_decisions.csv`
- `l2_factor_report.md`
- `l2_decay_plots.png`

## Non-goals

- No new L2 formulas
- No edits to `intraday_evaluation_v2` / freeze JSON / DDB minute packages

## Status map

```text
Phase 2.1  primitive validation     ✅  (see l2_alpha_discovery_v2)
Phase 2.2  L2 Feature Factory v1    ✅  research/docs/l2_feature_factory_v1.md
Phase 3    OOS / residual / cost on KEEP
Phase 4    L2 + EOD assemble
```
