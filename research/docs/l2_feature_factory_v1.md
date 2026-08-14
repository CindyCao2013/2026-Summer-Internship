# L2 Feature Factory v1 (Phase 2.2)

## Positioning

Phase 2.1 proved SSL2 primitives carry residual information but are not
production factors. Phase 2.2 builds the missing **feature engineering layer**
(DDB-style snapshot → minute → derived), then runs classic alpha screening.

```text
Phase 0  multi-DB audit                         ✅
Phase 1  SSL2 extractor (primitives)            ✅
Phase 2.1 primitive validation + residual       ✅
Phase 2.2 L2 Feature Factory (this doc)         ✅  selection screen on POC
Phase 3  OOS / residual / cost on KEEP          →  research/docs/l2_phase3_validation_v1.md
Phase 4  L2 + EOD assemble
```

Package: `research/l2_alpha/feature_factory/`
(`registry.py`, `primitives_sql.py`, `derived_sql.py`, `extract.py`, `export.py`).

**POC KEEP note:** `woi_mean10` and `woi_mean10_rank` share metrics (evaluator
already CS-ranks). Unique KEEP signal = 1.


## Architecture (DolphinDB analogue)

| DDB concept | Our implementation |
|-------------|-------------------|
| Snapshot + TimeSeriesEngine | ClickHouse `snapshot_feature_select_sql` → `minute_primitives_sql` |
| State / rolling derived | ClickHouse window functions in `derived_sql.py` |
| Metacode `base × agg` | `feature_factory/registry.py` |
| Evaluation | Reuse `_evaluate` + exact group excess (no eval_v2 edits) |

```text
SSL2 arrays
  → minute primitives (avg within minute, full session)
  → rolling mean/std/delta/slope/persistence/zscore
  → PREHEAT bartime filter
  → CS ranks (Python, per bartime)
  → RankIC / H-L / mono screen
```

## First 20 factors

See `research/l2_alpha/feature_factory/registry.py` (`L2_FF_ALL_COLUMNS`).

Cancel-derived features are **SSE-only** (SZSE withdraw columns absent).

## Commands

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# Unit tests
$PY -m unittest tests.test_l2_feature_factory_v1 -v

# Smoke export (1 day)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/l2_alpha/feature_factory/export.py \
  --start 2024-06-03 --end 2024-06-03 --limit-symbols 50

# POC discovery: 200 CSI1000 names, 2024H1 train screen
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_feature_factory_discovery_v1.py \
  --limit-symbols 200 --start 2024-01-01 --end 2024-06-30

# Follow-on (not first run): full CSI1000
# OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_feature_factory_discovery_v1.py --full
```

## Gates (train screen)

| Rule | Threshold |
|------|-----------|
| \|ICIR\| | > 2 |
| H-L Sharpe | > 3 |
| Decile | G1 lowest → G10 highest, Spearman ≥ 0.7 |

OOS + residual for Phase 3 **research set** (unique KEEP + near-misses),
not rank clones. No entropy / ML. See `l2_feature_audit_v23.md`.

## Outputs

`research/results/l2_feature_factory_v1/`

- `panel/YYYYMMDD.parquet`
- `l2_ff_metrics.csv`
- `l2_ff_decisions.csv`
- `l2_ff_report.md`

## Non-goals

- No new LOB primitives beyond existing formula family
- No mutual information / XGBoost / tick OrderID
- No edits to `intraday_evaluation_v2` / freeze JSON
