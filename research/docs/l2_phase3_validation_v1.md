# Phase 3 — L2 Feature Factory Validation

## Research question

Is WOI imbalance reversal (and Tier-2 research near-misses) still alpha in
2025, residual to RV/CVWAP/Amihud, and cost-feasible?

```text
Phase 2.3  Feature family audit                 ✅
Phase 3    OOS + residual + cost (this doc)     ← current
Phase 4    L2 + EOD / production factor library
```

## Freeze (no re-optimization)

| factor | bartime | horizon | direction | tier |
|--------|---------|---------|-----------|------|
| `woi_mean10` | 14:29 | Ret_30 | −1 | 1 production |
| `depth_imb_mean10` | 14:29 | Ret_30 | −1 | 2 research |
| `woi_delta30` | 13:59 | Ret_15 | +1 | 2 timing |
| `woi_std20` | 10:59 | Ret_30 | −1 | 2 risk-like |

Universe continuity: same ~230 names frozen from Phase 2.2 H1 panel
(`research/results/l2_phase3_validation/frozen_symbols.txt`).

Periods:

| Split | Range |
|-------|-------|
| Train | 2024-01-01 → 2024-06-30 |
| Validation | 2024-07-01 → 2024-12-31 |
| Test (OOS) | 2025-01-01 → 2025-08-18 |

## Gates

| Gate | Rule |
|------|------|
| Train | \|ICIR\| > 2, H-L Sharpe > 3, decile mono |
| OOS | H-L Sharpe > 2, decile mono |
| Residual | \|Residual ICIR\| > 1.5 vs RV/CVWAP/Amihud |
| Cost | 10 / 15 / 20 bp ladder (diagnostic) |

Survival labels: `KEEP` / `WATCH` / `DROP`.

## Commands

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python
SYM=research/results/l2_phase3_validation/frozen_symbols.txt
PANEL=research/results/l2_feature_factory_v1/panel

# 1) Export OOS Feature Factory panels (resume-safe; ~1.5s/day @ 230 names)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/l2_alpha/feature_factory/export.py \
  --start 2024-07-01 --end 2025-08-18 \
  --symbols-file $SYM \
  --bartimes 10:59,13:59,14:29 \
  --output $PANEL \
  --skip-existing

# 2) Phase 3 validation
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_phase3_validation_v1.py
```

## Outputs

`research/results/l2_phase3_validation/`

| File | Content |
|------|---------|
| `oos_metrics.csv` | Train / val / test H-L + IC |
| `residual_metrics.csv` | Residual ICIR / H-L by period |
| `cost_analysis.csv` | Cost ladder 10/15/20bp |
| `factor_survival.csv` | KEEP / WATCH / DROP |
| `l2_phase3_report.md` | Narrative |

## Non-goals

- No entropy / ML / new metacode features
- No bartime/horizon re-search
- No edits to `intraday_evaluation_v2` / freeze JSON
