# Temporal Feature Layer

**Status:** Stages 0–3 frozen; Stage 4 validation runner ready  
**Depends on:** L2 Flow Density v1 ✅

## Pipeline

```
minute returns
      → Stage 1  return_timing.py          Gu / Gd
      → Stage 2.5 return_distribution.py   Rū / Rd̄ / zero_count
      → Stage 2  timing_residual.py        εu / εd  (CS vs controls)
      → Stage 3  tgd.py                    εd ~ εu → MA20 → TGD20
      → Stage 4  run_tgd_validation_v1.py  10-group / neut / cost
```

## Stages

| Stage | Module | Output | Status |
|-------|--------|--------|--------|
| 0 | paper decomposition | Gu/Gd → τ/υ → TGD | ✅ |
| 1 | `return_timing.py` | Gu, Gd | ✅ |
| 2.5 | `return_distribution.py` | Rū, Rd̄ = **conditional** up/down means; zero_count | ✅ |
| 2 | `timing_residual.py` | εu, εd | ✅ |
| 3 | `tgd.py` | `tgd_eps`, `TGD20` | ✅ frozen |
| 4 | `run_tgd_validation_v1.py` | ICIR / 10-group / neut / cost | ✅ runner |

## Critical definitions

- **Rū / Rd̄**: `mean(r | r>0)` / `mean(r | r<0)` — not mean of all minutes  
- **Not** `TGD = Gd − Gu` (研报否定的 τ)  
- **TGD20** = MA20 of daily CS residual from `εd ~ εu`
- **No lookahead:** `signal = TGD20.shift(1)` (full-session Gu/Gd on day T → T+1 return only)

## Stage 3 API

```python
from core.l2_features import build_tgd20, tgd20_to_wide

tgd_long = build_tgd20(residual_df)   # needs epsilon_u, epsilon_d
wide = tgd20_to_wide(tgd_long)        # for Factor_Dev_Lib.groupTest (n=10)
```

## Stage 4 validation

```bash
OMP_NUM_THREADS=1 python run_tgd_validation_v1.py
OMP_NUM_THREADS=1 python run_tgd_validation_v1.py --sample-days 252   # smoke
```

Outputs → `research/reports/tgd_v1/`:

| Layer | Path |
|-------|------|
| Summary | `summary.md` |
| A IC | `ic/rank_ic.csv` |
| A portfolio | `portfolio/decile_return.png`, `hml_curve.png` |
| B neut | `neutralization/neut_summary.csv` |
| C stability | `stability/yearly_ic.csv` |
| D cost | `cost/turnover_cost.csv` |

Do **not** hard-match paper 5-group IR. Compare mechanism + RankIC/ICIR + neut + cost.

## Execution Optimization (post Stage 4)

Factor frozen. Optimize signal→trade only:

```bash
OMP_NUM_THREADS=1 python run_tgd_execution_opt_v1.py
```

Outputs → `research/reports/tgd_v1/execution/` (`execution_summary.md`).
Reusable primitives: `execution_layer.py`.

## Research report + integrity

```bash
OMP_NUM_THREADS=1 python run_tgd_replication_integrity.py
```

- Metrics schema / pack: `tgd_v1/replication/metrics.json`
- Mechanism + Gu/Gd/τ/υ family: `tgd_v1/replication/`
- Full report: `tgd_v1/TGD_factor_research_report.md`

## Tests

```bash
OMP_NUM_THREADS=1 python -m pytest core/l2_features/ -q
```

## Panel builder

`core/l2_features/tgd_panel_builder.py` — DDB monthly Gu/Gd aggregates + overnight + residual + TGD20 cache. Does not alter Stage-3 formulas.
