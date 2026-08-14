# Phase 2.3 — L2 Feature Family Audit

## Positioning

Phase 2.2 proved the Feature Factory finds signal. Phase 2.3 does **not**
add features. It audits overlap so Phase 3 OOS is not run on rank clones
or redundant mechanisms.

```text
Phase 2.2  L2 Feature Factory screen              ✅
Phase 2.3  Feature family / correlation audit     ✅
Phase 3    OOS + residual + cost (research set)   →  l2_phase3_validation_v1.md
Phase 4    L2 + EOD integration
```

## Rules

1. Drop `*_rank` from candidate counts — evaluator already CS-ranks.
2. Strict production KEEP (unique): `woi_mean10` only.
3. Research OOS set (broader): `woi_mean10`, `depth_imb_mean10`,
   `woi_delta30`, `woi_std20`.
4. Do not expand metacode / feature count before OOS.

## Command

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_l2_feature_audit_v23.py
```

## Outputs

Under `research/results/l2_feature_factory_v1/`:

| File | Content |
|------|---------|
| `l2_feature_corr.csv` | Mean CS Spearman @ 14:29 |
| `l2_feature_corr_all_slots.csv` | Mean CS Spearman (sampled slots) |
| `l2_ic_corr.csv` | IC-profile Spearman across bartime×horizon |
| `l2_feature_pair_overlap.csv` | Pair list + family tags |
| `l2_phase3_candidates.csv` | Frozen research set + train-best / 14:29 metrics |
| `l2_ff_decisions_dedup.csv` | Decisions without `*_rank` |
| `l2_feature_family_report.md` | Narrative summary |

## Phase 3 freeze policy

- `woi_mean10`: freeze `14:29` / `Ret_30` / `direction=-1` (train KEEP).
- Other research names: freeze **their own train-best** `(bartime, horizon, direction)` from `l2_ff_decisions_dedup.csv`. Do not force all to 14:29 — e.g. `woi_delta30` collapses at 14:29/Ret_30.
