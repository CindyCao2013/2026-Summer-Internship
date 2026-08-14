# FlowDensity20 — Implementation

## Code map

| Stage | Module | Role |
|-------|--------|------|
| Net flow / mktcap | `factor_formulas_l2_flow_p2.py` | `build_net_active_flow_mktcap` |
| L2 cache | `l2_data_loaders.py` | active buy/sell wide panels |
| Mechanism | `run_flow_density_mechanism_v1.py` | amount-orth diagnostics |
| Orthogonality | `run_factor_similarity_matrix_v1.py` | vs TGD20 / D1 |
| Spec | `factor_specs/FlowDensity20.yaml` | metadata |

## Build steps (conceptual)

```text
1. Load L2 daily cache → active buy/sell amounts
2. net = buy − sell; f = net / float_mktcap
3. Rolling 20d sum → cs_zscore → FlowDensity20
4. Wide panel + signal_shift=1 → RankIC / decile / execution
5. Optional: size+industry neutralization for headline book
```

## Parameters (harvest — not frozen)

| Param | Value |
|-------|------:|
| MA / sum window | 20 |
| min_periods | 10 |
| signal_shift | 1 |
| Cost baseline | 15 bp round-trip |
| Headline neutralization | size+industry |

## Do not change without review

- Claim formula frozen (amount-orth evidence required first)  
- Register amount/gross-active legs as independent validated factors  
- Default equal-weight blend with TGD20  

Full narrative: `factor_report.md` · mechanism: `mechanism/mechanism_summary.md`
