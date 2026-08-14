# IdealReversal — Implementation

## Code map

| Stage | Module | Role |
|-------|--------|------|
| Spec | `factor_cutting/ideal_reversal.py` | `IDEAL_REVERSAL_SPEC`, `compute_ideal_reversal` |
| W-cut engine | `factor_cutting/w_cut.py` | rank-split sum helper |
| Core engine | `factor_cutting/engine.py` | `rolling_rank_split_sum` |
| Trade count | `factor_cutting/trade_count.py` | daily trade_count loader |
| YAML spec | `factor_specs/IdealReversal.yaml` | metadata |
| Pack / exec | `run_milestone_2_2_1_ideal_reversal.py` | execution grid fill |

## Build steps (conceptual)

```text
1. Load EOD return, amount, trade_count panels
2. Build ATS = amount / trade_count
3. W-cut: sum(r | high ATS) − sum(r | low ATS) over 20d
4. Wide panel + signal_shift=1 → RankIC / decile / execution
```

## Frozen parameters

| Param | Value |
|-------|------:|
| Window | 20 |
| High / low count | 10 / 10 |
| signal_shift | 1 |
| TOP_FRAC (execution) | 0.10 |
| Cost baseline | 15 bp round-trip |

## Do not change under `IdealReversal`

- Promote Registry status without monotonicity fix  
- Register M_high / M_low as separate factor_ids  
- Retune knife without new factor_id  

Mechanism notes: `research/factor_cutting/mechanism.md`
