# IdealAmplitude — Implementation

## Code map

| Stage | Module | Role |
|-------|--------|------|
| Spec | `factor_cutting/ideal_amplitude.py` | `IDEAL_AMPLITUDE_SPEC`, `compute_ideal_amplitude` |
| Core engine | `factor_cutting/engine.py` | `rolling_quantile_mean_diff` |
| YAML spec | `factor_specs/IdealAmplitude.yaml` | metadata |
| Pack / exec | `run_milestone_3_0_ideal_amplitude.py` | full OS path + Report v2 |
| Mechanism dx | `run_ideal_amplitude_mechanism_diagnosis.py` | leg / payoff diagnostics |

## Build steps (conceptual)

```text
1. Load EOD high / low / close panels
2. Amp = H/L − 1; optionally drop one-word limit days
3. Quantile-split by close (λ=0.25) → V_high, V_low means
4. V = V_high − V_low
5. Wide panel + signal_shift=1 → RankIC / decile / execution
```

## Frozen parameters

| Param | Value |
|-------|------:|
| Window | 20 |
| λ (quantile frac) | 0.25 |
| min_effective_days | 10 |
| signal_shift | 1 |
| TOP_FRAC (execution) | 0.10 |
| Cost baseline | 15 bp round-trip |

## Do not change under `IdealAmplitude`

- Promote Registry status without monotonicity fix  
- Register V_high / V_low as separate factor_ids  
- Retune λ without new factor_id  

Mechanism notes: `research/factor_cutting/mechanism.md` · `mechanism/mechanism.md`
