# FlowDensity Amount-Orthogonalization v1

**Purpose:** factor attribution — *not* a freeze gate.
**Mechanism class:** `flow_liquidity_interaction`
**Attribution case:** `case_interaction_entangled`
**Status:** keep `validated_single_factor_candidate` (no auto-promote)

## Research question

> After removing anti-amount / low-activity exposure, does pure flow direction still have alpha?

## Comparison table


| Signal                     | Family                | RankIC  | ICIR  | H-L Sharpe | Mono   | Net@15bp | Note                                                       |
| -------------------------- | --------------------- | ------- | ----- | ---------- | ------ | -------- | ---------------------------------------------------------- |
| `FlowDensity_raw`          | canonical             | 0.0236  | 4.85  | 3.38       | 0.879  | 1.92     | size+ind net flow (confirmation signal)                    |
| `Amount`                   | liquidity_channel     | -0.0475 | -8.66 | 3.95       | -1.000 | 3.55     | size+ind amount/mktcap 20d                                 |
| `GrossActive`              | liquidity_channel     | -0.0476 | -8.67 | 3.99       | -1.000 | 3.52     | size+ind gross active/mktcap 20d                           |
| `Flow_perp_Amount`         | amount_orthogonal     | -0.0088 | -2.49 | 0.13       | 0.370  | -2.10    | ε from Flow_si ~ Amount_si (tradable residual panel)       |
| `Amount_perp_Flow`         | amount_orthogonal     | -0.0371 | -8.49 | 2.53       | -0.855 | 0.67     | ε from Amount_si ~ Flow_si (liquidity after removing flow) |
| `Flow_perp_Amount_then_SI` | amount_orthogonal_alt | -0.0085 | -2.42 | 0.08       | 0.382  | -2.07    | ε from Flow~Amount in raw space, then size+ind             |


## Attribution summary

- cs_corr(Flow, Amount) = **-0.617**
- Raw ICIR = **4.85**
- Flow⊥Amount ICIR = **-2.49** (signed retention -0.51)
- Amount ICIR = **-8.66**
- Amount⊥Flow ICIR = **-8.49** (liquidity channel after removing flow)

## Interpretation

Flow⊥Amount ICIR=-2.49 (signed retain=-0.51) — neither pure flow nor pure amount. Classify as Flow × Liquidity interaction.

## Factor map (working)


| Factor           | True information                          |
| ---------------- | ----------------------------------------- |
| TGD20            | Temporal return structure                 |
| FlowDensity_raw  | Flow + liquidity (interaction)            |
| Flow_perp_Amount | Pure flow candidate (if Case 1)           |
| Amount           | Anti-activity / liquidity premium channel |


## Do not

- Auto-freeze FlowDensity20 from this run
- Jump to TGD×Flow composite before orthogonality + explicit liquidity exposure note

## Next

1. Keep raw FlowDensity as `liquidity_flow_interaction` candidate
2. Optionally track `Flow_perp_Amount` as a research satellite (not production rename yet)
3. Run **TGD20 ⟂ FlowDensity** with both raw and perp variants in the independence table

## Artifacts

- `mechanism_amount_neutral.csv`
- `amount_orth_verdict.json`

