# 7. Risk and Execution

## 7.1 Turnover snapshot (best recipes from Pack headlines)

| Factor | Best Net Sharpe @15bp | Daily TO (best recipe) | Status |
|--------|----------------------|------------------------|--------|
| TGD20 | 2.32 | 0.297 | validated |
| FlowDensity20 | 2.88 | 0.165 | candidate |
| IdealAmplitude | 3.40 | 0.446 | testing (mono fail) |
| IdealReversal | 1.70 | 0.153 | testing (mono fail) |
| APM_SessionResidual | 1.50 | 0.280 | testing_candidate |
| D1 | 1.38 | 0.234 | candidate |
| SmartMoney10d | 0.31 | 0.165 (5d buffer) | research |

High Net Sharpe with high TO (IdealAmplitude) is **not** automatically investable — mono and capacity still gate.

## 7.2 Transaction cost sensitivity

Packs primarily freeze / rank at **15bp** round-trip. Full 15 / 30 / 50bp grids:

| Factor | 15bp | 30bp | 50bp |
|--------|------|------|------|
| Most Packs | available in execution CSVs | partial | partial / TODO |

See Pack `execution/` folders. Report-level consolidated cost table: **TODO**.

## 7.3 Capacity

ADV / amount participation studies are **not** consolidated in Report v1. Flag for Stage 2 risk memo.

## 7.4 Operational risks called out in Packs

- Coverage exceptions (L2 start ~2022 for TGD/Flow)
- APM adapted index residual (not pure paper PM)
- Flow amount-orthogonal sign risk
- Ideal* weak monotonicity
- SmartMoney daily TO destroy Net Sharpe
