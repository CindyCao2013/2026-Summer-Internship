# 2. Factor Universe

Inventory of factors covered by Report v1, sourced from `research/reports/factors/factor_library.csv`.

## Table 2.1 Factor library summary

| Factor | Source | Family | Status | ICIR | RankIC | Pack / notes |
|--------|--------|--------|--------|------|--------|--------------|
| TGD20 | paper | temporal | validated | 11.29 | 0.0415 | Pack v1 complete |
| D1_LiquidityQuality60d | internal | liquidity | candidate | 6.01 | 0.0573 | Pack v1 normalized |
| FlowDensity20 | paper/internal | flow | candidate | 4.85 | 0.0178 | Pack v1; SI ICIR 4.85 |
| APM_SessionResidual | paper_adapted | session_behavior | testing_candidate | 6.55 | 0.0225 | daily\|buffer_10_30 Net1.50 |
| IdealReversal | paper | behavior | testing | 9.46 | −0.031 | mono weak |
| IdealAmplitude | paper | behavior | testing | 9.97 | −0.038 | mono very weak |
| SmartMoney10d | paper | microstructure | research | 6.1 | −0.045 | parked; no Pack v1 |
| ActiveTradeProxy | proxy | microstructure | testing | 7.02 | 0.045 | **≠ APM**; excluded from headline |
| SUE_ConsensusEPS | paper | fundamental | design_only | — | — | C2; scout pending |

## 2.2 Family map (information sources)

```text
Temporal ── TGD20
Liquidity ── D1_LiquidityQuality60d
Flow ────── FlowDensity20
Session ─── APM_SessionResidual
Behavior ── IdealReversal, IdealAmplitude
Micro ───── SmartMoney10d (research)
Fundamental ─ SUE_ConsensusEPS (design_only; outside Report body)
```

(Schematic family map only — backtest figures live in Section 4 Pack links.)

## 2.3 Design locks

1. **ActiveTradeProxy ≠ APM_SessionResidual** — never rename/promote proxy as paper APM.
2. Status labels are frozen for this report; do not “upgrade” testing → validated in prose.
3. IdealAmplitude / IdealReversal share cutting family — treat as related mechanisms, not two independent alphas without residual analysis.
