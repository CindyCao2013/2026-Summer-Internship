# Factor Delivery Roadmap (ACTIVE)

Governance freeze: [`GOVERNANCE.md`](GOVERNANCE.md) · Board: [`factor_delivery_plan.csv`](factor_delivery_plan.csv)  
Metric freeze: [`METRICS_G10_EXCESS.md`](METRICS_G10_EXCESS.md)

## Product rule (2026-07-22 Mentor reset)

```text
NOW  — Single-factor discovery & optimization
       Headline = G10 Excess Sharpe (G1=short, G10=long)
       Gates: Excess > 3.5 AND Excess > all group/H-L Sharpes
PAUSE — Factor matrix / combination (need ~100 factors, corr < 0.7)
PAUSE — Portfolio / Composite / Topology / Governance
```

Buffer may be used in execution diagnostics but **must not** be the only lever to pass gates.

## Required report modules (lean)

1. Formula / construction  
2. Tradability mask + neutralization  
3. IC one-liner  
4. Decile H-L + `decile_return` (`groupTest` metrics)  
5. G10 Excess Sharpe vs group Sharpes + gate  
6. ALL / HS300 / CSI500 / CSI1000  
7. Decay T+1/5/10/20  
8. Verdict vs gates  

Template: [`templates/factor_report_template.md`](templates/factor_report_template.md)  
Runner: `run_mentor_single_factor_protocol.py`

## Tier A library progress

| # | Factor | Status |
|---|--------|--------|
| 1 | **TGD20** | Mentor slim rebuild + optimize toward Excess > 3.5 (current SI ≈ 2.16) |
| 2–6 | Flow / Ideal* / APM / … | delivered (Batch 1); re-slim when TGD protocol locked |
| **7** | AmihudShockReversal5d | testing_candidate |
| 8+ | Batch 2+ | planned |

## SKEW note

SKEW reuses the same mentor protocol (flip to Alpha so G10 = long).  
Window grids are separate; do not mix SKEW Net Sharpe with TGD G10 Excess.

## Next

1. Finish TGD20 mentor tables + slim report  
2. MA / neutralization / MAD grid tracking **G10 Excess Sharpe**  
3. Only after single-factor gates: revisit factor matrix (≥100, corr < 0.7)

## Pause

Registry · Composite · Portfolio · Topology · textbook OHLCV sprint · combo / corr matrix.
