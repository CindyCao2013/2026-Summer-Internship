# FlowDensity20 — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `FlowDensity20` |
| **display_name** | FlowDensity20 |
| **source** | Internal · L2 net active flow / mktcap |
| **family** | microstructure · liquidity |
| **data_level** | L2 / active flow |
| **status** | **candidate** |
| **formula** | **not frozen** |

---

## Economic intuition

FlowDensity20 is a **microstructure × liquidity interaction** signal: net active buy−sell flow scaled by market cap, smoothed over 20 days.

High net active flow in low-activity names can look like conviction; the same signed flow in high-amount names may just be churn. Amount-orthogonal tests show that stripping the activity channel destroys (and can reverse) the edge — hence **interaction, not pure flow**.

Catalogue as `liquidity_flow_interaction`, not pure smart-money flow.

---

## Factor card (one page)

```yaml
factor: FlowDensity20
name: FlowDensity20
source: L2 net active flow / mktcap (20d)
intuition: |
  Net active flow density predicts returns when conditioned on liquidity/activity.
  Not a pure directional flow factor.
signal: MA20( net_active_flow / mktcap ) → cs_zscore
direction: positive RankIC after size+industry (long high FlowDensity)
horizon: daily signal_shift=1; MA window=20
```

---

## Mechanism verdict (diagnostics — not sibling factors)

| Hypothesis | Result | Conclusion |
|------------|--------|------------|
| Pure flow direction is alpha | Flow⊥Amount ICIR ≈ −2.49 | fail as pure flow |
| Anti-activity channel dominates | Amount ICIR ≈ −8.6 | entangled with FlowDensity |
| Tradable after size+industry | ICIR ≈ 4.85 · H-L Sharpe ≈ 3.38 | accept as candidate interaction |

See `mechanism/` and `diagnostics/amount_orth_summary.md`.

---

## Do not

- Freeze as pure flow without amount-orth evidence  
- Equal-weight with TGD by default  
- Mine variant under same id without orthogonality review  

---

## Pack layout (v1 canonical)

```
research/reports/factors/FlowDensity20/
├── factor_definition.md   ← this file
├── formula.md
├── data_source.md
├── implementation.md
├── validation.md
├── summary.yaml
├── ic_analysis/
├── quantile_analysis/
├── stability/
└── execution/
```

Full narrative report (legacy): `factor_report.md`
