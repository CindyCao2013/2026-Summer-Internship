# APM_SessionResidual — Factor Definition

| Field | Value |
|-------|-------|
| **factor_id** | `APM_SessionResidual` |
| **display_name** | APM Session Residual |
| **source** | Paper adapted · 《APM / 主动买卖》 |
| **identity_class** | `adapted_replication` |
| **family** | session_behavior / microstructure |
| **data_level** | minute + EOD index |
| **status** | **testing_candidate** |
| **formula** | **frozen** (`apm_session_v1_adapted_eod_index`) |

---

## Economic intuition

Overnight vs afternoon (PM) **residual α** vs index captures differences in session trader mix / timing.  
Cross-sectional residual vs Ret20 removes medium-horizon return exposure.

**Answers:** does session-level residual behavior predict future returns?  
**Does not answer:** aggressive order-flow imbalance (`Active_*`) or daily overnight−day accrual timing alone (`ActiveTradeProxy`).

---

## Direction lock

```yaml
direction:
  paper: positive
  implementation: positive
  evaluation: long_high_apm
  sign_flip: false
```

Higher `apm_cs` → long. **Do not** multiply by −1.

---

## Factor card

```yaml
factor: APM_SessionResidual
signal: apm_cs
intuition: |
  Session residual α (overnight vs PM) after index adapter + Ret20 CS residual.
horizon: signal_shift=1; APM_stat window=20
execution_recipe: daily | buffer_10_30 @ 15bp
```

---

## Provenance guards

| Forbidden | Why |
|-----------|-----|
| Rename `ActiveTradeProxy` → APM | Provenance break |
| Use `Active_buy_*` under this id | Different alpha source |
| Claim `true_replication` | Index PM unmatched (EOD daytime proxy) |
| Silent sign flip | Paper orientation locked |

---

## Related docs

- Phase0: `docs/milestone_c1_activetrade_phase0_identity.md`
- Scout: `research/reports/apm_session_v1/scout/`
- Execution: `research/reports/apm_session_v1/execution/`
