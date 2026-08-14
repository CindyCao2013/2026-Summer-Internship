# Checkpoint — C1 ActiveTrade / APM Track CLOSED

**Date:** 2026-07-21  
**Status:** CLOSED  
**Asset:** `APM_SessionResidual` Pack v1 · `testing_candidate`  
**Pack:** `research/reports/factors/APM_SessionResidual/`

---

## One-line verdict

> Paper 《主动买卖 / APM》 became a reusable Factor Asset: adapted replication, positive IC, frozen `daily|buffer_10_30` recipe (Net≈1.50) — **not** Registry.

---

## Pipeline completed

```text
Phase0 Identity          ✅ adapted_go
Phase1 Session Panel     ✅ PASS
Phase2 Constructability  ✅ PASS
Phase3 Scout             ✅ PASS_research_FAIL_invest
Phase4 Execution         ✅ CASE A → testing_candidate
Phase5 Pack v1           ✅ DONE
```

## Identity lock (final for C1)

```yaml
factor_id: APM_SessionResidual
identity_class: adapted_replication
status: testing_candidate
registry: false
execution_recipe: highAPM|daily|buffer_10_30 @15bp
direction: positive / long_high_apm / no_sign_flip
```

**Distinct from:** `ActiveTradeProxy` (proxy) · `SmartMoney10d` (parked) · `FlowDensity20` (orthogonal).

## Why not validated yet

- Index residual still EOD daytime proxy (not minute true)
- CSI1000 scout only; no ALL / longer OOS promote gate

## Library after C1

| Factor | Status |
|--------|--------|
| TGD20 | validated |
| D1 / Flow | candidate |
| APM_SessionResidual | **testing_candidate** |
| Ideal* | testing |
| SmartMoney10d | research_candidate parked |
| ActiveTradeProxy | testing proxy |

## Mainline next

**C2 SUE_ConsensusEPS Pack Track** — fundamental surprise diversity.  
SmartMoney slow recipe deferred (execution already answered).  
Still paused: Registry · Composite · Portfolio.
