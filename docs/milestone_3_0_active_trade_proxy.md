# Phase III A3 — ActiveTradeProxy

**Date:** 2026-07-20  
**Status:** PASS as `testing` / **research_proxy**  
**Script:** `run_milestone_3_0_active_trade_proxy.py`  
**Pack:** `research/reports/factors/ActiveTradeProxy/`

---

## Honesty gate (critical)

| Claim | Truth |
|-------|-------|
| Paper ActiveTrade / APM replication | ❌ **No** |
| What shipped | Daily `overnight − daytime` 20d t-stat proxy |
| Paper needs | Session/minute residual vs index + CS residual vs Ret20 |

Do **not** promote as paper replication.

---

## Results (last 252d)

| Metric | Value |
|--------|------:|
| RankICIR | 7.02 |
| Plain Net Sharpe | 0.95 |
| Mono | 0.78 |
| Exec best Net | **2.47** (`every_20d|buffer_10_30`) |
| Exec TO | ~0.13 |

Pack validation: Generator v2 `ok=true`.

---

## III-A microstructure map (after A3)

```
TGD20 validated
D1 candidate
FlowDensity enhancer
IdealReversal testing
IdealAmplitude testing
ActiveTradeProxy testing (proxy only)
SmartMoney stub (minute)
```

Daily cutting-family slots for Ideal* + proxy ActiveTrade are filled.  
**True APM / SmartMoney** wait on minute layer.

---

## Next

```
III-B SUE / Fundamental   ← recommended mainline
SmartMoney                ← when minute data ready
Portfolio v2              ← frozen
```
