# III-A4 — SmartMoney / APM Feasibility & Replication Design

**Date:** 2026-07-20  
**Status:** DESIGN ONLY — no implementation · no Registry · no proxy rename  
**Parent:** Phase III-A Microstructure Completion (closed)  
**Constraint:** No formula invention · provenance-controlled identities

---

## Purpose

Before coding:

1. Locate minute / active fields  
2. Map to **paper** Smart Money & APM definitions (repo Stage-0 reverse engineering)  
3. Lock **factor identity**  
4. Decide: **true replication** / **adapted replication** / **impossible**

---

## 1. Data inventory (located)

### Primary store

```
dfs://QV_Trade_to_MinuteBar / Stock_one_minute
```

**Coverage:** 2018-09-03 → 2026-07-17 (live; same path TGD uses)

### Minute columns (full)

| Column | Present |
|--------|:-------:|
| Symbol, Date, Bartime, Barstart, Barend | ✅ |
| Open, High, Low, Close, Adjfactor | ✅ |
| Volume, Amount | ✅ |
| Active_buy_volume / Active_sell_volume | ✅ |
| Active_buy_amount / Active_sell_amount | ✅ |
| Active_buy_count / Active_sell_count | ✅ |
| Bid/Ask cancel volume·count | ✅ |

### Related caches (already on disk)

```
research/cache/l2_daily/l2_daily_20181127_20251231.parquet
research/cache/l2_daily/l2_imbalance_duration_20181127_20251231.parquet
```

Useful for Flow / imbalance diagnostics — **not** substitutes for SmartMoney knife or paper APM session residuals.

### Index residual inputs

| Need | Availability |
|------|----------------|
| Stock overnight / afternoon returns | ✅ EOD open/close + stock minute Close |
| Index daily returns | ✅ `Factor_Dev_Lib.get_Ret_Matrix(..., base_index=...)` / Wind EOD |
| Index **minute** bars in same DFS table | ❌ not in `Stock_one_minute` |

→ APM index residualization must use **EOD index session proxies** or a separate index minute source (not located in this audit). Mark as **adapted** if paper requires index minute.

---

## 2. Paper definitions (from Stage-0 — do not invent)

Source of truth in-repo:

- `research/factor_cutting/paper_summary.md`  
- `research/factor_cutting/factor_definition.yaml`  
- `factor_cutting/smart_money.py` / `active_trade.py` stubs  

### 2.1 Smart Money（《聪明钱因子模型》）

| Element | Paper / Stage-0 definition |
|---------|----------------------------|
| Object | Minute VWAP construction |
| Knife | \(S_t = \|R_t\| / V_t^{0.25}\) |
| Method | Within lookback (10d), take minutes in top **20% cumulative volume** by smart score order |
| Output | \(Q = \mathrm{VWAP}_{smart} / \mathrm{VWAP}_{all}\) |
| Direction | negative IC (high Q → underperform) |
| Data | **Minute** Close + Volume (Amount optional for VWAP) |

**Active buy/sell fields are NOT required by this paper definition.**  
They are bonus diagnostics / future extensions — must not silently redefine SmartMoney as “active buy VWAP.”

### 2.2 APM / ActiveTrade（《APM / 主动买卖》）

| Element | Paper / Stage-0 definition |
|---------|----------------------------|
| Object | Overnight vs **afternoon (PM)** residual α |
| Knife | Time-of-day bucket (overnight/AM vs afternoon) |
| Output | Residualized APM statistic, then **CS residual vs Ret20** |
| Direction | positive IC |
| Data | Session/minute returns + **index residualization** |

### 2.3 Explicit non-identity

| ID | Is paper? |
|----|-----------|
| `ActiveTradeProxy` | ❌ daily overnight−day t-stat only |
| Future `APM_*` | Only if session residual + Ret20 residual path matches design below |
| Future `SmartMoney*` | Only if \(S_t\) + VWAP ratio path matches §2.1 |

---

## 3. Field ↔ requirement matrix

### Smart Money

| Requirement | Field / method | Gap? |
|-------------|----------------|------|
| Minute return \(R_t\) | `ratios(Close)-1` (as TGD) | ✅ none |
| Minute volume \(V_t\) | `Volume` | ✅ none |
| Smart score \(S_t\) | \(\|R\|/V^{0.25}\) | ✅ computable |
| Cumvol top 20% | rolling 10d minute stream | ✅ engineering only |
| VWAP smart / all | Amount or Close×Volume | ✅ Amount preferred |
| Active buy/sell | available but **unused by paper** | optional diagnostic |

**Verdict: TRUE REPLICATION — feasible**

### Paper APM

| Requirement | Field / method | Gap? |
|-------------|----------------|------|
| Overnight return | EOD `Open/prev_Close-1` | ✅ |
| Afternoon (PM) return | Minute Close 13:01–15:00 session return | ✅ stock side |
| Residual vs index (session) | Index overnight / PM return | ⚠ **no index minute in table**; EOD index open/close ≈ adapted |
| Rolling residual α / t-stat | stock − β·index (or subtract) | ✅ designable |
| CS residual vs Ret20 | EOD Ret20 panel | ✅ |
| Active buy/sell | not in Stage-0 APM formula | not required |

**Verdict: ADAPTED REPLICATION — feasible**  
(True *if* paper’s index residual is daily/session EOD-equivalent; **not proven** without original PDF session definition. Do not claim “true” until PDF checklist signed.)

---

## 4. Factor identity design (lock before code)

### Identity A — preferred first implementation

```yaml
factor_id: SmartMoney10d   # proposed — NOT registered yet
display_name: Smart Money 10d (Kaiyuan Q)
paper: 聪明钱因子模型
identity_class: true_replication_candidate
data_level: minute
formula_frozen: false until pack soft-bar review

object: minute VWAP construction
knife: S_t = abs(ret_1m) / volume_1m ** 0.25
output: VWAP_smart / VWAP_all   # top 20% cumvol by S within 10d

banned_aliases:
  - ActiveTradeProxy
  - using Active_buy_* as the knife without renaming
```

**Do not invent** alternate knives (e.g. active-buy-only VWAP) under this `factor_id`.  
If exploring active-buy VWAP → separate id e.g. `ActiveBuyVWAP_Research` (future), never `SmartMoney*`.

### Identity B — APM (second)

```yaml
factor_id: APM_SessionResidual   # proposed — NOT registered yet
display_name: APM Session Residual (adapted)
paper: APM因子模型 / 主动买卖
identity_class: adapted_replication   # until PDF confirms index session definition
data_level: minute_plus_eod_index

object: overnight_residual vs afternoon_residual
knife: time_of_day buckets
output: APM_stat then CS residual vs Ret20

distinct_from:
  ActiveTradeProxy: daily ON-DAY t-stat, no index residual, no Ret20 residual, no PM minute session
```

**Do not rename** `ActiveTradeProxy` → `APM`. Keep both if APM ships.

### Identity C — impossible / rejected

| Proposal | Decision |
|----------|----------|
| Promote ActiveTradeProxy to ActiveTrade / APM | **Rejected** — provenance break |
| Daily-only SmartMoney | **Impossible** — knife is minute |
| SmartMoney = L2 imbalance parquet alone | **Impossible** — wrong object |

---

## 5. Decision summary

| Factor | Decision | Rationale |
|--------|----------|-----------|
| **SmartMoney** | **True replication — GO** | Close + Volume (+ Amount) sufficient; matches Stage-0 formula |
| **Paper APM** | **Adapted replication — GO (after SmartMoney)** | Stock PM minute OK; index residual via EOD index sessions until index minute found |
| **ActiveTradeProxy** | **Remain testing proxy** | No promotion |
| **Active_* as SmartMoney** | **Forbidden** | Formula invention / identity theft |

### Recommended implementation order

```
1) SmartMoney10d   ← true replication, cleaner field match
2) APM_SessionResidual ← adapted; supersedes research use of proxy for "APM story"
3) Keep ActiveTradeProxy as labeled proxy (optional deprecate later)
```

### Engineering pattern (when coding starts — not this milestone)

Reuse TGD month-chunk DDB aggregate pattern:

```
Stock_one_minute
  → server-side daily features (smart VWAP ratio / session returns)
  → research/cache/...
  → CS eval → pack → Registry testing
```

No Python row-loop over full minute history without cache.

---

## 6. Explicit non-goals (this milestone)

- ❌ No `compute_smart_money` body  
- ❌ No Registry row  
- ❌ No status change for Ideal* / ActiveTradeProxy  
- ❌ No composite / portfolio  
- ❌ No SUE start in this doc’s scope  

---

## 7. Gate to start coding (III-A4.1)

Coding may start only when:

1. This design accepted  
2. Chosen identity is **SmartMoney10d** (recommended) or **APM_SessionResidual** with `adapted` label locked  
3. PDF checklist optional but preferred for APM “true” upgrade  
4. Eval window + cost (month chunks) agreed  

---

## Related

- III-A closure: `docs/milestone_3_0_iiia_closure.md`  
- Proxy honesty: `docs/milestone_3_0_active_trade_proxy.md`  
- Stage-0 papers: `research/factor_cutting/paper_summary.md`  
- Spec stubs: `factor_cutting/smart_money.py`, `factor_cutting/active_trade.py`
